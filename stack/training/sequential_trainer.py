"""
Sequential multi-agent PPO trainer.

This module implements the training pipeline:

    1. Collect rollout using frozen old policies.
    2. Agent B selects a bay.
    3. Agent R observes the selected bay and selects a row.
    4. Execute ONE physical StackEnv transition.
    5. Store the joint transition.
    6. Compute centralized-critic GAE.
    7. Update the centralized critic.
    8. Update Agent B with PPO.
    9. Precompute the detached Bay sequence ratio once for the whole
       rollout, by re-evaluating Agent B's (now updated) actions.
   10. Update Agent R with PPO using the corrected advantage.

Architecture
------------

    AgentB
        |
        └── BayPolicy theta_B

    AgentR
        |
        └── RowPolicy theta_R

    CentralizedCritic phi

Important
---------
There is still only ONE physical environment transition:

    Agent B chooses bay
        |
        | no env.step()
        v
    Agent R chooses row
        |
        v
    HierarchicalEnv.step(bay, row)
        |
        v
    StackEnv.step(global_action)

The Bay -> Row sequence-ratio correction is part of the new
HAPPO-style sequential training extension.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ..agents.agent_b import AgentB
from ..agents.agent_r import AgentR

from ..envs.hierarchical_envs.vec_hierarchical_env import (
    VecHierarchicalEnv,
)

from ..models.centralized_critic import (
    CentralizedCritic,
)

from .rollout_buffer import (
    JointRolloutBuffer,
    RolloutBatch,
)

from .advantage import (
    compute_and_store_gae,
)

from .profiling import (
    Profiler,
)


# ======================================================================
# Shared PPO statistics accumulator
# ======================================================================


class _PPOStatsAccumulator:
    """
    Collects detached per-minibatch GPU scalar tensors across one
    update_*() call and reduces each to a Python float ONCE at the end.

    Used identically by update_critic/update_bay_actor/update_row_actor
    instead of each hand-declaring its own List[torch.Tensor]
    accumulators. Deferring the CPU<->GPU sync (.item()) until every
    minibatch/epoch has finished means one blocking sync per tracked
    statistic for the WHOLE update call, instead of one per minibatch.

    log() always detaches, so callers don't need to remember whether the
    tensor already came from inside a `with torch.no_grad():` block.
    """

    def __init__(self) -> None:
        self._values: Dict[
            str,
            List[torch.Tensor],
        ] = {}

    def log(
        self,
        name: str,
        value: torch.Tensor,
    ) -> None:

        self._values.setdefault(
            name,
            [],
        ).append(
            value.detach()
        )

    def means(self) -> Dict[str, float]:

        return {
            name: (
                float(
                    torch.stack(values)
                    .mean()
                    .item()
                )
                if values
                else float("nan")
            )
            for name, values in self._values.items()
        }


class SequentialPPOTrainer:
    """
    Sequential PPO trainer for:

        Agent B -> Agent R

    using one centralized state-value critic.

    Parameters
    ----------
    agent_b : AgentB
        Bay-selection agent.

    agent_r : AgentR
        Row-selection agent.

    critic : CentralizedCritic
        Centralized state-value critic V_phi(s).

    learning_rate : float
        Learning rate for all three optimizers.

    n_epochs : int
        PPO epochs per rollout.

    batch_size : int
        PPO minibatch size.

    gamma : float
        Discount factor.

    gae_lambda : float
        GAE lambda.

    clip_range : float
        PPO clipping epsilon.

    ent_coef : float
        Entropy-loss coefficient.

    vf_coef : float
        Value-loss coefficient.

    max_grad_norm : float
        Maximum gradient norm.

    normalize_advantage : bool
        Whether to normalize the base advantage before actor updates.

    num_envs : int
        Number of parallel environments collect_rollout() expects to be
        stepped together each iteration (see VecHierarchicalEnv). All
        rollout arrays/tensors are batched by this size.

    device : str | torch.device
        Training device.

    profiler : Profiler | None
        Optional wall-clock phase profiler (see profiling.py). When
        None, a disabled (zero-overhead) Profiler is used.
        train_iteration() always returns the same fixed set of timing
        keys (defaulting to 0.0 for any phase that didn't run or
        wasn't measured) regardless of whether profiling is enabled,
        so downstream CSV logging's column set never depends on this.
    """

    def __init__(
        self,
        agent_b: AgentB,
        agent_r: AgentR,
        critic: CentralizedCritic,
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.15,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        profiler: Optional[Profiler] = None,
    ) -> None:

        # ==============================================================
        # Validation
        # ==============================================================

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if num_envs <= 0:
            raise ValueError(
                "num_envs must be positive."
            )

        if n_epochs <= 0:
            raise ValueError(
                "n_epochs must be positive."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                "gamma must be in [0, 1]."
            )

        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must be in [0, 1]."
            )

        if clip_range <= 0:
            raise ValueError(
                "clip_range must be positive."
            )

        if max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive."
            )

        # ==============================================================
        # Device
        # ==============================================================

        self.device = torch.device(
            device
        )

        # ==============================================================
        # Agents + centralized critic
        # ==============================================================

        self.agent_b = agent_b.to(
            self.device
        )

        self.agent_r = agent_r.to(
            self.device
        )

        self.critic = critic.to(
            self.device
        )

        # ==============================================================
        # Hyperparameters
        # ==============================================================

        self.learning_rate = float(
            learning_rate
        )

        self.n_epochs = int(
            n_epochs
        )

        self.batch_size = int(
            batch_size
        )

        self.gamma = float(
            gamma
        )

        self.gae_lambda = float(
            gae_lambda
        )

        self.clip_range = float(
            clip_range
        )

        self.ent_coef = float(
            ent_coef
        )

        self.vf_coef = float(
            vf_coef
        )

        self.max_grad_norm = float(
            max_grad_norm
        )

        self.normalize_advantage = bool(
            normalize_advantage
        )

        self.num_envs = int(
            num_envs
        )

        self.profiler = (
            profiler
            if profiler is not None
            else Profiler(enabled=False)
        )

        # ==============================================================
        # Independent optimizers
        # ==============================================================
        #
        # Agent B owns theta_B
        # Agent R owns theta_R
        # Critic owns phi
        #
        # These parameter sets are independent.
        # ==============================================================

        self.bay_optimizer = torch.optim.Adam(
            self.agent_b.parameters(),
            lr=self.learning_rate,
        )

        self.row_optimizer = torch.optim.Adam(
            self.agent_r.parameters(),
            lr=self.learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
        )

        # ==============================================================
        # Persistent rollout state
        # ==============================================================
        #
        # A rollout buffer may become full in the middle of an episode.
        #
        # We therefore remember the current environment observation
        # instead of automatically resetting after every PPO update.
        #
        # _current_bay_observation holds shape (num_envs, obs_dim) once
        # rollout collection has started; episode return/length are
        # tracked per env-index since each parallel env finishes
        # episodes at different times.
        # ==============================================================

        self._current_bay_observation: Optional[
            np.ndarray
        ] = None

        # Cached V_old(s_t) carried forward from the previous rollout
        # step's V_old(s_{t+1}) forward pass (see collect_rollout()).
        # Always None at the start of a rollout - collect_rollout()
        # itself invalidates it at the top of every call, since
        # update_critic() changes self.critic's parameters between
        # rollouts and a cached value must never survive that boundary.
        self._current_value_tensor: Optional[
            torch.Tensor
        ] = None

        self._current_episode_return = np.zeros(
            self.num_envs,
            dtype=np.float64,
        )

        self._current_episode_length = np.zeros(
            self.num_envs,
            dtype=np.int64,
        )

        self.total_environment_steps = 0
        self.total_episodes = 0

    # ==================================================================
    # Rollout state reset
    # ==================================================================

    def reset_rollout_state(
        self,
    ) -> None:
        """
        Force the next rollout collection to start with env.reset().

        This should be called when switching to a new environment
        instance or manually restarting training state.
        """

        self._current_bay_observation = None
        self._current_value_tensor = None

        self._current_episode_return = np.zeros(
            self.num_envs,
            dtype=np.float64,
        )

        self._current_episode_length = np.zeros(
            self.num_envs,
            dtype=np.int64,
        )

    # ==================================================================
    # Durable training state
    # ==================================================================

    def training_state_dict(
        self,
    ) -> Dict[str, object]:
        """Return everything needed to continue after a process restart.

        The in-progress environment episode and rollout buffer are deliberately
        not serialized.  Checkpoints are written only after a complete PPO
        update, so restoration starts a new environment episode with the exact
        updated networks, optimizer moments and global counters.
        """

        return {
            "agent_b_state_dict": self.agent_b.state_dict(),
            "agent_r_state_dict": self.agent_r.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "bay_optimizer_state_dict": self.bay_optimizer.state_dict(),
            "row_optimizer_state_dict": self.row_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_environment_steps": int(self.total_environment_steps),
            "total_episodes": int(self.total_episodes),
        }

    def load_training_state_dict(
        self,
        state: Dict[str, object],
    ) -> None:
        """Restore networks, optimizers and counters from a checkpoint."""

        required_keys = {
            "agent_b_state_dict",
            "agent_r_state_dict",
            "critic_state_dict",
            "bay_optimizer_state_dict",
            "row_optimizer_state_dict",
            "critic_optimizer_state_dict",
            "total_environment_steps",
            "total_episodes",
        }
        missing_keys = sorted(required_keys.difference(state))
        if missing_keys:
            raise ValueError(
                "Training checkpoint is incomplete; missing: "
                + ", ".join(missing_keys)
            )

        self.agent_b.load_state_dict(state["agent_b_state_dict"])
        self.agent_r.load_state_dict(state["agent_r_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])

        self.bay_optimizer.load_state_dict(
            state["bay_optimizer_state_dict"]
        )
        self.row_optimizer.load_state_dict(
            state["row_optimizer_state_dict"]
        )
        self.critic_optimizer.load_state_dict(
            state["critic_optimizer_state_dict"]
        )

        total_environment_steps = int(state["total_environment_steps"])
        total_episodes = int(state["total_episodes"])
        if total_environment_steps < 0 or total_episodes < 0:
            raise ValueError("Checkpoint counters must be nonnegative.")

        self.total_environment_steps = total_environment_steps
        self.total_episodes = total_episodes

        # A newly constructed environment cannot continue an in-memory partial
        # episode.  The next rollout therefore begins with env.reset().
        self.reset_rollout_state()

    # ==================================================================
    # Tensor conversion helpers
    # ==================================================================

    def _obs_tensor(
        self,
        array: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert an observation/state array to float tensor.
        """

        return torch.as_tensor(
            array,
            dtype=torch.float32,
            device=self.device,
        )

    def _mask_tensor(
        self,
        array: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert an action mask to bool tensor.
        """

        return torch.as_tensor(
            array,
            dtype=torch.bool,
            device=self.device,
        )

    # ==================================================================
    # Advantage normalization
    # ==================================================================

    def _normalize_advantage_tensor(
        self,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize the BASE GAE advantage.

        Bay update:

            A_B = normalized(A)

        Row update:

            A_R =
                M_B
                *
                normalized(A)

        We intentionally do not normalize again after multiplying by
        M_B because M_B is the sequential correction.
        """

        if not self.normalize_advantage:
            return advantages

        # Standard deviation is not useful for a one-element batch.
        if advantages.numel() <= 1:
            return advantages

        return (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

    # ==================================================================
    # Rollout collection
    # ==================================================================

    def collect_rollout(
        self,
        env: VecHierarchicalEnv,
        buffer: JointRolloutBuffer,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, float]:
        """
        Fill one JointRolloutBuffer using self.num_envs parallel
        environments stepped together.

        ``env`` must expose the VecHierarchicalEnv interface: reset(),
        get_global_state(), get_bay_action_mask(), get_row_decision_input(),
        and step() all operating on batches of shape (num_envs, ...).
        VecHierarchicalEnv auto-resets any environment that finishes
        internally, so this method never needs to call env.reset() again
        once rollout collection has started.

        Decision sequence (batched across all num_envs environments)
        -----------------

            global state s_t
                    |
                    v
                Agent B
                    |
                    v
                bay_idx
                    |
                    | NO env.step()
                    v
            selected-bay observation
                    |
                    v
                Agent R
                    |
                    v
                row_idx
                    |
                    v
            HierarchicalEnv.step(...)
                    |
                    v
              ONE StackEnv.step(...) per environment
                    |
                    v
            reward, s_{t+1}

        During rollout no model parameters are updated. ``step_callback`` is
        invoked after each complete physical transition with the cumulative
        training-step count. It is used for exact periodic evaluation without
        ending or shortening the PPO rollout.
        """

        if buffer.device != self.device:
            raise ValueError(
                "Rollout buffer device does not match trainer device. "
                f"buffer={buffer.device}, "
                f"trainer={self.device}."
            )

        buffer.reset()

        # Unconditionally invalidate the cached V_old(s_t) from
        # whatever previous rollout last set it. Every new rollout's
        # first step must do a real critic forward pass: self.critic's
        # parameters may have changed (update_critic()) since that
        # cache was written, and a cached value must never be reused
        # across that boundary. Placed here, at the top of
        # collect_rollout() itself (like buffer.reset() right above),
        # rather than only in train_iteration()'s caller, so this holds
        # for every collect_rollout() call, not just ones that happen
        # to go through train_iteration().
        self._current_value_tensor = None

        # ==============================================================
        # Freeze networks during rollout
        # ==============================================================

        self.agent_b.eval()
        self.agent_r.eval()
        self.critic.eval()

        num_envs = self.num_envs

        # ==============================================================
        # Reset environment only when necessary
        # ==============================================================

        if self._current_bay_observation is None:

            self._current_bay_observation = np.asarray(
                env.reset(),
                dtype=np.float32,
            )

        completed_episode_returns: List[
            float
        ] = []

        completed_episode_lengths: List[
            int
        ] = []

        rollout_reward = 0.0

        # ==============================================================
        # Collect transitions
        # ==============================================================

        while not buffer.is_full():

            # ----------------------------------------------------------
            # Current centralized state
            # ----------------------------------------------------------

            global_state = np.asarray(
                env.get_global_state(),
                dtype=np.float32,
            )

            # ----------------------------------------------------------
            # Agent B observation
            # ----------------------------------------------------------
            #
            # Currently this is the same original global observation,
            # but we keep separate semantic names.
            # ----------------------------------------------------------

            bay_observation = np.asarray(
                self._current_bay_observation,
                dtype=np.float32,
            )

            bay_action_mask = np.asarray(
                env.get_bay_action_mask(),
                dtype=np.bool_,
            )

            if not bay_action_mask.any(axis=-1).all():
                raise RuntimeError(
                    "No valid bay exists at the beginning of "
                    "a hierarchical decision for at least one "
                    "environment."
                )

            # ----------------------------------------------------------
            # Convert to tensors
            # ----------------------------------------------------------
            #
            # global_state_tensor is NOT built here - it is only needed
            # by the critic cache-miss branch below, and building it
            # unconditionally on every step (including cache hits,
            # where it would never be read) would reintroduce exactly
            # the kind of per-step work this phase removes.
            # ----------------------------------------------------------

            bay_obs_tensor = self._obs_tensor(
                bay_observation
            )

            bay_mask_tensor = self._mask_tensor(
                bay_action_mask
            )

            # ==========================================================
            # OLD centralized value V_old(s_t)
            # ==========================================================
            #
            # The previous iteration of this while-loop already computed
            # V_old(s_{t+1}) from next_global_state (below), and this
            # step's global_state IS that same next_global_state (see
            # self._current_bay_observation, which get_global_state()
            # now returns unchanged). Reuse that tensor instead of
            # running the critic forward pass again on an identical
            # input. self._current_value_tensor is invalidated (set to
            # None) at the top of every collect_rollout() call, so it
            # can never survive across an update_critic() call that
            # changed self.critic's parameters - the very first step of
            # every rollout always recomputes for real.
            # ==========================================================

            if self._current_value_tensor is not None:
                value_tensor = self._current_value_tensor
            else:
                with self.profiler.region(
                    "critic_inference_seconds",
                    sync_cuda=True,
                ), torch.no_grad():

                    value_tensor = self.critic.predict_values(
                        self._obs_tensor(
                            global_state
                        )
                    )

            # ==========================================================
            # Agent B selects bay
            # ==========================================================

            with self.profiler.region(
                "bay_inference_seconds",
                sync_cuda=True,
            ), torch.no_grad():

                (
                    bay_action_tensor,
                    bay_log_prob_tensor,
                ) = self.agent_b.act(
                    observations=(
                        bay_obs_tensor
                    ),
                    action_masks=(
                        bay_mask_tensor
                    ),
                    deterministic=False,
                )

            # Single CPU<->GPU sync for the whole batch of num_envs
            # actions, instead of one .item() call per environment.
            bay_actions = (
                bay_action_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            # ==========================================================
            # Agent R gets selected-bay observation
            # ==========================================================
            #
            # IMPORTANT:
            #
            # Environment has NOT transitioned yet.
            # ==========================================================

            (
                row_observation,
                row_action_mask,
            ) = env.get_row_decision_input(
                bay_actions
            )

            row_observation = np.asarray(
                row_observation,
                dtype=np.float32,
            )

            row_action_mask = np.asarray(
                row_action_mask,
                dtype=np.bool_,
            )

            if not row_action_mask.any(axis=-1).all():
                raise RuntimeError(
                    "Agent B selected a bay with no valid row for at "
                    "least one environment. Bay mask and Row mask are "
                    "inconsistent."
                )

            row_obs_tensor = self._obs_tensor(
                row_observation
            )

            row_mask_tensor = self._mask_tensor(
                row_action_mask
            )

            # ==========================================================
            # Agent R selects row
            # ==========================================================

            with self.profiler.region(
                "row_inference_seconds",
                sync_cuda=True,
            ), torch.no_grad():

                (
                    row_action_tensor,
                    row_log_prob_tensor,
                ) = self.agent_r.act(
                    observations=(
                        row_obs_tensor
                    ),
                    action_masks=(
                        row_mask_tensor
                    ),
                    deterministic=False,
                )

            row_actions = (
                row_action_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            # ==========================================================
            # ONE physical environment transition PER environment
            # ==========================================================
            #
            # Pure CPU/NumPy work (env.step() takes/returns NumPy
            # arrays, no torch tensors cross this boundary), so no CUDA
            # sync is needed for this timer to be accurate.
            # ==========================================================

            with self.profiler.region(
                "env_step_seconds"
            ):

                (
                    next_global_state,
                    rewards,
                    terminated,
                    truncated,
                    _infos,
                ) = env.step(
                    bay_actions,
                    row_actions,
                )

            next_global_state = np.asarray(
                next_global_state,
                dtype=np.float32,
            )

            rewards = np.asarray(
                rewards,
                dtype=np.float64,
            )

            terminated = np.asarray(
                terminated,
                dtype=bool,
            )

            truncated = np.asarray(
                truncated,
                dtype=bool,
            )

            episode_done = (
                terminated
                | truncated
            )

            # ==========================================================
            # Bootstrap V_old(s_{t+1})
            # ==========================================================
            #
            # It is cheaper to batch-evaluate the critic for every
            # environment (even the ones that just finished) than to
            # branch per environment, then zero out the finished ones.
            # advantage.py treats both terminated and truncated as
            # episode boundaries.
            # ==========================================================

            with self.profiler.region(
                "critic_inference_seconds",
                sync_cuda=True,
            ), torch.no_grad():

                next_value_tensor = (
                    self.critic.predict_values(
                        self._obs_tensor(
                            next_global_state
                        )
                    )
                )

            # Cache the RAW (un-zeroed) value for reuse as next
            # iteration's V_old(s_t) above. Even for an env slot whose
            # episode just ended, next_global_state is already
            # VecHierarchicalEnv's auto-reset observation for the new
            # episode (see collect_rollout()'s docstring reference to
            # VecHierarchicalEnv), so this is a legitimate value
            # estimate to reuse - the buffer-stored copy below is
            # zeroed separately for GAE bootstrap; this cached tensor
            # is not.
            self._current_value_tensor = next_value_tensor

            values_np = (
                value_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            next_values_np = (
                next_value_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            next_values_np = np.where(
                episode_done,
                0.0,
                next_values_np,
            )

            bay_log_probs_np = (
                bay_log_prob_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            row_log_probs_np = (
                row_log_prob_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            # ==========================================================
            # Store one complete joint transition PER environment
            # ==========================================================
            #
            # One vectorized call instead of a Python loop calling add()
            # num_envs times. write_n is the number of rows actually
            # stored - equal to num_envs unless the buffer had less than
            # num_envs slots of remaining capacity, in which case only
            # the first write_n environments' transitions are stored,
            # exactly as the previous `if buffer.is_full(): break` loop
            # would have stopped after write_n additions.
            # ==========================================================

            write_n = buffer.add_batch(

                # centralized critic
                global_states=global_state,

                # Agent B
                bay_observations=bay_observation,
                bay_action_masks=bay_action_mask,
                bay_actions=bay_actions,
                bay_log_probs=bay_log_probs_np,

                # Agent R
                row_observations=row_observation,
                row_action_masks=row_action_mask,
                row_actions=row_actions,
                row_log_probs=row_log_probs_np,

                # environment
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,

                # critic
                values=values_np,
                next_values=next_values_np,
            )

            for i in range(write_n):

                # ------------------------------------------------------
                # Statistics
                # ------------------------------------------------------

                self.total_environment_steps += 1

                self._current_episode_return[i] += float(
                    rewards[i]
                )

                self._current_episode_length[i] += 1

                rollout_reward += float(
                    rewards[i]
                )

                if episode_done[i]:

                    completed_episode_returns.append(
                        float(
                            self._current_episode_return[i]
                        )
                    )

                    completed_episode_lengths.append(
                        int(
                            self._current_episode_length[i]
                        )
                    )

                    self.total_episodes += 1

                    self._current_episode_return[i] = 0.0
                    self._current_episode_length[i] = 0

                # Aritra's SB3 callback evaluates at exact environment-step
                # thresholds, even when a threshold falls inside a PPO
                # rollout. The callback uses a separate environment and
                # must not mutate this rollout buffer or counter.
                if step_callback is not None:
                    step_callback(
                        self.total_environment_steps
                    )

            # --------------------------------------------------------------
            # Continue every environment next iteration.
            #
            # VecHierarchicalEnv already auto-resets any environment that
            # terminated/truncated this step, so next_global_state is
            # always a valid observation to continue from for every index.
            # --------------------------------------------------------------

            self._current_bay_observation = (
                next_global_state
            )

        # ==============================================================
        # Rollout statistics
        # ==============================================================

        return {

            "rollout_steps": float(
                len(buffer)
            ),

            "rollout_reward_sum": float(
                rollout_reward
            ),

            "episodes_completed": float(
                len(
                    completed_episode_returns
                )
            ),

            "mean_episode_reward": (
                float(
                    np.mean(
                        completed_episode_returns
                    )
                )
                if completed_episode_returns
                else float("nan")
            ),

            "mean_episode_length": (
                float(
                    np.mean(
                        completed_episode_lengths
                    )
                )
                if completed_episode_lengths
                else float("nan")
            ),
        }

    # ==================================================================
    # Centralized critic update
    # ==================================================================

    def update_critic(
        self,
        buffer: JointRolloutBuffer,
    ) -> Dict[str, float]:
        """
        Update centralized state-value critic.

        Target:

            return_t
                =
            advantage_t
                +
            V_old(s_t)

        Loss:

            MSE(
                V_phi(s_t),
                return_t
            )

        Only critic parameters phi are updated.
        """

        self.critic.train()

        stats = _PPOStatsAccumulator()

        for _epoch in range(
            self.n_epochs
        ):

            for batch in buffer.get_batches(
                batch_size=self.batch_size,
                shuffle=True,
            ):

                predicted_values = (
                    self.critic(
                        batch.global_states
                    )
                )

                value_loss = F.mse_loss(
                    predicted_values,
                    batch.returns,
                )

                critic_loss = (
                    self.vf_coef
                    * value_loss
                )

                # ------------------------------------------------------
                # Critic optimization
                # ------------------------------------------------------

                self.critic_optimizer.zero_grad()

                critic_loss.backward()

                clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

                self.critic_optimizer.step()

                stats.log(
                    "critic_loss",
                    value_loss,
                )

        self.critic.eval()

        return stats.means()

    # ==================================================================
    # Agent B PPO update
    # ==================================================================

    def update_bay_actor(
        self,
        buffer: JointRolloutBuffer,
    ) -> Dict[str, float]:
        """
        FIRST actor update.

        Agent B uses the base advantage:

            A_B = A_t

        Bay PPO ratio:

            ratio_B
                =
            exp(
                log pi_B,new
                -
                log pi_B,old
            )

        PPO objective:

            min(
                ratio_B * A_B,
                clip(ratio_B) * A_B
            )
        """

        self.agent_b.train()

        stats = _PPOStatsAccumulator()

        # ==============================================================
        # PPO epochs
        # ==============================================================

        for _epoch in range(
            self.n_epochs
        ):

            for batch in buffer.get_batches(
                batch_size=self.batch_size,
                shuffle=True,
            ):

                # ======================================================
                # Base GAE advantage
                # ======================================================

                advantages = (
                    self._normalize_advantage_tensor(
                        batch.advantages
                    )
                )

                # ======================================================
                # Current Agent B policy
                # ======================================================

                (
                    new_log_probs,
                    entropy,
                ) = self.agent_b.evaluate_actions(
                    observations=(
                        batch.bay_observations
                    ),

                    actions=(
                        batch.bay_actions
                    ),

                    action_masks=(
                        batch.bay_action_masks
                    ),
                )

                # ======================================================
                # PPO ratio
                # ======================================================

                log_ratio = (
                    new_log_probs
                    - batch.old_bay_log_probs
                )

                ratio = torch.exp(
                    log_ratio
                )

                # ======================================================
                # PPO clipped surrogate
                # ======================================================

                surrogate_1 = (
                    ratio
                    * advantages
                )

                surrogate_2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    )
                    * advantages
                )

                policy_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                # PPO entropy bonus:
                #
                # total loss =
                #
                # policy_loss
                # -
                # ent_coef * entropy
                #
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ent_coef
                    * entropy_loss
                )

                # ======================================================
                # Optimize Agent B ONLY
                # ======================================================

                self.bay_optimizer.zero_grad()

                loss.backward()

                clip_grad_norm_(
                    self.agent_b.parameters(),
                    self.max_grad_norm,
                )

                self.bay_optimizer.step()

                # ======================================================
                # Statistics
                # ======================================================

                with torch.no_grad():

                    clip_fraction = (
                        (
                            torch.abs(
                                ratio - 1.0
                            )
                            > self.clip_range
                        )
                        .float()
                        .mean()
                    )

                    approx_kl = (
                        (
                            torch.exp(
                                log_ratio
                            )
                            - 1.0
                            - log_ratio
                        )
                        .mean()
                    )

                stats.log(
                    "bay_policy_loss",
                    policy_loss,
                )

                stats.log(
                    "bay_entropy",
                    entropy.mean(),
                )

                stats.log(
                    "bay_clip_fraction",
                    clip_fraction,
                )

                stats.log(
                    "bay_approx_kl",
                    approx_kl,
                )

        # ==============================================================
        # Bay update is FINISHED.
        #
        # Agent B must now remain fixed while Agent R is updated.
        # ==============================================================

        self.agent_b.eval()

        return stats.means()

    # ==================================================================
    # Bay sequential correction ratio
    # ==================================================================

    @torch.no_grad()
    def _compute_bay_sequence_ratio(
        self,
        batch: RolloutBatch,
    ) -> torch.Tensor:
        """
        Compute the Bay correction factor after Agent B has been updated.

        M_B
            =
        pi_B,updated(
            b_t | o_B,t
        )
        ------------------
        pi_B,old(
            b_t | o_B,t
        )

        In log-probability form:

            M_B
                =
            exp(
                log_pi_B_updated
                -
                log_pi_B_old
            )

        IMPORTANT
        ---------
        This ratio is detached from the computation graph.

        Agent R's backward pass must NEVER modify Agent B.
        """

        (
            updated_bay_log_probs,
            _entropy,
        ) = self.agent_b.evaluate_actions(
            observations=(
                batch.bay_observations
            ),

            actions=(
                batch.bay_actions
            ),

            action_masks=(
                batch.bay_action_masks
            ),
        )

        sequence_ratio = torch.exp(
            updated_bay_log_probs
            - batch.old_bay_log_probs
        )

        # --------------------------------------------------------------
        # Critical HAPPO-style sequential separation.
        # --------------------------------------------------------------

        return sequence_ratio.detach()

    # ==================================================================
    # Bay sequential correction ratio - precomputed once
    # ==================================================================

    def precompute_bay_correction(
        self,
        buffer: JointRolloutBuffer,
    ) -> None:
        """
        Compute the Bay correction ratio for EVERY sample in the
        rollout exactly once, and cache it on the buffer for
        update_row_actor() to reuse across all of its PPO
        epochs/minibatches.

        Must be called after update_bay_actor() (so Agent B already
        reflects its update) and before update_row_actor().

        Why this is exact, not an approximation
        ----------------------------------------
        _compute_bay_sequence_ratio() is @torch.no_grad() and Agent B
        is frozen (self.agent_b.eval()) for the entire duration of
        update_row_actor(). A given sample's ratio therefore depends
        only on (the now-fixed updated Agent B, that sample's
        bay_observation/bay_action/old_bay_log_prob) - never on which
        minibatch it happens to be shuffled into, or which epoch. So
        computing it once here, in original buffer order, and gathering
        it per minibatch by index (get_row_batches()) is mathematically
        identical to calling _compute_bay_sequence_ratio() fresh inside
        every minibatch - exactly like old_bay_log_probs/advantages,
        which are likewise computed once and gathered per minibatch
        rather than recomputed.
        """

        self.agent_b.eval()

        full_batch = next(
            buffer.get_batches(
                batch_size=buffer.size,
                shuffle=False,
            )
        )

        ratio = self._compute_bay_sequence_ratio(
            full_batch
        )

        buffer.set_bay_correction_ratio(
            ratio
        )

    # ==================================================================
    # Agent R PPO update
    # ==================================================================

    def update_row_actor(
        self,
        buffer: JointRolloutBuffer,
    ) -> Dict[str, float]:
        """
        SECOND actor update.

        Agent B has already been updated and is now fixed.

        The corrected Agent R advantage is:

            A_R
                =
            M_B * A_t

        where:

            M_B
                =
            pi_B,updated
            /
            pi_B,old

        and M_B is detached.

        Agent R still has its OWN PPO ratio:

            ratio_R
                =
            pi_R,new
            /
            pi_R,old

        Therefore:

            L_R
                =
            min(
                ratio_R * A_R,
                clip(ratio_R) * A_R
            )
        """

        # ==============================================================
        # Agent B MUST stay frozen.
        # ==============================================================

        self.agent_b.eval()

        self.agent_r.train()

        stats = _PPOStatsAccumulator()

        # ==============================================================
        # PPO epochs
        # ==============================================================

        for _epoch in range(
            self.n_epochs
        ):

            for (
                batch,
                sequence_ratio,
            ) in buffer.get_row_batches(
                batch_size=self.batch_size,
                shuffle=True,
            ):

                # ======================================================
                # Base team advantage
                # ======================================================

                base_advantages = (
                    self._normalize_advantage_tensor(
                        batch.advantages
                    )
                )

                # ======================================================
                # UPDATED Agent B / OLD Agent B ratio
                # ======================================================
                #
                # Precomputed once for the whole rollout by
                # precompute_bay_correction() (called between
                # update_bay_actor() and update_row_actor() in
                # train_iteration()) instead of being recomputed via a
                # fresh Agent B forward pass in every minibatch of every
                # epoch - see precompute_bay_correction()'s docstring
                # for why this is exact, not an approximation.
                # ======================================================

                # ======================================================
                # Sequential Agent R advantage
                # ======================================================

                row_advantages = (
                    sequence_ratio
                    * base_advantages
                )

                # ======================================================
                # Current Agent R policy
                # ======================================================

                (
                    new_log_probs,
                    entropy,
                ) = self.agent_r.evaluate_actions(
                    observations=(
                        batch.row_observations
                    ),

                    actions=(
                        batch.row_actions
                    ),

                    action_masks=(
                        batch.row_action_masks
                    ),
                )

                # ======================================================
                # Agent R's OWN PPO ratio
                # ======================================================

                log_ratio = (
                    new_log_probs
                    - batch.old_row_log_probs
                )

                ratio = torch.exp(
                    log_ratio
                )

                # ======================================================
                # PPO clipped objective for Agent R
                # ======================================================

                surrogate_1 = (
                    ratio
                    * row_advantages
                )

                surrogate_2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    )
                    * row_advantages
                )

                policy_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ent_coef
                    * entropy_loss
                )

                # ======================================================
                # Optimize Agent R ONLY
                # ======================================================

                self.row_optimizer.zero_grad()

                loss.backward()

                clip_grad_norm_(
                    self.agent_r.parameters(),
                    self.max_grad_norm,
                )

                self.row_optimizer.step()

                # ======================================================
                # Statistics
                # ======================================================

                with torch.no_grad():

                    clip_fraction = (
                        (
                            torch.abs(
                                ratio - 1.0
                            )
                            > self.clip_range
                        )
                        .float()
                        .mean()
                    )

                    approx_kl = (
                        (
                            torch.exp(
                                log_ratio
                            )
                            - 1.0
                            - log_ratio
                        )
                        .mean()
                    )

                    sequence_ratio_mean = (
                        sequence_ratio.mean()
                    )

                    sequence_ratio_std = (
                        sequence_ratio.std(
                            unbiased=False
                        )
                    )

                stats.log(
                    "row_policy_loss",
                    policy_loss,
                )

                stats.log(
                    "row_entropy",
                    entropy.mean(),
                )

                stats.log(
                    "row_clip_fraction",
                    clip_fraction,
                )

                stats.log(
                    "row_approx_kl",
                    approx_kl,
                )

                stats.log(
                    "bay_sequence_ratio_mean",
                    sequence_ratio_mean,
                )

                stats.log(
                    "bay_sequence_ratio_std",
                    sequence_ratio_std,
                )

        self.agent_r.eval()

        return stats.means()

    # ==================================================================
    # One complete training iteration
    # ==================================================================

    def train_iteration(
        self,
        env: VecHierarchicalEnv,
        buffer: JointRolloutBuffer,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, float]:
        """
        Perform ONE complete sequential PPO training iteration.

        Exact order
        -----------

            1. Collect rollout
            2. Compute GAE
            3. Update centralized critic
            4. Update Agent B
            5. Compute updated-Bay / old-Bay ratio
            6. Update Agent R
        """

        # ==============================================================
        # 1. Rollout using OLD policies
        # ==============================================================
        #
        # collect_rollout() itself invalidates the cached V_old(s_t)
        # from any previous rollout before it starts - see its own
        # docstring/comments - so this call always begins with a real
        # critic forward pass regardless of update_critic() (phase 3
        # below) having changed self.critic's parameters since.
        # ==============================================================

        # Note: step_callback (evaluate_if_due in run.py) can fire
        # synchronously from inside collect_rollout(), and its own
        # "evaluation_seconds" region uses this same self.profiler
        # instance - so on an iteration where evaluation runs,
        # "rollout_seconds" below includes that time too (they are not
        # mutually exclusive; "evaluation_seconds" is broken out
        # separately so it's still visible, not to be subtracted from
        # "rollout_seconds").
        with self.profiler.region(
            "rollout_seconds",
            sync_cuda=True,
        ):
            rollout_stats = (
                self.collect_rollout(
                    env=env,
                    buffer=buffer,
                    step_callback=step_callback,
                )
            )

        # ==============================================================
        # 2. GAE based on OLD critic predictions
        # ==============================================================

        (
            advantages,
            returns,
        ) = compute_and_store_gae(

            buffer=buffer,

            gamma=self.gamma,

            gae_lambda=self.gae_lambda,

            num_envs=self.num_envs,
        )

        gae_stats = {

            "advantage_mean":
                float(
                    advantages.mean()
                ),

            "advantage_std":
                float(
                    advantages.std()
                ),

            "return_mean":
                float(
                    returns.mean()
                ),
        }

        # ==============================================================
        # 3. Update centralized critic
        # ==============================================================

        with self.profiler.region(
            "critic_update_seconds",
            sync_cuda=True,
        ):
            critic_stats = (
                self.update_critic(
                    buffer
                )
            )

        # ==============================================================
        # 4. Agent B PPO update FIRST
        # ==============================================================

        with self.profiler.region(
            "bay_update_seconds",
            sync_cuda=True,
        ):
            bay_stats = (
                self.update_bay_actor(
                    buffer
                )
            )

        # ==============================================================
        # 5. Precompute the Bay sequence ratio for the WHOLE rollout
        #    exactly once, now that Agent B's update above is final.
        #
        #    update_row_actor() previously recomputed this via a fresh
        #    Agent B forward pass inside every one of its own
        #    n_epochs * n_minibatches minibatches, even though Agent B
        #    stays frozen for update_row_actor()'s entire duration - see
        #    precompute_bay_correction()'s docstring for why computing
        #    it once here and gathering it per minibatch is exact.
        # ==============================================================

        with self.profiler.region(
            "bay_correction_seconds",
            sync_cuda=True,
        ):
            self.precompute_bay_correction(
                buffer
            )

        # ==============================================================
        # 6.
        #
        # update_row_actor() internally:
        #
        #     precomputed Bay sequence ratio (already detached)
        #         ↓
        #     corrected Row advantage
        #         ↓
        #     Agent R PPO update
        # ==============================================================

        with self.profiler.region(
            "row_update_seconds",
            sync_cuda=True,
        ):
            row_stats = (
                self.update_row_actor(
                    buffer
                )
            )

        # ==============================================================
        # Combine statistics
        # ==============================================================

        stats: Dict[
            str,
            float,
        ] = {}

        stats.update(
            rollout_stats
        )

        stats.update(
            gae_stats
        )

        stats.update(
            critic_stats
        )

        stats.update(
            bay_stats
        )

        stats.update(
            row_stats
        )

        stats[
            "total_environment_steps"
        ] = float(
            self.total_environment_steps
        )

        stats[
            "total_episodes"
        ] = float(
            self.total_episodes
        )

        # ==============================================================
        # Profiling
        # ==============================================================
        #
        # Fixed key set, defaulted to 0.0, merged with whatever this
        # iteration's Profiler regions actually measured (an empty dict
        # when self.profiler is disabled, or when a region - e.g.
        # "evaluation_seconds" on an iteration where evaluate_if_due()
        # didn't fire - simply didn't run this time). This keeps
        # rollouts.csv's column set identical on every row regardless
        # of profiling being enabled or which phases happened to run,
        # since TrainingMonitor locks its column set from the first
        # row it ever writes.
        #
        # checkpoint_seconds/steps_per_second are NOT set here: the
        # checkpoint save happens in run.py's training loop, after
        # this method has already returned, so run.py is responsible
        # for adding those two keys (with the same 0.0-default
        # discipline) before logging this row.
        # ==============================================================

        profiling_stats: Dict[
            str,
            float,
        ] = {
            "rollout_seconds": 0.0,
            "env_step_seconds": 0.0,
            "bay_inference_seconds": 0.0,
            "row_inference_seconds": 0.0,
            "critic_inference_seconds": 0.0,
            "critic_update_seconds": 0.0,
            "bay_update_seconds": 0.0,
            "bay_correction_seconds": 0.0,
            "row_update_seconds": 0.0,
            "evaluation_seconds": 0.0,
        }

        profiling_stats.update(
            self.profiler.pop_stats()
        )

        stats.update(
            profiling_stats
        )

        return stats
