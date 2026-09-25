"""
Sequential (HAPPO-style) multi-agent PPO trainer.

One training iteration:

    1. Collect a rollout with the frozen current actors: Agent B picks a
       bay, Agent R picks a row inside it, and StackEnv takes ONE step
       with action bay * n_rows + row.
    2. Compute GAE from the centralized critic's values.
    3. Update the centralized critic.
    4. PPO-update the Bay actor with the base advantage A.
    5. Compute the detached Bay correction ratio
           M_B = pi_B,updated(b | o_B) / pi_B,old(b | o_B).
    6. PPO-update the Row actor with the corrected advantage M_B * A.

Steps 4 and 6 are the same clipped-surrogate update (_ppo_actor_update);
only the actor, its data and the advantage scaling differ. Steps 3, 4
and 6 share one epoch/minibatch schedule (_minibatches).

    SequentialPPOTrainer
        |
        ├── bay_actor   PointerActor over bays   theta_B   (Agent B)
        ├── row_actor   PointerActor over rows   theta_R   (Agent R)
        └── critic      CentralizedCritic        phi
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecEnv
from torch.nn.utils import clip_grad_norm_

from ..models.centralized_critic import CentralizedCritic
from ..models.pointer_actor import PointerActor
from .bay_row_layout import BayRowLayout, select_hierarchical_action
from .profiling import Profiler
from .rollout_buffer import JointRolloutBuffer, RolloutBatch

# Phase timings reported by every train_iteration() when profiling is
# enabled, 0.0 when a phase did not run, so rollouts.csv always has the
# same columns. "evaluation_seconds" is recorded by the caller's
# step_callback, and it is also counted inside "rollout_seconds".
PHASE_TIMINGS = (
    "rollout_seconds",
    "critic_update_seconds",
    "bay_update_seconds",
    "bay_correction_seconds",
    "row_update_seconds",
    "evaluation_seconds",
)


def _mean_stats(stats: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    """Average per-minibatch statistics with one device sync per key."""

    return {name: torch.stack(values).mean().item() for name, values in stats.items()}


class SequentialPPOTrainer:
    """
    Sequential PPO for Agent B -> Agent R with one centralized
    state-value critic.

    Hyperparameters come from HierarchicalConfig.trainer_kwargs().
    """

    def __init__(
        self,
        bay_actor: PointerActor,
        row_actor: PointerActor,
        critic: CentralizedCritic,
        layout: BayRowLayout,
        *,
        learning_rate: float,
        n_epochs: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        normalize_advantage: bool,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        profiler: Optional[Profiler] = None,
    ) -> None:

        self.device = torch.device(device)
        self.bay_actor = bay_actor.to(self.device)
        self.row_actor = row_actor.to(self.device)
        self.critic = critic.to(self.device)
        self.layout = layout

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.num_envs = num_envs
        self.profiler = profiler if profiler is not None else Profiler(enabled=False)

        # theta_B, theta_R and phi are independent parameter sets.
        self.bay_optimizer = torch.optim.Adam(self.bay_actor.parameters(), lr=learning_rate)
        self.row_optimizer = torch.optim.Adam(self.row_actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.total_environment_steps = 0
        self.total_episodes = 0
        self.reset_rollout_state()

    # ==================================================================
    # Rollout state (saved and restored by training/checkpoint.py)
    # ==================================================================

    def reset_rollout_state(self) -> None:
        """Start the next rollout with env.reset().

        A rollout buffer can fill up mid-episode, so otherwise the
        environments simply continue from the last observation.
        """

        self._current_observation: Optional[np.ndarray] = None
        # V_old(s_t), reused from the previous step's V_old(s_{t+1}).
        self._current_value_tensor: Optional[torch.Tensor] = None
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int64)

    # ==================================================================
    # 1. Rollout collection
    # ==================================================================

    def collect_rollout(
        self,
        env: VecEnv,
        buffer: JointRolloutBuffer,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, float]:
        """
        Fill ``buffer`` from ``env``, an SB3 VecEnv of StackEnv copies.

        Each step reads StackEnv's action mask once and makes the Bay ->
        Row decision with select_hierarchical_action(), the same helper
        evaluation uses. ``step_callback`` is called with the
        cumulative step count after every transition (used for exact
        periodic evaluation inside a rollout).
        """

        buffer.reset()

        # The critic changed since the previous rollout, so its cached
        # value is stale.
        self._current_value_tensor = None

        self.bay_actor.eval()
        self.row_actor.eval()
        self.critic.eval()

        if self._current_observation is None:
            self._current_observation = np.asarray(env.reset(), dtype=np.float32)

        completed_returns: List[float] = []
        completed_lengths: List[int] = []
        rollout_reward = 0.0

        while not buffer.is_full():

            state_tensor = torch.as_tensor(self._current_observation, dtype=torch.float32, device=self.device)

            if self._current_value_tensor is not None:
                value_tensor = self._current_value_tensor
            else:
                with torch.no_grad():
                    value_tensor = self.critic.predict_values(state_tensor)

            action = select_hierarchical_action(
                self.layout, self.bay_actor, self.row_actor, state_tensor, get_action_masks(env)
            )

            next_global_state, rewards, dones, _infos = env.step(action.stack_actions)
            next_global_state = np.asarray(next_global_state, dtype=np.float32)
            # DummyVecEnv returns float32 rewards, SubprocVecEnv float64:
            # use the precision training uses, for both.
            rewards = np.asarray(rewards, dtype=np.float32)

            # SB3 reports terminated | truncated as done and auto-resets
            # finished environments; both end the episode for GAE.
            episode_done = np.asarray(dones, dtype=bool)

            # V_old(s_{t+1}) for every environment. For a finished one,
            # next_global_state is already its reset observation, so this
            # is next step's V_old(s_t); GAE drops it as a bootstrap.
            with torch.no_grad():
                next_value_tensor = self.critic.predict_values(
                    torch.as_tensor(next_global_state, dtype=torch.float32, device=self.device)
                )
            self._current_value_tensor = next_value_tensor

            buffer.add_batch(
                global_states=state_tensor,
                bay_action_masks=action.bay_masks,
                bay_actions=action.bay_actions,
                bay_log_probs=action.bay_log_probs,
                row_observations=action.row_observations,
                row_action_masks=action.row_masks,
                row_actions=action.row_actions,
                row_log_probs=action.row_log_probs,
                rewards=rewards,
                dones=episode_done,
                values=value_tensor,
                next_values=next_value_tensor,
            )

            for i in range(self.num_envs):
                self.total_environment_steps += 1
                self._current_episode_return[i] += float(rewards[i])
                self._current_episode_length[i] += 1
                rollout_reward += float(rewards[i])

                if episode_done[i]:
                    completed_returns.append(float(self._current_episode_return[i]))
                    completed_lengths.append(int(self._current_episode_length[i]))
                    self.total_episodes += 1
                    self._current_episode_return[i] = 0.0
                    self._current_episode_length[i] = 0

                if step_callback is not None:
                    step_callback(self.total_environment_steps)

            self._current_observation = next_global_state

        return {
            "rollout_steps": float(len(buffer)),
            "rollout_reward_sum": rollout_reward,
            "episodes_completed": float(len(completed_returns)),
            "mean_episode_reward": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
        }

    # ==================================================================
    # Shared update machinery
    # ==================================================================

    def _minibatches(self, n_samples: int) -> Iterator[torch.Tensor]:
        """Shuffled minibatch indices for each of the n_epochs passes."""

        for _epoch in range(self.n_epochs):
            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                yield indices[start : start + self.batch_size]

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, module: nn.Module, loss: torch.Tensor) -> None:
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(module.parameters(), self.max_grad_norm)
        optimizer.step()

    def _normalize(self, advantages: torch.Tensor) -> torch.Tensor:
        if not self.normalize_advantage or advantages.numel() <= 1:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ==================================================================
    # 3. Centralized critic
    # ==================================================================

    def update_critic(self, rollout: RolloutBatch) -> Dict[str, float]:
        """Regress V_phi(s_t) onto the GAE returns A_t + V_old(s_t)."""

        self.critic.train()
        stats = defaultdict(list)

        for idx in self._minibatches(len(rollout.returns)):
            value_loss = F.mse_loss(self.critic(rollout.global_states[idx]), rollout.returns[idx])
            self._optimizer_step(self.critic_optimizer, self.critic, self.vf_coef * value_loss)
            stats["critic_loss"].append(value_loss.detach())

        self.critic.eval()
        return _mean_stats(stats)

    # ==================================================================
    # 4. and 6. PPO actor update
    # ==================================================================

    def _ppo_actor_update(
        self,
        name: str,
        actor: PointerActor,
        optimizer: torch.optim.Optimizer,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        advantage_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Clipped-surrogate PPO update of one actor:

            ratio = pi_new(a | o) / pi_old(a | o)
            A     = advantage_scale * normalized(advantages)
            L     = -min(ratio * A, clip(ratio, 1 +- eps) * A) - c_ent * H

        advantage_scale is the detached Bay correction M_B for the Row
        actor and None (i.e. 1) for the Bay actor. Advantages are
        normalized per minibatch BEFORE scaling, so M_B is not undone.
        Only ``actor``'s parameters are updated.
        """

        actor.train()
        stats = defaultdict(list)

        for idx in self._minibatches(len(actions)):

            minibatch_advantages = self._normalize(advantages[idx])
            if advantage_scale is not None:
                minibatch_advantages = advantage_scale[idx] * minibatch_advantages

            log_probs, entropy = actor.evaluate_actions(observations[idx], actions[idx], action_masks[idx])

            log_ratio = log_probs - old_log_probs[idx]
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

            policy_loss = -torch.min(ratio * minibatch_advantages, clipped_ratio * minibatch_advantages).mean()
            entropy_loss = -entropy.mean()
            self._optimizer_step(optimizer, actor, policy_loss + self.ent_coef * entropy_loss)

            with torch.no_grad():
                stats[f"{name}_policy_loss"].append(policy_loss)
                stats[f"{name}_entropy"].append(entropy.mean())
                stats[f"{name}_clip_fraction"].append((torch.abs(ratio - 1.0) > self.clip_range).float().mean())
                stats[f"{name}_approx_kl"].append((torch.exp(log_ratio) - 1.0 - log_ratio).mean())

        actor.eval()
        return _mean_stats(stats)

    # ==================================================================
    # 5. Bay correction ratio
    # ==================================================================

    @torch.no_grad()
    def bay_correction_ratio(self, rollout: RolloutBatch) -> torch.Tensor:
        """
        M_B = exp(log pi_B,updated - log pi_B,old) for every sample.

        Computed once for the whole rollout after the Bay update: Agent B
        is frozen during the Row update, so each sample's ratio does not
        depend on its minibatch or epoch. Detached, so the Row update can
        never change Agent B.
        """

        self.bay_actor.eval()
        updated_log_probs, _entropy = self.bay_actor.evaluate_actions(
            rollout.global_states, rollout.bay_actions, rollout.bay_action_masks
        )
        return torch.exp(updated_log_probs - rollout.old_bay_log_probs)

    # ==================================================================
    # One complete iteration
    # ==================================================================

    def train_iteration(
        self,
        env: VecEnv,
        buffer: JointRolloutBuffer,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, float]:
        """Rollout -> GAE -> critic -> Bay actor -> M_B -> Row actor."""

        with self.profiler.region("rollout_seconds", sync_cuda=True):
            stats = self.collect_rollout(env, buffer, step_callback)

        advantages, returns = buffer.compute_gae(self.gamma, self.gae_lambda)
        stats["advantage_mean"] = advantages.mean().item()
        stats["advantage_std"] = advantages.std(unbiased=False).item()
        stats["return_mean"] = returns.mean().item()

        rollout = buffer.rollout_batch()

        with self.profiler.region("critic_update_seconds", sync_cuda=True):
            stats.update(self.update_critic(rollout))

        with self.profiler.region("bay_update_seconds", sync_cuda=True):
            stats.update(
                self._ppo_actor_update(
                    "bay",
                    self.bay_actor,
                    self.bay_optimizer,
                    rollout.global_states,
                    rollout.bay_actions,
                    rollout.bay_action_masks,
                    rollout.old_bay_log_probs,
                    rollout.advantages,
                )
            )

        with self.profiler.region("bay_correction_seconds", sync_cuda=True):
            correction = self.bay_correction_ratio(rollout)

        with self.profiler.region("row_update_seconds", sync_cuda=True):
            stats.update(
                self._ppo_actor_update(
                    "row",
                    self.row_actor,
                    self.row_optimizer,
                    rollout.row_observations,
                    rollout.row_actions,
                    rollout.row_action_masks,
                    rollout.old_row_log_probs,
                    rollout.advantages,
                    advantage_scale=correction,
                )
            )

        stats["bay_sequence_ratio_mean"] = correction.mean().item()
        stats["bay_sequence_ratio_std"] = correction.std(unbiased=False).item()
        stats["total_environment_steps"] = float(self.total_environment_steps)
        stats["total_episodes"] = float(self.total_episodes)

        if self.profiler.enabled:
            stats.update(dict.fromkeys(PHASE_TIMINGS, 0.0))
            stats.update(self.profiler.pop_stats())
        return stats
