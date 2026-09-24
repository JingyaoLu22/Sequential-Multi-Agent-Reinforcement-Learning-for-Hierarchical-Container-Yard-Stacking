"""
Joint rollout buffer for the sequential multi-agent PPO pipeline.

One stored transition corresponds to ONE complete hierarchical decision:

    global state s_t
        |
        v
    Agent B chooses bay_action
        |
        v
    Agent R observes selected bay
        |
        v
    Agent R chooses row_action
        |
        v
    StackEnv.step(global_action)
        |
        v
    reward_t, s_{t+1}

The buffer stores information for BOTH actors and the centralized critic.

Important
---------
This buffer does NOT:

    - compute GAE
    - compute PPO losses
    - update actors
    - update critic
    - compute HAPPO sequence ratios
    - normalize advantages

Those operations belong to:

    advantage.py
    sequential_trainer.py
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces


# ======================================================================
# Mini-batch container
# ======================================================================


@dataclass
class RolloutBatch:
    """
    A mini-batch sampled from JointRolloutBuffer.

    All fields are torch tensors.

    Shapes
    ------
    global_states:
        (B, global_obs_dim)

    bay_observations:
        (B, bay_obs_dim)

    bay_action_masks:
        (B, n_bays)

    bay_actions:
        (B,)

    old_bay_log_probs:
        (B,)

    row_observations:
        (B, row_obs_dim)

    row_action_masks:
        (B, n_rows)

    row_actions:
        (B,)

    old_row_log_probs:
        (B,)

    advantages:
        (B,)

    returns:
        (B,)

    Notes
    -----
    rewards/values/next_values/terminated/truncated are intentionally
    NOT included here: no update_* method or PPO computation reads them
    from a mini-batch (GAE consumes them directly from the buffer's raw
    arrays via get_rollout_arrays(), before any batching happens).
    Including them would mean uploading and re-gathering them on every
    minibatch for no reader.
    """

    global_states: torch.Tensor

    bay_observations: torch.Tensor
    bay_action_masks: torch.Tensor
    bay_actions: torch.Tensor
    old_bay_log_probs: torch.Tensor

    row_observations: torch.Tensor
    row_action_masks: torch.Tensor
    row_actions: torch.Tensor
    old_row_log_probs: torch.Tensor

    advantages: torch.Tensor
    returns: torch.Tensor


# ======================================================================
# Single source of truth for the cache/batch layer
# ======================================================================
#
# JointRolloutBuffer._build_device_cache() and _make_batch() both need
# the exact same (field name, torch dtype) pairs as RolloutBatch. Rather
# than listing them three times by hand (and risking one place silently
# drifting from the others - a field forgotten in one spot doesn't
# raise, it just never reaches training), both methods loop over this
# one table.
# ======================================================================

_ROLLOUT_BATCH_FIELDS: Tuple[
    Tuple[str, torch.dtype],
    ...,
] = (
    ("global_states", torch.float32),
    ("bay_observations", torch.float32),
    ("bay_action_masks", torch.bool),
    ("bay_actions", torch.long),
    ("old_bay_log_probs", torch.float32),
    ("row_observations", torch.float32),
    ("row_action_masks", torch.bool),
    ("row_actions", torch.long),
    ("old_row_log_probs", torch.float32),
    ("advantages", torch.float32),
    ("returns", torch.float32),
)

# Fail LOUDLY at import time if RolloutBatch and this table ever drift
# apart, instead of a field silently never being cached/batched.
assert {name for name, _ in _ROLLOUT_BATCH_FIELDS} == {
    field.name for field in dataclass_fields(RolloutBatch)
}, (
    "_ROLLOUT_BATCH_FIELDS and RolloutBatch fields must match exactly."
)


# ======================================================================
# Joint rollout buffer
# ======================================================================


class JointRolloutBuffer:
    """
    Fixed-size rollout buffer for Bay + Row sequential PPO.

    Parameters
    ----------
    buffer_size : int
        Number of complete hierarchical transitions stored before
        performing an update.

    global_observation_space : spaces.Box
        Observation space used by the centralized critic.

    bay_observation_space : spaces.Box
        Observation space used by Agent B.

    row_observation_space : spaces.Box
        Observation space used by Agent R.

    n_bays : int
        Number of Agent B actions.

    n_rows : int
        Number of Agent R actions.

    device : str | torch.device
        Device used when converting mini-batches to torch tensors.

        The buffer itself stores rollout data as NumPy arrays on CPU.
    """

    def __init__(
        self,
        buffer_size: int,
        global_observation_space: spaces.Box,
        bay_observation_space: spaces.Box,
        row_observation_space: spaces.Box,
        n_bays: int,
        n_rows: int,
        device: str | torch.device = "cpu",
    ) -> None:

        # ==============================================================
        # Validation
        # ==============================================================

        if buffer_size <= 0:
            raise ValueError(
                "buffer_size must be positive."
            )

        if n_bays <= 0:
            raise ValueError(
                "n_bays must be positive."
            )

        if n_rows <= 0:
            raise ValueError(
                "n_rows must be positive."
            )

        for name, space in [
            (
                "global_observation_space",
                global_observation_space,
            ),
            (
                "bay_observation_space",
                bay_observation_space,
            ),
            (
                "row_observation_space",
                row_observation_space,
            ),
        ]:
            if not isinstance(
                space,
                spaces.Box,
            ):
                raise TypeError(
                    f"{name} must be "
                    f"gymnasium.spaces.Box."
                )

            if len(space.shape) != 1:
                raise ValueError(
                    f"{name} must be flat. "
                    f"Received shape={space.shape}."
                )

        # ==============================================================
        # Metadata
        # ==============================================================

        self.buffer_size = int(
            buffer_size
        )

        self.n_bays = int(
            n_bays
        )

        self.n_rows = int(
            n_rows
        )

        self.global_obs_dim = int(
            global_observation_space.shape[0]
        )

        self.bay_obs_dim = int(
            bay_observation_space.shape[0]
        )

        self.row_obs_dim = int(
            row_observation_space.shape[0]
        )

        self.device = torch.device(
            device
        )

        # ==============================================================
        # Allocate storage
        # ==============================================================

        self._allocate_storage()

        self.reset()

    # ==================================================================
    # Storage allocation
    # ==================================================================

    def _allocate_storage(
        self,
    ) -> None:
        """
        Allocate all fixed-size NumPy arrays.

        We store detached rollout data only.
        No computation graph is kept inside the buffer.
        """

        # --------------------------------------------------------------
        # Centralized critic state
        # --------------------------------------------------------------

        self.global_states = np.zeros(
            (
                self.buffer_size,
                self.global_obs_dim,
            ),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Agent B
        # --------------------------------------------------------------

        self.bay_observations = np.zeros(
            (
                self.buffer_size,
                self.bay_obs_dim,
            ),
            dtype=np.float32,
        )

        self.bay_action_masks = np.zeros(
            (
                self.buffer_size,
                self.n_bays,
            ),
            dtype=np.bool_,
        )

        self.bay_actions = np.zeros(
            self.buffer_size,
            dtype=np.int64,
        )

        self.old_bay_log_probs = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Agent R
        # --------------------------------------------------------------

        self.row_observations = np.zeros(
            (
                self.buffer_size,
                self.row_obs_dim,
            ),
            dtype=np.float32,
        )

        self.row_action_masks = np.zeros(
            (
                self.buffer_size,
                self.n_rows,
            ),
            dtype=np.bool_,
        )

        self.row_actions = np.zeros(
            self.buffer_size,
            dtype=np.int64,
        )

        self.old_row_log_probs = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Environment transition
        # --------------------------------------------------------------

        self.rewards = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        self.terminated = np.zeros(
            self.buffer_size,
            dtype=np.bool_,
        )

        self.truncated = np.zeros(
            self.buffer_size,
            dtype=np.bool_,
        )

        # --------------------------------------------------------------
        # Central critic predictions during rollout
        # --------------------------------------------------------------
        #
        # V_old(s_t)
        #
        self.values = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # V_old(s_{t+1})
        #
        # We store this explicitly rather than relying on:
        #
        #     values[t + 1]
        #
        # because a rollout can cross an episode boundary.
        # --------------------------------------------------------------

        self.next_values = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Filled later by advantage.py
        # --------------------------------------------------------------

        self.advantages = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

        self.returns = np.zeros(
            self.buffer_size,
            dtype=np.float32,
        )

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
    ) -> None:
        """
        Clear logical buffer contents.

        Existing arrays are reused to avoid unnecessary allocations.
        """

        self.pos = 0
        self.full = False

        self.advantages_ready = False

        # Populated once by _build_device_cache(), after
        # set_advantages_and_returns() finalizes the rollout. Cleared
        # here so a reused buffer object never serves stale GPU data.
        self._device_cache: Optional[
            Dict[str, torch.Tensor]
        ] = None

        # Populated once by set_bay_correction_ratio(), after
        # update_bay_actor() finishes (later than _device_cache above).
        # Cleared here for the same reason.
        self._bay_correction_cache: Optional[
            torch.Tensor
        ] = None

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def size(self) -> int:
        """
        Number of currently stored transitions.
        """

        if self.full:
            return self.buffer_size

        return self.pos

    def __len__(
        self,
    ) -> int:
        return self.size

    def is_full(
        self,
    ) -> bool:
        """
        Return True once buffer_size transitions have been collected.
        """

        return self.full

    # ==================================================================
    # Add one joint transition
    # ==================================================================

    def add(
        self,
        *,
        global_state: np.ndarray,

        bay_observation: np.ndarray,
        bay_action_mask: np.ndarray,
        bay_action: int,
        bay_log_prob,

        row_observation: np.ndarray,
        row_action_mask: np.ndarray,
        row_action: int,
        row_log_prob,

        reward: float,

        value,
        next_value,

        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Store ONE complete hierarchical transition.

        This method should be called only AFTER:

            Agent B chose a bay
            Agent R chose a row
            StackEnv.step(...) was executed

        Example
        -------
        buffer.add(
            global_state=global_state,

            bay_observation=bay_obs,
            bay_action_mask=bay_mask,
            bay_action=bay_action,
            bay_log_prob=bay_log_prob,

            row_observation=row_obs,
            row_action_mask=row_mask,
            row_action=row_action,
            row_log_prob=row_log_prob,

            reward=reward,

            value=value,
            next_value=next_value,

            terminated=terminated,
            truncated=truncated,
        )
        """

        if self.full:
            raise RuntimeError(
                "Cannot add transition: "
                "rollout buffer is already full."
            )

        idx = self.pos

        # ==============================================================
        # Validate observation shapes
        # ==============================================================

        global_state = np.asarray(
            global_state,
            dtype=np.float32,
        )

        bay_observation = np.asarray(
            bay_observation,
            dtype=np.float32,
        )

        row_observation = np.asarray(
            row_observation,
            dtype=np.float32,
        )

        if global_state.shape != (
            self.global_obs_dim,
        ):
            raise ValueError(
                "Unexpected global_state shape. "
                f"Expected "
                f"{(self.global_obs_dim,)}, "
                f"received "
                f"{global_state.shape}."
            )

        if bay_observation.shape != (
            self.bay_obs_dim,
        ):
            raise ValueError(
                "Unexpected bay observation shape. "
                f"Expected "
                f"{(self.bay_obs_dim,)}, "
                f"received "
                f"{bay_observation.shape}."
            )

        if row_observation.shape != (
            self.row_obs_dim,
        ):
            raise ValueError(
                "Unexpected row observation shape. "
                f"Expected "
                f"{(self.row_obs_dim,)}, "
                f"received "
                f"{row_observation.shape}."
            )

        # ==============================================================
        # Validate masks
        # ==============================================================

        bay_action_mask = np.asarray(
            bay_action_mask,
            dtype=np.bool_,
        )

        row_action_mask = np.asarray(
            row_action_mask,
            dtype=np.bool_,
        )

        if bay_action_mask.shape != (
            self.n_bays,
        ):
            raise ValueError(
                "Unexpected bay_action_mask shape. "
                f"Expected {(self.n_bays,)}, "
                f"received "
                f"{bay_action_mask.shape}."
            )

        if row_action_mask.shape != (
            self.n_rows,
        ):
            raise ValueError(
                "Unexpected row_action_mask shape. "
                f"Expected {(self.n_rows,)}, "
                f"received "
                f"{row_action_mask.shape}."
            )

        # ==============================================================
        # Validate selected actions
        # ==============================================================

        bay_action = int(
            bay_action
        )

        row_action = int(
            row_action
        )

        if not (
            0
            <= bay_action
            < self.n_bays
        ):
            raise ValueError(
                f"Invalid bay_action="
                f"{bay_action}."
            )

        if not (
            0
            <= row_action
            < self.n_rows
        ):
            raise ValueError(
                f"Invalid row_action="
                f"{row_action}."
            )

        # The sampled action should always be valid under the mask used
        # to generate it.
        if not bay_action_mask[
            bay_action
        ]:
            raise ValueError(
                "Stored bay action is invalid "
                "under its rollout action mask."
            )

        if not row_action_mask[
            row_action
        ]:
            raise ValueError(
                "Stored row action is invalid "
                "under its rollout action mask."
            )

        # ==============================================================
        # Store critic state
        # ==============================================================

        self.global_states[
            idx
        ] = global_state

        # ==============================================================
        # Store Agent B data
        # ==============================================================

        self.bay_observations[
            idx
        ] = bay_observation

        self.bay_action_masks[
            idx
        ] = bay_action_mask

        self.bay_actions[
            idx
        ] = bay_action

        self.old_bay_log_probs[
            idx
        ] = self._scalar_to_float(
            bay_log_prob
        )

        # ==============================================================
        # Store Agent R data
        # ==============================================================

        self.row_observations[
            idx
        ] = row_observation

        self.row_action_masks[
            idx
        ] = row_action_mask

        self.row_actions[
            idx
        ] = row_action

        self.old_row_log_probs[
            idx
        ] = self._scalar_to_float(
            row_log_prob
        )

        # ==============================================================
        # Store environment signal
        # ==============================================================

        self.rewards[
            idx
        ] = float(
            reward
        )

        self.terminated[
            idx
        ] = bool(
            terminated
        )

        self.truncated[
            idx
        ] = bool(
            truncated
        )

        # ==============================================================
        # Store OLD critic predictions
        # ==============================================================

        self.values[
            idx
        ] = self._scalar_to_float(
            value
        )

        self.next_values[
            idx
        ] = self._scalar_to_float(
            next_value
        )

        # ==============================================================
        # Move write pointer
        # ==============================================================

        self.pos += 1

        if self.pos >= self.buffer_size:
            self.full = True

    # ==================================================================
    # Add a batch of joint transitions (one per environment)
    # ==================================================================

    def add_batch(
        self,
        *,
        global_states: np.ndarray,

        bay_observations: np.ndarray,
        bay_action_masks: np.ndarray,
        bay_actions: np.ndarray,
        bay_log_probs: np.ndarray,

        row_observations: np.ndarray,
        row_action_masks: np.ndarray,
        row_actions: np.ndarray,
        row_log_probs: np.ndarray,

        rewards: np.ndarray,

        values: np.ndarray,
        next_values: np.ndarray,

        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> int:
        """
        Store one complete hierarchical transition PER environment in a
        single vectorized call.

        Every argument is an array whose first axis is num_envs, laid
        out in environment order (row i belongs to environment i) -
        exactly the arrays a VecHierarchicalEnv rollout step already
        produces.

        Equivalent to calling add() once per row, in order:

            for i in range(num_envs):
                if buffer.is_full():
                    break
                buffer.add(global_state=global_states[i], ...)

        including that equivalent loop's early-exit behaviour: if the
        buffer has less remaining capacity than num_envs, only the
        first `remaining` rows are validated and written, and the rest
        are silently dropped, matching what the loop above would do.

        Returns
        -------
        int
            Number of rows actually written (0 <= n_written <= num_envs).
        """

        if self.full:
            raise RuntimeError(
                "Cannot add transitions: "
                "rollout buffer is already full."
            )

        # ==============================================================
        # Cast + shape validation (applies to every row, same as add()
        # validating a single row's shape before checking capacity)
        # ==============================================================

        global_states = np.asarray(global_states, dtype=np.float32)
        bay_observations = np.asarray(bay_observations, dtype=np.float32)
        bay_action_masks = np.asarray(bay_action_masks, dtype=np.bool_)
        bay_actions = np.asarray(bay_actions, dtype=np.int64)
        bay_log_probs = np.asarray(bay_log_probs, dtype=np.float32)
        row_observations = np.asarray(row_observations, dtype=np.float32)
        row_action_masks = np.asarray(row_action_masks, dtype=np.bool_)
        row_actions = np.asarray(row_actions, dtype=np.int64)
        row_log_probs = np.asarray(row_log_probs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        next_values = np.asarray(next_values, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)

        if global_states.ndim != 2:
            raise ValueError(
                "global_states must be 2D (num_envs, global_obs_dim). "
                f"Received shape={global_states.shape}."
            )

        n = global_states.shape[0]

        expected_shapes = {
            "global_states": (n, self.global_obs_dim),
            "bay_observations": (n, self.bay_obs_dim),
            "bay_action_masks": (n, self.n_bays),
            "bay_actions": (n,),
            "bay_log_probs": (n,),
            "row_observations": (n, self.row_obs_dim),
            "row_action_masks": (n, self.n_rows),
            "row_actions": (n,),
            "row_log_probs": (n,),
            "rewards": (n,),
            "values": (n,),
            "next_values": (n,),
            "terminated": (n,),
            "truncated": (n,),
        }

        arrays = {
            "global_states": global_states,
            "bay_observations": bay_observations,
            "bay_action_masks": bay_action_masks,
            "bay_actions": bay_actions,
            "bay_log_probs": bay_log_probs,
            "row_observations": row_observations,
            "row_action_masks": row_action_masks,
            "row_actions": row_actions,
            "row_log_probs": row_log_probs,
            "rewards": rewards,
            "values": values,
            "next_values": next_values,
            "terminated": terminated,
            "truncated": truncated,
        }

        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(
                    f"Unexpected {name} shape. "
                    f"Expected {expected}, "
                    f"received {arrays[name].shape}."
                )

        # ==============================================================
        # How many rows actually fit (mirrors the is_full()-gated loop)
        # ==============================================================

        remaining = self.buffer_size - self.pos
        write_n = min(n, remaining)

        # Content validation below (action range, action-under-mask
        # validity) is checked only for rows that will actually be
        # written, exactly as the per-row `for i in range(num_envs): if
        # buffer.is_full(): break; buffer.add(...)` loop this replaces
        # would never even call add() - and thus never validate
        # anything - for a row beyond the buffer's remaining capacity.
        # Shape validation above is unconditional: it is a basic
        # well-formed-batch check on the caller's arrays as a whole,
        # not a per-row content check, so it has no equivalent in that
        # loop to diverge from.
        row_indices = np.arange(write_n)

        if write_n > 0 and (
            np.any((bay_actions[:write_n] < 0) | (bay_actions[:write_n] >= self.n_bays))
        ):
            raise ValueError(
                "bay_actions contains an out-of-range action."
            )

        if write_n > 0 and (
            np.any((row_actions[:write_n] < 0) | (row_actions[:write_n] >= self.n_rows))
        ):
            raise ValueError(
                "row_actions contains an out-of-range action."
            )

        if write_n > 0 and not bay_action_masks[
            row_indices, bay_actions[:write_n]
        ].all():
            raise ValueError(
                "Stored bay action is invalid "
                "under its rollout action mask."
            )

        if write_n > 0 and not row_action_masks[
            row_indices, row_actions[:write_n]
        ].all():
            raise ValueError(
                "Stored row action is invalid "
                "under its rollout action mask."
            )

        # ==============================================================
        # Write
        # ==============================================================

        idx = slice(self.pos, self.pos + write_n)

        self.global_states[idx] = global_states[:write_n]

        self.bay_observations[idx] = bay_observations[:write_n]
        self.bay_action_masks[idx] = bay_action_masks[:write_n]
        self.bay_actions[idx] = bay_actions[:write_n]
        self.old_bay_log_probs[idx] = bay_log_probs[:write_n]

        self.row_observations[idx] = row_observations[:write_n]
        self.row_action_masks[idx] = row_action_masks[:write_n]
        self.row_actions[idx] = row_actions[:write_n]
        self.old_row_log_probs[idx] = row_log_probs[:write_n]

        self.rewards[idx] = rewards[:write_n]
        self.terminated[idx] = terminated[:write_n]
        self.truncated[idx] = truncated[:write_n]

        self.values[idx] = values[:write_n]
        self.next_values[idx] = next_values[:write_n]

        self.pos += write_n

        if self.pos >= self.buffer_size:
            self.full = True

        return write_n

    # ==================================================================
    # Advantage / return storage
    # ==================================================================

    def set_advantages_and_returns(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        """
        Store base GAE advantages and critic targets.

        This will be called by advantage.py.

        Parameters
        ----------
        advantages : np.ndarray
            Shared/base advantage A_t.

        returns : np.ndarray
            Value target:

                R_t = A_t + V_old(s_t)

        Important
        ---------
        These are BASE advantages.

        We do NOT store the HAPPO row correction here.

        Later:

            Bay update:
                uses A_t

            Row update:
                uses detached Bay sequence factor * A_t
        """

        n = self.size

        advantages = np.asarray(
            advantages,
            dtype=np.float32,
        )

        returns = np.asarray(
            returns,
            dtype=np.float32,
        )

        if advantages.shape != (n,):
            raise ValueError(
                "Advantage shape mismatch. "
                f"Expected {(n,)}, "
                f"received "
                f"{advantages.shape}."
            )

        if returns.shape != (n,):
            raise ValueError(
                "Return shape mismatch. "
                f"Expected {(n,)}, "
                f"received "
                f"{returns.shape}."
            )

        self.advantages[
            :n
        ] = advantages

        self.returns[
            :n
        ] = returns

        # Rollout data is now finalized. Upload every field to the
        # device ONCE here, so update_critic/update_bay_actor/
        # update_row_actor never each re-upload the whole buffer from
        # CPU numpy on every epoch/minibatch.
        #
        # advantages_ready is set only AFTER this succeeds, so it can
        # never be True while _device_cache is still None.
        self._build_device_cache()

        self.advantages_ready = True

    # ==================================================================
    # Device tensor cache
    # ==================================================================

    def _build_device_cache(
        self,
    ) -> None:
        """
        Upload every RolloutBatch field to self.device ONCE as a tensor.

        Without this, every PPO minibatch (across every epoch, across
        all three independent update_* passes) would re-slice the CPU
        numpy arrays and re-copy that slice to the device from scratch.
        Building the cache once here means get_batches()/_make_batch()
        only ever perform on-device tensor indexing afterward - no
        repeated CPU<->GPU transfers.

        rewards/values/next_values/terminated/truncated are deliberately
        NOT cached here - see RolloutBatch's docstring for why.

        Every numpy slice is copied (`.copy()`) before conversion.
        `torch.as_tensor` does not copy when device="cpu" and dtypes
        already match, so without this a cached tensor would silently
        alias this buffer's own mutable storage - reset()+add() on a
        REUSED buffer instance would then corrupt any tensor/batch a
        caller is still holding from the previous rollout. On a CUDA
        device the host->device copy already happens either way, so
        this only changes behavior (for the better) on CPU.

        Fields and dtypes come from _ROLLOUT_BATCH_FIELDS - the single
        table shared with _make_batch(), so this method and RolloutBatch
        can't silently drift apart.
        """

        n = self.size

        self._device_cache = {
            name: self._to_tensor(
                getattr(self, name)[:n].copy(),
                dtype=dtype,
            )
            for name, dtype in _ROLLOUT_BATCH_FIELDS
        }

    # ==================================================================
    # Mini-batch iteration
    # ==================================================================

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[
        RolloutBatch,
        None,
        None,
    ]:
        """
        Yield mini-batches for PPO training.

        Parameters
        ----------
        batch_size : int
            Number of transitions per mini-batch.

        shuffle : bool
            Shuffle rollout indices before creating batches.

        Notes
        -----
        PPO usually performs several epochs over this generator.

        For every epoch, call get_batches(...) again so a new shuffled
        ordering can be generated.
        """

        if self._device_cache is None:
            raise RuntimeError(
                "Device tensor cache has not been built. This should "
                "not happen once advantages_ready is True - "
                "set_advantages_and_returns() always builds it."
            )

        for batch_indices in self._shuffled_batch_indices(
            batch_size,
            shuffle,
        ):

            yield self._make_batch(
                batch_indices
            )

    # ==================================================================
    # Bay correction ratio (Agent R update only)
    # ==================================================================

    def set_bay_correction_ratio(
        self,
        ratio: torch.Tensor,
    ) -> None:
        """
        Store the precomputed Bay sequence-correction ratio for every
        sample in the rollout, to be reused by every Agent R minibatch
        of every PPO epoch instead of each one recomputing it via a
        fresh Agent B forward pass.

        Deliberately kept separate from RolloutBatch/_device_cache
        (built by _build_device_cache() at GAE time, BEFORE Agent B has
        even been updated) rather than added as another RolloutBatch
        field: this ratio only exists after update_bay_actor() has
        already run, later in the same train_iteration() call, so it
        cannot be part of that earlier, already-finalized cache.

        Parameters
        ----------
        ratio : torch.Tensor
            Shape (self.size,), already on self.device.
        """

        ratio = torch.as_tensor(
            ratio,
            dtype=torch.float32,
            device=self.device,
        )

        if ratio.shape != (self.size,):
            raise ValueError(
                "Bay correction ratio shape mismatch. "
                f"Expected {(self.size,)}, "
                f"received {tuple(ratio.shape)}."
            )

        self._bay_correction_cache = ratio

    def get_row_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[
        Tuple[RolloutBatch, torch.Tensor],
        None,
        None,
    ]:
        """
        Same mini-batching as get_batches(), additionally yielding each
        mini-batch's precomputed Bay correction ratio (see
        set_bay_correction_ratio()) gathered by the same indices.

        Used only by update_row_actor(), which needs both the ordinary
        RolloutBatch fields and this ratio. update_critic()/
        update_bay_actor() keep using get_batches() unchanged.
        """

        if self._bay_correction_cache is None:
            raise RuntimeError(
                "set_bay_correction_ratio() must be called before "
                "get_row_batches()."
            )

        if self._device_cache is None:
            raise RuntimeError(
                "Device tensor cache has not been built. This should "
                "not happen once advantages_ready is True - "
                "set_advantages_and_returns() always builds it."
            )

        for batch_indices in self._shuffled_batch_indices(
            batch_size,
            shuffle,
        ):

            yield (
                self._make_batch(
                    batch_indices
                ),
                self._bay_correction_cache[
                    batch_indices
                ],
            )

    def _shuffled_batch_indices(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> Generator[
        torch.Tensor,
        None,
        None,
    ]:
        """
        Shared index-batching logic behind get_batches()/
        get_row_batches(): validate, optionally shuffle once, then
        yield contiguous slices of (possibly shuffled) indices.

        Shuffle indices are generated directly on self.device, so
        minibatch selection never needs a CPU<->GPU round trip.
        """

        if self.size == 0:
            raise RuntimeError(
                "Cannot sample from an empty rollout buffer."
            )

        if not self.advantages_ready:
            raise RuntimeError(
                "Advantages and returns have not "
                "been computed yet."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if shuffle:
            indices = torch.randperm(
                self.size,
                device=self.device,
            )
        else:
            indices = torch.arange(
                self.size,
                device=self.device,
            )

        for start in range(
            0,
            self.size,
            batch_size,
        ):

            yield indices[
                start:
                start + batch_size
            ]

    # ==================================================================
    # Build torch mini-batch
    # ==================================================================

    def _make_batch(
        self,
        indices: torch.Tensor,
    ) -> RolloutBatch:
        """
        Select one mini-batch from the cached device tensors.

        ``indices`` is a LongTensor already on self.device (see
        get_batches()). Every field lookup below is therefore a pure
        on-device gather - no CPU<->GPU transfer happens here, since
        _build_device_cache() already uploaded the whole rollout once.

        Fields come from _ROLLOUT_BATCH_FIELDS - the single table shared
        with _build_device_cache(), so this method and RolloutBatch
        can't silently drift apart.
        """

        cache = self._device_cache

        return RolloutBatch(
            **{
                name: cache[name][indices]
                for name, _ in _ROLLOUT_BATCH_FIELDS
            }
        )

    # ==================================================================
    # Raw rollout access
    # ==================================================================

    def get_rollout_arrays(
        self,
    ) -> dict:
        """
        Return the currently populated rollout arrays.

        Primarily used by advantage.py.

        Arrays are returned only up to self.size.
        """

        n = self.size

        return {
            "rewards":
                self.rewards[:n],

            "values":
                self.values[:n],

            "next_values":
                self.next_values[:n],

            "terminated":
                self.terminated[:n],

            "truncated":
                self.truncated[:n],
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _to_tensor(
        self,
        array: np.ndarray,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert NumPy array to tensor on training device.
        """

        return torch.as_tensor(
            array,
            dtype=dtype,
            device=self.device,
        )

    @staticmethod
    def _scalar_to_float(
        value,
    ) -> float:
        """
        Convert Python / NumPy / Torch scalar to detached float.

        This is important for rollout log probabilities and critic
        predictions because the buffer must NOT retain autograd graphs.
        """

        if isinstance(
            value,
            torch.Tensor,
        ):

            if value.numel() != 1:
                raise ValueError(
                    "Expected scalar tensor, "
                    f"received shape "
                    f"{tuple(value.shape)}."
                )

            return float(
                value.detach()
                .cpu()
                .item()
            )

        return float(
            value
        )