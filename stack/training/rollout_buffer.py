"""
On-device rollout storage and GAE for sequential HPPO.

Every field is a (T, N, ...) tensor on the training device, where
T = rollout steps and N = parallel environments, so GAE runs over all
environments at once. Flattened, sample t * N + n is environment n's
transition at step t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .bay_row_layout import BayRowLayout


@dataclass
class RolloutBatch:
    """The whole rollout, flattened to (T * N, ...) device tensors.

    Agent B and the critic both observe global_states.
    """

    global_states: torch.Tensor
    bay_action_masks: torch.Tensor
    bay_actions: torch.Tensor
    old_bay_log_probs: torch.Tensor
    row_observations: torch.Tensor
    row_action_masks: torch.Tensor
    row_actions: torch.Tensor
    old_row_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class JointRolloutBuffer:
    """
    Joint Bay/Row transitions of ``num_envs`` environments stepped
    together. ``buffer_size`` counts transitions, so it must be a
    multiple of ``num_envs``.
    """

    def __init__(
        self,
        layout: BayRowLayout,
        buffer_size: int,
        num_envs: int,
        device: str | torch.device,
    ) -> None:

        if buffer_size <= 0 or buffer_size % num_envs:
            raise ValueError(
                f"buffer_size={buffer_size} must be a positive multiple "
                f"of num_envs={num_envs}."
            )

        self.n_steps = buffer_size // num_envs
        self.num_envs = num_envs
        self.device = torch.device(device)

        def storage(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros((self.n_steps, num_envs, *shape), dtype=dtype, device=self.device)

        self.global_states = storage(layout.observation_space.shape[0])
        self.bay_action_masks = storage(layout.n_bays, dtype=torch.bool)
        self.bay_actions = storage(dtype=torch.long)
        self.bay_log_probs = storage()
        self.row_observations = storage(layout.row_observation_space.shape[0])
        self.row_action_masks = storage(layout.n_rows, dtype=torch.bool)
        self.row_actions = storage(dtype=torch.long)
        self.row_log_probs = storage()
        self.rewards = storage()
        # terminated | truncated: both end the episode without bootstrap.
        self.dones = storage(dtype=torch.bool)
        self.values = storage()
        self.next_values = storage()
        self.advantages = storage()
        self.returns = storage()

        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def is_full(self) -> bool:
        return self.step == self.n_steps

    def __len__(self) -> int:
        return self.step * self.num_envs

    def add_batch(
        self,
        global_states,
        bay_action_masks,
        bay_actions,
        bay_log_probs,
        row_observations,
        row_action_masks,
        row_actions,
        row_log_probs,
        rewards,
        dones,
        values,
        next_values,
    ) -> None:
        """
        Store one step for all environments. Every argument is a tensor
        or array with leading dimension num_envs. ``next_values`` is the
        raw V_old(s_{t+1}); GAE ignores it where ``dones`` is set.
        """

        if self.is_full():
            raise RuntimeError("Rollout buffer is full.")

        transition = {
            "global_states": global_states,
            "bay_action_masks": bay_action_masks,
            "bay_actions": bay_actions,
            "bay_log_probs": bay_log_probs,
            "row_observations": row_observations,
            "row_action_masks": row_action_masks,
            "row_actions": row_actions,
            "row_log_probs": row_log_probs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "next_values": next_values,
        }

        for name, value in transition.items():
            getattr(self, name)[self.step] = torch.as_tensor(value, device=self.device)

        self.step += 1

    @torch.no_grad()
    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation for all environments at once:

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            R_t     = A_t + V(s_t)

        Returns flattened (T * N,) advantages and returns.
        """

        steps = self.step
        not_done = (~self.dones[:steps]).float()
        last_advantage = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(steps)):
            delta = self.rewards[t] + gamma * self.next_values[t] * not_done[t] - self.values[t]
            last_advantage = delta + gamma * gae_lambda * not_done[t] * last_advantage
            self.advantages[t] = last_advantage

        self.returns[:steps] = self.advantages[:steps] + self.values[:steps]

        return self.advantages[:steps].reshape(-1), self.returns[:steps].reshape(-1)

    def rollout_batch(self) -> RolloutBatch:
        """Flattened views (no copy) of the stored rollout; valid until
        the next reset()/add_batch()."""

        def flat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[: self.step].reshape(len(self), *tensor.shape[2:])

        return RolloutBatch(
            global_states=flat(self.global_states),
            bay_action_masks=flat(self.bay_action_masks),
            bay_actions=flat(self.bay_actions),
            old_bay_log_probs=flat(self.bay_log_probs),
            row_observations=flat(self.row_observations),
            row_action_masks=flat(self.row_action_masks),
            row_actions=flat(self.row_actions),
            old_row_log_probs=flat(self.row_log_probs),
            advantages=flat(self.advantages),
            returns=flat(self.returns),
        )
