"""
Sequential Hierarchical Policy for bay and stack selection.

Implements a hierarchical policy with three networks, each with its own
Transformer encoder:
  1. Bay actor selects which bay to place the container in
  2. Row actor selects which row within that bay; it sees only the
     selected bay's slice of the observation
  3. Centralized critic estimates V(s) from the full observation

The action space remains Discrete(n_stacks) for compatibility with
StackEnv and MaskablePPO: stack action = bay * n_rows_per_bay + row.

Unlike the joint hierarchical policy, which is trained by MaskablePPO with
a single loss on log p(bay) + log p(stack | bay), the three networks are
updated one after another by SequentialHPPO (models/sequential_hppo.py).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from gymnasium import spaces

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from models.transformer_policy import MaskableTransformerPolicy
from models.transformer_bay_policy import MaskableBayTransformerPolicy


class MaskableSequentialTransformerPolicy(MaskableActorCriticPolicy):
    """
    Sequential HPPO policy for SequentialHPPO: a Bay actor
    (MaskableBayTransformerPolicy), a Row actor that sees only the selected
    bay (MaskableBayTransformerPolicy with one row per action) and a
    centralized critic (the value of MaskableTransformerPolicy), each with
    its own encoder.

    Required policy_kwargs:

        n_stacks       : int  — total yard stacks (== action_space.n)
        n_bays         : int  — number of physical (odd) bays
        n_rows_per_bay : int  — rows per bay
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        n_stacks: int | None = None,
        n_bays: int | None = None,
        n_rows_per_bay: int | None = None,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vf_dim: int = 128,
        tanh_clipping: float = 10.0,
        include_container_in_encoder: bool = True,
        container_start: int | None = None,
        container_dim: int | None = None,
        **kwargs,
    ) -> None:
        assert n_stacks is not None, "n_stacks required"
        assert n_bays is not None, "n_bays required"
        assert n_rows_per_bay is not None, "n_rows_per_bay required"

        self.n_stacks = n_stacks
        self.n_bays = n_bays
        self.n_rows_per_bay = n_rows_per_bay
        self._seq_network_kwargs = dict(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            vf_dim=vf_dim,
            tanh_clipping=tanh_clipping,
            include_container_in_encoder=include_container_in_encoder,
            container_start=container_start,
            container_dim=container_dim,
        )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        # stack_features_v3 has one token per stack, bay by bay, so a bay is a
        # contiguous slice of the observation; all tokens have the same bounds.
        row_obs_dim = self.observation_space.shape[0] // self.n_bays
        row_observation_space = spaces.Box(
            low=self.observation_space.low[:row_obs_dim],
            high=self.observation_space.high[:row_obs_dim],
            dtype=self.observation_space.dtype,
        )
        self.bay_actor = MaskableBayTransformerPolicy(
            self.observation_space,
            spaces.Discrete(self.n_bays),
            lr_schedule,
            n_stacks=self.n_stacks,
            n_rows_per_bay=self.n_rows_per_bay,
            **self._seq_network_kwargs,
        )
        self.row_actor = MaskableBayTransformerPolicy(
            row_observation_space,
            spaces.Discrete(self.n_rows_per_bay),
            lr_schedule,
            n_stacks=self.n_rows_per_bay,
            n_rows_per_bay=1,
            **self._seq_network_kwargs,
        )
        self.critic = MaskableTransformerPolicy(
            self.observation_space,
            self.action_space,
            lr_schedule,
            **self._seq_network_kwargs,
        )
        # One optimizer for the three networks, as in MaskableActorCriticPolicy;
        # each SequentialHPPO step has gradients for one network only.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def bay_masks(self, action_masks: torch.Tensor) -> torch.Tensor:
        """(B, n_stacks) -> (B, n_bays): a bay is valid if any of its rows is."""
        return action_masks.reshape(-1, self.n_bays, self.n_rows_per_bay).any(dim=-1)

    def row_inputs(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor | None,
        bay_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """The Row actor's observation (the selected bay's slice) and mask."""
        index = torch.arange(bay_actions.shape[0], device=bay_actions.device)
        row_obs = obs.reshape(-1, self.n_bays, obs.shape[-1] // self.n_bays)[index, bay_actions]
        if action_masks is None:
            return row_obs, None
        return row_obs, action_masks.reshape(-1, self.n_bays, self.n_rows_per_bay)[index, bay_actions]

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bay actor -> Row actor on the selected bay -> stack action; the value
        is the critic's and the log-probability is log pi_bay + log pi_row.
        """
        if action_masks is not None:
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=obs.device).reshape(-1, self.n_stacks)

        bay_dist = self.bay_actor.get_distribution(obs, None if action_masks is None else self.bay_masks(action_masks))
        bay_actions = bay_dist.get_actions(deterministic=deterministic)

        row_obs, row_masks = self.row_inputs(obs, action_masks, bay_actions)
        row_dist = self.row_actor.get_distribution(row_obs, row_masks)
        row_actions = row_dist.get_actions(deterministic=deterministic)

        log_prob = bay_dist.log_prob(bay_actions) + row_dist.log_prob(row_actions)
        return bay_actions * self.n_rows_per_bay + row_actions, self.critic.predict_values(obs), log_prob

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic.predict_values(obs)

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Used by model.predict() during evaluation.
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic, action_masks=action_masks)
        return actions
