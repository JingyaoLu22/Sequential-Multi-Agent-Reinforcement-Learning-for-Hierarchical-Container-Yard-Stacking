"""Centralized state-value function for Sequential HPPO."""

from __future__ import annotations

import torch
from gymnasium import spaces
from torch import nn

from .transformer_policy import TransformerFeaturesExtractor


class CentralizedCritic(nn.Module):
    """Predict one value for the full StackEnv observation.

    The encoder is intentionally independent from both actors.  Encoded stack
    tokens are mean-pooled, combined with the current-container context, and
    passed through a value head.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_stacks: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vf_dim: int = 128,
        include_container_in_encoder: bool = True,
        container_start: int = 2,
        container_dim: int = 1,
    ) -> None:
        super().__init__()

        self.n_stacks = n_stacks
        self.embed_dim = embed_dim

        self.encoder = TransformerFeaturesExtractor(
            observation_space=observation_space,
            n_stacks=n_stacks,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            include_container_in_encoder=include_container_in_encoder,
            container_start=container_start,
            container_dim=container_dim,
        )
        self.critic_step_proj = nn.Linear(container_dim, embed_dim, bias=False)
        self.critic_head = nn.Sequential(nn.Linear(embed_dim, vf_dim), nn.ReLU())
        self.value_head = nn.Linear(vf_dim, 1)

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        """Values (B,) of global states (B, obs_dim)."""

        features = self.encoder(global_states)
        enc_size = self.n_stacks * self.embed_dim

        pooled = features[:, :enc_size].view(-1, self.n_stacks, self.embed_dim).mean(dim=1)
        context = self.critic_step_proj(features[:, enc_size:])

        return self.value_head(self.critic_head(pooled + context)).squeeze(-1)
