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
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("CentralizedCritic requires a Box observation space.")
        if n_stacks <= 0 or embed_dim <= 0 or vf_dim <= 0:
            raise ValueError("n_stacks, embed_dim, and vf_dim must be positive.")
        if container_dim <= 0:
            raise ValueError("container_dim must be positive.")

        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self.container_dim = container_dim
        self._enc_size = n_stacks * embed_dim

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

    def _split_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split extractor output into stack embeddings and context."""
        if features.ndim != 2:
            raise ValueError(f"Expected encoded features (B, F), got {features.shape}.")
        stack_features = features[:, : self._enc_size].reshape(
            -1, self.n_stacks, self.embed_dim
        )
        container_features = features[:, self._enc_size :]
        if container_features.shape[-1] != self.container_dim:
            raise RuntimeError(
                f"Expected {self.container_dim} container features, "
                f"got {container_features.shape[-1]}."
            )
        return stack_features, container_features

    def forward_latent(self, global_states: torch.Tensor) -> torch.Tensor:
        """Return the critic hidden representation with shape ``(B, vf_dim)``."""
        if global_states.ndim == 1:
            global_states = global_states.unsqueeze(0)
        if global_states.ndim != 2:
            raise ValueError(
                f"Global states must have shape (obs_dim,) or (B, obs_dim), "
                f"got {global_states.shape}."
            )
        encoded = self.encoder(global_states.float())
        stack_features, container_features = self._split_features(encoded)
        pooled = stack_features.mean(dim=1)
        context = self.critic_step_proj(container_features)
        return self.critic_head(pooled + context)

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        """Return centralized values with shape ``(B,)``."""
        return self.value_head(self.forward_latent(global_states)).squeeze(-1)

    @torch.no_grad()
    def predict_values(self, global_states: torch.Tensor) -> torch.Tensor:
        """Return values without constructing an autograd graph."""
        return self(global_states)
