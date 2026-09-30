"""
Masked Transformer + Pointer Network actor for both Agent B and Agent R,
built from transformer_policy.py's TransformerFeaturesExtractor and
PointerDecoder:

    obs (B, n_stacks * F) -> encoder -> stack tokens (B, n_stacks, D)
      -> mean-pool each run of tokens_per_action tokens -> (B, n_actions, D)
      -> PointerDecoder -> logits (B, n_actions), invalid actions -inf

Agent B sees all n_bays * n_rows tokens pooled per bay (tokens_per_action
= n_rows); Agent R sees the selected bay's n_rows tokens, one per action.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical

from .transformer_policy import PointerDecoder, TransformerFeaturesExtractor


class PointerActor(nn.Module):
    """Categorical actor over n_stacks // tokens_per_action actions; each
    action's tokens must be contiguous (StackEnv's tokens are bay-major)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_stacks: int,
        tokens_per_action: int = 1,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        tanh_clipping: float = 10.0,
        include_container_in_encoder: bool = True,
        container_start: int = 2,
        container_dim: int = 1,
    ) -> None:
        super().__init__()
        if n_stacks % tokens_per_action:
            raise ValueError(f"n_stacks={n_stacks} must be a multiple of tokens_per_action={tokens_per_action}.")

        self.n_stacks = n_stacks
        self.tokens_per_action = tokens_per_action
        self.n_actions = n_stacks // tokens_per_action
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
        self.decoder = PointerDecoder(
            embed_dim=embed_dim,
            n_stacks=self.n_actions,
            group_num=container_dim,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
        )

    def forward(self, observations: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Action logits (B, n_actions), -inf where the mask is False."""

        features = self.encoder(observations)
        enc_size = self.n_stacks * self.embed_dim
        action_embeddings = features[:, :enc_size].view(
            -1, self.n_actions, self.tokens_per_action, self.embed_dim
        ).mean(dim=2)

        logits = self.decoder(action_embeddings, features[:, enc_size:])
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, float("-inf"))
        return logits

    # Categorical(validate_args=False) everywhere: validation would sync with
    # the GPU on every call, and select_hierarchical_action() already checks
    # (once, on the CPU) that every row has a valid action.

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actions (B,) and their log-probabilities (B,)."""

        logits = self(observations, action_masks)
        distribution = Categorical(logits=logits, validate_args=False)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-probabilities and entropies (B,) of stored actions under the
        current parameters, for the PPO ratio."""

        distribution = Categorical(logits=self(observations, action_masks), validate_args=False)
        return distribution.log_prob(actions), distribution.entropy()
