"""
Masked Transformer + Pointer Network actor shared by Agent B and Agent R.

Both actors are the existing transformer_policy.py architecture - the
same TransformerFeaturesExtractor encoder and PointerDecoder - and differ
only in which stack tokens they see and how tokens map to actions:

    Agent B (bays): all n_bays * n_rows stack tokens of the global
                    observation, mean-pooled per bay (tokens_per_action
                    = n_rows) into n_bays action tokens.

    Agent R (rows): the n_rows stack tokens of the selected bay, one
                    action per token (tokens_per_action = 1).

    obs (B, n_stacks * F)
      └─ TransformerFeaturesExtractor → GE (B, n_stacks, D), container (B, G)
      └─ mean-pool each tokens_per_action run → (B, n_actions, D)
      └─ PointerDecoder → logits (B, n_actions)
      └─ invalid actions → -inf
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical

from .transformer_policy import (
    PointerDecoder,
    TransformerFeaturesExtractor,
)


class PointerActor(nn.Module):
    """
    Categorical actor over n_stacks // tokens_per_action actions.

    Stack tokens must be ordered so that each action's tokens are
    contiguous, as StackEnv's bay-major stack_features_v3 layout is.
    """

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

        if n_stacks <= 0 or tokens_per_action <= 0 or n_stacks % tokens_per_action:
            raise ValueError(
                f"n_stacks={n_stacks} must be a positive multiple of "
                f"tokens_per_action={tokens_per_action}."
            )

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

    def forward(
        self,
        observations: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked action logits (B, n_actions); mask True = valid."""

        features = self.encoder(observations)
        enc_size = self.n_stacks * self.embed_dim

        action_embeddings = features[:, :enc_size].view(
            -1, self.n_actions, self.tokens_per_action, self.embed_dim
        ).mean(dim=2)

        logits = self.decoder(action_embeddings, features[:, enc_size:])

        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, float("-inf"))

        return logits

    def get_distribution(
        self,
        observations: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Categorical:
        # No argument validation: it synchronizes with the GPU on every
        # call. Callers guarantee every row has a valid action
        # (select_hierarchical_action() checks it once, on the CPU).
        return Categorical(logits=self(observations, action_masks), validate_args=False)

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

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = distribution.sample()

        return actions, distribution.log_prob(actions)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-probabilities (B,) and entropies (B,) of stored actions
        under the current parameters, for the PPO ratio."""

        distribution = self.get_distribution(observations, action_masks)

        return distribution.log_prob(actions), distribution.entropy()


def load_actor_state_dict(
    actor: PointerActor,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Strictly load actor parameters, also from checkpoints saved while
    the actor was wrapped in AgentB/AgentR (keys prefixed "policy.")."""

    actor.load_state_dict(
        {key.removeprefix("policy."): value for key, value in state_dict.items()}
    )
