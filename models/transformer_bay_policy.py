"""
Bay-level Transformer Pointer Network policy for the high-level agent.

Reuses TransformerFeaturesExtractor and PointerDecoder from
transformer_policy.py without modification.  The key addition is a
bay pooling layer that aggregates per-stack encoder embeddings into
per-bay embeddings before the pointer decoder produces bay-level logits.

Stack ordering in _create_stack_features is bay-major: for each odd bay
the rows are iterated in order, so contiguous chunks of n_rows_per_bay
stacks belong to the same physical bay.  This enables a simple reshape +
mean-pool for bay aggregation.

Architecture:

    obs (B, N_stacks * F)
      └─ TransformerFeaturesExtractor (shared, unchanged)
           → (B, N_stacks * D + G)
      └─ BayTransformerActorCritic
           _split_features → GE (B, N_stacks, D),  container_feats (B, G)
           _pool_to_bays   → GE_bay (B, N_bays, D)
           actor : PointerDecoder(GE_bay, container_feats) → (B, N_bays)
           critic: mean-pool(GE) + project(container_feats) → MLP → (B, vf_dim)
      └─ MaskableBayTransformerPolicy
           action_net = Identity  (pointer logits pass through)
           value_net  = Linear(vf_dim, 1)
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from models.transformer_policy import (
    TransformerFeaturesExtractor,
    PointerDecoder,
)


class BayTransformerActorCritic(nn.Module):
    """
    Receives the shared encoder output (B, N_stacks*D + G) and produces:
      - latent_pi: (B, N_bays)  — pointer logits over bays
      - latent_vf: (B, vf_dim)  — value estimate

    Parameters
    ----------
    feature_dim    : int — N_stacks * embed_dim + group_num
    n_stacks       : int — total yard stacks (bays * rows)
    n_bays         : int — number of physical (odd) bays == action_space.n
    n_rows_per_bay : int — rows per bay (yard_shape[1])
    embed_dim      : int — transformer model dimension
    group_num      : int — container one-hot size
    n_heads        : int — attention heads for PointerDecoder
    vf_dim         : int — hidden size of critic MLP
    tanh_clipping  : float — passed to PointerDecoder
    """

    def __init__(
        self,
        feature_dim: int,
        n_stacks: int,
        n_bays: int,
        n_rows_per_bay: int,
        embed_dim: int,
        group_num: int,
        n_heads: int = 4,
        vf_dim: int = 128,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()

        assert feature_dim == n_stacks * embed_dim + group_num, (
            f"feature_dim ({feature_dim}) != n_stacks*embed_dim+group_num "
            f"({n_stacks}*{embed_dim}+{group_num})"
        )
        assert n_stacks == n_bays * n_rows_per_bay, (
            f"n_stacks ({n_stacks}) != n_bays*n_rows_per_bay "
            f"({n_bays}*{n_rows_per_bay})"
        )

        self.n_stacks = n_stacks
        self.n_bays = n_bays
        self.n_rows_per_bay = n_rows_per_bay
        self.embed_dim = embed_dim
        self._enc_size = n_stacks * embed_dim

        # Used by SB3 ActorCriticPolicy for building action_net / value_net
        self.latent_dim_pi = n_bays
        self.latent_dim_vf = vf_dim

        # Pointer decoder operates on bay-level embeddings
        self.decoder = PointerDecoder(
            embed_dim=embed_dim,
            n_stacks=n_bays,  # decoder sees bays as "stacks"
            group_num=group_num,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
        )

        # Critic: mean-pool all stack embeddings + container projection → MLP
        self.critic_step_proj = nn.Linear(group_num, embed_dim, bias=False)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, vf_dim),
            nn.ReLU(),
        )

    def _split_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split (B, N_stacks*D + G) into GE and container_feats.
        """
        enc_feats = features[:, : self._enc_size]
        container_feats = features[:, self._enc_size :]
        GE = enc_feats.view(features.shape[0], self.n_stacks, self.embed_dim)
        return GE, container_feats

    def _pool_to_bays(self, GE: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool stack embeddings within each bay.

        GE : (B, N_stacks, D)  where stacks are ordered bay-major
        Returns: (B, N_bays, D)
        """
        B, N, D = GE.shape
        # Reshape: (B, n_bays, n_rows_per_bay, D)
        GE_by_bay = GE.view(B, self.n_bays, self.n_rows_per_bay, D)
        # Mean-pool over rows within each bay
        return GE_by_bay.mean(dim=2)  # (B, n_bays, D)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        GE, container_feats = self._split_features(features)
        GE_bay = self._pool_to_bays(GE)  # (B, n_bays, D)
        return self.decoder(GE_bay, container_feats)  # (B, n_bays)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        GE, container_feats = self._split_features(features)
        pooled = GE.mean(dim=1)  # (B, D)
        C_k = self.critic_step_proj(container_feats)  # (B, D)
        return self.critic_head(pooled + C_k)  # (B, vf_dim)


class MaskableBayTransformerPolicy(MaskableActorCriticPolicy):
    """
    Drop-in replacement for MlpPolicy in MaskablePPO for the high-level
    bay-selection agent.

    The encoder processes all N_stacks tokens (same as the stack-level
    policy), but the actor produces logits over N_bays via bay pooling
    + pointer decoder.

    Required policy_kwargs
    ----------------------
    n_stacks       : int — total number of yard stacks (must be explicit
                     since action_space.n == n_bays, not n_stacks)
    n_rows_per_bay : int — number of rows per bay (yard_shape[1])

    Optional policy_kwargs
    ----------------------
    embed_dim      : int   (default 128)
    n_heads        : int   (default 4)
    n_layers       : int   (default 2)
    dropout        : float (default 0.1)
    vf_dim         : int   (default 128)
    tanh_clipping  : float (default 10.0)
    include_container_in_encoder : bool (default True)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        n_stacks: int = None,
        n_rows_per_bay: int = None,
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
        assert n_stacks is not None, "n_stacks must be provided for BayTransformerPolicy"
        assert n_rows_per_bay is not None, "n_rows_per_bay must be provided"

        self._bay_n_stacks = n_stacks
        self._bay_n_rows_per_bay = n_rows_per_bay
        self._bay_embed_dim = embed_dim
        self._bay_n_heads = n_heads
        self._bay_vf_dim = vf_dim
        self._bay_tanh_clipping = tanh_clipping

        # Inject the shared TransformerFeaturesExtractor
        # n_stacks is the actual number of stacks (not bays)
        kwargs["features_extractor_class"] = TransformerFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(
            n_stacks=n_stacks,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            include_container_in_encoder=include_container_in_encoder,
            container_start=container_start,
            container_dim=container_dim,
        )

        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # features_dim = N_stacks * embed_dim + group_num
        group_num = self.features_dim - self._bay_n_stacks * self._bay_embed_dim

        n_bays = self.action_space.n

        self.mlp_extractor = BayTransformerActorCritic(
            feature_dim=self.features_dim,
            n_stacks=self._bay_n_stacks,
            n_bays=n_bays,
            n_rows_per_bay=self._bay_n_rows_per_bay,
            embed_dim=self._bay_embed_dim,
            group_num=group_num,
            n_heads=self._bay_n_heads,
            vf_dim=self._bay_vf_dim,
            tanh_clipping=self._bay_tanh_clipping,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        # Pointer logits pass through unchanged (same pattern as stack-level policy)
        self.action_net = nn.Identity()
