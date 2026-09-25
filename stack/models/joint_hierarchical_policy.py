"""
Joint Hierarchical Policy for simultaneous bay and stack selection.

Implements an autoregressive policy where:
  1. Bay head selects which bay to place the container in
  2. Stack head selects which stack within that bay (conditioned on bay)

Both heads receive the full observation but have separate backbones.
The action space remains Discrete(n_stacks) for compatibility with
StackEnv and MaskablePPO — the bay selection is internal to the policy.

The joint log-probability is log p(bay) + log p(stack | bay), used
as a single scalar in the standard PPO clipped surrogate objective.

Note: the bay index is implicitly encoded in the stack action
(bay_idx = action // n_rows_per_bay), so during evaluate_actions
the bay action can be recovered from the stored stack action without
extra storage.

Supports two architectures:
  - MLP: three independent [256, 256, out] networks
  - Transformer: two independent TransformerEncoder + PointerDecoder
    pairs (bay encoder with bay pooling, stack encoder), critic reuses
    the stack encoder output.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from .transformer_policy import PointerDecoder


# ====================================================================
# MLP actor-critic module
# ====================================================================


class JointMlpActorCritic(nn.Module):
    """
    Separate-backbone MLP actor-critic for joint bay+stack selection.

    Three independent MLPs (no shared parameters):
      - bay_net   : obs → bay logits   (B, n_bays)
      - stack_net : obs → stack logits (B, n_stacks)
      - critic_net: obs → value latent (B, vf_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        n_stacks: int,
        n_bays: int,
        n_rows_per_bay: int,
        vf_dim: int = 128,
    ) -> None:
        super().__init__()

        assert n_stacks == n_bays * n_rows_per_bay

        self.n_stacks = n_stacks
        self.n_bays = n_bays
        self.n_rows_per_bay = n_rows_per_bay

        # SB3 reads these to build action_net and value_net
        self.latent_dim_pi = n_stacks
        self.latent_dim_vf = vf_dim

        self.bay_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_bays),
        )
        self.stack_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_stacks),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, vf_dim),
        )

    def forward_bay(self, features: torch.Tensor) -> torch.Tensor:
        return self.bay_net(features)

    def forward_stack(self, features: torch.Tensor) -> torch.Tensor:
        return self.stack_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_net(features)

    def forward_all(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.forward_bay(features),
            self.forward_stack(features),
            self.forward_critic(features),
        )


# ====================================================================
# Transformer actor-critic module
# ====================================================================


class JointTransformerActorCritic(nn.Module):
    """
    Separate-encoder Transformer actor-critic for joint bay+stack selection.

    Two independent TransformerEncoder + PointerDecoder pairs:
      - Bay path  : bay_encoder → pool → bay_decoder  → (B, n_bays)
      - Stack path: stack_encoder → stack_decoder              → (B, n_stacks)
      - Critic    : reuses stack_encoder output → mean-pool + proj → MLP → (B, vf_dim)

    Observation preprocessing (reshape + container extraction) is shared
    but has no learnable parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        n_stacks: int,
        n_bays: int,
        n_rows_per_bay: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vf_dim: int = 128,
        tanh_clipping: float = 10.0,
        include_container_in_encoder: bool = True,
        container_start: int | None = None,
        container_dim: int | None = None,
    ) -> None:
        super().__init__()

        assert n_stacks == n_bays * n_rows_per_bay
        assert feature_dim % n_stacks == 0

        f_per_stack = feature_dim // n_stacks

        # When container_start and container_dim are explicitly provided
        # (e.g. for stack_features_v3), skip the auto-derive logic.
        if container_start is not None and container_dim is not None:
            group_num = container_dim
            cont_start = container_start
            cont_end = container_start + container_dim
        else:
            if (f_per_stack - 5) % 5 != 0:
                raise ValueError(
                    f"f_per_stack={f_per_stack} does not satisfy "
                    f"(f_per_stack - 5) % 5 == 0.  Requires "
                    f"observation_type='stack_features' with pos_embeddings=False "
                    f"or explicit container_start/container_dim."
                )
            group_num = (f_per_stack - 5) // 5
            cont_start = group_num + 2
            cont_end = 2 * group_num + 2

        self.n_stacks = n_stacks
        self.n_bays = n_bays
        self.n_rows_per_bay = n_rows_per_bay
        self.embed_dim = embed_dim
        self.group_num = group_num
        self.include_container_in_encoder = include_container_in_encoder

        self._cont_start = cont_start
        self._cont_end = cont_end

        enc_f = f_per_stack if include_container_in_encoder else f_per_stack - group_num

        # SB3 reads these to build action_net and value_net
        self.latent_dim_pi = n_stacks
        self.latent_dim_vf = vf_dim

        # Bay encoder
        self.bay_input_proj = nn.Linear(enc_f, embed_dim)
        bay_enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.bay_encoder = nn.TransformerEncoder(bay_enc_layer, num_layers=n_layers)
        self.bay_decoder = PointerDecoder(
            embed_dim=embed_dim,
            n_stacks=n_bays,
            group_num=group_num,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
        )

        # Stack encoder
        self.stack_input_proj = nn.Linear(enc_f, embed_dim)
        stack_enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.stack_encoder = nn.TransformerEncoder(
            stack_enc_layer, num_layers=n_layers
        )
        self.stack_decoder = PointerDecoder(
            embed_dim=embed_dim,
            n_stacks=n_stacks,
            group_num=group_num,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
        )

        # Critic (reuses stack encoder)
        self.critic_step_proj = nn.Linear(group_num, embed_dim, bias=False)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, vf_dim),
            nn.ReLU(),
        )

    # Helper functions for shared preprocessing and pooling

    def _preprocess(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape flat obs (B, N*F) → (B, N, F) and extract container features
        """
        B = features.shape[0]
        x = features.view(B, self.n_stacks, -1)  # (B, N, F)
        container_feats = x[:, 0, self._cont_start : self._cont_end].clone()  # (B, G)
        if not self.include_container_in_encoder:
            x = torch.cat(
                [x[:, :, : self._cont_start], x[:, :, self._cont_end :]], dim=-1
            )
        return x, container_feats

    def _pool_to_bays(self, GE: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool stack embeddings within each bay: (B, N, D) → (B, n_bays, D)
        """
        B, N, D = GE.shape
        return GE.view(B, self.n_bays, self.n_rows_per_bay, D).mean(dim=2)

    # ------------------------------------------------------------------
    # Per-head forwards
    # ------------------------------------------------------------------

    def forward_bay(self, features: torch.Tensor) -> torch.Tensor:
        x, container_feats = self._preprocess(features)
        x = self.bay_input_proj(x)
        GE = self.bay_encoder(x)
        GE_bay = self._pool_to_bays(GE)
        return self.bay_decoder(GE_bay, container_feats)

    def forward_stack(self, features: torch.Tensor) -> torch.Tensor:
        GE_stack, container_feats = self._run_stack_encoder(features)
        return self.stack_decoder(GE_stack, container_feats)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        GE_stack, container_feats = self._run_stack_encoder(features)
        pooled = GE_stack.mean(dim=1)
        C_k = self.critic_step_proj(container_feats)
        return self.critic_head(pooled + C_k)

    def _run_stack_encoder(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run stack encoder once and return (GE_stack, container_feats).
        """
        x, container_feats = self._preprocess(features)
        x = self.stack_input_proj(x)
        GE_stack = self.stack_encoder(x)
        return GE_stack, container_feats

    # ------------------------------------------------------------------
    # Efficient Joint forward (avoids running stack encoder twice)
    # ------------------------------------------------------------------

    def forward_all(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, container_feats = self._preprocess(features)

        # Bay branch
        bay_x = self.bay_input_proj(x)
        GE_bay = self.bay_encoder(bay_x)
        bay_logits = self.bay_decoder(self._pool_to_bays(GE_bay), container_feats)

        # Stack + critic branch (shared encoder pass)
        stack_x = self.stack_input_proj(x)
        GE_stack = self.stack_encoder(stack_x)
        stack_logits = self.stack_decoder(GE_stack, container_feats)

        pooled = GE_stack.mean(dim=1)
        C_k = self.critic_step_proj(container_feats)
        latent_vf = self.critic_head(pooled + C_k)

        return bay_logits, stack_logits, latent_vf


# ====================================================================
# Policy mixin — overrides necessary functions for compatibilty with sb3
# ====================================================================


class _JointPolicyMixin:
    """
    Mixin that overrides forward(), evaluate_actions(), and _predict() for sb3 compatibility.

    Expects self.mlp_extractor to expose:
      - forward_all(features) → (bay_logits, stack_logits, latent_vf)
      - forward_critic(features) → latent_vf
      - n_bays, n_rows_per_bay, n_stacks attributes
    """

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------

    def _prepare_masks(
        self, action_masks: np.ndarray | torch.Tensor | None, device: torch.device
    ) -> torch.Tensor | None:
        """
        Convert action_masks to a bool tensor with a batch dimension.
        """
        if action_masks is None:
            return None
        if not isinstance(action_masks, torch.Tensor):
            action_masks = torch.as_tensor(
                action_masks, dtype=torch.bool, device=device
            )
        action_masks = action_masks.bool()
        if action_masks.dim() == 1:
            action_masks = action_masks.unsqueeze(0)
        return action_masks

    def _derive_bay_mask(self, stack_mask: torch.Tensor) -> torch.Tensor:
        """
        Derive (B, n_bays) mask from (B, n_stacks) mask.
        """
        n_bays = self.mlp_extractor.n_bays
        n_rows = self.mlp_extractor.n_rows_per_bay
        return stack_mask.view(-1, n_bays, n_rows).any(dim=-1)

    def _build_conditional_stack_mask(
        self,
        bay_actions: torch.Tensor,
        stack_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build (B, n_stacks) mask: only valid stacks in the selected bay.
        """
        n_stacks = self.mlp_extractor.n_stacks
        n_rows = self.mlp_extractor.n_rows_per_bay
        bay_of_stack = (
            torch.arange(n_stacks, device=bay_actions.device) // n_rows
        )  # (n_stacks,)
        in_bay = bay_of_stack.unsqueeze(0) == bay_actions.unsqueeze(1)  # (B, n_stacks)
        return in_bay & stack_mask

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward and action logits calculation: sample bay → conditional stack → value.
        """
        features = self.extract_features(obs)
        bay_logits, stack_logits, latent_vf = self.mlp_extractor.forward_all(features)
        values = self.value_net(latent_vf)

        stack_mask = self._prepare_masks(action_masks, features.device)

        # --- Bay sampling ---
        if stack_mask is not None:
            bay_mask = self._derive_bay_mask(stack_mask)
            bay_logits = bay_logits.masked_fill(~bay_mask, float("-inf"))

        bay_dist = Categorical(logits=bay_logits)
        if deterministic:
            bay_actions = bay_logits.argmax(dim=-1)
        else:
            bay_actions = bay_dist.sample()
        bay_log_prob = bay_dist.log_prob(bay_actions)

        # --- Conditional stack sampling ---
        if stack_mask is not None:
            cond_mask = self._build_conditional_stack_mask(bay_actions, stack_mask)
            stack_logits = stack_logits.masked_fill(~cond_mask, float("-inf"))

        stack_dist = Categorical(logits=stack_logits)
        if deterministic:
            stack_actions = stack_logits.argmax(dim=-1)
        else:
            stack_actions = stack_dist.sample()
        stack_log_prob = stack_dist.log_prob(stack_actions)

        joint_log_prob = bay_log_prob + stack_log_prob
        stack_actions = stack_actions.reshape((-1, *self.action_space.shape))

        return stack_actions, values, joint_log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Evaluate stored stack actions under the current policy.
        """
        features = self.extract_features(obs)
        bay_logits, stack_logits, latent_vf = self.mlp_extractor.forward_all(features)
        values = self.value_net(latent_vf)

        stack_mask = self._prepare_masks(action_masks, features.device)
        n_rows = self.mlp_extractor.n_rows_per_bay

        # Recover the bay from the stored stack action
        bay_actions = actions // n_rows

        # --- Bay distribution ---
        if stack_mask is not None:
            bay_mask = self._derive_bay_mask(stack_mask)
            bay_logits = bay_logits.masked_fill(~bay_mask, float("-inf"))
        bay_dist = Categorical(logits=bay_logits)
        bay_log_prob = bay_dist.log_prob(bay_actions)
        bay_entropy = bay_dist.entropy()

        # --- Conditional stack distribution ---
        if stack_mask is not None:
            cond_mask = self._build_conditional_stack_mask(bay_actions, stack_mask)
            stack_logits = stack_logits.masked_fill(~cond_mask, float("-inf"))
        stack_dist = Categorical(logits=stack_logits)
        stack_log_prob = stack_dist.log_prob(actions)
        stack_entropy = stack_dist.entropy()

        return (
            values,
            bay_log_prob + stack_log_prob,
            bay_entropy + stack_entropy,
        )

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Used by model.predict() during evaluation.
        """
        actions, _, _ = self.forward(
            observation, deterministic=deterministic, action_masks=action_masks
        )
        return actions


# ====================================================================
# Concrete policy classes
# ====================================================================


class MaskableJointMlpPolicy(_JointPolicyMixin, MaskableActorCriticPolicy):
    """
    Joint hierarchical MLP policy for MaskablePPO.

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
        vf_dim: int = 128,
        **kwargs,
    ) -> None:
        assert n_stacks is not None, "n_stacks required"
        assert n_bays is not None, "n_bays required"
        assert n_rows_per_bay is not None, "n_rows_per_bay required"

        self._joint_n_stacks = n_stacks
        self._joint_n_bays = n_bays
        self._joint_n_rows_per_bay = n_rows_per_bay
        self._joint_vf_dim = vf_dim

        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = JointMlpActorCritic(
            feature_dim=self.features_dim,
            n_stacks=self._joint_n_stacks,
            n_bays=self._joint_n_bays,
            n_rows_per_bay=self._joint_n_rows_per_bay,
            vf_dim=self._joint_vf_dim,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        # Logits are produced by the mlp_extractor; action_net is a no-op.
        self.action_net = nn.Identity()


class MaskableJointTransformerPolicy(_JointPolicyMixin, MaskableActorCriticPolicy):
    """
    Joint hierarchical Transformer policy for MaskablePPO.

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

        self._joint_n_stacks = n_stacks
        self._joint_n_bays = n_bays
        self._joint_n_rows_per_bay = n_rows_per_bay
        self._joint_embed_dim = embed_dim
        self._joint_n_heads = n_heads
        self._joint_n_layers = n_layers
        self._joint_dropout = dropout
        self._joint_vf_dim = vf_dim
        self._joint_tanh_clipping = tanh_clipping
        self._joint_include_container = include_container_in_encoder
        self._joint_container_start = container_start
        self._joint_container_dim = container_dim

        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = JointTransformerActorCritic(
            feature_dim=self.features_dim,
            n_stacks=self._joint_n_stacks,
            n_bays=self._joint_n_bays,
            n_rows_per_bay=self._joint_n_rows_per_bay,
            embed_dim=self._joint_embed_dim,
            n_heads=self._joint_n_heads,
            n_layers=self._joint_n_layers,
            dropout=self._joint_dropout,
            vf_dim=self._joint_vf_dim,
            tanh_clipping=self._joint_tanh_clipping,
            include_container_in_encoder=self._joint_include_container,
            container_start=self._joint_container_start,
            container_dim=self._joint_container_dim,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
