"""
Differentiated-Observation Joint Hierarchical Policy.
Training converges to a policy clearly better than
random, but has not been observed to reach optimal stacking solutions.

Implements an autoregressive policy where the bay head and stack head
receive different observation features:
  - Bay head:   bay-level summary features  (B, n_bays, bay_f_dim)
  - Stack head: per-stack features (relative indexing) for the *selected* bay
               (B, n_rows_per_bay, stack_f_dim)

The flat observation produced by StackEnv (obs_typehierarchical_diff_obs)
is laid out as[bay_features.flatten() | stack_features.flatten()].

The action space remainsDiscrete(n_stacks) for MaskablePPO compatibility.
The global action encodes bay_idx * n_rows_per_bay + local_stack_idx.

Critic : mean-pool(bay_encoder) + mean-pool(stack_encoder)
  + proj(container_feats)  →  MLP  →  (B, vf_dim)

Architecture

    obs (B, total_dim) → split into bay_features & stack_features
      → DiffObsTransformerActorCritic (mlp_extractor):
          ├─ bay_input_proj → bay_encoder → bay_decoder  → bay_logits (B, n_bays)
          │    mask with bay_mask
          │    sample bay
          ├─ extract stack_features[bay] → (B, n_rows, stack_f_dim)
          │    stack_input_proj → stack_encoder → stack_decoder → stack_logits (B, n_rows)
          │    mask with local_mask
          │    sample local_stack
          └─ critic: pool(bay_enc) + pool(stack_enc) + proj(cont) → MLP → (B, vf_dim)
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
# Transformer actor-critic module with differentiated observations
# ====================================================================


class DiffObsTransformerActorCritic(nn.Module):
    """
    Autoregressive Transformer actor-critic with differentiated observations
    for bay and stack heads.

    Parameters
    ----------
    n_bays         : int — number of bays
    n_rows_per_bay : int — stacks per bay
    bay_f_dim      : int — features per bay token (group_num + 4)
    stack_f_dim    : int — features per stack token (5 * group_num + 5)
    group_num      : int — number of container groups
    embed_dim      : int — transformer embedding dimension
    n_heads        : int — attention heads
    n_layers       : int — transformer encoder layers
    dropout        : float
    vf_dim         : int — critic MLP hidden size
    tanh_clipping  : float — logit clipping for PointerDecoder
    """

    def __init__(
        self,
        n_bays: int,
        n_rows_per_bay: int,
        bay_f_dim: int,
        stack_f_dim: int,
        group_num: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vf_dim: int = 128,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()

        self.n_bays = n_bays
        self.n_rows_per_bay = n_rows_per_bay
        self.n_stacks = n_bays * n_rows_per_bay
        self.bay_f_dim = bay_f_dim
        self.stack_f_dim = stack_f_dim
        self.group_num = group_num
        self.embed_dim = embed_dim

        # SB3 reads these to build action_net and value_net
        self.latent_dim_pi = self.n_stacks  # not used directly (logits come from decoder)
        self.latent_dim_vf = vf_dim

        # Container one-hot slice within stack features:
        # stack features layout: [group_counts(G), num_occupied, num_empty,
        #                         current_group_onehot(G), left_max_group(G),
        #                         right_max_group(G), vessel_remaining(G),
        #                         is_empty, has_remaining, positional_index]
        self._cont_start = group_num + 2
        self._cont_end = 2 * group_num + 2

        # Precompute split sizes
        self._bay_total = n_bays * bay_f_dim
        self._stack_total = self.n_stacks * stack_f_dim

        # --- Bay encoder ---
        self.bay_input_proj = nn.Linear(bay_f_dim, embed_dim)
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

        # --- Stack encoder (processes n_rows_per_bay tokens for selected bay) ---
        self.stack_input_proj = nn.Linear(stack_f_dim, embed_dim)
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
            n_stacks=n_rows_per_bay,
            group_num=group_num,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
        )

        # --- Critic (joint: bay + stack encoder outputs) ---
        self.critic_cont_proj = nn.Linear(group_num, embed_dim, bias=False)
        self.critic_head = nn.Sequential(
            nn.Linear(3 * embed_dim, vf_dim),
            nn.ReLU(),
        )

    # ------------------------------------------------------------------
    # Observation splitting
    # ------------------------------------------------------------------

    def _split_obs(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split flat obs into bay features, stack features, and container features.

        Args:
            features: (B, total_dim)

        Returns:
            bay_feats     : (B, n_bays, bay_f_dim)
            stack_feats   : (B, n_bays, n_rows_per_bay, stack_f_dim)
            container_feats : (B, group_num) — current container one-hot
        """
        B = features.shape[0]
        bay_flat = features[:, : self._bay_total]
        stack_flat = features[:, self._bay_total :]

        bay_feats = bay_flat.view(B, self.n_bays, self.bay_f_dim)
        stack_feats = stack_flat.view(
            B, self.n_bays, self.n_rows_per_bay, self.stack_f_dim
        )

        # Extract container one-hot from first stack of first bay
        container_feats = stack_feats[:, 0, 0, self._cont_start : self._cont_end].clone()

        return bay_feats, stack_feats, container_feats

    # ------------------------------------------------------------------
    # Per-head forwards
    # ------------------------------------------------------------------

    def _encode_bays(
        self, bay_feats: torch.Tensor
    ) -> torch.Tensor:
        """Encode bay features: (B, n_bays, bay_f_dim) → (B, n_bays, D)."""
        x = self.bay_input_proj(bay_feats)
        return self.bay_encoder(x)

    def _encode_stacks(
        self, selected_stack_feats: torch.Tensor
    ) -> torch.Tensor:
        """Encode stack features for one bay: (B, n_rows, stack_f_dim) → (B, n_rows, D)."""
        x = self.stack_input_proj(selected_stack_feats)
        return self.stack_encoder(x)

    # ------------------------------------------------------------------
    # Joint forward
    # ------------------------------------------------------------------

    def forward_all(
        self,
        features: torch.Tensor,
        bay_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: bay logits, stack logits (for selected bay), critic latent.

        Args:
            features    : (B, total_dim)
            bay_actions : (B,) — selected bay indices

        Returns:
            bay_logits   : (B, n_bays)
            stack_logits : (B, n_rows_per_bay)
            latent_vf    : (B, vf_dim)
        """
        B = features.shape[0]
        bay_feats, stack_feats, container_feats = self._split_obs(features)

        # --- Bay path ---
        GE_bay = self._encode_bays(bay_feats)            # (B, n_bays, D)
        bay_logits = self.bay_decoder(GE_bay, container_feats)  # (B, n_bays)

        # --- Stack path (only the selected bay) ---
        selected_stacks = stack_feats[
            torch.arange(B, device=features.device), bay_actions
        ]  # (B, n_rows_per_bay, stack_f_dim)
        GE_stack = self._encode_stacks(selected_stacks)   # (B, n_rows, D)
        stack_logits = self.stack_decoder(GE_stack, container_feats)  # (B, n_rows)

        # --- Critic (joint) ---
        bay_pool = GE_bay.mean(dim=1)      # (B, D)
        stack_pool = GE_stack.mean(dim=1)   # (B, D)
        cont_proj = self.critic_cont_proj(container_feats)  # (B, D)
        latent_vf = self.critic_head(
            torch.cat([bay_pool, stack_pool, cont_proj], dim=-1)
        )  # (B, vf_dim)

        return bay_logits, stack_logits, latent_vf

    def forward_bay_only(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return bay logits and container features (used for sampling bay first)."""
        bay_feats, stack_feats, container_feats = self._split_obs(features)
        GE_bay = self._encode_bays(bay_feats)
        bay_logits = self.bay_decoder(GE_bay, container_feats)
        return bay_logits, container_feats

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Critic-only forward using bay index 0 for stack encoding (fallback)."""
        bay_feats, stack_feats, container_feats = self._split_obs(features)
        GE_bay = self._encode_bays(bay_feats)
        # Use first bay stacks for critic when no bay action available
        GE_stack = self._encode_stacks(stack_feats[:, 0])
        bay_pool = GE_bay.mean(dim=1)
        stack_pool = GE_stack.mean(dim=1)
        cont_proj = self.critic_cont_proj(container_feats)
        return self.critic_head(
            torch.cat([bay_pool, stack_pool, cont_proj], dim=-1)
        )


# ====================================================================
# Policy mixin — autoregressive forward / evaluate
# ====================================================================


class _DiffObsJointPolicyMixin:
    """
    Mixin that overrides forward(), evaluate_actions(), and
    _predict() for autoregressive bay → stack sampling with
    differentiated observations.

    Expects self.mlp_extractor to be a DiffObsTransformerActorCritic.
    """

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------

    def _prepare_masks(
        self, action_masks: np.ndarray | torch.Tensor | None, device: torch.device
    ) -> torch.Tensor | None:
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
        """Derive (B, n_bays) mask from (B, n_stacks) mask."""
        n_bays = self.mlp_extractor.n_bays
        n_rows = self.mlp_extractor.n_rows_per_bay
        return stack_mask.view(-1, n_bays, n_rows).any(dim=-1)

    def _build_local_stack_mask(
        self,
        bay_actions: torch.Tensor,
        stack_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build (B, n_rows_per_bay) local mask for selected bay."""
        n_bays = self.mlp_extractor.n_bays
        n_rows = self.mlp_extractor.n_rows_per_bay
        B = bay_actions.shape[0]
        # Reshape to (B, n_bays, n_rows) and select the chosen bay
        all_masks = stack_mask.view(B, n_bays, n_rows)
        return all_masks[torch.arange(B, device=bay_actions.device), bay_actions]

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive forward: sample bay → encode selected bay stacks → sample local stack → value."""
        features = self.extract_features(obs)
        n_rows = self.mlp_extractor.n_rows_per_bay

        stack_mask = self._prepare_masks(action_masks, features.device)

        # --- Step 1: Get bay logits (only needs bay features) ---
        bay_logits, _ = self.mlp_extractor.forward_bay_only(features)

        if stack_mask is not None:
            bay_mask = self._derive_bay_mask(stack_mask)
            bay_logits = bay_logits.masked_fill(~bay_mask, float("-inf"))

        bay_dist = Categorical(logits=bay_logits)
        if deterministic:
            bay_actions = bay_logits.argmax(dim=-1)
        else:
            bay_actions = bay_dist.sample()
        bay_log_prob = bay_dist.log_prob(bay_actions)

        # --- Step 2: Full forward with known bay actions ---
        _, stack_logits, latent_vf = self.mlp_extractor.forward_all(
            features, bay_actions
        )
        values = self.value_net(latent_vf)

        # --- Step 3: Sample local stack ---
        if stack_mask is not None:
            local_mask = self._build_local_stack_mask(bay_actions, stack_mask)
            stack_logits = stack_logits.masked_fill(~local_mask, float("-inf"))

        stack_dist = Categorical(logits=stack_logits)
        if deterministic:
            local_actions = stack_logits.argmax(dim=-1)
        else:
            local_actions = stack_dist.sample()
        stack_log_prob = stack_dist.log_prob(local_actions)

        # Convert to global action
        global_actions = bay_actions * n_rows + local_actions
        joint_log_prob = bay_log_prob + stack_log_prob
        global_actions = global_actions.reshape((-1, *self.action_space.shape))

        return global_actions, values, joint_log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Evaluate stored global actions under the current policy."""
        features = self.extract_features(obs)
        n_rows = self.mlp_extractor.n_rows_per_bay

        stack_mask = self._prepare_masks(action_masks, features.device)

        # Recover bay and local stack from global action
        bay_actions = (actions // n_rows).long()
        local_actions = (actions % n_rows).long()

        # Full forward with recovered bay actions
        bay_logits, stack_logits, latent_vf = self.mlp_extractor.forward_all(
            features, bay_actions
        )
        values = self.value_net(latent_vf)

        # --- Bay distribution ---
        if stack_mask is not None:
            bay_mask = self._derive_bay_mask(stack_mask)
            bay_logits = bay_logits.masked_fill(~bay_mask, float("-inf"))
        bay_dist = Categorical(logits=bay_logits)
        bay_log_prob = bay_dist.log_prob(bay_actions)
        bay_entropy = bay_dist.entropy()

        # --- Local stack distribution ---
        if stack_mask is not None:
            local_mask = self._build_local_stack_mask(bay_actions, stack_mask)
            stack_logits = stack_logits.masked_fill(~local_mask, float("-inf"))
        stack_dist = Categorical(logits=stack_logits)
        stack_log_prob = stack_dist.log_prob(local_actions)
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
        actions, _, _ = self.forward(
            observation, deterministic=deterministic, action_masks=action_masks
        )
        return actions


# ====================================================================
# Concrete policy class
# ====================================================================


class MaskableDiffObsJointTransformerPolicy(
    _DiffObsJointPolicyMixin, MaskableActorCriticPolicy
):
    """
    Differentiated-observation joint hierarchical Transformer policy for MaskablePPO.

    Required policy_kwargs :

        n_bays         : int  — number of bays
        n_rows_per_bay : int  — rows per bay
        bay_f_dim      : int  — bay feature dimension (group_num + 4)
        stack_f_dim    : int  — stack feature dimension (5 * group_num + 5)
        group_num      : int  — number of container groups
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        n_bays: int | None = None,
        n_rows_per_bay: int | None = None,
        bay_f_dim: int | None = None,
        stack_f_dim: int | None = None,
        group_num: int | None = None,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vf_dim: int = 128,
        tanh_clipping: float = 10.0,
        **kwargs,
    ) -> None:
        assert n_bays is not None, "n_bays required"
        assert n_rows_per_bay is not None, "n_rows_per_bay required"
        assert bay_f_dim is not None, "bay_f_dim required"
        assert stack_f_dim is not None, "stack_f_dim required"
        assert group_num is not None, "group_num required"

        self._diff_n_bays = n_bays
        self._diff_n_rows_per_bay = n_rows_per_bay
        self._diff_bay_f_dim = bay_f_dim
        self._diff_stack_f_dim = stack_f_dim
        self._diff_group_num = group_num
        self._diff_embed_dim = embed_dim
        self._diff_n_heads = n_heads
        self._diff_n_layers = n_layers
        self._diff_dropout = dropout
        self._diff_vf_dim = vf_dim
        self._diff_tanh_clipping = tanh_clipping

        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DiffObsTransformerActorCritic(
            n_bays=self._diff_n_bays,
            n_rows_per_bay=self._diff_n_rows_per_bay,
            bay_f_dim=self._diff_bay_f_dim,
            stack_f_dim=self._diff_stack_f_dim,
            group_num=self._diff_group_num,
            embed_dim=self._diff_embed_dim,
            n_heads=self._diff_n_heads,
            n_layers=self._diff_n_layers,
            dropout=self._diff_dropout,
            vf_dim=self._diff_vf_dim,
            tanh_clipping=self._diff_tanh_clipping,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        # Logits are produced by the mlp_extractor; action_net is a no-op.
        self.action_net = nn.Identity()
