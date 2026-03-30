"""
Transformer encoder and Pointer Network decoder model.
The policy architecture is designed for the "stack_features" observation layout without positional embeddings.

Let N be number of stacks in the yard (== num_actions) and F be the number of features per stack.
include_container_in_encoder controls whether the current-container (one-hot) is included in the encoder input or only used as side context for the decoder and critic.
B is the batch dimension. D is the embed_dim (hidden dim) for the transformer.
so the input observation is a flat vector of shape (N*F,). sb3 requires it to be flattened,
so reshaping it into (B, N, F) is the first step in the features extractor.

Then a forward pass through the transformer encoder produces
graph embeddings (GE) of shape (B, N, embed_dim).

The graph embeddings (GE) and the current_container features are used
in the PointerDecoder to produce the stack action logits (B, N)
and in the critic head to produce value estimates (B, vf_dim).

Observation layout (observation_type="stack_features", pos_embeddings=False):
    flat shape (N*F,)  where
        N = num_stacks = yard_bays * yard_rows  (== num_actions)
        F = 5 * group_num + 5  (features_per_stack)

Forward pass
------------
obs (B, N*F)
  └─ TransformerFeaturesExtractor (Encoder)
       reshape → (B,N,F)
       if include_container_in_encoder:
           keep full (B, N, F) for the encoder
       else:
           strip current_group_onehot → (B, N, F-G)
       input_proj → TransformerEncoder → GE (B, N, embed_dim)
  └─ TransformerActorCritic (Pointer Decoder + Critic)
       actor  : PointerDecoder(GE, container_feats) → (B, N)  pointer logits
       critic : mean-pool(GE) + project(container_feats) → critic MLP → (B, vf_dim)
  └─ MaskableTransformerPolicy (Actor and Critic heads)
       action_net = nn.Identity()   ← pointer logits pass through unchanged
       value_net  = nn.Linear(vf_dim, 1)
  └─ MaskablePPO (apply action masks)
       logits[action_mask == 0] = -inf

PointerDecoder (AM-style, Kool et al. 2019)
-------------------------------------------
Summary of Pointer Decoder :
GE (B, N, D)
  project_node → K, V, L   each (B, N, D)    [3-chunk split of Linear(D→3D)]
  project_global(GE.mean) → G  (B, D)         [global graph context]
  project_step(container_feats) → C_k (B, D)  [step context from current container]
  Q = (G + C_k).unsqueeze(1)                  (B, 1, D)
  H, _ = MultiheadAttention(Q, K, V)          (B, 1, D)  [cross-attention glimpse]
  G_k = out_proj(H)                           (B, 1, D)
  logits = bmm(G_k, L.T) / sqrt(D) → squeeze (B, N)
  logits = tanh_clipping * tanh(logits)
"""

import math
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Reshapes the flat (N*F,) observation into (B,N,F), optionally strips the
    current-container one-hot from each stack token, projects each token to
    embed_dim, runs a TransformerEncoder, then returns:
        cat(GE.flatten(1), container_feats)  shape (B, N*embed_dim + group_num)

    group_num is auto-inferred from the stack_features layout:
        features_per_stack = 5 * group_num + 5  (requires pos_embeddings=False)

    The current-container one-hot lives at positions [group_num+2 : 2*group_num+2]
    within each stack's feature slice — identical across all N stacks — so it is
    always read from stack index 0 for the decoder/critic.  When
    include_container_in_encoder is True the one-hot is kept in the encoder
    input so that self-attention can attend to the current container identity.

    Parameters
    ----------
    observation_space          : spaces.Box  shape=(N*F,)
    n_stacks                   : int   — N, number of yard stacks (== num_actions)
    embed_dim                  : int   — transformer model dimension
    n_heads                    : int   — number of attention heads
    n_layers                   : int   — number of TransformerEncoder layers
    dropout                    : float
    include_container_in_encoder : bool — if True (default) the current-container
        one-hot is included in the encoder input; if False it is stripped out
        and only used as side context for the decoder and critic.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_stacks: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        include_container_in_encoder: bool = True,
    ) -> None:
        # features_dim set after computing n_stacks * embed_dim
        super().__init__(observation_space, features_dim=1)  # placeholder

        obs_dim = observation_space.shape[0]
        assert obs_dim % n_stacks == 0, (
            f"obs_dim ({obs_dim}) must be divisible by n_stacks ({n_stacks})"
        )
        f_per_stack = obs_dim // n_stacks

        # Extracting group_num from stack_features layout: f_per_stack = 5*group_num + 5
        if (f_per_stack - 5) % 5 != 0:
            raise ValueError(
                f"f_per_stack={f_per_stack} does not satisfy "
                f"(f_per_stack - 5) % 5 == 0.  This policy requires "
                f"observation_type='stack_features' with pos_embeddings=False."
            )
        group_num = (f_per_stack - 5) // 5

        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self.group_num = group_num
        self.include_container_in_encoder = include_container_in_encoder
        # Slice offsets for the current-container one-hot within each stack feature vector:
        self._cont_start = group_num + 2
        self._cont_end = 2 * group_num + 2

        # Encoder input: keep or strip the current_group_onehot (G features)
        enc_f_per_stack = (
            f_per_stack if include_container_in_encoder else f_per_stack - group_num
        )
        self.input_proj = nn.Linear(enc_f_per_stack, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # features_dim = encoder output (N*embed_dim) + container one-hot (group_num)
        self._features_dim = n_stacks * embed_dim + group_num

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, N*F)
        b = observations.shape[0]
        x = observations.view(b, self.n_stacks, -1)  # (B, N, F)
        # Extract current_container features
        container_feats = x[:, 0, self._cont_start : self._cont_end].clone()  # (B, G)
        if not self.include_container_in_encoder:
            # Strip the one-hot so the encoder only sees yard-state features
            x = torch.cat(
                [x[:, :, : self._cont_start], x[:, :, self._cont_end :]], dim=-1
            )  # (B, N, F-G)
        x = self.input_proj(x)  # (B, N, embed_dim)
        x = self.encoder(x)  # (B, N, embed_dim)
        enc_flat = x.flatten(start_dim=1)  # (B, N*embed_dim)

        # container_feats is appended to the output from encoder (without any learnable parameters)
        # so that it can be used by decoder
        return torch.cat([enc_flat, container_feats], dim=-1)  # (B, N*embed_dim + G)


class PointerDecoder(nn.Module):
    """
    Attention-Model pointer decoder (Kool et al. 2019), adapted for the stack
    placement problem.  Runs at every timestep so it is not autoregressive.

    Given graph embeddings GE (B, N, D) from the encoder and the current-
    container one-hot (B, G), it computes:

        K, V, L = chunk( Linear(D→3D)(GE) )           3 x (B, N, D)
        G       = Linear(D→D)( GE.mean(dim=1) )        (B, D)
        C_k     = Linear(G→D)( container_feats )        (B, D)
        Q       = (G + C_k).unsqueeze(1)                (B, 1, D)
        H, _    = MultiheadAttention(Q, K, V)           (B, 1, D)
        G_k     = Linear(D→D)(H)                        (B, 1, D)
        logits  = (G_k . L.T) / sqrt(D)                 (B, N)

    Parameters
    ----------
    embed_dim      : int
    n_stacks       : int   — N, number of stacks (== num_actions)
    group_num      : int   — G, size of container one-hot
    n_heads        : int   — heads for the glimpse cross-attention
    tanh_clipping  : float — clip logits to [-C, C]
    """

    def __init__(
        self,
        embed_dim: int,
        n_stacks: int,
        group_num: int,
        n_heads: int,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_stacks = n_stacks
        self.tanh_clipping = tanh_clipping
        self._scale = math.sqrt(embed_dim)

        # 3-way split: K, V, L — one projection for all three
        self.project_node = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # Global graph context
        self.project_global = nn.Linear(embed_dim, embed_dim, bias=False)
        # Step context from current container one-hot
        self.project_step = nn.Linear(group_num, embed_dim, bias=False)
        # Multi-head cross-attention for the glimpse vector
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, bias=False
        )
        # Output projection on the glimpse
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        GE: torch.Tensor,
        container_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            GE              : (B, N, D)  — encoder graph embeddings
            container_feats : (B, G)     — current-container one-hot
        Returns:
            logits          : (B, N)
        """
        # Split node embeddings into keys, values and logit keys
        K, V, L = self.project_node(GE).chunk(3, dim=-1)  # each (B, N, D)

        # Global context vector mean pooling
        G_ctx = self.project_global(GE.mean(dim=1))  # (B, D)

        # Current Step context from current container
        C_k = self.project_step(container_feats)  # (B, D)

        # Query = global context + step context
        Q = (G_ctx + C_k).unsqueeze(1)  # (B, 1, D)

        # Glimpse via multi-head cross-attention
        H, _ = self.cross_attn(Q, K, V)  # (B, 1, D)
        G_k = self.out_proj(H)  # (B, 1, D)

        # Dot-product scoring against logit keys
        logits = torch.bmm(G_k, L.transpose(1, 2)) / self._scale  # (B, 1, N)
        logits = logits.squeeze(1)  # (B, N)

        # Tanh clipping helps stability by preventing large logit values
        if self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)

        return logits


class TransformerActorCritic(nn.Module):
    """
    Receives the shared encoder output (B, N*embed_dim + G) and produces:
      - latent_pi : (B, N)       — pointer logits from PointerDecoder
      - latent_vf : (B, vf_dim)  — value estimates

    Parameters
    ----------
    feature_dim    - N*embed_dim + group_num  (features_dim from extractor)
    n_stacks       - N
    embed_dim      - D, transformer embedding dimension
    group_num      - G, container one-hot size (auto-derived)
    n_heads        - attention heads for PointerDecoder
    vf_dim         - hidden size of the critic MLP head
    tanh_clipping  - passed to PointerDecoder
    """

    def __init__(
        self,
        feature_dim: int,
        n_stacks: int,
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

        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self._enc_size = n_stacks * embed_dim  # split boundary in the feature vector

        # Used by SB3 ActorCriticPolicy for building action_net / value_net
        self.latent_dim_pi = n_stacks
        self.latent_dim_vf = vf_dim

        # Pointer decoder for actor
        self.decoder = PointerDecoder(
            embed_dim=embed_dim,
            n_stacks=n_stacks,
            group_num=group_num,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
        )

        # Critic: mean-pool encoder output + container projection -> MLP
        self.critic_step_proj = nn.Linear(group_num, embed_dim, bias=False)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, vf_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def _split_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split (B, N*D+G) into GE (B, N, D) and container_feats (B, G)."""
        enc_feats = features[:, : self._enc_size]  # (B, N*embed_dim)
        container_feats = features[:, self._enc_size :]  # (B, G)
        GE = enc_feats.view(
            features.shape[0], self.n_stacks, self.embed_dim
        )  # (B, N, D)
        return GE, container_feats

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, N*embed_dim + G)
        GE, container_feats = self._split_features(features)
        return self.decoder(GE, container_feats)  # (B, N)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, N*embed_dim + G)
        GE, container_feats = self._split_features(features)
        pooled = GE.mean(dim=1)  # (B, embed_dim)
        C_k = self.critic_step_proj(container_feats)  # (B, embed_dim)
        return self.critic_head(pooled + C_k)  # (B, vf_dim)


class MaskableTransformerPolicy(MaskableActorCriticPolicy):
    """
    Drop-in replacement for MlpPolicy in MaskablePPO.
    Required for compatibility with SB3's MaskablePPO.

    Optional policy_kwargs
    ----------------------
    n_stacks                   : int   — defaults to action_space.n
    embed_dim                  : int   (default 128)
    n_heads                    : int   (default 4)  — used for both encoder and pointer decoder
    n_layers                   : int   (default 2)
    dropout                    : float (default 0.1)
    vf_dim                     : int   (default 128)
    tanh_clipping              : float (default 10.0) — 0 disables clipping
    include_container_in_encoder : bool (default True) — include current-container to encoder input
                            or strip it out and only use as side context for decoder and critic
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        n_stacks: Optional[int] = None,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vf_dim: int = 128,
        tanh_clipping: float = 10.0,
        include_container_in_encoder: bool = True,
        **kwargs,
    ) -> None:
        # Extract num of stacks if not explicitily provided
        if n_stacks is None:
            n_stacks = action_space.n

        self._transformer_n_stacks = n_stacks
        self._transformer_embed_dim = embed_dim
        self._transformer_n_heads = n_heads
        self._transformer_vf_dim = vf_dim
        self._transformer_tanh_clipping = tanh_clipping

        # Inject the custom features extractor so SB3 builds it automatically
        # For sb3 compatibility
        kwargs["features_extractor_class"] = TransformerFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(
            n_stacks=n_stacks,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            include_container_in_encoder=include_container_in_encoder,
        )

        # Disable orthogonal init (performs poorly for transformers)
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # Derive group_num: features_dim = N*embed_dim + group_num
        group_num = (
            self.features_dim - self._transformer_n_stacks * self._transformer_embed_dim
        )
        self.mlp_extractor = TransformerActorCritic(
            feature_dim=self.features_dim,
            n_stacks=self._transformer_n_stacks,
            embed_dim=self._transformer_embed_dim,
            group_num=group_num,
            n_heads=self._transformer_n_heads,
            vf_dim=self._transformer_vf_dim,
            tanh_clipping=self._transformer_tanh_clipping,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        # Replace the standard Linear(latent_dim_pi, action_space.n) with Identity
        # for sb3 compatibility
        self.action_net = nn.Identity()
