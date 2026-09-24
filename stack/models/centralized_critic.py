"""
Centralized value critic for the separated multi-agent PPO pipeline.

The critic is used ONLY during training.

Both actors have their own policies:

    Agent B:
        global observation -> BayPolicy

    Agent R:
        selected-bay local observation -> RowPolicy

The centralized critic receives the FULL original StackEnv observation:

    global_state_t
        |
        v
    Transformer encoder
        |
        v
    stack embeddings GE
        |
        v
    mean pooling over all stacks
        |
        + current-container context
        |
        v
    critic MLP
        |
        v
    V(s_t)

Important
---------
This critic predicts ONE shared state value.

The environment produces one team reward for the joint hierarchical
decision:

    bay action
        ->
    row action
        ->
    ONE StackEnv.step(...)
        ->
    ONE reward

Therefore we do not invent separate Bay and Row reward/value functions.

The sequential/HAPPO distinction between the two actor updates will be
implemented later in advantage.py and sequential_trainer.py.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from gymnasium import spaces

from .transformer_policy import TransformerFeaturesExtractor


class CentralizedCritic(nn.Module):
    """
    Centralized Transformer value function.

    Parameters
    ----------
    observation_space : spaces.Box
        FULL global StackEnv observation space.

        This should be:

            env.global_observation_space

        which is currently identical to Agent B's global observation.

    n_stacks : int
        Total number of stacks in the whole yard:

            n_bays * n_rows_per_bay

    embed_dim : int
        Transformer embedding dimension.

        Original default:
            128

    n_heads : int
        Transformer attention heads.

        Original default:
            4

    n_layers : int
        Number of Transformer encoder layers.

        Original default:
            2

    dropout : float
        Transformer dropout.

        Original policy-level training default:
            0.1

    vf_dim : int
        Hidden critic dimension.

        Original default:
            128

    include_container_in_encoder : bool
        Whether the current-container feature remains inside every
        stack token sent through the Transformer.

    container_start : int
        Start index of the current-container feature.

        For stack_features_v3:

            [0] majority_group
            [1] num_occupied
            [2] current_container_group
            ...

        therefore:

            container_start = 2

    container_dim : int
        Number of current-container context features.

        For stack_features_v3:

            container_dim = 1
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

        # ==============================================================
        # Validation
        # ==============================================================

        if not isinstance(
            observation_space,
            spaces.Box,
        ):
            raise TypeError(
                "CentralizedCritic requires a "
                "gymnasium.spaces.Box observation space."
            )

        if n_stacks <= 0:
            raise ValueError(
                "n_stacks must be positive."
            )

        if embed_dim <= 0:
            raise ValueError(
                "embed_dim must be positive."
            )

        if vf_dim <= 0:
            raise ValueError(
                "vf_dim must be positive."
            )

        # ==============================================================
        # Architecture metadata
        # ==============================================================

        self.n_stacks = int(n_stacks)
        self.embed_dim = int(embed_dim)
        self.container_dim = int(
            container_dim
        )
        self.vf_dim = int(vf_dim)

        # Boundary inside TransformerFeaturesExtractor output:
        #
        # [
        #     flattened GE,
        #     container_feats
        # ]
        #
        self._enc_size = (
            self.n_stacks
            * self.embed_dim
        )

        # ==============================================================
        # Independent centralized Transformer
        # ==============================================================
        #
        # IMPORTANT:
        #
        # This is a separate network instance.
        #
        # It has the SAME architecture as the original Transformer
        # encoder, but it does NOT share parameters with:
        #
        #     BayPolicy.encoder
        #     RowPolicy.encoder
        #
        # Therefore critic gradients cannot modify either actor.
        # ==============================================================

        self.encoder = (
            TransformerFeaturesExtractor(
                observation_space=(
                    observation_space
                ),
                n_stacks=self.n_stacks,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                include_container_in_encoder=(
                    include_container_in_encoder
                ),
                container_start=(
                    container_start
                ),
                container_dim=(
                    container_dim
                ),
            )
        )

        # ==============================================================
        # Original critic context projection
        # ==============================================================
        #
        # Original critic:
        #
        #     pooled = GE.mean(dim=1)
        #
        #     C_k =
        #         critic_step_proj(container_feats)
        #
        #     critic_input =
        #         pooled + C_k
        #
        # ==============================================================

        self.critic_step_proj = nn.Linear(
            container_dim,
            embed_dim,
            bias=False,
        )

        # ==============================================================
        # Original critic MLP
        # ==============================================================
        #
        # Original:
        #
        #     Linear(embed_dim, vf_dim)
        #     ReLU()
        #
        # ==============================================================

        self.critic_head = nn.Sequential(
            nn.Linear(
                embed_dim,
                vf_dim,
            ),
            nn.ReLU(),
        )

        # ==============================================================
        # Original SB3 value_net equivalent
        # ==============================================================
        #
        # Previously SB3 created:
        #
        #     value_net = Linear(vf_dim, 1)
        #
        # We now make it explicit because the critic is no longer
        # contained inside an SB3 ActorCriticPolicy.
        # ==============================================================

        self.value_head = nn.Linear(
            vf_dim,
            1,
        )

    # ==================================================================
    # Feature splitting
    # ==================================================================

    def _split_features(
        self,
        features: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Split TransformerFeaturesExtractor output into:

            GE:
                (B, n_stacks, embed_dim)

            container_feats:
                (B, container_dim)
        """

        if features.dim() != 2:
            raise ValueError(
                "Expected encoded features with shape "
                "(batch, feature_dim). "
                f"Received {tuple(features.shape)}."
            )

        enc_feats = features[
            :,
            : self._enc_size,
        ]

        container_feats = features[
            :,
            self._enc_size :,
        ]

        GE = enc_feats.view(
            features.shape[0],
            self.n_stacks,
            self.embed_dim,
        )

        if (
            container_feats.shape[-1]
            != self.container_dim
        ):
            raise RuntimeError(
                "Unexpected container feature dimension. "
                f"Expected {self.container_dim}, "
                f"got {container_feats.shape[-1]}."
            )

        return (
            GE,
            container_feats,
        )

    # ==================================================================
    # Critic latent representation
    # ==================================================================

    def forward_latent(
        self,
        global_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce the original critic latent representation.

        Input
        -----
        global_states:

            single:
                (global_obs_dim,)

            batch:
                (B, global_obs_dim)

        Output
        ------
        latent_vf:

            (B, vf_dim)
        """

        # --------------------------------------------------------------
        # Support both one state and batches.
        # --------------------------------------------------------------

        if global_states.dim() == 1:
            global_states = (
                global_states.unsqueeze(0)
            )

        if global_states.dim() != 2:
            raise ValueError(
                "Global states must have shape "
                "(obs_dim,) or (batch, obs_dim). "
                f"Received "
                f"{tuple(global_states.shape)}."
            )

        global_states = (
            global_states.float()
        )

        # ==============================================================
        # Transformer encoder
        # ==============================================================

        features = self.encoder(
            global_states
        )

        GE, container_feats = (
            self._split_features(
                features
            )
        )

        # ==============================================================
        # Global mean pooling
        # ==============================================================
        #
        # Critic sees ALL stacks.
        #
        # This is intentionally NOT bay pooling.
        #
        # Original critic did:
        #
        #     GE.mean(dim=1)
        #
        # ==============================================================

        pooled = GE.mean(
            dim=1
        )

        # pooled:
        #
        #     (B, embed_dim)

        # ==============================================================
        # Current-container context
        # ==============================================================

        C_k = (
            self.critic_step_proj(
                container_feats
            )
        )

        # C_k:
        #
        #     (B, embed_dim)

        # ==============================================================
        # Critic latent
        # ==============================================================

        latent_vf = self.critic_head(
            pooled + C_k
        )

        # latent_vf:
        #
        #     (B, vf_dim)

        return latent_vf

    # ==================================================================
    # Value prediction
    # ==================================================================

    def forward(
        self,
        global_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict centralized state values.

        Returns
        -------
        values : torch.Tensor

            shape:

                (B,)

            representing:

                V_phi(s_t)
        """

        latent_vf = self.forward_latent(
            global_states
        )

        values = self.value_head(
            latent_vf
        )

        # Original value_net produces:
        #
        #     (B, 1)
        #
        # For our custom rollout/GAE implementation it is more
        # convenient to use:
        #
        #     (B,)
        #
        return values.squeeze(-1)

    # ==================================================================
    # No-gradient value prediction
    # ==================================================================

    @torch.no_grad()
    def predict_values(
        self,
        global_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict V(s) without constructing an autograd graph.

        Used during rollout collection and bootstrap value calculation.
        """

        return self.forward(
            global_states
        )