"""
Bay actor policy for the separated multi-agent PPO pipeline.

Agent B observes the ORIGINAL global StackEnv observation and selects
one bay.

Architecture
------------
global stack_features_v3 observation
        |
        v
TransformerFeaturesExtractor
        |
        v
GE: (B, n_bays * n_rows, embed_dim)
        |
        v
reshape by bay
        |
        v
mean-pool rows inside each bay
        |
        v
GE_bay: (B, n_bays, embed_dim)
        |
        v
PointerDecoder
        |
        v
bay logits: (B, n_bays)

Important
---------
This module contains ONLY the Bay actor.

It does NOT contain:
    - a value network
    - PPO loss
    - optimizer
    - rollout buffer
    - centralized critic
    - sequential/HAPPO update

Those components are implemented separately in the new MARL pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces

from .transformer_policy import (
    TransformerFeaturesExtractor,
    PointerDecoder,
)


class BayPolicy(nn.Module):
    """
    Transformer + Pointer Network actor for Agent B.

    Parameters
    ----------
    observation_space : spaces.Box
        Global StackEnv observation space.

        For stack_features_v3:

            shape =
            (
                n_bays
                * n_rows_per_bay
                * features_per_stack,
            )

    n_bays : int
        Number of physical usable bays.

    n_rows_per_bay : int
        Number of rows/stacks inside each bay.

    embed_dim : int
        Transformer embedding size.

    n_heads : int
        Number of attention heads.

    n_layers : int
        Number of Transformer encoder layers.

    dropout : float
        Transformer dropout.

        The original policy-level default was 0.1.

    tanh_clipping : float
        PointerDecoder logit clipping parameter.

    include_container_in_encoder : bool
        Whether the current-container feature remains inside every
        stack token passed to the Transformer.

    container_start : int
        Start index of current-container features in one stack token.

        For stack_features_v3:

            current_container_group = feature[2]

        therefore:

            container_start = 2

    container_dim : int
        Number of current-container features.

        For stack_features_v3:

            container_dim = 1
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_bays: int,
        n_rows_per_bay: int,
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

        # ==============================================================
        # Basic validation
        # ==============================================================

        if not isinstance(
            observation_space,
            spaces.Box,
        ):
            raise TypeError(
                "BayPolicy requires a gymnasium.spaces.Box "
                "observation space."
            )

        if n_bays <= 0:
            raise ValueError(
                "n_bays must be positive."
            )

        if n_rows_per_bay <= 0:
            raise ValueError(
                "n_rows_per_bay must be positive."
            )

        # ==============================================================
        # Architecture metadata
        # ==============================================================

        self.n_bays = int(n_bays)
        self.n_rows_per_bay = int(
            n_rows_per_bay
        )

        # Agent B sees ALL stacks.
        self.n_stacks = (
            self.n_bays
            * self.n_rows_per_bay
        )

        self.embed_dim = int(embed_dim)
        self.container_dim = int(
            container_dim
        )

        self._enc_size = (
            self.n_stacks
            * self.embed_dim
        )

        # ==============================================================
        # Transformer encoder
        # ==============================================================
        #
        # This is the same TransformerFeaturesExtractor extracted from
        # the original transformer_policy.py.
        #
        # Agent B gives it the FULL global observation.
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
        # Original bay-level PointerDecoder
        # ==============================================================
        #
        # The original BayTransformerActorCritic encoded all stacks,
        # pooled rows within each bay, then passed those bay embeddings
        # to PointerDecoder.
        #
        # We preserve exactly that actor path.
        # ==============================================================

        self.decoder = PointerDecoder(
            embed_dim=embed_dim,

            # Decoder sees one token PER BAY after pooling.
            n_stacks=self.n_bays,

            # For stack_features_v3 this is 1 because the current
            # container group is represented by one scalar.
            group_num=container_dim,

            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
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

            GE
                shape:
                (B, n_stacks, embed_dim)

        and:

            container_feats
                shape:
                (B, container_dim)

        Extractor output layout:

            [
                flattened GE,
                container features,
            ]
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
                "Unexpected container feature "
                "dimension. "
                f"Expected {self.container_dim}, "
                f"got "
                f"{container_feats.shape[-1]}."
            )

        return (
            GE,
            container_feats,
        )

    # ==================================================================
    # Bay pooling
    # ==================================================================

    def _pool_to_bays(
        self,
        GE: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool stack embeddings within each bay.

        Original stack ordering is bay-major:

            Bay 0:
                Row 0
                Row 1
                ...
            Bay 1:
                Row 0
                Row 1
                ...

        Therefore:

            GE
            (B, n_bays*n_rows, D)

        can be reshaped into:

            (B, n_bays, n_rows, D)

        and mean-pooled over rows.

        Returns
        -------
        torch.Tensor
            Bay embeddings:

                (B, n_bays, embed_dim)
        """

        if GE.dim() != 3:
            raise ValueError(
                "GE must have shape "
                "(batch, n_stacks, embed_dim)."
            )

        batch_size, n_tokens, dim = (
            GE.shape
        )

        if n_tokens != self.n_stacks:
            raise ValueError(
                f"Expected {self.n_stacks} "
                f"stack embeddings, "
                f"got {n_tokens}."
            )

        if dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim="
                f"{self.embed_dim}, "
                f"got {dim}."
            )

        GE_by_bay = GE.view(
            batch_size,
            self.n_bays,
            self.n_rows_per_bay,
            self.embed_dim,
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # This is the SAME bay aggregation used by the original
        # transformer_bay_policy.py.
        # --------------------------------------------------------------

        GE_bay = GE_by_bay.mean(
            dim=2
        )

        return GE_bay

    # ==================================================================
    # Raw actor forward
    # ==================================================================

    def forward(
        self,
        observations: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """
        Compute Bay Agent logits.

        Parameters
        ----------
        observations : torch.Tensor

            Shape:

                (B, global_obs_dim)

            or for one observation:

                (global_obs_dim,)

        action_masks : torch.Tensor | None

            Shape:

                (B, n_bays)

            or:

                (n_bays,)

            True = valid action
            False = invalid action

        Returns
        -------
        torch.Tensor

            Masked bay logits:

                (B, n_bays)
        """

        # --------------------------------------------------------------
        # Support both single observation and batch input.
        # --------------------------------------------------------------

        if observations.dim() == 1:
            observations = (
                observations.unsqueeze(0)
            )

        if observations.dim() != 2:
            raise ValueError(
                "Bay observations must have "
                "shape (obs_dim,) or "
                "(batch, obs_dim). "
                f"Received "
                f"{tuple(observations.shape)}."
            )

        # --------------------------------------------------------------
        # Preserve original numeric input behaviour expected by the
        # neural network.
        # --------------------------------------------------------------

        observations = (
            observations.float()
        )

        # --------------------------------------------------------------
        # Original Transformer feature extractor
        # --------------------------------------------------------------

        features = self.encoder(
            observations
        )

        # --------------------------------------------------------------
        # Restore graph embeddings + container context
        # --------------------------------------------------------------

        GE, container_feats = (
            self._split_features(
                features
            )
        )

        # --------------------------------------------------------------
        # Original Bay Transformer addition:
        #
        # stack embeddings -> bay embeddings
        # --------------------------------------------------------------

        GE_bay = self._pool_to_bays(
            GE
        )

        # --------------------------------------------------------------
        # Original PointerDecoder
        # --------------------------------------------------------------

        logits = self.decoder(
            GE_bay,
            container_feats,
        )

        # logits:
        #
        #     (B, n_bays)

        # --------------------------------------------------------------
        # Apply bay-level action mask AFTER neural network logits.
        #
        # This mirrors the original policy logic:
        #
        #     logits.masked_fill(
        #         ~mask,
        #         -inf
        #     )
        # --------------------------------------------------------------

        logits = self._apply_action_mask(
            logits,
            action_masks,
        )

        return logits

    # ==================================================================
    # Action masking
    # ==================================================================

    def _apply_action_mask(
        self,
        logits: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ],
    ) -> torch.Tensor:
        """
        Apply bay validity mask to actor logits.

        Invalid bays receive -inf so Categorical gives them probability
        zero.

        The validity rules themselves are NOT created here.

        They come from:

            BayEnv.action_masks()

        which projects StackEnv's original valid stack actions onto bays.
        """

        if action_masks is None:
            return logits

        if not isinstance(
            action_masks,
            torch.Tensor,
        ):
            action_masks = torch.as_tensor(
                action_masks,
                dtype=torch.bool,
                device=logits.device,
            )
        else:
            action_masks = action_masks.to(
                device=logits.device,
                dtype=torch.bool,
            )

        if action_masks.dim() == 1:
            action_masks = (
                action_masks.unsqueeze(0)
            )

        if action_masks.shape != logits.shape:
            raise ValueError(
                "Bay action mask shape does not "
                "match logits. "
                f"logits={tuple(logits.shape)}, "
                f"mask="
                f"{tuple(action_masks.shape)}."
            )

        # --------------------------------------------------------------
        # A policy distribution cannot be constructed if every action
        # is invalid.
        #
        # This should normally never happen during a valid rollout.
        # Catching it here makes environment/mask bugs obvious.
        # --------------------------------------------------------------

        if not torch.all(
            action_masks.any(dim=-1)
        ):
            raise RuntimeError(
                "BayPolicy received an action "
                "mask with no valid bay."
            )

        return logits.masked_fill(
            ~action_masks,
            float("-inf"),
        )

    # ==================================================================
    # Distribution
    # ==================================================================

    def get_distribution(
        self,
        observations: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
    ) -> Categorical:
        """
        Construct the categorical bay-action distribution.
        """

        logits = self.forward(
            observations,
            action_masks,
        )

        return Categorical(
            logits=logits
        )

    # ==================================================================
    # Action selection
    # ==================================================================

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
        deterministic: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Select bay actions during rollout/inference.

        Returns
        -------
        actions : torch.Tensor
            Shape:

                (B,)

        log_probs : torch.Tensor
            log π_B(b | o_B)

            Shape:

                (B,)

        Notes
        -----
        Old log probabilities returned here will later be stored in the
        joint rollout buffer for PPO.
        """

        logits = self.forward(
            observations,
            action_masks,
        )

        distribution = Categorical(
            logits=logits
        )

        if deterministic:
            actions = logits.argmax(
                dim=-1
            )
        else:
            actions = (
                distribution.sample()
            )

        log_probs = (
            distribution.log_prob(
                actions
            )
        )

        return (
            actions,
            log_probs,
        )

    # ==================================================================
    # PPO action evaluation
    # ==================================================================

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Evaluate stored Bay actions under the CURRENT policy.

        This method will later be used by sequential PPO training.

        Given rollout actions b_t:

            log_prob
                =
            log π_B,new(
                b_t | o_B,t
            )

        This allows computation of:

            ratio_B
                =
            exp(
                log_prob_new
                -
                log_prob_old
            )

        Returns
        -------
        log_probs : torch.Tensor
            Shape:

                (B,)

        entropy : torch.Tensor
            Shape:

                (B,)
        """

        distribution = (
            self.get_distribution(
                observations,
                action_masks,
            )
        )

        actions = actions.long()

        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        log_probs = (
            distribution.log_prob(
                actions
            )
        )

        entropy = (
            distribution.entropy()
        )

        return (
            log_probs,
            entropy,
        )