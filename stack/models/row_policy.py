"""
Row actor policy for the separated multi-agent PPO pipeline.

Agent R observes ONLY the stacks/rows inside the bay selected by Agent B
and selects one row/stack for container placement.

Architecture
------------
selected-bay stack_features_v3 observation
        |
        v
TransformerFeaturesExtractor
        |
        v
GE_row: (B, n_rows, embed_dim)
        |
        v
PointerDecoder
        |
        v
row logits: (B, n_rows)

Important
---------
This module contains ONLY the Row actor.

It does NOT contain:
    - value network
    - centralized critic
    - PPO loss
    - optimizer
    - rollout buffer
    - sequential HAPPO update
    - environment transition

The Transformer and PointerDecoder are reused from the existing transformer_policy.py
and follow the original transformer_policy.py actor architecture.
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


class RowPolicy(nn.Module):
    """
    Transformer + Pointer Network actor for Agent R.

    Agent R operates only inside the bay selected by Agent B.

    Parameters
    ----------
    observation_space : spaces.Box
        Local selected-bay observation space.

        For stack_features_v3:

            shape =
            (
                n_rows * features_per_stack,
            )

        Example for small_with_margin:

            n_rows = 4
            features_per_stack = 10

        therefore:

            observation shape = (40,)

    n_rows : int
        Number of rows/stacks inside one bay.

        This is also Agent R's number of actions.

    embed_dim : int
        Transformer embedding dimension.

    n_heads : int
        Number of attention heads.

    n_layers : int
        Number of Transformer encoder layers.

    dropout : float
        Transformer dropout.

        Kept consistent with the original policy-level default.

    tanh_clipping : float
        PointerDecoder tanh clipping.

    include_container_in_encoder : bool
        Whether the current-container feature remains in each stack
        token passed to the Transformer.

    container_start : int
        Position of current-container feature inside one stack token.

        For stack_features_v3:

            [0] majority_group
            [1] num_occupied
            [2] current_container_group
            ...

        therefore:

            container_start = 2

    container_dim : int
        Number of current-container features.

        For stack_features_v3 the current container group is represented
        by one scalar:

            container_dim = 1
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_rows: int,
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
        # Validation
        # ==============================================================

        if not isinstance(
            observation_space,
            spaces.Box,
        ):
            raise TypeError(
                "RowPolicy requires a gymnasium.spaces.Box "
                "observation space."
            )

        if n_rows <= 0:
            raise ValueError(
                "n_rows must be positive."
            )

        # ==============================================================
        # Architecture metadata
        # ==============================================================

        self.n_rows = int(n_rows)

        # For Agent R:
        #
        # one row == one stack token == one possible local action.
        self.n_stacks = self.n_rows

        self.embed_dim = int(embed_dim)
        self.container_dim = int(
            container_dim
        )

        # Boundary between:
        #
        #     flattened Transformer embeddings
        #
        # and:
        #
        #     appended current-container features
        #
        self._enc_size = (
            self.n_stacks
            * self.embed_dim
        )

        # ==============================================================
        # Transformer encoder
        # ==============================================================
        #
        # Same TransformerFeaturesExtractor as the original stack-level
        # policy.
        #
        # The ONLY important difference is:
        #
        # original:
        #     n_stacks = all stacks in yard
        #
        # Agent R:
        #     n_stacks = rows inside selected bay
        #
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
        # Pointer decoder
        # ==============================================================
        #
        # Unlike BayPolicy, NO bay pooling happens here.
        #
        # Each Transformer token already represents exactly one action:
        #
        #     row 0
        #     row 1
        #     ...
        #
        # ==============================================================

        self.decoder = PointerDecoder(
            embed_dim=embed_dim,
            n_stacks=self.n_rows,
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

            GE_row
                shape:
                (B, n_rows, embed_dim)

        and:

            container_feats
                shape:
                (B, container_dim)

        Extractor output layout:

            [
                flattened GE_row,
                container features
            ]
        """

        if features.dim() != 2:
            raise ValueError(
                "Expected encoded features with shape "
                "(batch, feature_dim). "
                f"Received {tuple(features.shape)}."
            )

        # --------------------------------------------------------------
        # Transformer embeddings
        # --------------------------------------------------------------

        enc_feats = features[
            :,
            : self._enc_size,
        ]

        # --------------------------------------------------------------
        # Current-container context
        # --------------------------------------------------------------

        container_feats = features[
            :,
            self._enc_size :,
        ]

        # --------------------------------------------------------------
        # Restore token structure
        # --------------------------------------------------------------

        GE_row = enc_feats.view(
            features.shape[0],
            self.n_rows,
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
            GE_row,
            container_feats,
        )

    # ==================================================================
    # Actor forward
    # ==================================================================

    def forward(
        self,
        observations: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """
        Compute Agent R row logits.

        Parameters
        ----------
        observations : torch.Tensor
            Local selected-bay observation.

            Single observation:

                (local_obs_dim,)

            Batch:

                (B, local_obs_dim)

        action_masks : torch.Tensor | None
            Row validity mask.

            Single:

                (n_rows,)

            Batch:

                (B, n_rows)

            Convention:

                True  = valid
                False = invalid

        Returns
        -------
        torch.Tensor
            Masked row logits:

                (B, n_rows)
        """

        # --------------------------------------------------------------
        # Support one observation or batch
        # --------------------------------------------------------------

        if observations.dim() == 1:
            observations = (
                observations.unsqueeze(0)
            )

        if observations.dim() != 2:
            raise ValueError(
                "Row observations must have "
                "shape (obs_dim,) or "
                "(batch, obs_dim). "
                f"Received "
                f"{tuple(observations.shape)}."
            )

        observations = (
            observations.float()
        )

        # ==============================================================
        # Transformer
        # ==============================================================

        features = self.encoder(
            observations
        )

        # ==============================================================
        # Recover row embeddings
        # ==============================================================

        GE_row, container_feats = (
            self._split_features(
                features
            )
        )

        # ==============================================================
        # PointerDecoder
        # ==============================================================
        #
        # IMPORTANT:
        #
        # There is NO:
        #
        #     reshape by bay
        #     mean(dim=2)
        #
        # here.
        #
        # Each token is already one row/action.
        # ==============================================================

        logits = self.decoder(
            GE_row,
            container_feats,
        )

        # logits:
        #
        #     (B, n_rows)

        # ==============================================================
        # Local row action masking
        # ==============================================================

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
        Apply Agent R's local row mask.

        Invalid rows receive -inf.

        The validity rules themselves are NOT defined here.

        They come from:

            RowEnv.action_masks(bay_idx)

        which projects the ORIGINAL StackEnv global valid actions onto
        rows inside the bay selected by Agent B.
        """

        if action_masks is None:
            return logits

        # --------------------------------------------------------------
        # Convert numpy/list masks if necessary
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Single mask -> batch dimension
        # --------------------------------------------------------------

        if action_masks.dim() == 1:
            action_masks = (
                action_masks.unsqueeze(0)
            )

        if action_masks.shape != logits.shape:
            raise ValueError(
                "Row action mask shape does not "
                "match logits. "
                f"logits={tuple(logits.shape)}, "
                f"mask="
                f"{tuple(action_masks.shape)}."
            )

        # --------------------------------------------------------------
        # Every sample must contain at least one valid row.
        #
        # Normally this is guaranteed because Agent B's bay mask only
        # permits bays containing at least one valid StackEnv action.
        # --------------------------------------------------------------

        if not torch.all(
            action_masks.any(dim=-1)
        ):
            raise RuntimeError(
                "RowPolicy received an action "
                "mask with no valid row."
            )

        # --------------------------------------------------------------
        # Mask invalid actions
        # --------------------------------------------------------------

        return logits.masked_fill(
            ~action_masks,
            float("-inf"),
        )

    # ==================================================================
    # Action distribution
    # ==================================================================

    def get_distribution(
        self,
        observations: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
    ) -> Categorical:
        """
        Return categorical distribution over rows.
        """

        logits = self.forward(
            observations,
            action_masks,
        )

        return Categorical(
            logits=logits
        )

    # ==================================================================
    # Rollout action selection
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
        Select Agent R row actions.

        Returns
        -------
        actions : torch.Tensor

            row_idx

            shape:

                (B,)

        log_probs : torch.Tensor

            log π_R(
                row_idx
                |
                local selected-bay observation
            )

            shape:

                (B,)

        The returned log probabilities will later be stored in the
        joint rollout buffer as old_row_log_prob.
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
    # PPO evaluation
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
        Evaluate stored Agent R actions under the CURRENT policy.

        During rollout we store:

            log π_R,old(
                r_t | o_R,t
            )

        During PPO update this method calculates:

            log π_R,new(
                r_t | o_R,t
            )

        allowing:

            ratio_R
                =
            exp(
                log_prob_new
                -
                log_prob_old
            )

        Returns
        -------
        log_probs : torch.Tensor
            Current-policy log probabilities.

        entropy : torch.Tensor
            Categorical entropy.

        Both have shape:

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