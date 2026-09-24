"""
Agent R for the separated hierarchical multi-agent PPO pipeline.

Agent R is responsible for:

    selecting a ROW / STACK inside the bay chosen by Agent B.

It owns exactly one RowPolicy.

Responsibilities
----------------
AgentR:
    - owns RowPolicy parameters theta_R
    - converts local observations/masks to tensors
    - samples or deterministically selects row actions
    - evaluates stored actions for PPO
    - supports save/load

AgentR does NOT:
    - choose bays
    - execute environment transitions
    - calculate PPO losses
    - own the PPO optimizer
    - calculate advantages
    - update the centralized critic
    - calculate Bay sequence ratios

Those responsibilities belong to other modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from gymnasium import spaces

from ..models.row_policy import RowPolicy


class AgentR(nn.Module):
    """
    Row-selection agent.

    Parameters
    ----------
    observation_space : spaces.Box
        Local selected-bay observation space.

    n_rows : int
        Number of rows/stacks inside one bay.

    embed_dim : int
        Transformer embedding dimension.

    n_heads : int
        Number of attention heads.

    n_layers : int
        Number of Transformer encoder layers.

    dropout : float
        Transformer dropout.

    tanh_clipping : float
        PointerDecoder tanh clipping.

    include_container_in_encoder : bool
        Whether current-container information remains inside each
        stack token passed to the Transformer.

    container_start : int
        Start position of current-container feature.

        For stack_features_v3:

            [2] current_container_group

        therefore:

            container_start = 2

    container_dim : int
        Current-container feature dimension.

        For stack_features_v3:

            container_dim = 1

    device : str | torch.device
        Device on which Agent R runs.
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
        device: Union[
            str,
            torch.device,
        ] = "cpu",
    ) -> None:

        super().__init__()

        # ==============================================================
        # Metadata
        # ==============================================================

        self.device = torch.device(
            device
        )

        self.n_rows = int(
            n_rows
        )

        # ==============================================================
        # Agent R owns its OWN policy
        # ==============================================================

        self.policy = RowPolicy(
            observation_space=(
                observation_space
            ),

            n_rows=n_rows,

            embed_dim=embed_dim,

            n_heads=n_heads,

            n_layers=n_layers,

            dropout=dropout,

            tanh_clipping=(
                tanh_clipping
            ),

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

        self.to(
            self.device
        )

    # ==================================================================
    # Tensor helpers
    # ==================================================================

    def _observation_to_tensor(
        self,
        observation,
    ) -> torch.Tensor:
        """
        Convert local observation to float tensor.
        """

        if isinstance(
            observation,
            torch.Tensor,
        ):

            return observation.to(
                device=self.device,
                dtype=torch.float32,
            )

        return torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

    def _mask_to_tensor(
        self,
        action_mask,
    ) -> Optional[torch.Tensor]:
        """
        Convert local row mask to bool tensor.
        """

        if action_mask is None:
            return None

        if isinstance(
            action_mask,
            torch.Tensor,
        ):

            return action_mask.to(
                device=self.device,
                dtype=torch.bool,
            )

        return torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        )

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        observations: torch.Tensor,
        action_masks: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """
        Return RowPolicy logits.
        """

        return self.policy(
            observations,
            action_masks,
        )

    # ==================================================================
    # Environment-facing action selection
    # ==================================================================

    @torch.no_grad()
    def select_action(
        self,
        observation,
        action_mask,
        deterministic: bool = False,
    ) -> Tuple[
        int,
        float,
    ]:
        """
        Select ONE row for ONE selected bay.

        Parameters
        ----------
        observation
            Local selected-bay observation.

        action_mask
            Row-level validity mask.

        deterministic
            False during training rollout.
            True during evaluation/inference.

        Returns
        -------
        row_idx : int
            Zero-based local row action.

        log_prob : float
            log pi_R(row_idx | local_observation)

        Example
        -------
        row_idx, log_prob = agent_r.select_action(
            row_obs,
            row_mask,
        )
        """

        observation_tensor = (
            self._observation_to_tensor(
                observation
            )
        )

        mask_tensor = (
            self._mask_to_tensor(
                action_mask
            )
        )

        actions, log_probs = (
            self.policy.act(
                observations=(
                    observation_tensor
                ),

                action_masks=(
                    mask_tensor
                ),

                deterministic=(
                    deterministic
                ),
            )
        )

        if actions.numel() != 1:
            raise RuntimeError(
                "AgentR.select_action() expected exactly "
                "one action. "
                f"Received shape={tuple(actions.shape)}."
            )

        return (
            int(
                actions.item()
            ),
            float(
                log_probs.item()
            ),
        )

    # ==================================================================
    # Batch action selection
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
        Tensor/batch action interface used by the trainer.

        Returns
        -------
        actions:
            row indices, shape (B,)

        log_probs:
            log pi_R, shape (B,)
        """

        return self.policy.act(
            observations=observations,
            action_masks=action_masks,
            deterministic=deterministic,
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
        Evaluate stored Row actions under the CURRENT Agent R policy.

        Used to compute:

            ratio_R
              =
            pi_R,new / pi_R,old

        during Row PPO update.
        """

        return self.policy.evaluate_actions(
            observations=observations,
            actions=actions,
            action_masks=action_masks,
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
    ):
        """
        Return Agent R's categorical row-action distribution.
        """

        return self.policy.get_distribution(
            observations=observations,
            action_masks=action_masks,
        )

    # ==================================================================
    # Save
    # ==================================================================

    def save(
        self,
        path: Union[
            str,
            Path,
        ],
    ) -> None:
        """
        Save Agent R neural-network parameters.

        Optimizer state is deliberately not stored here because
        optimizers belong to SequentialPPOTrainer.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )

    # ==================================================================
    # Load
    # ==================================================================

    def load(
        self,
        path: Union[
            str,
            Path,
        ],
        strict: bool = True,
    ) -> None:
        """
        Load Agent R model parameters.
        """

        path = Path(
            path
        )

        state_dict = torch.load(
            path,
            map_location=self.device,
        )

        self.load_state_dict(
            state_dict,
            strict=strict,
        )

    # ==================================================================
    # Simple inference
    # ==================================================================

    def predict(
        self,
        observation,
        action_mask,
        deterministic: bool = True,
    ) -> int:
        """
        Return only the selected zero-based row index.

        Evaluation can use:

            row_idx = agent_r.predict(
                row_obs,
                row_mask,
            )
        """

        row_idx, _ = (
            self.select_action(
                observation=observation,
                action_mask=action_mask,
                deterministic=deterministic,
            )
        )

        return row_idx