"""
Agent B for the separated hierarchical multi-agent PPO pipeline.

Agent B is responsible for:

    selecting a BAY.

It owns exactly one BayPolicy.

Responsibilities
----------------
AgentB:
    - owns BayPolicy parameters theta_B
    - converts environment observations/masks to tensors
    - samples or deterministically selects bay actions
    - evaluates stored actions for PPO
    - supports save/load

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from ..models.bay_policy import BayPolicy


class AgentB(nn.Module):
    """
    Bay-selection agent.

    Parameters
    ----------
    observation_space : spaces.Box
        Global observation space used by Agent B.

    n_bays : int
        Number of possible bay actions.

    n_rows_per_bay : int
        Number of rows/stacks inside every bay.

    embed_dim : int
        Transformer embedding dimension.

    n_heads : int
        Number of attention heads.

    n_layers : int
        Number of Transformer layers.

    dropout : float
        Transformer dropout.

    tanh_clipping : float
        PointerDecoder clipping constant.

    container_start : int
        Position of current-container feature inside one
        stack_features_v3 token.

        For stack_features_v3:

            index 2 = current_container_group

    container_dim : int
        Number of current-container features.

        For stack_features_v3:

            container_dim = 1

    device : str | torch.device
        Device on which this agent runs.
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

        self.n_bays = int(
            n_bays
        )

        self.n_rows_per_bay = int(
            n_rows_per_bay
        )

        # ==============================================================
        # Agent B owns its OWN policy
        # ==============================================================

        self.policy = BayPolicy(
            observation_space=observation_space,

            n_bays=n_bays,

            n_rows_per_bay=(
                n_rows_per_bay
            ),

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
        Convert NumPy/list/Tensor observation to float tensor.
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
        Convert Bay action mask to bool tensor.
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
        Return BayPolicy logits.

        Mainly useful for training/debugging.
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
        Select ONE bay for ONE environment state.

        Parameters
        ----------
        observation
            Global Agent B observation.

        action_mask
            Bay-level validity mask.

        deterministic
            False during training rollout.
            True during evaluation/inference.

        Returns
        -------
        bay_idx : int
            Zero-based bay action.

        log_prob : float
            log pi_B(bay_idx | observation)

        Example
        -------
        bay_idx, log_prob = agent_b.select_action(
            bay_obs,
            bay_mask,
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

        # This method represents ONE environment decision,
        # therefore the output should contain exactly one action.
        if actions.numel() != 1:
            raise RuntimeError(
                "AgentB.select_action() expected exactly "
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
    # Batch action sampling
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
        Tensor-based action interface used by the trainer.

        Unlike select_action(), this method supports batches.

        Returns
        -------
        actions:
            (B,)

        log_probs:
            (B,)
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
        Evaluate stored Bay actions under the CURRENT Agent B policy.

        Used for:

            Bay PPO ratio

        and later:

            updated Bay / old Bay sequence ratio.
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
        Return Agent B's categorical action distribution.
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
        Save Agent B model parameters.

        Only the neural-network parameters are saved.

        Optimizer state belongs to SequentialPPOTrainer and is not
        stored here.
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
        Load Agent B model parameters.
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
    # Convenience inference
    # ==================================================================

    def predict(
        self,
        observation,
        action_mask,
        deterministic: bool = True,
    ) -> int:
        """
        Simple inference interface.

        Returns only the selected bay index.

        Evaluation code can therefore use:

            bay_idx = agent_b.predict(
                bay_obs,
                bay_mask,
            )
        """

        bay_idx, _ = (
            self.select_action(
                observation=observation,
                action_mask=action_mask,
                deterministic=deterministic,
            )
        )

        return bay_idx