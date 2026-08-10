"""
Frozen RL Low-Level Agent.

Wraps a trained MaskablePPO checkpoint and exposes the same
``get_action(observation, valid_actions, selected_bay) -> int`` interface
as the rule-based ``LowLevelAgent``.  This allows a previously trained
low-level RL agent to be used as a fixed policy inside
``HierarchicalHighLevelEnv`` while the high-level agent is being trained.
"""

from __future__ import annotations

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO


class FrozenRLLowLevelAgent:
    """
    Wraps a trained MaskablePPO model to serve as a fixed low-level agent.

    Parameters
    ----------
    model_path : str
        Path to the saved MaskablePPO checkpoint (without .zip extension).
    n_actions : int
        Total number of discrete actions (yard stacks) in the inner env.
    n_rows : int
        Number of rows per bay in the yard (``yard_shape[1]``).
    device : str
        Device to load the model on (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_path: str,
        n_actions: int,
        n_rows: int,
        device: str = "cpu",
    ) -> None:
        self.model = MaskablePPO.load(model_path, device=device)
        self.n_actions = n_actions
        self.n_rows = n_rows

    # ------------------------------------------------------------------
    # Bay-row mapping (mirrors StackEnv._action_to_bay_row)
    # ------------------------------------------------------------------

    def _action_to_bay(self, action: int) -> int:
        """Return the odd bay number for a given action index."""
        bay_idx = action // self.n_rows
        return 2 * bay_idx + 1

    # ------------------------------------------------------------------
    # Public interface (matches LowLevelAgent.get_action signature)
    # ------------------------------------------------------------------

    def get_action(
        self,
        observation: dict,
        valid_actions: list[int],
        selected_bay: int,
    ) -> int:
        """
        Select a stack action within *selected_bay* using the frozen model.

        Parameters
        ----------
        observation : dict or np.ndarray
            Current environment observation.  When called from
            ``HierarchicalHighLevelEnv`` with a frozen RL agent, this is the
            raw observation array (same format the model was trained on).
        valid_actions : list[int]
            All currently valid yard-stack action indices.
        selected_bay : int
            Odd bay number chosen by the high-level agent.

        Returns
        -------
        int
            Action index of the selected yard stack.
        """
        # Build mask: only stacks in the selected bay that are also valid
        mask = np.zeros(self.n_actions, dtype=bool)
        for a in valid_actions:
            if self._action_to_bay(a) == selected_bay:
                mask[a] = True

        # If no valid actions in the selected bay, fall back to any valid action
        if not mask.any():
            for a in valid_actions:
                mask[a] = True

        # The observation passed here is the raw env obs (np.ndarray)
        obs = observation
        if isinstance(observation, dict):
            # If something sends a dict, try to extract the array
            obs = observation.get("observation", observation)

        action, _ = self.model.predict(
            obs, deterministic=True, action_masks=mask
        )
        return int(action)
