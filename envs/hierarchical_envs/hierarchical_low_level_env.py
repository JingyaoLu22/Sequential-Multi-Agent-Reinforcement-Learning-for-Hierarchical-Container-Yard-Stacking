"""
Hierarchical Low-Level Environment Wrapper.

Wraps StackEnv and embeds a fixed high-level agent (rule-based by default)
that selects which bay to place the current container in.  The RL agent
(low-level) then selects which stack (bay x row) within that bay.

The action space is identical to StackEnv (Discrete over all yard stacks),
but the action mask restricts valid actions to only stacks in the bay
chosen by the high-level agent.  This allows a standard MaskablePPO
(MLP or Pointer Net) to be trained as the low-level policy without any
architecture changes.

Future use:
    Once the low-level agent is trained, it can be frozen and plugged into
    StackHighLevelEnv to train a high-level Pointer Net.  The constructor
    also accepts an arbitrary ``high_level_agent`` object so the rule-based
    policy can later be swapped for a learned one.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from envs.stack_gym import StackEnv, StateIds
from agents.hierarchical_rule_based_agent import HighLevelAgent


class HierarchicalLowLevelEnv(gym.Env):
    """
    Gymnasium wrapper that adds a fixed high-level bay-selection policy on
    top of StackEnv.  The RL agent only chooses among stacks in the
    high-level's selected bay.

    Parameters
    ----------
    config : dict
        Passed directly to ``StackEnv``.
    high_level_policy_type : str
        Policy name for the built-in ``HighLevelAgent``
        (``"rule_based_grouped"`` | ``"rule_based"`` | ``"random"``).
        Ignored when ``high_level_agent`` is provided.
    high_level_agent : object | None
        Any object with a ``.get_action(observation, valid_actions) -> int``
        method that returns a bay number.  When provided, the built-in
        ``HighLevelAgent`` is not created.
    render_mode : str | None
        Forwarded to the inner ``StackEnv``.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: Optional[Dict] = None,
        high_level_policy_type: str = "rule_based_grouped",
        high_level_agent: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # --- inner environment -------------------------------------------
        self.inner_env = StackEnv(config=config, render_mode=render_mode)

        # Mirror spaces from inner env (unchanged)
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space

        # --- high-level agent --------------------------------------------
        if high_level_agent is not None:
            self.high_level_agent = high_level_agent
        else:
            self.high_level_agent = HighLevelAgent(
                vessel_shape=self.inner_env.vessel_shape,
                yard_shape=self.inner_env.yard_shape,
                num_slot_attrs=self.inner_env.num_slot_attrs,
                policy_type=high_level_policy_type,
            )

        # Bay selected by the high-level agent for the current timestep
        self.selected_bay: Optional[int] = None

        # Expose inner env attributes needed by training utilities
        self.render_mode = render_mode

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs_for_high_level(self) -> Dict[str, np.ndarray]:
        """
        Build the ``flat_parsed`` style dict that ``HighLevelAgent`` expects
        directly from the inner env's raw state arrays.
        """
        # yard_state: only odd-bay rows (filter out even-bay padding)
        yard_state = self.inner_env.yard_state  # (total_yard_coords, 5)

        # current container attributes
        idx = self.inner_env.current_vessel_container
        if idx is not None:
            current_container = self.inner_env.vessel_state[idx]  # (5,)
        else:
            current_container = np.zeros(self.inner_env.num_slot_attrs, dtype=int)

        return {
            "yard_state": yard_state,
            "current_container": current_container,
        }

    # ------------------------------------------------------------------
    # Bay selection
    # ------------------------------------------------------------------

    def _select_bay(self) -> None:
        """
        Run the high-level agent to pick a bay for the current container
        and store the result in ``self.selected_bay``.
        """
        valid_actions = self.inner_env._get_valid_yard_actions()
        if len(valid_actions) == 0:
            self.selected_bay = None
            return

        obs_for_hl = self._get_obs_for_high_level()
        self.selected_bay = self.high_level_agent.get_action(
            obs_for_hl, valid_actions.tolist()
        )

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def action_masks(self) -> List[bool]:
        """
        Return a boolean mask over all actions.  Only stacks inside the
        high-level's ``selected_bay`` that are also valid (not full) are
        ``True``.
        """
        if self.inner_env.current_vessel_container is None or self.selected_bay is None:
            return [False] * self.action_space.n

        # Base validity from inner env (respects full stacks / odd bays)
        base_mask = self.inner_env.action_masks()

        # Further restrict to stacks in the selected bay
        restricted_mask = [False] * self.action_space.n
        for action_idx in range(self.action_space.n):
            if base_mask[action_idx]:
                bay, _row = self.inner_env._action_to_bay_row(action_idx)
                if bay == self.selected_bay:
                    restricted_mask[action_idx] = True

        return restricted_mask

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[Union[np.ndarray, Dict], Dict]:
        obs, info = self.inner_env.reset(seed=seed, **kwargs)

        # Select bay for the first container
        self._select_bay()
        info["selected_bay"] = self.selected_bay

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.inner_env.step(action)
        info["selected_bay"] = self.selected_bay

        # If episode continues, select bay for the next container
        if not (terminated or truncated):
            self._select_bay()
            info["selected_bay"] = self.selected_bay

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self.inner_env.render()

    def close(self) -> None:
        self.inner_env.close()
