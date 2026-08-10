"""
Hierarchical High-Level Environment Wrapper.

Wraps StackEnv and embeds a fixed low-level agent (rule-based or frozen RL)
that selects which stack within a bay to place a container.  The RL agent
(high-level) selects which bay to place the current container in.

The action space is Discrete(num_physical_bays) where each action maps
to an odd bay number.  The observation space is identical to StackEnv.
Action masks restrict valid actions to bays that have at least one non-full
stack.

Future use:
    Once the high-level agent is trained, it can be combined with the
    previously trained low-level agent for full end-to-end hierarchical
    inference.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from envs.stack_gym import StackEnv
from agents.hierarchical_rule_based_agent import LowLevelAgent
from agents.frozen_low_level_agent import FrozenRLLowLevelAgent


class HierarchicalHighLevelEnv(gym.Env):
    """
    Gymnasium wrapper where the RL agent selects a bay and a fixed low-level
    agent selects the specific stack within that bay.

    Parameters
    ----------
    config : dict
        Passed directly to StackEnv.
    low_level_policy_type : str
        Policy name for the built-in LowLevelAgent
        ("rule_based_grouped" | "rule_based" | "random").
        Ignored when low_level_agent or low_level_model_path is
        provided.
    low_level_agent : object | None
        Any object with a
        .get_action(observation, valid_actions, selected_bay) -> int
        method.  When provided, takes priority.
    low_level_model_path : str | None
        Path to a saved MaskablePPO checkpoint for the low-level agent.
        When provided (and low_level_agent is None), a
        FrozenRLLowLevelAgent is created.
    render_mode : str | None
        Forwarded to the inner StackEnv.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: Optional[Dict] = None,
        low_level_policy_type: str = "rule_based_grouped",
        low_level_agent: Optional[Any] = None,
        low_level_model_path: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # --- inner environment -------------------------------------------
        self.inner_env = StackEnv(config=config, render_mode=render_mode)

        # --- bay numbering -----------------------------------------------
        all_bays = self.inner_env._generate_bay_coords(self.inner_env.yard_shape[0])
        self.baynum_list: List[int] = [b for b in all_bays if b % 2 != 0]
        self.num_physical_bays: int = len(self.baynum_list)

        # --- spaces ------------------------------------------------------
        self.observation_space = self.inner_env.observation_space
        self.action_space = gym.spaces.Discrete(self.num_physical_bays)

        # --- low-level agent ---------------------------------------------
        if low_level_agent is not None:
            self.low_level_agent = low_level_agent
            self._ll_is_frozen_rl = isinstance(low_level_agent, FrozenRLLowLevelAgent)
        elif low_level_model_path is not None:
            self.low_level_agent = FrozenRLLowLevelAgent(
                model_path=low_level_model_path,
                n_actions=self.inner_env.action_space.n,
                n_rows=self.inner_env.yard_shape[1],
            )
            self._ll_is_frozen_rl = True
        else:
            self.low_level_agent = LowLevelAgent(
                vessel_shape=self.inner_env.vessel_shape,
                yard_shape=self.inner_env.yard_shape,
                num_slot_attrs=self.inner_env.num_slot_attrs,
                policy_type=low_level_policy_type,
            )
            self._ll_is_frozen_rl = False

        # Expose inner env attributes needed by training utilities
        self.render_mode = render_mode

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs_for_low_level(self) -> Dict[str, np.ndarray]:
        """
        Build the flat_parsed style dict that rule-based LowLevelAgent
        expects, directly from the inner env's raw state arrays.
        """
        yard_state = self.inner_env.yard_state

        idx = self.inner_env.current_vessel_container
        if idx is not None:
            current_container = self.inner_env.vessel_state[idx]
        else:
            current_container = np.zeros(self.inner_env.num_slot_attrs, dtype=int)

        return {
            "yard_state": yard_state,
            "current_container": current_container,
        }

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def action_masks(self) -> List[bool]:
        """
        Return a boolean mask over bay actions.  A bay is valid (True)
        if it contains at least one non-full stack.
        """
        if self.inner_env.current_vessel_container is None:
            return [False] * self.num_physical_bays

        valid_actions = self.inner_env._get_valid_yard_actions()
        if len(valid_actions) == 0:
            return [False] * self.num_physical_bays

        # Collect set of bays that have at least one valid stack
        valid_bays = set()
        for a in valid_actions:
            bay, _row = self.inner_env._action_to_bay_row(int(a))
            valid_bays.add(bay)

        mask = [False] * self.num_physical_bays
        for bay_idx, bay_num in enumerate(self.baynum_list):
            if bay_num in valid_bays:
                mask[bay_idx] = True

        return mask

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[Union[np.ndarray, Dict], Dict]:
        obs, info = self.inner_env.reset(seed=seed, **kwargs)
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        # Map high-level action (bay index) to bay number
        selected_bay = self.baynum_list[action]

        # Get valid yard actions from inner env
        valid_actions = self.inner_env._get_valid_yard_actions()

        # Build observation for low-level agent
        if self._ll_is_frozen_rl:
            # Frozen RL agent expects the raw observation array
            obs_for_ll = self.inner_env._create_observation()
        else:
            # Rule-based agent expects dict format
            obs_for_ll = self._get_obs_for_low_level()

        # Low-level agent selects the specific stack within the bay
        slot_action = self.low_level_agent.get_action(
            obs_for_ll, valid_actions.tolist(), selected_bay
        )

        # Step the inner environment
        obs, reward, terminated, truncated, info = self.inner_env.step(slot_action)
        info["selected_bay"] = selected_bay

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self.inner_env.render()

    def close(self) -> None:
        self.inner_env.close()
