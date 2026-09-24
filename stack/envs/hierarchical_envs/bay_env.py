"""
Bay-level observation/action adapter for the hierarchical MARL pipeline.

This class does NOT own an independent StackEnv and does NOT call env.step().

It receives a reference to the single shared StackEnv and exposes:

    1. The original global StackEnv observation for Agent B.
    2. A bay-level action space.
    3. A bay-level action mask derived from StackEnv's original
       stack-level validity logic.
    4. Mapping helpers between zero-based bay actions used by Agent B
       and the physical odd bay numbering used inside StackEnv.

Important:
    The underlying observation representation is NOT changed here.
    In particular, when StackEnv uses "stack_features_v3", BayEnv returns
    exactly the same feature vector produced by StackEnv.

    BayEnv is only an adapter/view over the shared StackEnv.
"""

from __future__ import annotations

from typing import List

import gymnasium as gym
import numpy as np

from ..stack_gym import StackEnv


class BayEnv:
    """
    Bay-level adapter used by Agent B.

    Parameters
    ----------
    stack_env : StackEnv
        The single shared physical environment.

    Notes
    -----
    Agent B uses zero-based bay actions:

        0, 1, 2, ..., n_bays - 1

    while StackEnv internally represents physical bays as:

        1, 3, 5, ..., 2*n_bays - 1

    This class handles the conversion between the two representations.

    No environment transition happens when Agent B selects a bay.
    The actual StackEnv.step(...) call happens later, after Agent R
    has selected a row/stack inside the chosen bay.
    """

    def __init__(self, stack_env: StackEnv) -> None:
        self.stack_env = stack_env

        # --------------------------------------------------------------
        # Bay numbering
        # --------------------------------------------------------------
        #
        # Reuse the same physical bay numbering logic used in the
        # original HierarchicalHighLevelEnv.
        #
        # Example:
        #     physical bays = [1, 3, 5, 7]
        #     Agent B actions = [0, 1, 2, 3]
        #
        all_bays = self.stack_env._generate_bay_coords(
            self.stack_env.yard_shape[0]
        )

        self.baynum_list: List[int] = [
            int(bay)
            for bay in all_bays
            if bay % 2 != 0
        ]

        self.num_bays = len(self.baynum_list)

        # --------------------------------------------------------------
        # Observation space
        # --------------------------------------------------------------
        #
        # Agent B keeps the ORIGINAL global StackEnv observation.
        #
        # Normally this is a Box when action_mask="default".
        #
        # StackEnv can optionally wrap observations as:
        #
        # {
        #     "observation": ...,
        #     "mask": ...
        # }
        #
        # In our new MARL pipeline masks are stored separately, so in
        # that case we expose only the original observation component.
        #
        if isinstance(self.stack_env.observation_space, gym.spaces.Dict):
            if "observation" not in self.stack_env.observation_space.spaces:
                raise ValueError(
                    "BayEnv expected StackEnv observation_space to contain "
                    "an 'observation' entry."
                )

            self.observation_space = (
                self.stack_env.observation_space.spaces["observation"]
            )
        else:
            self.observation_space = self.stack_env.observation_space

        # --------------------------------------------------------------
        # Action space
        # --------------------------------------------------------------
        #
        # One action for each physical bay.
        #
        self.action_space = gym.spaces.Discrete(self.num_bays)

    # ==================================================================
    # Observation
    # ==================================================================

    def get_observation(self) -> np.ndarray:
        """
        Return Agent B's global observation.

        The actual feature construction is delegated completely to
        StackEnv._create_observation().

        Therefore, for thesis configuration:

            observation_type == "stack_features_v3"

        the returned vector is exactly the original StackEnv
        stack_features_v3 representation.

        Returns
        -------
        np.ndarray
            Original global observation produced by StackEnv.
        """

        observation = self.stack_env._create_observation()

        # StackEnv optionally embeds an action mask inside the observation.
        # Our custom MARL pipeline stores masks separately, so only return
        # the original observation vector here.
        if isinstance(observation, dict):
            if "observation" not in observation:
                raise ValueError(
                    "StackEnv returned a dict observation without an "
                    "'observation' key."
                )

            observation = observation["observation"]

        return np.asarray(observation, dtype=np.float32)

    # ==================================================================
    # Action masking
    # ==================================================================

    def action_masks(self) -> List[bool]:
        """
        Build Agent B's bay-level action mask.

        StackEnv already contains all physical placement constraints in:

            StackEnv._get_valid_yard_actions()

        That function handles, among other things:

            - stack capacity
            - valid odd bays
            - container-size compatibility
            - IMO compatibility

        We DO NOT reproduce those rules here.

        Instead, valid stack actions are projected onto bays:

            bay is valid
                iff
            at least one valid stack exists inside that bay.

        Returns
        -------
        List[bool]
            Boolean mask of length num_bays.
        """

        # No container is currently waiting to be placed.
        if self.stack_env.current_vessel_container is None:
            return [False] * self.num_bays

        # Reuse original StackEnv validity logic.
        valid_stack_actions = self.stack_env._get_valid_yard_actions()

        if len(valid_stack_actions) == 0:
            return [False] * self.num_bays

        valid_bay_numbers = set()

        for stack_action in valid_stack_actions:
            physical_bay, _row = self.stack_env._action_to_bay_row(
                int(stack_action)
            )

            valid_bay_numbers.add(int(physical_bay))

        return [
            bay_number in valid_bay_numbers
            for bay_number in self.baynum_list
        ]

    # ==================================================================
    # Bay-index conversion
    # ==================================================================

    def bay_index_to_number(self, bay_idx: int) -> int:
        """
        Convert Agent B's zero-based action to StackEnv's physical bay number.

        Example
        -------
        bay_idx = 0 -> physical bay 1
        bay_idx = 1 -> physical bay 3
        bay_idx = 2 -> physical bay 5
        """

        if not self.action_space.contains(bay_idx):
            raise ValueError(
                f"Invalid bay_idx={bay_idx}. "
                f"Expected 0 <= bay_idx < {self.num_bays}."
            )

        return self.baynum_list[bay_idx]

    def bay_number_to_index(self, bay_number: int) -> int:
        """
        Convert StackEnv physical odd bay numbering back to Agent B index.

        Example
        -------
        physical bay 1 -> bay_idx 0
        physical bay 3 -> bay_idx 1
        physical bay 5 -> bay_idx 2
        """

        if bay_number not in self.baynum_list:
            raise ValueError(
                f"Invalid physical bay number {bay_number}. "
                f"Valid bays are {self.baynum_list}."
            )

        return self.baynum_list.index(bay_number)

    # ==================================================================
    # Convenience helpers
    # ==================================================================

    def get_valid_bay_indices(self) -> np.ndarray:
        """
        Return zero-based indices of currently valid Bay Agent actions.
        """

        mask = np.asarray(self.action_masks(), dtype=bool)

        return np.flatnonzero(mask)

    def is_bay_valid(self, bay_idx: int) -> bool:
        """
        Check whether a specific Agent B bay action is currently valid.
        """

        if not self.action_space.contains(bay_idx):
            return False

        return bool(self.action_masks()[bay_idx])