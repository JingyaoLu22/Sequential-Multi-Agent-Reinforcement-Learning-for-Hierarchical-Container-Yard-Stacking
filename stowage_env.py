import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional
from enum import Enum


class StateIds(Enum):
    BAY = 0
    ROW = 1
    TIER = 2
    IS_OCCUPIED = 3


class StowageEnv(gym.Env):
    """
    Args:
        container_type (str): Type of container to place in the yard. Can be one of ["one", "two", "mixed"].
            - "one": Place only the 20-feet containers in the yard.
            - "two": Place only the 40-feet containers in the yard.
            - "mixed": Place both 20-feet and 40-feet containers in the yard.
    """

    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(self, config: Dict = None, render_mode: Optional[str] = None):
        # Render part
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        if config is None:
            config = {}

        self.vessel_shape = config.get("vessel_shape", (1, 2, 2))
        self.yard_shape = config.get("yard_shape", (2, 2, 2))
        self.num_containers = config.get("num_containers", 4)
        self.container_type = config.get("container_type", "one")

        # Calculate total physical slots
        self.total_vessel_slots = self.vessel_shape[0] * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_slots = self.yard_shape[0] * self.yard_shape[1] * self.yard_shape[2]

        self.num_containers = min(self.num_containers, self.total_yard_slots)

        # Calculate total coordinates
        self.num_vessel_bay = (
            self.vessel_shape[0] // 2 + self.vessel_shape[0]
        )  # number of bay coords. e.g., need 6 coords to encode 4 bays
        self.num_yard_bay = self.yard_shape[0] // 2 + self.yard_shape[0]
        self.total_vessel_coords = self.num_vessel_bay * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * self.yard_shape[1] * self.yard_shape[2]
        self.obs_coords = self.total_vessel_coords + self.total_yard_coords + 1  # +1 for current target

        # Sequencer
        self.current_vessel_slot = None
        self.vessel_slots_filled = 0

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(
                max(self.num_vessel_bay, self.num_yard_bay),  # bay upper limit
                max(self.vessel_shape[1], self.yard_shape[1]),  # row upper limit
                max(self.vessel_shape[2], self.yard_shape[2]),  # tier upper limit
                1,  # occupied upper limit
            ),
            shape=(self.obs_coords, 4),
            dtype=np.int32,
        )
        self.action_space = gym.spaces.Discrete(self.yard_shape[0] * self.yard_shape[1] * self.yard_shape[2])
        # Each slot stores 4 values: bay, row, tier, occupied(0/1)

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        # Check if the action is valid (container exists and is accessible)
        valid_actions = self._get_valid_yard_actions().tolist()

        if action not in valid_actions:
            # Invalid action - return large negative reward
            reward = -100.0
            observation = self._create_observation()
            info["yard_mask"] = valid_actions
            return observation, reward, terminated, truncated, info

        original_bay = self.yard_state[action, StateIds.BAY.value]
        original_row = self.yard_state[action, StateIds.ROW.value]
        original_tier = self.yard_state[action, StateIds.TIER.value]
        # Identify the container stacking on top of the action container
        same_bay_row_mask = (
            (self.yard_state[:, StateIds.BAY.value] == original_bay)
            & (self.yard_state[:, StateIds.ROW.value] == original_row)
            & (self.yard_state[:, StateIds.TIER.value] > original_tier)
            & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
        )

        upper_slots = np.where(same_bay_row_mask)[
            0
        ]  # get the indices of the containers that are above the action container in ascending order of tier
        sorted_upper_slots = []  # sort the upper slots in descending order of tier
        if len(upper_slots) > 0:
            sorted_upper_slots = sorted(
                upper_slots, key=lambda x: self.yard_state[x, StateIds.TIER.value], reverse=True
            )

            self.yard_state[sorted_upper_slots[0], StateIds.IS_OCCUPIED.value] = (
                0  # remove the top container in the same stack
            )

            for i in range(1, len(sorted_upper_slots)):  # From top to bottom, move the containers one slot down
                current_slot = sorted_upper_slots[i]
                self.yard_state[current_slot, StateIds.TIER.value] -= 1

        else:
            self.yard_state[action, StateIds.IS_OCCUPIED.value] = 0
        self.vessel_state[self.current_vessel_slot, StateIds.IS_OCCUPIED.value] = 1

        self.vessel_slots_filled += 1
        self.current_vessel_slot = self._get_next_vessel_slot()

        shifters = len(sorted_upper_slots)
        reward = -shifters

        terminated = self.current_vessel_slot is None
        valid_actions = self._get_valid_yard_actions()

        observation = self._create_observation()
        info.update({"yard_mask": valid_actions, "shifters": shifters, "vessel_slots_filled": self.vessel_slots_filled})

        return observation, reward, terminated, truncated, info

    def reset(self):
        yard_bays = self._generate_bay_coords(self.yard_shape[0])
        _, R, T = self.yard_shape
        self.yard_state = np.zeros((self.total_yard_coords, 4), dtype=int)
        # Generate coords in bay-row-tier order
        self.yard_state[:, StateIds.BAY.value] = np.repeat(yard_bays, R * T)
        self.yard_state[:, StateIds.ROW.value] = np.tile(np.arange(1, R + 1).repeat(T), len(yard_bays))
        self.yard_state[:, StateIds.TIER.value] = np.tile(np.arange(1, T + 1), len(yard_bays) * R)

        physical_vessel_bays = self.vessel_shape[0]
        vessel_bays = self._generate_bay_coords(physical_vessel_bays)
        _, Rv, Tv = self.vessel_shape
        self.vessel_state = np.zeros((self.total_vessel_coords, 4), dtype=int)

        self.vessel_state[:, StateIds.BAY.value] = np.repeat(vessel_bays, Rv * Tv)
        self.vessel_state[:, StateIds.ROW.value] = np.tile(np.arange(1, Rv + 1).repeat(Tv), len(vessel_bays))
        self.vessel_state[:, StateIds.TIER.value] = np.tile(np.arange(1, Tv + 1), len(vessel_bays) * Rv)

        self.vessel_slots_filled = 0

        self._initialize_containers()

        # Get the next vessel slot to fill
        self.current_vessel_slot = self._get_next_vessel_slot()

        # Create observation
        observation = self._create_observation()
        info = {}

        return observation, info

    def _generate_bay_coords(self, physical_bays: int) -> list:
        """Generate bay coordinates based on the number of physical bays."""
        bay_groups = []
        group_start = 1

        for _ in range(physical_bays // 2):
            bay_groups.extend([group_start, group_start + 1, group_start + 2])
            group_start += 4

        if physical_bays % 2 == 1:
            bay_groups.append(group_start)

        return bay_groups

    def _get_next_vessel_slot(self) -> Optional[Tuple[int, int, int]]:
        if self.container_type == "one":
            valid_slots = np.where(
                (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
                & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
            )[0]
            if len(valid_slots) == 0:
                return None
            return valid_slots[0]

    def _create_observation(self):
        state = np.concatenate((self.vessel_state, self.yard_state), axis=0)
        if self.current_vessel_slot is not None:
            state = np.concatenate((state, self.vessel_state[self.current_vessel_slot].reshape(1, -1)), axis=0)
        else:
            state = np.concatenate((state, np.zeros((1, 4), dtype=int)), axis=0)
        return state

    def _get_valid_yard_actions(self) -> np.ndarray:
        occupied_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 0
        if self.container_type == "one":
            bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 == 1
            valid_actions = np.where(occupied_mask & bay_mask)[0]
        return valid_actions

    def render(self):
        pass

    def _get_bay_groups(self, type="yard") -> dict:
        """generate bay groups based on the yard shape
        Returns:
            dict: dictionary containing bay groups, e.g. {1: [1, 2, 3], 2: [5, 6, 7]}
        """
        groups = {}
        group_id = 1
        if type == "yard":
            bays = sorted(np.unique(self.yard_state[:, StateIds.BAY.value]))
        else:
            bays = sorted(np.unique(self.vessel_state[:, StateIds.BAY.value]))

        current_group = []
        for bay in bays:
            if current_group and bay - current_group[-1] > 1:
                groups[group_id] = current_group
                group_id += 1
                current_group = []
            current_group.append(bay)

        if current_group:
            groups[group_id] = current_group

        return groups

    def _initialize_containers(self):
        if self.container_type == "one":
            odd_bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 != 0
            available_slots = np.where(odd_bay_mask)[0]
            num_to_set = min(self.num_containers, len(available_slots))
            if num_to_set > 0:
                self.yard_state[available_slots[:num_to_set], StateIds.IS_OCCUPIED.value] = 1
