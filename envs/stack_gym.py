import warnings
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional
from enum import Enum
import colorsys


class StateIds(Enum):
    BAY = 0
    ROW = 1
    TIER = 2
    IS_OCCUPIED = 3
    GROUP = 4


class StackEnv(gym.Env):
    """
    Stacking Environment for moving containers from vessel to yard
    
    Containers start fully loaded on the vessel. Environment selects containers sequentially and randomly
    from the vessel (top containers in each stack) at each time step and the agent is tasjed with selecting a slot for
    the container in the yard. The goal is to:
    1. Place similar-group containers close together
    2. Avoid placing dissimilar containers close together
    3. Optimize yard usage by reusing bays (penalty for placing containers in previously empty stacks)
    """
    
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(self, config: Dict = None, render_mode: Optional[str] = None):
        if config is None:
            config = {}

        self.vessel_shape = config.get("vessel_shape", (1, 2, 2))
        self.yard_shape = config.get("yard_shape", (2, 2, 2))
        self.num_containers = config.get("num_containers", 4)
        self.group_num = config.get("group_num", 1)
        self.group_placement = config.get("group_placement", "fixed")
        self.seed = config.get("seed", 0)
        self.action_mask = config.get("action_mask", "default")
        self.reward_scheme = config.get("reward_scheme", "progress")
        
        # Calculate total physical slots
        self.total_vessel_slots = self.vessel_shape[0] * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_slots = self.yard_shape[0] * self.yard_shape[1] * self.yard_shape[2]
        self.total_timesteps = 0
        self.total_containers_remaining = 0

        self.num_containers = min(self.num_containers, self.total_vessel_slots)
        if self.num_containers > self.total_vessel_slots:
            warnings.warn(
                f"Number of containers is set to {self.num_containers} as it exceeds the total vessel slots {self.total_vessel_slots}"
            )
        self.num_slot_attrs = 5

        # Calculate total coordinates (used later for numbering and removing even numbered bays)
        self.num_vessel_bay = self.vessel_shape[0] // 2 + self.vessel_shape[0]
        self.num_yard_bay = self.yard_shape[0] // 2 + self.yard_shape[0]

        self.total_vessel_coords = self.num_vessel_bay * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * self.yard_shape[1] * self.yard_shape[2]
        
        self.obs_coords = self.total_vessel_coords + self.total_yard_coords + 1  # +1 for current target

        # Sequencer
        self.current_vessel_container = None  # Index of container currently being retrieved
        self.current_retrieval_group = None   # Group being retrieved
        self.containers_retrieved = 0

        # Track which bay/row cells have been occupied in yard
        self.yard_bay_row_occupied = set()

        # Observation and action spaces
        # Maximum for observation space based on various yard, vessel and container attributes
        # shape=(self.obs_coords, 5). Each slot stores 5 values: bay, row, tier, occupied(0/1), group number of the container
        observation_space = gym.spaces.Box(
            low=0,
            high=max(
                max(self.num_vessel_bay, self.num_yard_bay),    # bay upper limit
                max(self.vessel_shape[1], self.yard_shape[1]),  # row upper limit
                max(self.vessel_shape[2], self.yard_shape[2]),  # tier upper limit
                1,                                              # is_occupied upper limit
                self.group_num,                                 # group number upper limit
            ),
            shape=(self.obs_coords * self.num_slot_attrs,),
            dtype=np.int64,
        )
        if self.action_mask == "default":
            self.observation_space = observation_space
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": observation_space,
                    "mask": gym.spaces.Box(low=0, high=1, shape=(self.total_yard_coords,), dtype=np.bool_),
                }
            )

        # Action is choosing a yard slot to place the current vessel container
        self.action_space = gym.spaces.Discrete(self.total_yard_coords)

        # Render part
        self.render_mode = render_mode
        self.screen_width = (
            35 * max(self.vessel_shape[1] * self.vessel_shape[0], self.yard_shape[1] * self.yard_shape[0]) + 60
        )
        self.screen_height = 60 * max(self.vessel_shape[2], self.yard_shape[2]) + 150
        self.screen = None

    def step(self, action):
        """
        Execute one step: place current container from vessel to yard at a specified action (yard slot index)
        """
        self.total_timesteps += 1  
        truncated = False
        info = {}
        
        # Get valid actions for current state
        valid_actions = self._get_valid_yard_actions()
        valid_actions_list = valid_actions.tolist() if isinstance(valid_actions, np.ndarray) else list(valid_actions)

        # Check if action is valid (in valid_actions list)
        # Invalid action (either occupied slot, even bay, not bottommost tier, or out of range slot number)
        # Invlalid actions do not change the environment and incur a large negative reward (-100)
        # May need to modify this later based on how action masking is implemented in hrl
        if action not in valid_actions_list:
            reward = -100.0
            observation = self._create_observation()
            info["yard_mask"] = valid_actions_list
            terminated = False
            return observation, reward, terminated, truncated, info

        # Check if placement bay is adjacent to vessel container's bay
        vessel_container_bay = self.vessel_state[self.current_vessel_container, StateIds.BAY.value]
        yard_placement_bay = self.yard_state[action, StateIds.BAY.value]
        
        # Container can only be placed in same bay or adjacent bays (left/right) (REMOVING THIS FOR NOW)
        # Huge negative reward if agent tries to place container in non-adjacent bay.
        # In this case, action does not change environment
        # allowed_bays = [vessel_container_bay - 2, vessel_container_bay, vessel_container_bay + 2]
        # if yard_placement_bay not in allowed_bays:
            # reward = -100.0
            # observation = self._create_observation()
            # info["yard_mask"] = valid_actions_list
            # terminated = False
            # return observation, reward, terminated, truncated, info

        # Calculate reward
        reward = self._calculate_reward(action)

        # Place container in yard
        self._place_container_in_yard(action, self.current_vessel_container)

        # Remove container from vessel
        self.vessel_state[self.current_vessel_container, StateIds.IS_OCCUPIED.value] = 0
        self.total_containers_remaining -= 1

        # Get next vessel container to retrieve
        self.current_vessel_container = self._get_next_vessel_container()

        # Episode terminates when all containers retrieved or no more valid groups
        terminated = (self.current_vessel_container is None) or (self.total_containers_remaining == 0)
        valid_actions = self._get_valid_yard_actions() if not terminated else np.array([], dtype=int)
        valid_actions_list = valid_actions.tolist() if isinstance(valid_actions, np.ndarray) else list(valid_actions)

        observation = self._create_observation()
        info.update({
            "yard_mask": valid_actions_list,
            "containers_retrieved": self.containers_retrieved,
            "containers_remaining": self.total_containers_remaining,
        })

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        """Reset environment wrapper"""
        self._reset()
        observation = self._create_observation()
        info = {}
        return observation, info

    def _reset(self):
        """Internal reset logic :

        Generate coords in bay-row-tier order
        To generate all possible bay-row-tier combinations, we can try grouping like this:
        Starting grouping varied tiers into rows as: row1 -> tier1, tier2, tier3, row2 -> tier1, tier2, tier3...
        Then grouping rows into bays as: bay1 -> row1(including 3 tiers), row2(incl. 3 tiers), bay2 -> row1, row2...
        Then we need tiers as num_bays * num_rows * (tier1, tier2, tier3), which can be achieved by using np.tile
        For rows we need to repeat the rows for each bay, we also need to repeat each element for a whole tier array
        like num_bays * (row1, row1, row1, row2, row2, row2, ...). This can be achieved by using np.tile for outer array
        and np.repeat for inner array
        For bays we just need to repeat each elements num_rows * num_tiers times
        
        Example of yard_state and vessel_state after initialization for shape (3,2,2) for (bays,rows,tiers).
        Each row represents a slot with 5 attributes: BAY, ROW, TIER,
        IS_OCCUPIED (boolean whther slot is occupied), GROUP (which group of container it holds).

        Note : Rows with even numbered bays (2,4,6,...) are made invalid using masks and and is not being used.
        So 3 bays being used here are actually bays 1,3,5 in physical layout.
        
            Index  BAY  ROW	TIER IS_OCCUPIED GROUP
            0	    1	 1	 1	     0	      0
            1	    1	 1	 2	     0	      0
            2	    1	 2	 1	     0	      0
            3	    1	 2	 2	     0	      0
            4	    2	 1	 1	     0	      0
            5	    2	 1	 2	     0	      0
            6	    2	 2	 1	     0	      0
            7	    2	 2	 2	     0	      0
            8	    3	 1	 1	     0	      0
            9	    3	 1	 2	     0	      0
            10	    3	 2	 1	     0	      0
            11	    3	 2	 2	     0	      0
            12	    5	 1	 1	     0	      0
            13	    5	 1	 2	     0	      0
            14	    5	 2	 1	     0	      0
            15	    5	 2	 2	     0	      0

        """
        # yard_state has shape (total_yard_coords, 5)
        self.total_timesteps = 0
        self.containers_retrieved = 0
        self.yard_bay_row_occupied = set()

        # Initialize yard
        # _generate_bay_coords generates initial bay coordinates before masking.
        # In the above example, _generate_bay_coords(3) would return [1,2,3,5] (2 will be excluded later using masks)
        # So only bays 1,3,5 will be used for yard placement
        yard_bays = self._generate_bay_coords(self.yard_shape[0])
        _, R, T = self.yard_shape
        self.yard_state = np.zeros((self.total_yard_coords, self.num_slot_attrs), dtype=int)
        self.yard_state[:, StateIds.BAY.value] = np.repeat(yard_bays, R * T)
        self.yard_state[:, StateIds.ROW.value] = np.tile(np.arange(1, R + 1).repeat(T), len(yard_bays))
        self.yard_state[:, StateIds.TIER.value] = np.tile(np.arange(1, T + 1), len(yard_bays) * R)

        # Initialize vessel (similar logic as yard)
        vessel_bays = self._generate_bay_coords(self.vessel_shape[0])
        _, Rv, Tv = self.vessel_shape
        self.vessel_state = np.zeros((self.total_vessel_coords, self.num_slot_attrs), dtype=int)
        self.vessel_state[:, StateIds.BAY.value] = np.repeat(vessel_bays, Rv * Tv)
        self.vessel_state[:, StateIds.ROW.value] = np.tile(np.arange(1, Rv + 1).repeat(Tv), len(vessel_bays))
        self.vessel_state[:, StateIds.TIER.value] = np.tile(np.arange(1, Tv + 1), len(vessel_bays) * Rv)

        # Assign groups to vessel containers
        if self.group_num > 1:
            odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 == 1
            odd_bay_indices = np.where(odd_bay_mask)[0]                       # even bay indices are ignored
            valid_slots = odd_bay_indices

            slots_per_group = len(valid_slots) // self.group_num
            for group in range(self.group_num):
                start_pos = group * slots_per_group
                end_pos = (group + 1) * slots_per_group if group < self.group_num - 1 else len(valid_slots)
                group_indices = valid_slots[start_pos:end_pos]
                if len(group_indices) > 0:
                    self.vessel_state[group_indices, StateIds.GROUP.value] = group

        # Load vessel with containers
        self._initialize_vessel_loaded()

        # Start retrieval with first available group
        self.current_retrieval_group = 0
        self.current_vessel_container = self._get_next_vessel_container()
        self.total_containers_remaining = self.num_containers

    def _initialize_vessel_loaded(self):
        """
        Load vessel with containers - mirrors stowage but loads vessel instead of yard
        """
        odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 != 0
        available_slots = np.where(odd_bay_mask)[0]               # even bay indices are ignored
        num_to_set = min(self.num_containers, len(available_slots))

        if num_to_set > 0:
            selected_slots = available_slots[:num_to_set]
            self.vessel_state[selected_slots, StateIds.IS_OCCUPIED.value] = 1

            if self.group_num > 1:
                containers_per_group = num_to_set // self.group_num
 
                if self.group_placement == "fixed":
                    # Fixed placement of groups
                    for group in range(self.group_num):
                        start_idx = group * containers_per_group
                        end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                        if start_idx < end_idx:
                            self.vessel_state[selected_slots[start_idx:end_idx], StateIds.GROUP.value] = group
                
                else:
                    # Random placement of groups
                    rng = np.random.RandomState(self.seed)
                    shuffled_indices = rng.permutation(num_to_set)
                    shuffled_slots = selected_slots[shuffled_indices]
                    for group in range(self.group_num):
                        start_idx = group * containers_per_group
                        end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                        if start_idx < end_idx:
                            self.vessel_state[shuffled_slots[start_idx:end_idx], StateIds.GROUP.value] = group

    def _generate_bay_coords(self, physical_bays: int) -> list:
        """
        Generate initial bay coordinates based on the number of physical bays
        For example, _generate_bay_coords(7) would return [1, 2, 3, 5, 6, 7, 9, 10, 11, 13]
        (2,6,10 will be excluded later using masks outside this function)
        So for 7 bays, the final bay numbering will be [1,3,5,7,9,11,13]
        """
        bay_groups = []
        group_start = 1

        for _ in range(physical_bays // 2):
            bay_groups.extend([group_start, group_start + 1, group_start + 2])
            group_start += 4

        if physical_bays % 2 == 1:
            bay_groups.append(group_start)

        return bay_groups

    def _get_next_vessel_container(self):
        """
        Get next accessible top container from vessel - randomly selected from all topmost containers
        """
        occupied_mask = self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 1
        occupied_indices = np.where(occupied_mask)[0]
        
        if len(occupied_indices) == 0:
            return None
        
        bays = self.vessel_state[occupied_indices, StateIds.BAY.value]
        rows = self.vessel_state[occupied_indices, StateIds.ROW.value]
        tiers = self.vessel_state[occupied_indices, StateIds.TIER.value]
        
        topmost_containers = []
        unique_bay_rows = set(zip(bays, rows))
        
        # For each stack, find the highest tier container
        for bay, row in unique_bay_rows:
            mask = (bays == bay) & (rows == row)
            indices_in_stack = occupied_indices[mask]
            tiers_in_stack = tiers[mask]
            top_idx_in_stack = np.argmax(tiers_in_stack)
            topmost_containers.append(indices_in_stack[top_idx_in_stack])
        
        if len(topmost_containers) == 0:
            return None
        
        # Randomly select one of the topmost containers
        rng = np.random.RandomState(self.seed + self.total_timesteps)
        selected_idx = rng.choice(topmost_containers)
        
        # Update current retrieval group based on selected container
        self.current_retrieval_group = int(self.vessel_state[selected_idx, StateIds.GROUP.value])
        
        return selected_idx

    def _place_container_in_yard(self, yard_slot, vessel_container_idx):
        """
        Place vessel container in yard at specified slot (only lowest unoccupied slot per stack is filled)
        """
        
        bay = self.yard_state[yard_slot, StateIds.BAY.value]
        row = self.yard_state[yard_slot, StateIds.ROW.value]
        group = self.vessel_state[vessel_container_idx, StateIds.GROUP.value]
        
        # Find lowest unoccupied tier in this stack
        bay_row_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (self.yard_state[:, StateIds.ROW.value] == row)
        bay_row_indices = np.where(bay_row_mask)[0]
        
        occupied_in_stack = self.yard_state[bay_row_indices, StateIds.IS_OCCUPIED.value]
        tiers_in_stack = self.yard_state[bay_row_indices, StateIds.TIER.value]
        
        # Sort by tier to find first unoccupied
        sorted_indices = np.argsort(tiers_in_stack)
        placement_idx = None
        for idx in sorted_indices:
            if self.yard_state[bay_row_indices[idx], StateIds.IS_OCCUPIED.value] == 0:
                placement_idx = bay_row_indices[idx]
                break
        
        if placement_idx is not None:
            self.yard_state[placement_idx, StateIds.IS_OCCUPIED.value] = 1
            self.yard_state[placement_idx, StateIds.GROUP.value] = group
            
            # Add stack to list of occupied stacks (used in reward function)
            self.yard_bay_row_occupied.add((bay, row))
            self.containers_retrieved += 1

    def _calculate_reward(self, yard_action):
        """
        Function takes in yard_action (index of slot chosen to place container) and calculates reward for this placement
        Calculate reward based on exact rules:
        1. Penalty (-1) for occupying new unoccupied ground slot in stack
        2. Reward (+1/-1) for placing container in a stack having same/dissimilar containers in the same stack.
        3. Reward (+0.5/-0.5) for placing container having same/dissimilar containers in +/-1 adjacent rows (same bay only)
        4. Reward (+0.25/-0.25) for placing container having same/dissimilar containers in other stacks in same bay (not same row or adjacent rows)
        
        2,3 and 4 rewards are added for each similar/dissimilar container found in the same stack or adjacent rows.
        """
        placement_bay = self.yard_state[yard_action, StateIds.BAY.value]
        placement_row = self.yard_state[yard_action, StateIds.ROW.value]
        container_group = self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
        
        reward = 0.0
        
        # Rule 1: Penalty for occupying new ground slot in unoccupied stack (REMOVING THIS FOR NOW)
        # if (placement_bay, placement_row) not in self.yard_bay_row_occupied:
            # reward -= 0.5
        
        # Rule 2: Reward/penalty for placing container in a stack having same/dissimilar containers in the same stack.
        bay_row_mask = (self.yard_state[:, StateIds.BAY.value] == placement_bay) & \
                        (self.yard_state[:, StateIds.ROW.value] == placement_row)
        bay_row_indices = np.where(bay_row_mask)[0]

        same_stack_occupied_mask = self.yard_state[bay_row_indices, StateIds.IS_OCCUPIED.value] == 1
        same_stack_occupied_slots = bay_row_indices[same_stack_occupied_mask]

        if len(same_stack_occupied_slots) > 0:
            same_stack_container_groups = self.yard_state[same_stack_occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(same_stack_container_groups == container_group)
            diff_group_count = np.sum(same_stack_container_groups != container_group)
            reward += (same_group_count - diff_group_count) * 1.0

        # Find  +/- 1 adjacent rows in the same bay
        adjacent_rows = []
        if placement_row > 1:
            adjacent_rows.append(placement_row - 1)
        if placement_row < self.yard_shape[1]:
            adjacent_rows.append(placement_row + 1)

        
        # Rule 4 reward/penalty for other stacks in same bay (not same row or adjacent rows)
        same_bay_mask = (self.yard_state[:, StateIds.BAY.value] == placement_bay) & \
                        (self.yard_state[:, StateIds.ROW.value] != placement_row) & \
                         ~np.isin(self.yard_state[:,StateIds.ROW.value],adjacent_rows)
        
        same_bay_indices = np.where(same_bay_mask)[0]

        same_bay_occupied_slots_mask = self.yard_state[same_bay_indices, StateIds.IS_OCCUPIED.value] == 1
        same_bay_occupied_slots = same_bay_indices[same_bay_occupied_slots_mask]

        if len(same_bay_occupied_slots) > 0:
            same_bay_container_groups = self.yard_state[same_bay_occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(same_bay_container_groups == container_group)
            diff_group_count = np.sum(same_bay_container_groups != container_group)
            reward += (same_group_count - diff_group_count) * 0.25

        # Rule 3 reward/penalty for adjacent rows in same bay
        adj_stack_mask = (self.yard_state[:,StateIds.BAY.value] == placement_bay) & \
                            np.isin(self.yard_state[:, StateIds.ROW.value], adjacent_rows)
        adj_stack_indices = np.where(adj_stack_mask)[0]

        adj_occupied_slots_mask = self.yard_state[adj_stack_indices, StateIds.IS_OCCUPIED.value] == 1
        adj_occupied_slots = adj_stack_indices[adj_occupied_slots_mask]

        if len(adj_occupied_slots) > 0:
            adj_stack_container_groups = self.yard_state[adj_occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(adj_stack_container_groups == container_group)
            diff_group_count = np.sum(adj_stack_container_groups != container_group)
            reward += (same_group_count - diff_group_count) * 0.5

        

        
        return reward

    def _get_valid_yard_actions(self) -> np.ndarray:
        """
        Get valid yard placement slots - only bottommost unoccupied tier stack in odd bays
        """
        if self.current_vessel_container is None:
            return np.array([], dtype=int)
        
        valid_actions = []
        
        # Get all unoccupied slots in odd bays
        unoccupied_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 0
        bay_values = self.yard_state[:, StateIds.BAY.value]
        bay_mask = (bay_values % 2) == 1
        valid_mask = unoccupied_mask & bay_mask
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.array([], dtype=int)
        
        # Find unique (bay, row) combinations from valid slots
        bays = self.yard_state[valid_indices, StateIds.BAY.value]
        rows = self.yard_state[valid_indices, StateIds.ROW.value]
        tiers = self.yard_state[valid_indices, StateIds.TIER.value]
        
        unique_bay_rows = set(zip(bays, rows))
        
        # For each (bay, row), find the lowest unoccupied tier
        for bay, row in unique_bay_rows:
            mask = (bays == bay) & (rows == row)
            indices_in_slot = valid_indices[mask]
            tiers_in_slot = tiers[mask]
            min_tier_idx = np.argmin(tiers_in_slot)
            valid_actions.append(indices_in_slot[min_tier_idx])
        
        return np.array(valid_actions, dtype=int)

    def action_masks(self):
        """
        For compatibility with SB3
        """
        if self.current_vessel_container is None:
            return [False] * self.action_space.n

        valid_actions = self._get_valid_yard_actions()
        return [action in valid_actions for action in range(self.action_space.n)]

    def _create_observation(self):
        """
        Create flattened observation vector
        Observation vector consists of: 
        1. vessel state (total_vessel_coords i.e. number of vessel slots), 
        2. yard state (total_vessel_coords i.e. number of yard slots) 
        3. Current vessel selected container state (1).
        Therefore, obs_coords = total_vessel_coords + total_yard_coords + 1
        Each slot has 5 attributes (num_slot_attrs): bay, row, tier, is_occupied(0/1), group number of container
        Final flattened observation shape: (obs_coords * num_slot_attrs,)
        """
        state = np.concatenate((self.vessel_state, self.yard_state), axis=0)
        if self.current_vessel_container is not None:
            state = np.concatenate((state, self.vessel_state[self.current_vessel_container].reshape(1, -1)), axis=0)
        else:
            state = np.concatenate((state, np.zeros((1, 5), dtype=int)), axis=0)
        state = state.flatten()
        
        if self.action_mask == "default":
            return state
        else:
            mask = self.action_masks()
            return {"observation": state, "mask": mask}

    def render(self):
        """
        Render the environment as an RGB array (similar to stowage env)
        """
        if self.render_mode != "rgb_array":
            return None

        try:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            import pygame
        except ImportError:
            raise ImportError("pygame is not installed")

        if not pygame.get_init():
            pygame.init()
        if self.screen is None:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.screen.fill((255, 255, 255))

        padding, title_height, section_gap = 10, 20, 50
        vessel_height = (self.screen_height - 3 * padding - 2 * title_height) * 0.4
        yard_height = (self.screen_height - 3 * padding - 2 * title_height) * 0.6

        font = pygame.font.Font(None, 24)
        self.screen.blit(font.render("Vessel", True, (0, 0, 0)), (padding, padding))
        self.screen.blit(font.render("Yard", True, (0, 0, 0)), (padding, padding + vessel_height + section_gap))

        self._draw_grid(
            self.vessel_state,
            padding + title_height,
            vessel_height,
            self.vessel_shape[0],
            self.vessel_shape[1],
            self.vessel_shape[2],
            True,
        )
        self._draw_grid(
            self.yard_state,
            3 * padding + 2 * title_height + vessel_height + section_gap,
            yard_height,
            self.yard_shape[0],
            self.yard_shape[1],
            self.yard_shape[2],
            False,
        )

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _draw_grid(self, state, top, height, bays, rows, tiers, is_vessel):
        """
        Draw a grid section (vessel or yard) with fixed cell size
        """
        import pygame

        cell_width, cell_height, left_margin, label_margin = self._setup_grid_dimensions(bays, rows)
        fonts = {"small": pygame.font.Font(None, 20), "tiny": pygame.font.Font(None, 18)}
        colors = self._setup_colors()
        self._draw_tier_labels(top, tiers, cell_height, left_margin, label_margin, fonts["small"])
        row_order = self._get_row_order(rows, is_vessel)
        cell_info = self._build_cell_info(state, bays, is_vessel)
        self._draw_grid_cells(
            cell_info,
            top,
            left_margin,
            bays,
            rows,
            tiers,
            row_order,
            cell_width,
            cell_height,
            label_margin,
            colors,
            fonts,
            is_vessel,
        )
        self._draw_bay_dividers(left_margin, top, bays, rows, cell_width, tiers, cell_height, colors["bay_grid"])

    def _setup_grid_dimensions(self, bays, rows):
        cell_width, cell_height = 35, 35
        padding, label_margin = 30, 15
        left_margin = max(padding, (self.screen_width - bays * rows * cell_width) / 2)
        return cell_width, cell_height, left_margin, label_margin

    def _setup_colors(self):
        group_colors = []
        for i in range(self.group_num):
            hue = i / self.group_num
            r, g, b = colorsys.hsv_to_rgb(hue, 0.3, 0.95)
            light_color = (int(r * 255), int(g * 255), int(b * 255))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.7)
            dark_color = (int(r * 255), int(g * 255), int(b * 255))
            group_colors.append((light_color, dark_color))

        return {
            "group_colors": group_colors,
            "empty": (255, 255, 255),
            "thin_grid": (180, 180, 180),
            "bay_grid": (0, 0, 0),
            "target": (255, 0, 0),
        }

    def _get_row_order(self, rows, is_vessel):
        """
        Determine row ordering based on vessel or yard
        """
        if is_vessel:
            if rows % 2 == 0:
                left = list(range(rows - 1, 0, -2))
                right = list(range(2, rows + 1, 2))
                return left + right
            else:
                left = list(range(rows, 0, -2))
                right = list(range(2, rows, 2))
                return left + right
        else:
            return list(range(1, rows + 1))

    def _draw_tier_labels(self, top, tiers, cell_height, left_margin, label_margin, font):
        """
        Draw tier labels on the left side
        """
        for t in range(1, tiers + 1):
            y = top + (tiers - t) * cell_height + cell_height / 2
            tier_label = font.render(f"{t}", True, (0, 0, 0))
            self.screen.blit(tier_label, (left_margin - label_margin, y - tier_label.get_height() / 2))

    def _build_cell_info(self, state, bays, is_vessel):
        """
        Build cell information dictionary for rendering
        """
        cell_info = {}

        if is_vessel:
            # Handle vessel containers
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                is_target = self._is_target_cell(i, is_vessel)
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1:
                    cell_info[(bay, row, tier)] = self._create_cell_props(is_occupied, is_target, i, group)

            
            # handle even bays
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                # Only apply even bay logic if it's occupied (maybe required later but not used currently)
                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            existing = cell_info.get(
                                (adj_bay, row, tier), self._create_cell_props(False, False, None, group)
                            )
                            existing["filled"] = True
                            existing["idx"] = i
                            cell_info[(adj_bay, row, tier)] = existing
        else:
            # Handle yard containers
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1:
                    cell_info[(bay, row, tier)] = self._create_cell_props(is_occupied, False, i, group)

            # handle even yard bays (should not be occupied but just in case)
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                # Only apply even bay logic if it's occupied (maybe required later but not used currently)
                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            cell_info[(adj_bay, row, tier)] = self._create_cell_props(is_occupied, False, i, group)

        return cell_info

    def _is_target_cell(self, idx, is_vessel):
        if is_vessel:
            return self.current_vessel_container == idx if self.current_vessel_container is not None else False
        return False

    def _create_cell_props(self, filled, target, idx, group):
        # Used for inheritance
        return {"filled": filled, "target": target, "idx": idx, "group": group}

    def _get_cell_border_style(self, cell):
        if cell["target"]:
            return (255, 0, 0), 3       # Red border for curently selcted container in vessel for yard placement
        return (180, 180, 180), 1

    def _draw_grid_cells(
        self,
        cell_info,
        top,
        left_margin,
        bays,
        rows,
        tiers,
        row_order,
        cell_width,
        cell_height,
        label_margin,
        colors,
        fonts,
        is_vessel,
    ):
        """
        Draw grid cells with labels
        """
        for b in range(bays):
            bay_num = b * 2 + 1
            bay_x = left_margin + b * rows * cell_width

            bay_center_x = bay_x + (rows * cell_width) / 2
            bay_label = fonts["small"].render(f"Bay {bay_num}", True, (0, 0, 0))
            self.screen.blit(bay_label, bay_label.get_rect(center=(bay_center_x, top - 10)))

            for pos, r in enumerate(row_order):
                # Draw row labels
                x_label = bay_x + pos * cell_width + cell_width / 2
                row_label = fonts["small"].render(f"{r}", True, (0, 0, 0))
                self.screen.blit(
                    row_label, row_label.get_rect(center=(x_label, top + tiers * cell_height + label_margin / 2))
                )
                
                # Draw cells for each tier
                for t in range(1, tiers + 1):
                    x = bay_x + pos * cell_width
                    y = top + (tiers - t) * cell_height

                    default_cell = self._create_cell_props(False, False, None, 0)
                    cell = cell_info.get((bay_num, r, t), default_cell)

                    self._draw_cell(x, y, cell_width, cell_height, cell, colors, fonts, is_vessel)

    def _draw_cell(self, x, y, width, height, cell, colors, fonts, is_vessel):
        """
        Draw an individual cell
        """
        import pygame

        group_colors = colors["group_colors"]
        group_idx = min(cell["group"], len(group_colors) - 1)

        if cell["filled"]:
            color = group_colors[group_idx][1]
        elif is_vessel:
            color = group_colors[group_idx][0]
        else:
            color = colors["empty"]

        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, rect)

        line_color, line_width = self._get_cell_border_style(cell)
        pygame.draw.rect(self.screen, line_color, rect, line_width)
        if cell["idx"] is not None:
            if is_vessel:
                text_color = (255, 255, 255) if cell["filled"] else (50, 50, 50)
                label = fonts["tiny"].render(f"{cell['idx']}", True, text_color)
                self.screen.blit(label, label.get_rect(center=(x + width / 2, y + height / 2)))
            else:
                text_color = (255, 255, 255) if cell["filled"] else (50, 50, 50)
                label = fonts["tiny"].render(f"{cell['idx']}", True, text_color)
                self.screen.blit(label, label.get_rect(center=(x + width / 2, y + height / 2)))

    def _draw_bay_dividers(self, left_margin, top, bays, rows, cell_width, tiers, cell_height, color):
        """
        Draw vertical divider lines between bays
        """
        import pygame

        for b in range(bays + 1):
            x = left_margin + b * rows * cell_width
            pygame.draw.line(
                self.screen,
                color,
                (x, top),
                (x, top + tiers * cell_height),
                2,
            )
