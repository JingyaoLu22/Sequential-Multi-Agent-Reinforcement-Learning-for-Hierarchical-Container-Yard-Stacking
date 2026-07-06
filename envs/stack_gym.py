import warnings
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, List, Any, Union
from enum import Enum
import colorsys


class StateIds(Enum):
    BAY = 0
    ROW = 1
    TIER = 2
    IS_OCCUPIED = 3
    GROUP = 4
    SIZE = 5  # 0 = 20ft, 1 = 40ft (only used when container_sizes is enabled)
    IS_IMO = 6  # 0 = normal, 1 = IMO/dangerous (only used when enable_imo is enabled)


class StackEnv(gym.Env):
    """
    Stacking Environment for moving containers from vessel to yard

    Containers start fully loaded on the vessel. Environment selects containers sequentially and randomly
    from the vessel (top containers in each stack) at each time step and the agent is tasked with selecting a slot for
    the container in the yard. The goal is to place similar group containers close together.
    """

    # metadata needed for using gym wrapper
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(
        self, config: Optional[Dict] = None, render_mode: Optional[str] = None
    ) -> None:
        """
        Initialize the stacking environment.

        Args:
            config: Configuration dictionary with the following keys:
                - vessel_shape: (bays, rows, tiers) dimensions of vessel storage
                - yard_shape: (bays, rows, tiers) dimensions of yard storage
                - num_containers: Total containers to move from vessel to yard
                - group_num: Number of distinct container groups
                - group_placement: Strategy for container group placement ("fixed" or "random")
                - seed: Random seed for reproducibility
                - action_mask: Action masking strategy
                - reward_scheme: Reward calculation scheme ("default" or "simple_single_stack")
                - reward_design: Reward function variant used by _calculate_reward ("default" or "majority_reward")
                - observation_type: Observation format ("flat", "stack_features", "stack_features_simplev2",
                  "stack_features_v3", "flat_parsed", "hierarchical_diff_obs")
                - reward_norm: Whether to normalize/scale reward
                - reward_clip: Whether to clip reward to a certain range
                - pos_embeddings: Whether to use sinusoidal positional encoding (instead of a scalar
                  stack index) in the "stack_features" observation type
                - stack_fill_penalty: If True, penalize opening a new empty stack when another
                  same-group stack with available space exists
                - container_sizes: Whether containers have 40ft sizes, enabling per-container size
                  tracking and stack size-compatibility constraints.
                - enable_imo: Whether to simulate IMO (dangerous goods) containers, enabling placement
                  compatibility constraints between IMO containers of different groups
                - random_group_sizes: Whether to randomize container counts per group around an equal
                  average (min 1 per group) instead of splitting containers evenly across groups
            render_mode: Rendering mode for visualization ("rgb_array")

        Main Attributes:
            vessel_shape: Vessel storage dimensions tuple
            yard_shape: Yard storage dimensions tuple
            num_containers: Number of containers to place
            group_num: Number of container groups
            group_placement: Container group placement strategy (fixed or random)
            seed: Random seed value
            action_mask_with_obs: If set this returns action_mask along with observation. (not used currently but maybe needed later)
            reward_scheme: Reward calculation scheme (default or simple_single_stack)
            observation_type: State observation format (details in _create_observation method)
            pos_embeddings: Whether stack_features uses sinusoidal positional encoding instead of a scalar index
            stack_fill_penalty: Whether opening a new stack is only penalized when a same-group stack has space
            container_sizes: Whether containers carry a 20ft/40ft size attribute
            enable_imo: Whether IMO (dangerous goods) containers and their placement constraints are simulated
            random_group_sizes: Whether per-group container counts are randomized instead of split evenly
            imo_groups: Groups designated as IMO (dangerous goods) groups, re-set each episode
            total_vessel_slots: Total container slots in vessel
            total_yard_slots: Total container slots in yard
            total_timesteps: Current episode timestep counter
            total_containers_remaining: Containers pending placement
            num_slot_attrs: Attributes per slot state.
            num_actions: Number of discrete actions. Each action corresponds to a single stack in the yard (bay x row combination).
            num_vessel_bay: Total bays in vessel (including even bays that are not used for placement)
            num_yard_bay: Total bays in yard (including even bays that are not used for placement)
            total_vessel_coords: Total coordinates in vessel state (including even bays that are not used for placement)
            total_yard_coords: Total coordinates in yard state (including even bays that are not used for placement)
            obs_coords: Size of observation space coordinates (including even bays that are not used for placement)
            current_vessel_container: Current container index being retrieved
            current_retrieval_group: Current group being removed
            containers_retrieved: Number of containers removed so far
            yard_bay_row_occupied: Track occupied stack (bay x row) positions
            observation_space: Gymnasium observation space definition
            reward_norm: Whether to normalize reward
            reward_clip: Whether to clip reward to a certain range
            action_space: Gymnasium action space (discrete actions)
            render_mode: Rendering visualization mode
            screen_width: Pixel width for rendering display
            screen_height: Pixel height for rendering display
        """
        if config is None:
            config = {}

        self.vessel_shape = config.get("vessel_shape", (1, 2, 2))
        self.yard_shape = config.get("yard_shape", (2, 2, 2))
        self.num_containers = config.get("num_containers", 4)
        self.group_num = config.get("group_num", 1)
        self.group_placement = config.get("group_placement", "fixed")
        self.seed = config.get("seed")
        self.action_mask_with_obs = config.get("action_mask", "default")
        self.reward_scheme = config.get("reward_scheme", "default")
        self.reward_design = config.get("reward_design", "default")
        self.observation_type = config.get("observation_type", "flat")
        self.reward_norm = config.get("reward_norm", False)
        self.reward_clip = config.get("reward_clip", False)
        self.pos_embeddings = config.get("pos_embeddings", False)
        self.stack_fill_penalty = config.get("stack_fill_penalty", False)
        self.container_sizes = config.get("container_sizes", False)
        self.enable_imo = config.get("enable_imo", False)
        self.random_group_sizes = config.get("random_group_sizes", False)
        self._same_group_stack_available: bool = False
        self.imo_groups: set = set()

        if self.seed is None:
            # Use instance-level RandomState for thread-safe parallel execution
            rng = np.random.RandomState()
            self.seed = rng.randint(0, 10000)

        # Calculate total physical slots
        self.total_vessel_slots = (
            self.vessel_shape[0] * self.vessel_shape[1] * self.vessel_shape[2]
        )
        self.total_yard_slots = (
            self.yard_shape[0] * self.yard_shape[1] * self.yard_shape[2]
        )
        self.total_timesteps = 0
        self.total_containers_remaining = 0

        self.num_containers = min(self.num_containers, self.total_vessel_slots)
        if self.num_containers > self.total_vessel_slots:
            warnings.warn(
                f"Number of containers is set to {self.num_containers} as it exceeds the total vessel slots {self.total_vessel_slots}"
            )
        self.num_slot_attrs = 5
        if self.container_sizes:
            self.num_slot_attrs += 1  # SIZE at index 5
        if self.enable_imo:
            # IS_IMO comes after SIZE (or after GROUP if no sizes)
            self.imo_attr_idx = self.num_slot_attrs
            self.num_slot_attrs += 1
        else:
            self.imo_attr_idx = None

        self.num_actions = (
            self.yard_shape[0] * self.yard_shape[1]
        )  # num_bays * num_rows

        # Calculate total coordinates (used later for numbering and removing even numbered bays)
        self.num_vessel_bay = self.vessel_shape[0] // 2 + self.vessel_shape[0]
        self.num_yard_bay = self.yard_shape[0] // 2 + self.yard_shape[0]

        self.total_vessel_coords = (
            self.num_vessel_bay * self.vessel_shape[1] * self.vessel_shape[2]
        )
        self.total_yard_coords = (
            self.num_yard_bay * self.yard_shape[1] * self.yard_shape[2]
        )

        self.obs_coords = (
            self.total_yard_coords + 1
        )  # yard_state + current container (no vessel_state)

        self.current_vessel_container = (
            None  # Index of container currently being retrieved
        )
        self.current_retrieval_group = None  # Group being retrieved
        self.containers_retrieved = 0

        # Track which bay/row cells have been occupied in yard
        self.yard_bay_row_occupied = set()

        # Pre-compute sinusoidal positional encoding for stack_features (fixed, computed once)
        if self.pos_embeddings and self.observation_type == "stack_features":
            num_stacks = self.yard_shape[0] * self.yard_shape[1]
            max_frequency = self.yard_shape[1]
            self.stack_pos_encoding = self._stack_positional_encoding(
                num_stacks, max_frequency
            )
        else:
            self.stack_pos_encoding = None

        # Observation and action spaces
        # Maximum for observation space based on yard and container attributes (vessel_state excluded)
        # shape=(self.obs_coords, 5). Each slot stores 5 values: bay, row, tier, occupied(0/1), group number of the container
        observation_space = self._get_observation_space()

        if self.action_mask_with_obs == "default":
            self.observation_space = observation_space
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": observation_space,
                    "mask": gym.spaces.Box(
                        low=0, high=1, shape=(self.num_actions,), dtype=np.bool_
                    ),
                }
            )

        # Action is choosing a stack (bay x row) to place the current vessel container by selecting an action index corresponding to each stack.
        # Action index x = ((b-1)/2)*n + (r-1) where b is odd bay number (1,3,5,...), r is row (1,2,3,...), n is num_rows

        self.action_space = gym.spaces.Discrete(self.num_actions)

        # Render part
        self.render_mode = render_mode
        self.screen_width = (
            35
            * max(
                self.vessel_shape[1] * self.vessel_shape[0],
                self.yard_shape[1] * self.yard_shape[0],
            )
            + 60
        )
        self.screen_height = 60 * max(self.vessel_shape[2], self.yard_shape[2]) + 150
        self.screen = None

    def _action_to_bay_row(self, action: int) -> Tuple[int, int]:
        """
        Convert action index (stack) to bay and row numbering
        Action index x = ((b-1)/2)*n + (r-1)
        where b is odd bay number (1,3,5,...), r is row number (1,2,...,n), n is num_rows

        Returns:
            tuple: (bay, row) where bay is odd bay number (1,3,5,...) and row is 1-indexed
        """
        n = self.yard_shape[1]  # num_rows
        bay = action // n  # which physical bay (0-indexed)
        row = (action % n) + 1  # which row (1-indexed)

        # Convert to actual bay number (1,3,5,7,...)
        bay = 2 * bay + 1

        return bay, row

    def _bay_row_to_action(self, bay: int, row: int) -> int:
        """
        Convert (bay, row) numbering pair to action index (stack)
        Action index x = ((b-1)/2)*n + (r-1)

        Args:
            bay: odd bay number (1,3,5,7,...)
            row: row number (1,2,3,...)

        Returns:
            int: action index
        """
        n = self.yard_shape[1]  # num_rows
        action = ((bay - 1) // 2) * n + (row - 1)
        return action

    def _action_to_yard_slot(self, action: int) -> Optional[int]:
        """
        Convert action index (stack) to yard_slot index (for accessing self.yard_state observation matrix)
        Finds the lowest unoccupied tier in the specified (bay, row) stack

        Returns:
            int: yard_slot row index in self.yard_state, or None if stack is full
        """
        bay, row = self._action_to_bay_row(action)

        # Find all slots matching this bay and row
        bay_row_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (
            self.yard_state[:, StateIds.ROW.value] == row
        )
        bay_row_indices = np.where(bay_row_mask)[0]

        if len(bay_row_indices) == 0:
            return None

        # Find lowest unoccupied tier
        tiers_in_stack = self.yard_state[bay_row_indices, StateIds.TIER.value]
        sorted_indices = np.argsort(tiers_in_stack)

        for idx in sorted_indices:
            slot_idx = bay_row_indices[idx]
            if self.yard_state[slot_idx, StateIds.IS_OCCUPIED.value] == 0:
                return slot_idx

        return None  # Stack is full

    def step(
        self, action: int
    ) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        """
        Execute one step: place current container from vessel to yard at a specified action (bay, row combination)
        Action represents a stack (bay x row) where the container will be placed at the lowest available tier.
        """
        self.total_timesteps += 1
        truncated = False
        terminated = False
        info = {}

        # Get valid actions for current state
        valid_actions = self._get_valid_yard_actions()
        valid_actions_list = (
            valid_actions.tolist()
            if isinstance(valid_actions, np.ndarray)
            else list(valid_actions)
        )

        # Check if action is valid (in valid_actions list)
        # Invalid action (either full stack, even bay, or out of range action number)
        # Invalid actions lead to huge negative reward and termination of episode
        if action is None or action not in valid_actions_list:
            reward = -200.0
            observation = self._create_observation()
            info["yard_mask"] = valid_actions_list
            terminated = True
            return observation, reward, terminated, truncated, info

        # Convert action to yard_slot (actual row slot index in self.yard_state)
        yard_slot = self._action_to_yard_slot(action)
        if yard_slot is None:
            # This shouldn't happen if action masking works correctly
            reward = -200.0
            observation = self._create_observation()
            info["yard_mask"] = valid_actions_list
            terminated = True
            return observation, reward, terminated, truncated, info

        # Cache whether a same-group stack with space exists (used in reward rule 0)
        self._cache_pre_placement_stack_fill(yard_slot)

        # IMO violation check: if current container is IMO and target/adjacent stacks
        # have IMO containers of a different group, terminate with large negative reward
        if self.enable_imo and int(self.vessel_state[self.current_vessel_container, self.imo_attr_idx]) == 1:
            placement_bay = self.yard_state[yard_slot, StateIds.BAY.value]
            placement_row = self.yard_state[yard_slot, StateIds.ROW.value]
            current_group = int(self.vessel_state[self.current_vessel_container, StateIds.GROUP.value])
            if not self._check_imo_compatible(placement_bay, placement_row, current_group):
                reward = -1000.0
                observation = self._create_observation()
                info["yard_mask"] = valid_actions_list
                info["imo_violation"] = True
                terminated = True
                return observation, reward, terminated, truncated, info

        # Calculate reward
        reward = self._calculate_reward(yard_slot)

        # Place container in yard
        self._place_container_in_yard(yard_slot, self.current_vessel_container)

        # Remove container from vessel
        self.vessel_state[self.current_vessel_container, StateIds.IS_OCCUPIED.value] = 0
        self.total_containers_remaining -= 1

        # Get next vessel container to retrieve
        self.current_vessel_container = self._get_next_vessel_container()

        # Episode terminates when all containers retrieved or no more valid groups
        terminated = (self.current_vessel_container is None) or (
            self.total_containers_remaining == 0
        )
        valid_actions = (
            self._get_valid_yard_actions() if not truncated else np.array([], dtype=int)
        )
        valid_actions_list = (
            valid_actions.tolist()
            if isinstance(valid_actions, np.ndarray)
            else list(valid_actions)
        )

        # If no valid actions remain (e.g. all stacks are size-incompatible), truncate
        if not terminated and len(valid_actions_list) == 0:
            truncated = True
            info["all_actions_masked"] = True

        observation = self._create_observation()
        info.update(
            {
                "yard_mask": valid_actions_list,
                "containers_retrieved": self.containers_retrieved,
                "containers_remaining": self.total_containers_remaining,
            }
        )

        return observation, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """Reset environment wrapper needed for gymnasium compatibility. 
        Calls internal _reset method to reset the environment state."""
        if seed is not None:
            self.seed = seed
        else:
            # Increment seed deterministically so each parallel env
            # maintains its own diverging seed for parallel envs
            self.seed += 1
        self._reset()
        observation = self._create_observation()
        info = {}
        return observation, info

    def _reset(self) -> None:
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

        Example of self,.yard_state and self.vessel_state after initialization for shape (3,2,2) for (bays,rows,tiers).
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

        The above example is shown for self.observation_type = "flat".
        Other types of observations are described in the _create_observation method but all other types of observations
        are derived from the yard_state as described above.

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
        self.yard_state = np.zeros(
            (self.total_yard_coords, self.num_slot_attrs), dtype=int
        )
        self.yard_state[:, StateIds.BAY.value] = np.repeat(yard_bays, R * T)
        self.yard_state[:, StateIds.ROW.value] = np.tile(
            np.arange(1, R + 1).repeat(T), len(yard_bays)
        )
        self.yard_state[:, StateIds.TIER.value] = np.tile(
            np.arange(1, T + 1), len(yard_bays) * R
        )

        # Initialize vessel (similar logic as yard)
        vessel_bays = self._generate_bay_coords(self.vessel_shape[0])
        _, Rv, Tv = self.vessel_shape
        self.vessel_state = np.zeros(
            (self.total_vessel_coords, self.num_slot_attrs), dtype=int
        )
        self.vessel_state[:, StateIds.BAY.value] = np.repeat(vessel_bays, Rv * Tv)
        self.vessel_state[:, StateIds.ROW.value] = np.tile(
            np.arange(1, Rv + 1).repeat(Tv), len(vessel_bays)
        )
        self.vessel_state[:, StateIds.TIER.value] = np.tile(
            np.arange(1, Tv + 1), len(vessel_bays) * Rv
        )

        # Assign groups to vessel containers
        if self.group_num > 1:
            odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 == 1
            odd_bay_indices = np.where(odd_bay_mask)[0]  # even bay indices are ignored
            valid_slots = odd_bay_indices

            slots_per_group = len(valid_slots) // self.group_num
            for group in range(self.group_num):
                start_pos = group * slots_per_group
                end_pos = (
                    (group + 1) * slots_per_group
                    if group < self.group_num - 1
                    else len(valid_slots)
                )
                group_indices = valid_slots[start_pos:end_pos]
                if len(group_indices) > 0:
                    self.vessel_state[group_indices, StateIds.GROUP.value] = group

        # Load vessel with containers
        self._initialize_vessel_loaded()

        # Start retrieval with first available group
        self.current_retrieval_group = 0
        self.current_vessel_container = self._get_next_vessel_container()
        self.total_containers_remaining = self.num_containers

    def _initialize_vessel_loaded(self) -> None:
        """
        Load vessel with containers - mirrors stowage but loads vessel instead of yard
        """
        odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 != 0
        available_slots = np.where(odd_bay_mask)[0]  # even bay indices are ignored
        num_to_set = min(self.num_containers, len(available_slots))

        if num_to_set > 0:
            selected_slots = available_slots[:num_to_set]
            self.vessel_state[selected_slots, StateIds.IS_OCCUPIED.value] = 1

            if self.group_num > 1:
                # Compute per-group container counts
                if self.random_group_sizes and self.group_num > 1:
                    # Randomized containers per group logic
                    # Each group gets at least 1; distribute remaining via multinomial
                    rng_gs = np.random.RandomState(self.seed + 3571)
                    remainder = num_to_set - self.group_num
                    if remainder > 0:
                        sampled = rng_gs.multinomial(remainder, [1.0 / self.group_num] * self.group_num)
                        group_counts = 1 + sampled
                    else:
                        # Edge case: fewer containers than groups — one each, some groups get 0
                        group_counts = np.zeros(self.group_num, dtype=int)
                        group_counts[:num_to_set] = 1
                else:
                    # Evenly distribute number of containers per group
                    containers_per_group = num_to_set // self.group_num
                    group_counts = np.array(
                        [containers_per_group] * (self.group_num - 1)
                        + [num_to_set - containers_per_group * (self.group_num - 1)]
                    )

                cumsum = np.cumsum(np.concatenate([[0], group_counts]))

                if self.group_placement == "fixed":
                    # Fixed placement of groups
                    for group in range(self.group_num):
                        start_idx = cumsum[group]
                        end_idx = cumsum[group + 1]

                        if start_idx < end_idx:
                            self.vessel_state[
                                selected_slots[start_idx:end_idx], StateIds.GROUP.value
                            ] = group

                else:
                    # Random placement of groups
                    rng = np.random.RandomState(self.seed)
                    shuffled_indices = rng.permutation(num_to_set)
                    shuffled_slots = selected_slots[shuffled_indices]
                    for group in range(self.group_num):
                        start_idx = cumsum[group]
                        end_idx = cumsum[group + 1]

                        if start_idx < end_idx:
                            self.vessel_state[
                                shuffled_slots[start_idx:end_idx], StateIds.GROUP.value
                            ] = group

            # Assign container sizes (20ft/40ft) when enabled
            # 30% of containers within each group are 40ft
            if self.container_sizes:
                size_rng = np.random.RandomState(self.seed + 7919)
                for group in range(self.group_num):
                    group_mask = (
                        (self.vessel_state[selected_slots, StateIds.GROUP.value] == group)
                        & (self.vessel_state[selected_slots, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    group_slots = selected_slots[group_mask]
                    if len(group_slots) > 0:
                        num_40ft = max(1, int(len(group_slots) * 0.3))
                        perm = size_rng.permutation(len(group_slots))
                        self.vessel_state[group_slots[perm[:num_40ft]], StateIds.SIZE.value] = 1

            # Assign IMO (dangerous) status when enabled
            # 50% of groups are randomly designated as IMO groups
            # 25% of containers within those groups are marked as IMO
            if self.enable_imo:
                imo_rng = np.random.RandomState(self.seed + 13331)
                imo_group_count = max(1, self.group_num // 2)
                self.imo_groups = set(
                    imo_rng.choice(self.group_num, size=imo_group_count, replace=False).tolist()
                )
                for group in self.imo_groups:
                    group_mask = (
                        (self.vessel_state[selected_slots, StateIds.GROUP.value] == group)
                        & (self.vessel_state[selected_slots, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    group_slots = selected_slots[group_mask]
                    if len(group_slots) > 0:
                        num_imo = max(1, int(len(group_slots) * 0.25))
                        perm = imo_rng.permutation(len(group_slots))
                        self.vessel_state[group_slots[perm[:num_imo]], self.imo_attr_idx] = 1
            else:
                self.imo_groups = set()

    def _generate_bay_coords(self, physical_bays: int) -> List[int]:
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

    def _get_next_vessel_container(self) -> Optional[int]:
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
        # NOTE : self.total_timesteps is used within rng to ensure different stacks are selected while selecting container from vessel
        # Without this, the containers from same stack is selcted repeatedly for unloading
        # until that stack is empty and then it moves to next stack.
        rng = np.random.RandomState(self.seed + self.total_timesteps)
        selected_idx = rng.choice(topmost_containers)

        # Update current retrieval group based on selected container
        self.current_retrieval_group = int(
            self.vessel_state[selected_idx, StateIds.GROUP.value]
        )

        return selected_idx

    def _place_container_in_yard(
        self, yard_slot: int, vessel_container_idx: int
    ) -> None:
        """
        Place the vessel container (at vessel_container_idx) in yard at specified slot (yard_slot).
        Note: yard_slot is already the correct index (lowest unoccupied tier) as determined by _action_to_yard_slot().
        """
        bay = self.yard_state[yard_slot, StateIds.BAY.value]
        row = self.yard_state[yard_slot, StateIds.ROW.value]
        group = self.vessel_state[vessel_container_idx, StateIds.GROUP.value]

        # Place container at the specified slot
        self.yard_state[yard_slot, StateIds.IS_OCCUPIED.value] = 1
        self.yard_state[yard_slot, StateIds.GROUP.value] = group

        # Copy container size when container_sizes is enabled
        if self.container_sizes:
            size = self.vessel_state[vessel_container_idx, StateIds.SIZE.value]
            self.yard_state[yard_slot, StateIds.SIZE.value] = size

        # Copy IMO status when enable_imo is enabled
        if self.enable_imo:
            imo_status = self.vessel_state[vessel_container_idx, self.imo_attr_idx]
            self.yard_state[yard_slot, self.imo_attr_idx] = imo_status

        # Add stack to list of occupied stacks (used in reward function)
        self.yard_bay_row_occupied.add((bay, row))
        self.containers_retrieved += 1

    def _calculate_reward(self, yard_action: int) -> float:
        """
        Routes to the appropriate reward function based on self.reward_design.
            - default        -> _calculate_default_reward
            - majority_reward -> _calculate_majority_reward (not used in Thesis)
        """
        if self.reward_design == "majority_reward":
            reward = self._calculate_majority_reward(yard_action)
        elif self.reward_design == "default":
            reward = self._calculate_default_reward(yard_action)
        return reward

    def _calculate_default_reward(self, yard_action: int) -> float:
        """
        Function takes in yard_action (index of slot chosen to place container) and calculates reward for this placement
        Calculate reward based on exact rules:

        0. Penalty for occupying new ground slot in unoccupied stack (to encourage stacking and not spreading out)
        1. Reward (+1/-1) for placing container in a stack having same/dissimilar containers in the same stack.
        2. Reward (+0.5/-0.5) for placing container having same/dissimilar containers in +/-1 adjacent rows (same bay only)
        3. Reward (+0.25/-0.25) for placing container having same/dissimilar containers in other stacks in same bay (not same row or adjacent rows)

        1,2 and 3 rewards are added for each similar/dissimilar container found in the same stack or adjacent rows.
        Multipliers are applied to each rule to scale the rewards for each rule.
        Note : Simple reward scheme only applies rule 1.

        """
        placement_bay = self.yard_state[yard_action, StateIds.BAY.value]
        placement_row = self.yard_state[yard_action, StateIds.ROW.value]
        container_group = self.vessel_state[
            self.current_vessel_container, StateIds.GROUP.value
        ]

        reward = 0.0
        
        rule_0_multiplier = 2.0
        rule_1_multiplier = 2.0

        rule_2_multiplier = 1.0
        rule_3_multiplier = 0.2

        # Rule 0: Penalty for occupying new ground slot in unoccupied stack.
        # If stack_fill_penalty is True, only penalise when another same-group stack
        # with available space exists (agent skipped a better stacking opportunity).
        # If stack_fill_penalty is False, always penalise opening a new stack.
        if (placement_bay, placement_row) not in self.yard_bay_row_occupied:
            if not self.stack_fill_penalty or self._same_group_stack_available:
                reward -= rule_0_multiplier

        # Rule 1: Reward/penalty for placing container in a stack having same/dissimilar containers in the same stack.
        bay_row_mask = (self.yard_state[:, StateIds.BAY.value] == placement_bay) & (
            self.yard_state[:, StateIds.ROW.value] == placement_row
        )
        bay_row_indices = np.where(bay_row_mask)[0]

        same_stack_occupied_mask = (
            self.yard_state[bay_row_indices, StateIds.IS_OCCUPIED.value] == 1
        )
        same_stack_occupied_slots = bay_row_indices[same_stack_occupied_mask]

        if len(same_stack_occupied_slots) > 0:
            same_stack_container_groups = self.yard_state[
                same_stack_occupied_slots, StateIds.GROUP.value
            ]
            same_group_count = np.sum(same_stack_container_groups == container_group)
            diff_group_count = np.sum(same_stack_container_groups != container_group)
            total_occupied_in_stack = same_group_count + diff_group_count
            if self.reward_norm and total_occupied_in_stack > 0:
                reward += (
                    (same_group_count - diff_group_count) / total_occupied_in_stack
                ) * rule_1_multiplier
            else:
                reward += (same_group_count - diff_group_count) * rule_1_multiplier

        if self.reward_scheme == "simple_single_stack":
            if self.reward_clip:
                reward = np.clip(reward, -5.0, 5.0)
            return reward

        # Find  +/- 1 adjacent rows in the same bay
        adjacent_rows = []
        if placement_row > 1:
            adjacent_rows.append(placement_row - 1)
        if placement_row < self.yard_shape[1]:
            adjacent_rows.append(placement_row + 1)

        # Rule 3 reward/penalty for other stacks in same bay (not same row or adjacent rows)
        same_bay_mask = (
            (self.yard_state[:, StateIds.BAY.value] == placement_bay)
            & (self.yard_state[:, StateIds.ROW.value] != placement_row)
            & ~np.isin(self.yard_state[:, StateIds.ROW.value], adjacent_rows)
        )

        same_bay_indices = np.where(same_bay_mask)[0]

        same_bay_occupied_slots_mask = (
            self.yard_state[same_bay_indices, StateIds.IS_OCCUPIED.value] == 1
        )
        same_bay_occupied_slots = same_bay_indices[same_bay_occupied_slots_mask]

        if len(same_bay_occupied_slots) > 0:
            same_bay_container_groups = self.yard_state[
                same_bay_occupied_slots, StateIds.GROUP.value
            ]
            same_group_count = np.sum(same_bay_container_groups == container_group)
            diff_group_count = np.sum(same_bay_container_groups != container_group)
            total_occupied_in_bay = same_group_count + diff_group_count
            if self.reward_norm and total_occupied_in_bay > 0:
                reward += (
                    (same_group_count - diff_group_count) / total_occupied_in_bay
                ) * rule_3_multiplier
            else:
                reward += (same_group_count - diff_group_count) * rule_3_multiplier

        # Rule 2 reward/penalty for adjacent rows in same bay
        adj_stack_mask = (
            self.yard_state[:, StateIds.BAY.value] == placement_bay
        ) & np.isin(self.yard_state[:, StateIds.ROW.value], adjacent_rows)
        adj_stack_indices = np.where(adj_stack_mask)[0]

        adj_occupied_slots_mask = (
            self.yard_state[adj_stack_indices, StateIds.IS_OCCUPIED.value] == 1
        )
        adj_occupied_slots = adj_stack_indices[adj_occupied_slots_mask]

        if len(adj_occupied_slots) > 0:
            adj_stack_container_groups = self.yard_state[
                adj_occupied_slots, StateIds.GROUP.value
            ]
            same_group_count = np.sum(adj_stack_container_groups == container_group)
            diff_group_count = np.sum(adj_stack_container_groups != container_group)
            total_occupied_adj = same_group_count + diff_group_count
            if self.reward_norm and total_occupied_adj > 0:
                reward += (
                    (same_group_count - diff_group_count) / total_occupied_adj
                ) * rule_2_multiplier
            else:
                reward += (same_group_count - diff_group_count) * rule_2_multiplier

        if self.reward_clip:
            reward = np.clip(reward, -5.0, 5.0)
        return reward

    def _cache_pre_placement_stack_fill(self, yard_slot: int) -> None:
        """Pre-compute whether any other stack in the target bay has the same-group
        container at its top AND has space remaining.

        Only computed when stack_fill_penalty is True.
        The cached boolean is used by rule 0 in _calculate_default_reward.
        """
        if not self.stack_fill_penalty:
            self._same_group_stack_available = False
            return

        placement_bay = self.yard_state[yard_slot, StateIds.BAY.value]
        placement_row = self.yard_state[yard_slot, StateIds.ROW.value]
        container_group = self.vessel_state[
            self.current_vessel_container, StateIds.GROUP.value
        ]
        max_tiers = self.yard_shape[2]

        # Iterate over all rows in this bay except the target row
        all_rows_in_bay = np.unique(
            self.yard_state[
                self.yard_state[:, StateIds.BAY.value] == placement_bay,
                StateIds.ROW.value,
            ]
        )
        for row in all_rows_in_bay:
            if row == placement_row:
                continue
            stack_mask = (
                (self.yard_state[:, StateIds.BAY.value] == placement_bay)
                & (self.yard_state[:, StateIds.ROW.value] == row)
                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
            )
            occupied_indices = np.where(stack_mask)[0]
            if len(occupied_indices) == 0 or len(occupied_indices) >= max_tiers:
                continue
            # Find the top container (highest tier value)
            tiers = self.yard_state[occupied_indices, StateIds.TIER.value]
            top_idx = occupied_indices[np.argmax(tiers)]
            if self.yard_state[top_idx, StateIds.GROUP.value] == container_group:
                # When container_sizes is enabled, only count stacks with compatible sizes
                if self.container_sizes:
                    current_size = int(
                        self.vessel_state[
                            self.current_vessel_container, StateIds.SIZE.value
                        ]
                    )
                    stack_size = int(
                        self.yard_state[occupied_indices[0], StateIds.SIZE.value]
                    )
                    if stack_size != current_size:
                        continue
                self._same_group_stack_available = True
                return

        self._same_group_stack_available = False

    def _get_majority_group(self, bay: int, row: int) -> Optional[int]:
        """
        Return the most common group in the given (bay, row) stack among occupied slots.
        Returns None if the stack has no occupied slots (i.e. it is empty).
        Used by majority reward scheme (not used in Thesis).
        """
        bay_row_mask = (
            (self.yard_state[:, StateIds.BAY.value] == bay)
            & (self.yard_state[:, StateIds.ROW.value] == row)
            & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
        )
        occupied_indices = np.where(bay_row_mask)[0]
        if len(occupied_indices) == 0:
            return None
        groups = self.yard_state[occupied_indices, StateIds.GROUP.value]
        counts = np.bincount(groups)
        return int(np.argmax(counts))

    def _calculate_majority_reward(self, yard_action: int) -> float:
        """
        Alternate reward design based on majority group matching (not used in Thesis).

        Rules (applied before the container is placed):

        Stack rules (evaluated on the currently placed stack):
          Rule 3: Penalty for occupying a previously empty stack       -> -0.1
          Rule 1: Majority group in placed stack == container group    ->  0.0
          Rule 2: Majority group in placed stack != container group    -> -1.0
          (Rules 1/2 only apply when the stack is non-empty.)

        Neighbourhood rules (l = majority of left adjacent stack,
                             r = majority of right adjacent stack,
                             c = container_group;
                             empty adjacent stacks are excluded):
          Rule 4: All non-empty adjacents match c                      ->  0.0
          Rule 5: Exactly one of two non-empty adjacents matches c     -> -0.3
          Rule 6: All non-empty adjacents (1 or 2) do NOT match c     -> -0.6
          (Rules 4-6 are skipped if there are no non-empty adjacent stacks.)
        """
        placement_bay = self.yard_state[yard_action, StateIds.BAY.value]
        placement_row = self.yard_state[yard_action, StateIds.ROW.value]
        container_group = int(
            self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
        )

        reward = 0.0

        stack_is_empty = (
            placement_bay,
            placement_row,
        ) not in self.yard_bay_row_occupied

        # Rule 3: penalty for occupying a new (empty) stack
        if stack_is_empty:
            reward -= 0.1
        else:
            # Rules 1 & 2: majority group comparison
            majority = self._get_majority_group(placement_bay, placement_row)
            if majority is not None and majority != container_group:
                reward -= 1.0
            # majority == container_group -> +0 (no change)

        # Neighbourhood rules
        # Collect majority groups of non-empty adjacent stacks (left then right)
        non_empty_adj_majorities = []
        if placement_row > 1:
            left_maj = self._get_majority_group(placement_bay, placement_row - 1)
            if left_maj is not None:
                non_empty_adj_majorities.append(left_maj)
        if placement_row < self.yard_shape[1]:
            right_maj = self._get_majority_group(placement_bay, placement_row + 1)
            if right_maj is not None:
                non_empty_adj_majorities.append(right_maj)

        if len(non_empty_adj_majorities) > 0:
            match_count = sum(
                1 for m in non_empty_adj_majorities if m == container_group
            )
            if match_count == len(non_empty_adj_majorities):
                # Rule 4: all non-empty adjacents match c
                reward += 0.0
            elif match_count > 0:
                # Rule 5: exactly one of two matches c
                reward -= 0.3
            else:
                # Rule 6: none match c
                reward -= 0.6

        return reward

    def _get_valid_yard_actions(self) -> np.ndarray:
        """
        Get valid yard placement actions - returns action indices or stacks for each valid (bay x row) combination
        that have at least one unoccupied tier in odd bays within the stack.
        Action index x = ((b-1)/2)*n + (r-1) where b is odd bay, r is row, n is num_rows
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

        unique_bay_rows = set(zip(bays, rows))

        # When container_sizes is enabled, exclude stacks with incompatible sizes
        if self.container_sizes:
            current_size = int(
                self.vessel_state[self.current_vessel_container, StateIds.SIZE.value]
            )
            compatible_bay_rows = set()
            for bay, row in unique_bay_rows:
                stack_mask = (
                    (self.yard_state[:, StateIds.BAY.value] == bay)
                    & (self.yard_state[:, StateIds.ROW.value] == row)
                    & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                )
                occupied_indices = np.where(stack_mask)[0]
                if len(occupied_indices) == 0:
                    # Empty stack — any size can go here
                    compatible_bay_rows.add((bay, row))
                elif int(self.yard_state[occupied_indices[0], StateIds.SIZE.value]) == current_size:
                    # Stack has same size containers
                    compatible_bay_rows.add((bay, row))
                # else: stack has different size — skip
            unique_bay_rows = compatible_bay_rows

        # When enable_imo is enabled and current container is IMO,
        # exclude stacks where same stack or adjacent stacks have IMO containers of a different group
        if self.enable_imo and int(self.vessel_state[self.current_vessel_container, self.imo_attr_idx]) == 1:
            current_group = int(self.vessel_state[self.current_vessel_container, StateIds.GROUP.value])
            imo_compatible_bay_rows = set()
            for bay, row in unique_bay_rows:
                if self._check_imo_compatible(bay, row, current_group):
                    imo_compatible_bay_rows.add((bay, row))
            unique_bay_rows = imo_compatible_bay_rows

        # Convert each (bay, row) to action index
        for bay, row in unique_bay_rows:
            action = self._bay_row_to_action(bay, row)
            valid_actions.append(action)

        return np.array(valid_actions, dtype=int)

    def _check_imo_compatible(self, bay: int, row: int, current_group: int) -> bool:
        """
        Check if placing an IMO container of current_group at (bay, row) is compatible.
        Returns False if the target stack or any adjacent stack (left/right) contains
        IMO containers of a DIFFERENT group.
        IMO containers of the same group are always compatible.
        """
        stacks_to_check = [(bay, row)]
        # Left adjacent
        if row - 1 >= 1:
            stacks_to_check.append((bay, row - 1))
        # Right adjacent
        if row + 1 <= self.yard_shape[1]:
            stacks_to_check.append((bay, row + 1))

        for check_bay, check_row in stacks_to_check:
            stack_mask = (
                (self.yard_state[:, StateIds.BAY.value] == check_bay)
                & (self.yard_state[:, StateIds.ROW.value] == check_row)
                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                & (self.yard_state[:, self.imo_attr_idx] == 1)
            )
            imo_indices = np.where(stack_mask)[0]
            if len(imo_indices) > 0:
                # Check if any IMO container belongs to a different group
                groups_in_stack = self.yard_state[imo_indices, StateIds.GROUP.value]
                if np.any(groups_in_stack != current_group):
                    return False
        return True

    def action_masks(self) -> List[bool]:
        """
        For compatibility with SB3 action masking.
        """
        if self.current_vessel_container is None:
            return [False] * self.action_space.n

        valid_actions = self._get_valid_yard_actions()
        return [action in valid_actions for action in range(self.action_space.n)]

    def _stack_positional_encoding(
        self, num_stacks: int, max_frequency: int
    ) -> np.ndarray:
        """
        Create sinusoidal positional encoding for stacks.
        Returns ndarray of shape (num_stacks, 2 * max_frequency).
        Stack indices are normalized to [0, 1) before applying sin/cos at each frequency.
        (Did not use in thesis as results were not promising, but kept for future reference.)
        """
        p = np.arange(num_stacks) / num_stacks
        features = []
        for k in range(1, max_frequency + 1):
            features.append(np.sin(2 * np.pi * k * p))
            features.append(np.cos(2 * np.pi * k * p))
        return np.stack(features, axis=1).astype(np.float32)

    def _get_observation_space(self) -> Union[gym.spaces.Box, gym.spaces.Dict]:
        """
        Get observation space based on observation_type
        Returns appropriate gym.spaces.Box for the specified observation type
        Observation types are described in the _create_observation() method in detail but in summary:
            - flat: Original flat observation space with yard_state and current container attributes in a single array
            - stack_features: Stack-based feature space with one-hot group encodings per stack
            - flat_parsed: Dictionary observation for rule-based agents with pre-parsed yard_state and current_container
            - stack_features_simplev2: Simplified stack-based feature space using one-hot group encodings
            - stack_features_v3 (best and used in Thesis): Compact stack-based feature space using scalar group indices instead of one-hot encodings
            - hierarchical_diff_obs ( not used in Thesis): Bay-level features concatenated with per-bay stack-level features, for the
              differentiated-observation joint hierarchical policy
        """
        if self.observation_type == "flat":
            # Original flat observation space
            return gym.spaces.Box(
                low=0,
                high=max(
                    self.num_yard_bay,  # bay upper limit
                    self.yard_shape[1],  # row upper limit
                    self.yard_shape[2],  # tier upper limit
                    1,  # is_occupied upper limit
                    self.group_num,  # group number upper limit
                ),
                shape=(self.obs_coords * self.num_slot_attrs,),
                dtype=np.int64,
            )
        elif self.observation_type == "stack_features":
            num_stacks = self.yard_shape[0] * self.yard_shape[1]  # bays * rows
            if self.pos_embeddings:
                # Replace scalar positional_index with 2 * max_frequency sin/cos features
                features_per_stack = 5 * self.group_num + 4 + 2 * self.yard_shape[1]
                if self.container_sizes:
                    features_per_stack += 3
                if self.enable_imo:
                    features_per_stack += 2  # is_current_container_imo, adj_has_different_group_imo
                return gym.spaces.Box(
                    low=-1,
                    high=max(
                        self.yard_shape[2],  # max count per group in a stack
                        self.num_containers,  # vessel_remaining_per_group upper limit
                        1,  # binary features upper limit
                    ),
                    shape=(num_stacks * features_per_stack,),
                    dtype=np.float32,
                )
            else:
                features_per_stack = 5 * self.group_num + 5
                if self.container_sizes:
                    features_per_stack += 3  # current_size, stack_size, size_match
                if self.enable_imo:
                    features_per_stack += 2  # is_current_container_imo, adj_has_different_group_imo
                return gym.spaces.Box(
                    low=0,
                    high=max(
                        self.yard_shape[2],  # max count per group in a stack
                        self.num_containers,  # vessel_remaining_per_group upper limit
                        num_stacks,  # positional index upper limit
                        1,  # binary features upper limit
                    ),
                    shape=(num_stacks * features_per_stack,),
                    dtype=np.float32,
                )
        elif self.observation_type == "flat_parsed":
            return gym.spaces.Dict(
                {
                    "yard_state": gym.spaces.Box(
                        low=0,
                        high=max(
                            self.num_yard_bay,  # bay upper limit
                            self.yard_shape[1],  # row upper limit
                            self.yard_shape[2],  # tier upper limit
                            1,  # is_occupied upper limit
                            self.group_num,  # group number upper limit
                        ),
                        shape=(self.total_yard_coords, self.num_slot_attrs),
                        dtype=np.int64,
                    ),
                    "current_container": gym.spaces.Box(
                        low=0,
                        high=max(
                            self.num_vessel_bay,  # bay upper limit
                            self.vessel_shape[1],  # row upper limit
                            self.vessel_shape[2],  # tier upper limit
                            1,  # is_occupied upper limit
                            self.group_num,  # group number upper limit
                        ),
                        shape=(self.num_slot_attrs,),
                        dtype=np.int64,
                    ),
                }
            )
        elif self.observation_type == "stack_features_simplev2":
            # Simplified stack features with one-hot encoding
            features_per_stack = (
                3 * self.group_num + 2
            )  # 3 one-hot groups + num_occupied + stack_index
            if self.container_sizes:
                features_per_stack += 3  # current_size, stack_size, size_match
            if self.enable_imo:
                features_per_stack += 2  # is_current_container_imo, adj_has_different_group_imo
            num_stacks = self.yard_shape[0] * self.yard_shape[1]  # bays * rows

            return gym.spaces.Box(
                low=0,
                high=max(
                    1,  # one-hot encoded features and binary features
                    self.yard_shape[2],  # num_occupied upper limit
                    num_stacks,  # stack_index upper limit
                ),
                shape=(num_stacks * features_per_stack,),
                dtype=np.float32,
            )
        elif self.observation_type == "stack_features_v3":
            # Scalar stack features: 5 features per stack (no one-hot encoding)
            # [majority_group, num_occupied, current_container_group,
            #  adj_majority_group, stack_index]
            features_per_stack = 5
            if self.container_sizes:
                features_per_stack += 3  # current_size, stack_size, size_match
            if self.enable_imo:
                features_per_stack += 2  # is_current_container_imo, adj_has_different_group_imo
            num_stacks = self.yard_shape[0] * self.yard_shape[1]  # bays * rows

            return gym.spaces.Box(
                low=-1,  # sentinel for empty/no-group
                high=max(
                    self.group_num,  # group index upper limit
                    self.yard_shape[2],  # num_occupied upper limit
                    num_stacks,  # stack_index upper limit
                ),
                shape=(num_stacks * features_per_stack,),
                dtype=np.float32,
            )
        elif self.observation_type == "hierarchical_diff_obs":
            # Hierarchical differentiated observation: bay features + stack features
            # Bay features per bay: bay_index, group_counts(group_num), num_empty_stacks,
            #                       num_not_full_matching, num_matching_stacks
            # => bay_f_dim = group_num + 4
            # Stack features per stack: same as stack_features but with relative positional index
            # => stack_f_dim = 5 * group_num + 5
            n_bays = self.yard_shape[0]
            n_rows = self.yard_shape[1]
            num_stacks = n_bays * n_rows
            bay_f_dim = self.group_num + 4
            stack_f_dim = 5 * self.group_num + 5
            total_dim = n_bays * bay_f_dim + num_stacks * stack_f_dim

            return gym.spaces.Box(
                low=0,
                high=max(
                    n_bays,  # bay_index upper limit
                    self.yard_shape[2] * n_rows,  # max containers per group in a bay
                    n_rows,  # num_empty_stacks upper limit
                    self.yard_shape[2],  # max count per group in a stack
                    self.num_containers,  # vessel_remaining_per_group upper limit
                    num_stacks,  # positional index upper limit
                    1,  # binary features upper limit
                ),
                shape=(total_dim,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def _create_stack_features(self) -> np.ndarray:
        """
        Transform yard_state into stack-based features

        For each stack (bay, row combination), computes:
        - count_group1, count_group2, ..., count_groupn: counts of each group in the stack
        - num_occupied: number of occupied slots in the stack
        - num_empty: number of empty slots in the stack
        - current_group (1-hot): one-hot encoding of current container's group
        - left_row_max_group (1-hot): one-hot encoding of dominant group in left adjacent row (same bay)
        - right_row_max_group (1-hot): one-hot encoding of dominant group in right adjacent row (same bay)
        - vessel_remaining_per_group (for each group) : count of remaining containers per group in vessel
        - is_empty: 1 if stack is completely empty, 0 otherwise
        - has_remaining_slots: 1 if stack has at least one empty slot, 0 otherwise
        - if pos_embeddings=False: positional_index: sequential index of the stack (0, 1, 2, ...)
        - if pos_embeddings=True:  sin/cos positional encoding (2 * yard_rows features per stack)

        Returns:
            np.ndarray: Feature vector for all stacks
        """
        # Get unique stacks (bay, row combinations) - only odd bays
        yard_bays = np.unique(self.yard_state[:, StateIds.BAY.value])
        yard_bays = yard_bays[yard_bays % 2 == 1]  # only odd bays
        yard_rows = np.arange(1, self.yard_shape[1] + 1)

        num_stacks = len(yard_bays) * len(yard_rows)
        if self.pos_embeddings:
            features_per_stack = 5 * self.group_num + 4 + 2 * self.yard_shape[1]
        else:
            features_per_stack = 5 * self.group_num + 5
        if self.container_sizes:
            features_per_stack += 3
        if self.enable_imo:
            features_per_stack += 2
        stack_features = np.zeros((num_stacks, features_per_stack), dtype=np.float32)

        # Get current container group
        current_group = -1
        if self.current_vessel_container is not None:
            current_group = int(
                self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
            )

        # Current container size
        current_size = -1
        if self.container_sizes and self.current_vessel_container is not None:
            current_size = int(
                self.vessel_state[self.current_vessel_container, StateIds.SIZE.value]
            )

        # Calculate remaining containers per group in vessel
        vessel_group_counts = np.zeros(self.group_num, dtype=np.float32)
        vessel_occupied_mask = self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 1
        vessel_occupied_slots = self.vessel_state[vessel_occupied_mask]
        for group in range(self.group_num):
            vessel_group_counts[group] = np.sum(
                vessel_occupied_slots[:, StateIds.GROUP.value] == group
            )

        stack_idx = 0
        for bay in yard_bays:
            for row in yard_rows:
                # Find all slots in this stack
                stack_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (
                    self.yard_state[:, StateIds.ROW.value] == row
                )
                stack_slots = self.yard_state[stack_mask]

                feature_idx = 0

                # Count each group in this stack (IS_OCCUPIED distinguishes empty slots from group-0 containers)
                for group in range(self.group_num):
                    group_mask = (stack_slots[:, StateIds.IS_OCCUPIED.value] == 1) & (
                        stack_slots[:, StateIds.GROUP.value] == group
                    )
                    stack_features[stack_idx, feature_idx] = np.sum(group_mask)
                    feature_idx += 1

                # num_occupied
                num_occupied = np.sum(stack_slots[:, StateIds.IS_OCCUPIED.value])
                stack_features[stack_idx, feature_idx] = num_occupied
                feature_idx += 1

                # num_empty
                num_empty = len(stack_slots) - num_occupied
                stack_features[stack_idx, feature_idx] = num_empty
                feature_idx += 1

                # current_group (one-hot) - group 0 is valid, use 0-indexed offset
                if current_group >= 0:
                    stack_features[stack_idx, feature_idx + current_group] = 1.0
                feature_idx += self.group_num

                # left_row_max_group (one-hot) - left adjacent row in same bay
                left_row = row - 1
                if left_row >= 1:
                    left_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == left_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    left_groups = self.yard_state[left_mask, StateIds.GROUP.value]
                    if len(left_groups) > 0:
                        # Find most common group in left row (IS_OCCUPIED already filtered)
                        left_max_group = np.bincount(left_groups.astype(int)).argmax()
                        stack_features[stack_idx, feature_idx + left_max_group] = 1.0
                feature_idx += self.group_num

                # right_row_max_group (one-hot) - right adjacent row in same bay
                right_row = row + 1
                if right_row <= self.yard_shape[1]:
                    right_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == right_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    right_groups = self.yard_state[right_mask, StateIds.GROUP.value]
                    if len(right_groups) > 0:
                        # Find most common group in right row (IS_OCCUPIED already filtered)
                        right_max_group = np.bincount(right_groups.astype(int)).argmax()
                        stack_features[stack_idx, feature_idx + right_max_group] = 1.0
                feature_idx += self.group_num

                # vessel_remaining_per_group (count of remaining containers per group in vessel)
                stack_features[
                    stack_idx, feature_idx : feature_idx + self.group_num
                ] = vessel_group_counts
                feature_idx += self.group_num

                # is_empty
                stack_features[stack_idx, feature_idx] = (
                    1.0 if num_occupied == 0 else 0.0
                )
                feature_idx += 1

                # has_remaining_slots
                stack_features[stack_idx, feature_idx] = 1.0 if num_empty > 0 else 0.0
                feature_idx += 1

                # positional encoding
                if self.pos_embeddings:
                    # sinusoidal positional embedding (2 * max_frequency features, pre-computed)
                    pe_size = 2 * self.yard_shape[1]
                    stack_features[stack_idx, feature_idx : feature_idx + pe_size] = (
                        self.stack_pos_encoding[stack_idx]
                    )
                    feature_idx += pe_size
                else:
                    # scalar positional index
                    stack_features[stack_idx, feature_idx] = stack_idx
                    feature_idx += 1

                # Container size features (when enabled)
                if self.container_sizes:
                    # current_container_size (0=20ft, 1=40ft)
                    stack_features[stack_idx, feature_idx] = float(current_size) if current_size >= 0 else 0.0
                    feature_idx += 1
                    # stack_majority_size
                    stack_majority_size = -1
                    occupied_mask_size = stack_slots[:, StateIds.IS_OCCUPIED.value] == 1
                    if np.any(occupied_mask_size):
                        sizes = stack_slots[occupied_mask_size, StateIds.SIZE.value]
                        stack_majority_size = int(np.bincount(sizes.astype(int)).argmax())
                    stack_features[stack_idx, feature_idx] = float(stack_majority_size) if stack_majority_size >= 0 else 0.0
                    feature_idx += 1
                    # size_match
                    if current_size >= 0 and (stack_majority_size < 0 or stack_majority_size == current_size):
                        stack_features[stack_idx, feature_idx] = 1.0
                    feature_idx += 1

                # IMO features (when enabled)
                if self.enable_imo:
                    # is_current_container_imo
                    if self.current_vessel_container is not None:
                        stack_features[stack_idx, feature_idx] = float(
                            self.vessel_state[self.current_vessel_container, self.imo_attr_idx]
                        )
                    feature_idx += 1
                    # adj_has_different_group_imo
                    if self.current_vessel_container is not None:
                        cur_group = int(self.vessel_state[self.current_vessel_container, StateIds.GROUP.value])
                        has_diff_imo = 0.0
                        for adj_row in [row - 1, row + 1]:
                            if adj_row < 1 or adj_row > self.yard_shape[1]:
                                continue
                            adj_mask = (
                                (self.yard_state[:, StateIds.BAY.value] == bay)
                                & (self.yard_state[:, StateIds.ROW.value] == adj_row)
                                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                                & (self.yard_state[:, self.imo_attr_idx] == 1)
                            )
                            adj_imo_indices = np.where(adj_mask)[0]
                            if len(adj_imo_indices) > 0:
                                adj_groups = self.yard_state[adj_imo_indices, StateIds.GROUP.value]
                                if np.any(adj_groups != cur_group):
                                    has_diff_imo = 1.0
                                    break
                        stack_features[stack_idx, feature_idx] = has_diff_imo
                    feature_idx += 1

                stack_idx += 1

        return stack_features.flatten()

    def _create_stack_features_simplev2(self) -> np.ndarray:
        """
        Transform yard_state into simplified stack-based features with one-hot encoding

        For each stack (bay, row combination), computes:
        - max_group (one-hot): One-hot encoding of group with most containers in this stack
        - num_occupied_in_stack: Number of occupied slots in the stack
        - current_container_group (one-hot): One-hot encoding of current container's group
        - left_right_row_max_group (one-hot): One-hot encoding of dominant group in adjacent rows
        - stack_index: Sequential index of the stack (0, 1, 2, ...)

        Returns:
            np.ndarray: Flattened feature vector for all stacks (shape: num_stacks * (3*group_num + 2))
        """
        # Get unique stacks (bay, row combinations) - only odd bays
        yard_bays = np.unique(self.yard_state[:, StateIds.BAY.value])
        yard_bays = yard_bays[yard_bays % 2 == 1]  # only odd bays
        yard_rows = np.arange(1, self.yard_shape[1] + 1)

        num_stacks = len(yard_bays) * len(yard_rows)
        features_per_stack = 3 * self.group_num + 2
        if self.container_sizes:
            features_per_stack += 3
        if self.enable_imo:
            features_per_stack += 2
        stack_features = np.zeros((num_stacks, features_per_stack), dtype=np.float32)

        # Get current container group (same for all stacks)
        current_container_group = 0
        if self.current_vessel_container is not None:
            current_container_group = int(
                self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
            )

        # Current container size
        current_size = -1
        if self.container_sizes and self.current_vessel_container is not None:
            current_size = int(
                self.vessel_state[self.current_vessel_container, StateIds.SIZE.value]
            )

        stack_idx = 0
        for bay in yard_bays:
            for row in yard_rows:
                feature_idx = 0

                # Find all slots in this stack
                stack_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (
                    self.yard_state[:, StateIds.ROW.value] == row
                )
                stack_slots = self.yard_state[stack_mask]

                # Feature: max_group (one-hot) - dominant group in this stack
                occupied_mask = stack_slots[:, StateIds.IS_OCCUPIED.value] == 1
                max_group = -1  # sentinel: stack is empty
                if np.any(occupied_mask):
                    occupied_groups = stack_slots[occupied_mask, StateIds.GROUP.value]
                    # IS_OCCUPIED already filters empty slots; group 0 is a valid container group
                    max_group = np.bincount(occupied_groups.astype(int)).argmax()

                # Set one-hot encoding for max_group (groups 0 to group_num-1)
                if max_group >= 0:
                    stack_features[stack_idx, feature_idx + max_group] = 1.0
                feature_idx += self.group_num

                # Feature: num_occupied_in_stack
                num_occupied = np.sum(occupied_mask)
                stack_features[stack_idx, feature_idx] = num_occupied
                feature_idx += 1

                # Feature: current_container_group (one-hot, same for all stacks; group 0 is valid)
                if current_container_group >= 0:
                    stack_features[stack_idx, feature_idx + current_container_group] = (
                        1.0
                    )
                feature_idx += self.group_num

                # Feature: left_right_row_max_group (one-hot) - dominant group in adjacent rows
                adjacent_groups = []

                # Left adjacent row
                left_row = row - 1
                if left_row >= 1:
                    left_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == left_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    left_groups = self.yard_state[left_mask, StateIds.GROUP.value]
                    if len(left_groups) > 0:
                        adjacent_groups.extend(left_groups)

                # Right adjacent row
                right_row = row + 1
                if right_row <= self.yard_shape[1]:
                    right_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == right_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    right_groups = self.yard_state[right_mask, StateIds.GROUP.value]
                    if len(right_groups) > 0:
                        adjacent_groups.extend(right_groups)

                # Find dominant group in combined adjacent rows and set one-hot encoding
                left_right_max_group = -1  # sentinel: no adjacent occupied containers
                if len(adjacent_groups) > 0:
                    adjacent_groups_array = np.array(adjacent_groups)
                    # IS_OCCUPIED already filtered; group 0 is a valid container group
                    left_right_max_group = np.bincount(
                        adjacent_groups_array.astype(int)
                    ).argmax()

                # Set one-hot encoding for left_right_row_max_group (groups 0 to group_num-1)
                if left_right_max_group >= 0:
                    stack_features[stack_idx, feature_idx + left_right_max_group] = 1.0
                feature_idx += self.group_num

                # Feature: stack_index
                stack_features[stack_idx, feature_idx] = stack_idx
                feature_idx += 1

                # Container size features (when enabled)
                if self.container_sizes:
                    # current_container_size
                    stack_features[stack_idx, feature_idx] = float(current_size) if current_size >= 0 else 0.0
                    feature_idx += 1
                    # stack_majority_size
                    stack_majority_size = -1
                    if np.any(occupied_mask):
                        sizes = stack_slots[occupied_mask, StateIds.SIZE.value]
                        stack_majority_size = int(np.bincount(sizes.astype(int)).argmax())
                    stack_features[stack_idx, feature_idx] = float(stack_majority_size) if stack_majority_size >= 0 else 0.0
                    feature_idx += 1
                    # size_match
                    if current_size >= 0 and (stack_majority_size < 0 or stack_majority_size == current_size):
                        stack_features[stack_idx, feature_idx] = 1.0
                    feature_idx += 1

                # IMO features (when enabled)
                if self.enable_imo:
                    # is_current_container_imo
                    if self.current_vessel_container is not None:
                        stack_features[stack_idx, feature_idx] = float(
                            self.vessel_state[self.current_vessel_container, self.imo_attr_idx]
                        )
                    feature_idx += 1
                    # adj_has_different_group_imo
                    if self.current_vessel_container is not None:
                        cur_group = int(self.vessel_state[self.current_vessel_container, StateIds.GROUP.value])
                        has_diff_imo = 0.0
                        for adj_row in [row - 1, row + 1]:
                            if adj_row < 1 or adj_row > self.yard_shape[1]:
                                continue
                            adj_mask = (
                                (self.yard_state[:, StateIds.BAY.value] == bay)
                                & (self.yard_state[:, StateIds.ROW.value] == adj_row)
                                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                                & (self.yard_state[:, self.imo_attr_idx] == 1)
                            )
                            adj_imo_indices = np.where(adj_mask)[0]
                            if len(adj_imo_indices) > 0:
                                adj_groups = self.yard_state[adj_imo_indices, StateIds.GROUP.value]
                                if np.any(adj_groups != cur_group):
                                    has_diff_imo = 1.0
                                    break
                        stack_features[stack_idx, feature_idx] = has_diff_imo
                    feature_idx += 1

                stack_idx += 1

        return stack_features.flatten()

    def _create_stack_features_v3(self) -> np.ndarray:
        """
        THIS IS THE BEST OBSERVATION SPACE SO FAR (USED IN THESIS)
        Transform yard_state into scalar stack-based features (no one-hot encoding).

        For each stack (bay, row combination), computes 5 scalar features:
        - [0] majority_group: group index with most containers in this stack (-1 if empty)
        - [1] num_occupied: number of occupied slots in the stack
        - [2] current_container_group: group index of the container being placed (-1 if none)
        - [3] adj_majority_group: majority group in left+right adjacent stacks combined (-1 if none)
        - [4] stack_index: sequential positional index (0, 1, 2, ...)
        Optional (when container_sizes=True):
        - [5] current_container_size: size of the container being placed (20 or 40 feet).
        - [6] stack_majority_size: majority size of containers already in this stack (-1 if empty)
        - [7] size_match: 1 if current container is compatible with stack (stack empty or same size), 0 if incompatible, -1 if no current container
        Optional (when enable_imo=True):
        - [N] is_current_container_imo: 1 if current container is IMO, 0 otherwise (-1 if no container)
        - [N+1] adj_has_different_group_imo: 1 if adjacent stacks have IMO from different group, 0 otherwise (-1 if no container)

        Returns:
            np.ndarray: Flattened feature vector for all stacks
        """
        yard_bays = np.unique(self.yard_state[:, StateIds.BAY.value])
        yard_bays = yard_bays[yard_bays % 2 == 1]  # only odd bays
        yard_rows = np.arange(1, self.yard_shape[1] + 1)

        num_stacks = len(yard_bays) * len(yard_rows)
        features_per_stack = 5
        if self.container_sizes:
            features_per_stack += 3
        if self.enable_imo:
            features_per_stack += 2
        stack_features = np.full(
            (num_stacks, features_per_stack), -1.0, dtype=np.float32
        )

        # Current container group
        current_container_group = -1.0
        if self.current_vessel_container is not None:
            current_container_group = float(
                self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
            )

        # Current container size
        current_container_size = -1.0
        if self.container_sizes and self.current_vessel_container is not None:
            current_container_size = float(
                self.vessel_state[self.current_vessel_container, StateIds.SIZE.value]
            )

        stack_idx = 0
        for bay in yard_bays:
            for row in yard_rows:
                # Find all slots in this stack
                stack_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (
                    self.yard_state[:, StateIds.ROW.value] == row
                )
                stack_slots = self.yard_state[stack_mask]

                # [0] majority_group — scalar (-1 if empty)
                occupied_mask = stack_slots[:, StateIds.IS_OCCUPIED.value] == 1
                if np.any(occupied_mask):
                    occupied_groups = stack_slots[occupied_mask, StateIds.GROUP.value]
                    stack_features[stack_idx, 0] = float(
                        np.bincount(occupied_groups.astype(int)).argmax()
                    )
                # else: stays -1

                # [1] num_occupied
                stack_features[stack_idx, 1] = float(np.sum(occupied_mask))

                # [2] current_container_group — scalar (-1 if none)
                stack_features[stack_idx, 2] = current_container_group

                # [3] adj_majority_group — majority of combined left+right adjacent stacks
                adjacent_groups = []
                left_row = row - 1
                if left_row >= 1:
                    left_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == left_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    left_groups = self.yard_state[left_mask, StateIds.GROUP.value]
                    if len(left_groups) > 0:
                        adjacent_groups.extend(left_groups)

                right_row = row + 1
                if right_row <= self.yard_shape[1]:
                    right_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == right_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    right_groups = self.yard_state[right_mask, StateIds.GROUP.value]
                    if len(right_groups) > 0:
                        adjacent_groups.extend(right_groups)

                if len(adjacent_groups) > 0:
                    stack_features[stack_idx, 3] = float(
                        np.bincount(np.array(adjacent_groups).astype(int)).argmax()
                    )
                # else: stays -1

                # [4] stack_index
                stack_features[stack_idx, 4] = float(stack_idx)

                # Container size features (when enabled)
                if self.container_sizes:
                    # [5] current_container_size
                    stack_features[stack_idx, 5] = current_container_size
                    # [6] stack_majority_size (-1 if empty)
                    if np.any(occupied_mask):
                        sizes = stack_slots[occupied_mask, StateIds.SIZE.value]
                        stack_features[stack_idx, 6] = float(
                            np.bincount(sizes.astype(int)).argmax()
                        )
                    # else: stays -1
                    # [7] size_match (1 if compatible, 0 if not, -1 if no container)
                    if current_container_size >= 0:
                        stack_size = stack_features[stack_idx, 6]
                        if stack_size < 0 or stack_size == current_container_size:
                            stack_features[stack_idx, 7] = 1.0
                        else:
                            stack_features[stack_idx, 7] = 0.0

                # IMO features (when enabled)
                if self.enable_imo:
                    imo_feat_start = 5 + (3 if self.container_sizes else 0)
                    # [N] is_current_container_imo
                    if self.current_vessel_container is not None:
                        stack_features[stack_idx, imo_feat_start] = float(
                            self.vessel_state[self.current_vessel_container, self.imo_attr_idx]
                        )
                    # else: stays -1

                    # [N+1] adj_has_different_group_imo: 1 if left/right adjacent stacks
                    # have IMO containers from a group different than current container's group
                    if self.current_vessel_container is not None:
                        current_group = int(
                            self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
                        )
                        has_diff_imo = 0.0
                        for adj_row in [row - 1, row + 1]:
                            if adj_row < 1 or adj_row > self.yard_shape[1]:
                                continue
                            adj_mask = (
                                (self.yard_state[:, StateIds.BAY.value] == bay)
                                & (self.yard_state[:, StateIds.ROW.value] == adj_row)
                                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                                & (self.yard_state[:, self.imo_attr_idx] == 1)
                            )
                            adj_imo_indices = np.where(adj_mask)[0]
                            if len(adj_imo_indices) > 0:
                                adj_groups = self.yard_state[adj_imo_indices, StateIds.GROUP.value]
                                if np.any(adj_groups != current_group):
                                    has_diff_imo = 1.0
                                    break
                        stack_features[stack_idx, imo_feat_start + 1] = has_diff_imo
                    # else: stays -1

                stack_idx += 1

        return stack_features.flatten()

    def _create_bay_features(self) -> np.ndarray:
        """
        Create bay-level summary features for the hierarchical differentiated observation (for diffobs observation type).

        For each bay computes:
        - bay_index: 1-indexed sequential bay number (1, 2, 3, ...)
        - count_group_0, ..., count_group_{n-1}: number of containers per group in the bay
        - num_completely_empty_stacks: stacks with 0 containers in the bay
        - num_not_full_matching: not-full stacks whose majority group == current container's group
        - num_matching_stacks: all non-empty stacks (full or not) whose majority group == current container's group

        Returns:
            np.ndarray: shape (n_bays, bay_f_dim) where bay_f_dim = group_num + 4
        """
        yard_bays = np.unique(self.yard_state[:, StateIds.BAY.value])
        yard_bays = yard_bays[yard_bays % 2 == 1]  # only odd bays
        yard_rows = np.arange(1, self.yard_shape[1] + 1)
        n_bays = len(yard_bays)
        bay_f_dim = self.group_num + 4

        bay_features = np.zeros((n_bays, bay_f_dim), dtype=np.float32)

        # Get current container group
        current_group = -1
        if self.current_vessel_container is not None:
            current_group = int(
                self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
            )

        for bay_idx, bay in enumerate(yard_bays):
            feature_idx = 0

            # bay_index (1-indexed sequential)
            bay_features[bay_idx, feature_idx] = bay_idx + 1
            feature_idx += 1

            # Get all slots in this bay
            bay_mask = self.yard_state[:, StateIds.BAY.value] == bay
            bay_slots = self.yard_state[bay_mask]
            occupied_mask = bay_slots[:, StateIds.IS_OCCUPIED.value] == 1

            # count_group_0 ... count_group_{n-1}
            for group in range(self.group_num):
                group_count = np.sum(
                    occupied_mask & (bay_slots[:, StateIds.GROUP.value] == group)
                )
                bay_features[bay_idx, feature_idx] = group_count
                feature_idx += 1

            # Per-stack stats within this bay
            num_empty_stacks = 0
            num_not_full_matching = 0
            num_matching_stacks = 0

            for row in yard_rows:
                stack_mask = (
                    (self.yard_state[:, StateIds.BAY.value] == bay)
                    & (self.yard_state[:, StateIds.ROW.value] == row)
                )
                stack_slots = self.yard_state[stack_mask]
                num_occupied = np.sum(stack_slots[:, StateIds.IS_OCCUPIED.value])
                num_total = len(stack_slots)
                is_full = num_occupied == num_total

                if num_occupied == 0:
                    num_empty_stacks += 1
                else:
                    # Stack is non-empty, compute majority group
                    majority_group = self._get_majority_group(bay, row)
                    if majority_group is not None and current_group >= 0:
                        if majority_group == current_group:
                            # non-empty stack with matching majority
                            num_matching_stacks += 1
                            if not is_full:
                                num_not_full_matching += 1

            bay_features[bay_idx, feature_idx] = num_empty_stacks
            feature_idx += 1
            bay_features[bay_idx, feature_idx] = num_not_full_matching
            feature_idx += 1
            bay_features[bay_idx, feature_idx] = num_matching_stacks

        return bay_features

    def _create_stack_features_relative(self) -> np.ndarray:
        """
        For diffobs observation type.
        Same as _create_stack_features() but with relative positional index per bay
        (0, 1, ..., n_rows-1) instead of global stack index.
        Stacks are ordered bay-major: all rows of bay 0, then all rows of bay 1, etc.

        Returns:
            np.ndarray: shape (n_stacks, stack_f_dim) where stack_f_dim = 5 * group_num + 5
        """
        yard_bays = np.unique(self.yard_state[:, StateIds.BAY.value])
        yard_bays = yard_bays[yard_bays % 2 == 1]
        yard_rows = np.arange(1, self.yard_shape[1] + 1)

        num_stacks = len(yard_bays) * len(yard_rows)
        features_per_stack = 5 * self.group_num + 5
        stack_features = np.zeros((num_stacks, features_per_stack), dtype=np.float32)

        current_group = -1
        if self.current_vessel_container is not None:
            current_group = int(
                self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
            )

        vessel_group_counts = np.zeros(self.group_num, dtype=np.float32)
        vessel_occupied_mask = self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 1
        vessel_occupied_slots = self.vessel_state[vessel_occupied_mask]
        for group in range(self.group_num):
            vessel_group_counts[group] = np.sum(
                vessel_occupied_slots[:, StateIds.GROUP.value] == group
            )

        stack_idx = 0
        for bay in yard_bays:
            relative_idx = 0
            for row in yard_rows:
                stack_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (
                    self.yard_state[:, StateIds.ROW.value] == row
                )
                stack_slots = self.yard_state[stack_mask]

                feature_idx = 0

                for group in range(self.group_num):
                    group_mask = (stack_slots[:, StateIds.IS_OCCUPIED.value] == 1) & (
                        stack_slots[:, StateIds.GROUP.value] == group
                    )
                    stack_features[stack_idx, feature_idx] = np.sum(group_mask)
                    feature_idx += 1

                num_occupied = np.sum(stack_slots[:, StateIds.IS_OCCUPIED.value])
                stack_features[stack_idx, feature_idx] = num_occupied
                feature_idx += 1

                num_empty = len(stack_slots) - num_occupied
                stack_features[stack_idx, feature_idx] = num_empty
                feature_idx += 1

                if current_group >= 0:
                    stack_features[stack_idx, feature_idx + current_group] = 1.0
                feature_idx += self.group_num

                # left_row_max_group
                left_row = row - 1
                if left_row >= 1:
                    left_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == left_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    left_groups = self.yard_state[left_mask, StateIds.GROUP.value]
                    if len(left_groups) > 0:
                        left_max_group = np.bincount(left_groups.astype(int)).argmax()
                        stack_features[stack_idx, feature_idx + left_max_group] = 1.0
                feature_idx += self.group_num

                # right_row_max_group
                right_row = row + 1
                if right_row <= self.yard_shape[1]:
                    right_mask = (
                        (self.yard_state[:, StateIds.BAY.value] == bay)
                        & (self.yard_state[:, StateIds.ROW.value] == right_row)
                        & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
                    )
                    right_groups = self.yard_state[right_mask, StateIds.GROUP.value]
                    if len(right_groups) > 0:
                        right_max_group = np.bincount(right_groups.astype(int)).argmax()
                        stack_features[stack_idx, feature_idx + right_max_group] = 1.0
                feature_idx += self.group_num

                stack_features[
                    stack_idx, feature_idx : feature_idx + self.group_num
                ] = vessel_group_counts
                feature_idx += self.group_num

                stack_features[stack_idx, feature_idx] = (
                    1.0 if num_occupied == 0 else 0.0
                )
                feature_idx += 1

                stack_features[stack_idx, feature_idx] = 1.0 if num_empty > 0 else 0.0
                feature_idx += 1

                # Relative positional index within bay (0, 1, ..., n_rows-1)
                stack_features[stack_idx, feature_idx] = relative_idx

                stack_idx += 1
                relative_idx += 1

        return stack_features

    def _create_hierarchical_diff_obs(self) -> np.ndarray:
        """
        Create the hierarchical differentiated observation. Training with RL not working with this currently.

        Layout: [bay_features.flatten() | stack_features.flatten()]
        - bay_features: (n_bays, bay_f_dim) flattened
        - stack_features: (n_stacks, stack_f_dim) flattened, with relative positional indices

        Returns:
            np.ndarray: flat observation vector
        """
        bay_features = self._create_bay_features()    # (n_bays, bay_f_dim)
        stack_features = self._create_stack_features_relative()  # (n_stacks, stack_f_dim)
        return np.concatenate([bay_features.flatten(), stack_features.flatten()])

    def _create_observation(self) -> Union[np.ndarray, Dict]:
        """
        Create observation based on observation_type.

        When action_mask_with_obs != "default", all types except "flat_parsed" return a dict
        {"observation": <array>, "mask": <action_mask>} instead of a plain array.

        observation_type options:
        - "flat": Original flattened observation vector (yard_state + current container)
        - "stack_features": Stack-based feature representation with one-hot encodings
        - "stack_features_simplev2": Simplified stack features with one-hot encodings (fewer features than "stack_features")
        - "stack_features_v3": Scalar stack features — no one-hot encoding (used in thesis)
        - "flat_parsed": Dictionary with separate yard_state and current_container arrays (for rule-based agents, no mask)
        - "hierarchical_diff_obs": Concatenation of bay-level and stack-level features (experimental)

        For "flat":
            Observation vector consists of:
            1. yard state (total_yard_coords i.e. number of yard slots)
            2. Current vessel selected container state (1).
            Therefore, obs_coords = total_yard_coords + 1
            Each slot has num_slot_attrs attributes: bay, row, tier, is_occupied(0/1), group number of container
            Final flattened observation shape: (obs_coords * num_slot_attrs,)

        For "stack_features":
            Per-stack features (shape: num_stacks * features_per_stack):
            - count_group0 .. count_groupN: count of each group in the stack (one per group)
            - num_occupied: number of occupied slots
            - num_empty: number of empty slots
            - current_group (one-hot): one-hot of current container's group
            - left_row_max_group (one-hot): dominant group in left adjacent row
            - right_row_max_group (one-hot): dominant group in right adjacent row
            - vessel_remaining_per_group: remaining vessel containers per group
            - is_empty: 1 if stack fully empty, 0 otherwise
            - has_remaining_slots: 1 if at least one empty slot, 0 otherwise
            - positional_index (pos_embeddings=False) OR sin/cos encoding (pos_embeddings=True)

        For "stack_features_simplev2":
            Per-stack features (shape: num_stacks * (3*group_num + 2[+3][+2])):
            - max_group (one-hot): dominant group in this stack
            - num_occupied_in_stack: number of occupied slots
            - current_container_group (one-hot): current container's group
            - left_right_row_max_group (one-hot): dominant group across both adjacent rows
            - stack_index: sequential positional index
            Optional (container_sizes=True): current_container_size, stack_majority_size, size_match
            Optional (enable_imo=True): is_current_container_imo, adj_has_different_group_imo

        For "stack_features_v3":
            Per-stack scalar features — see _create_stack_features_v3() docstring for full spec.
            Base 5 features + optional size features (3) + optional IMO features (2).
            Recommended for RL training (used in thesis).

        For "flat_parsed":
            Dictionary with:
            - "yard_state": 2D array (total_yard_coords, num_slot_attrs)
            - "current_container": 1D array (num_slot_attrs,)

        For "hierarchical_diff_obs":
            Concatenation of bay_features.flatten() and stack_features.flatten().
            bay_features shape: (n_bays, bay_f_dim); stack_features use relative positional indices.
            Note: RL training with this observation type is currently experimental and not stable.

        """
        mask = self.action_masks() if self.action_mask_with_obs != "default" else None

        if self.observation_type == "flat":
            # Original flat observation
            state = self.yard_state.copy()
            if self.current_vessel_container is not None:
                state = np.concatenate(
                    (
                        state,
                        self.vessel_state[self.current_vessel_container].reshape(1, -1),
                    ),
                    axis=0,
                )
            else:
                state = np.concatenate((state, np.zeros((1, self.num_slot_attrs), dtype=int)), axis=0)
            state = state.flatten()

            if self.action_mask_with_obs == "default":
                return state
            else:
                return {"observation": state, "mask": mask}

        elif self.observation_type == "stack_features":
            # Stack-based feature observation
            stack_features = self._create_stack_features()

            if self.action_mask_with_obs == "default":
                return stack_features
            else:
                return {"observation": stack_features, "mask": mask}

        elif self.observation_type == "flat_parsed":
            # Dictionary observation for rule-based agents
            # Returns pre-parsed yard_state and current_container
            yard_state = self.yard_state.copy()

            if self.current_vessel_container is not None:
                current_container = self.vessel_state[
                    self.current_vessel_container
                ].copy()
            else:
                current_container = np.zeros(self.num_slot_attrs, dtype=np.int64)

            # Note: flat_parsed is intended for rule-based agents, so no mask is returned
            return {"yard_state": yard_state, "current_container": current_container}

        elif self.observation_type == "stack_features_simplev2":
            # Simplified stack-based feature observation (no one-hot encoding)
            stack_features = self._create_stack_features_simplev2()

            if self.action_mask_with_obs == "default":
                return stack_features
            else:
                return {"observation": stack_features, "mask": mask}

        elif self.observation_type == "stack_features_v3":
            # Scalar stack features (5 features per stack, no one-hot)
            stack_features = self._create_stack_features_v3()

            if self.action_mask_with_obs == "default":
                return stack_features
            else:
                return {"observation": stack_features, "mask": mask}

        elif self.observation_type == "hierarchical_diff_obs":
            # Hierarchical differentiated observation (bay + stack features)
            diff_obs = self._create_hierarchical_diff_obs()

            if self.action_mask_with_obs == "default":
                return diff_obs
            else:
                return {"observation": diff_obs, "mask": mask}

        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def render(self, enhanced_visibility: bool = False) -> Optional[np.ndarray]:
        """
        Render the environment as an RGB array (similar to stowage env)

        Args:
            enhanced_visibility: When True, uses an enhanced layout:
                - Yard bays are wrapped into rows of at most 5 bays each
                - Vessel section shows only the current selected container box (no full grid)
                - Bay number labels use a larger font
                Default is False (original behavior is preserved).
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

        if enhanced_visibility:
            return self._render_enhanced()

        if self.screen is None:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.screen.fill((255, 255, 255))

        padding, title_height, section_gap = 10, 20, 50
        vessel_height = (self.screen_height - 3 * padding - 2 * title_height) * 0.4
        yard_height = (self.screen_height - 3 * padding - 2 * title_height) * 0.6

        font = pygame.font.Font(None, 24)
        self.screen.blit(font.render("Vessel", True, (0, 0, 0)), (padding, padding))
        self.screen.blit(
            font.render("Yard", True, (0, 0, 0)),
            (padding, padding + vessel_height + section_gap),
        )

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

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _render_enhanced(self) -> np.ndarray:
        """
        Enhanced visibility render:
        - Yard bays wrap after every 5 bays (multiple display rows)
        - Vessel section shows only the current selected container box
        - Bay labels use a larger font
        """
        import pygame

        CELL_W, CELL_H = 35, 35
        PADDING = 15
        LEFT_MARGIN = 45   # space for tier labels on the left
        ROW_LABEL_H = 20   # space for row-number labels below each row of cells
        BAY_LABEL_FONT_SIZE = 36

        yard_bays = self.yard_shape[0]
        yard_rows = self.yard_shape[1]
        yard_tiers = self.yard_shape[2]

        font_title = pygame.font.Font(None, 46)
        font_bay = pygame.font.Font(None, BAY_LABEL_FONT_SIZE)
        font_small = pygame.font.Font(None, 20)
        font_tiny = pygame.font.Font(None, 18)
        colors = self._setup_colors()
        fonts = {"small": font_small, "tiny": font_tiny}

        # Measure rendered bay-label height for layout calculations
        bay_label_h = font_bay.render("Bay 1", True, (0, 0, 0)).get_height() + 6
        title_h = font_title.render("X", True, (0, 0, 0)).get_height() + 4

        # Dynamic split: roughly half the bays on top row, rest on bottom row
        # (always 2 display rows; if only 1 bay total, keep it on one row)
        if yard_bays <= 1:
            bays_per_row_list = [yard_bays]
        else:
            top_count = (yard_bays + 1) // 2  # ceiling half
            bot_count = yard_bays - top_count
            bays_per_row_list = [top_count, bot_count]
        num_bay_rows = len(bays_per_row_list)
        max_bays_in_row = max(bays_per_row_list)

        screen_width = max(350, LEFT_MARGIN + max_bays_in_row * yard_rows * CELL_W + PADDING)

        vessel_section_h = 100
        single_bay_row_h = bay_label_h + yard_tiers * CELL_H + ROW_LABEL_H + 10

        total_h = (
            PADDING + title_h            # "Current Container" title
            + vessel_section_h           # container box
            + PADDING + title_h          # "Yard" title
            + num_bay_rows * single_bay_row_h
            + PADDING
        )

        enh_screen = pygame.Surface((screen_width, total_h))
        enh_screen.fill((255, 255, 255))

        # Temporarily redirect self.screen so existing draw helpers target the enhanced surface
        old_screen = self.screen
        self.screen = enh_screen

        try:
            # ── Vessel section ──────────────────────────────────────────
            self.screen.blit(
                font_title.render("Current Container", True, (0, 0, 0)),
                (PADDING, PADDING),
            )
            vessel_top = PADDING + title_h
            self._draw_enhanced_container_box(vessel_top, vessel_section_h, colors, fonts)

            # ── Yard section ────────────────────────────────────────────
            yard_label_y = vessel_top + vessel_section_h + PADDING
            self.screen.blit(
                font_title.render("Yard", True, (0, 0, 0)),
                (PADDING, yard_label_y),
            )
            yard_top_start = yard_label_y + title_h

            cell_info = self._build_cell_info(self.yard_state, yard_bays, False)
            row_order = self._get_row_order(yard_rows, False)

            cumulative_b = 0
            for bay_row_idx in range(num_bay_rows):
                row_top = yard_top_start + bay_row_idx * single_bay_row_h
                cells_top = row_top + bay_label_h  # cells sit below the bay labels

                # Tier labels on the left
                for t in range(1, yard_tiers + 1):
                    cy = cells_top + (yard_tiers - t) * CELL_H + CELL_H / 2
                    tier_lbl = font_small.render(f"{t}", True, (0, 0, 0))
                    self.screen.blit(
                        tier_lbl,
                        (LEFT_MARGIN - 18, int(cy - tier_lbl.get_height() / 2)),
                    )

                start_b = cumulative_b
                end_b = start_b + bays_per_row_list[bay_row_idx]
                cumulative_b = end_b

                for b in range(start_b, end_b):
                    bay_num = b * 2 + 1
                    local_b = b - start_b
                    bay_x = LEFT_MARGIN + local_b * yard_rows * CELL_W

                    # Bay label (large font)
                    bay_cx = bay_x + (yard_rows * CELL_W) / 2
                    bay_lbl = font_bay.render(f"Bay {bay_num}", True, (0, 0, 0))
                    self.screen.blit(
                        bay_lbl,
                        bay_lbl.get_rect(center=(int(bay_cx), row_top + bay_label_h // 2)),
                    )

                    for pos, r in enumerate(row_order):
                        # Row number labels below cells
                        x_lbl = bay_x + pos * CELL_W + CELL_W / 2
                        row_lbl = font_small.render(f"{r}", True, (0, 0, 0))
                        self.screen.blit(
                            row_lbl,
                            row_lbl.get_rect(
                                center=(int(x_lbl), cells_top + yard_tiers * CELL_H + ROW_LABEL_H // 2)
                            ),
                        )

                        # Cells for each tier
                        for t in range(1, yard_tiers + 1):
                            cx = bay_x + pos * CELL_W
                            cy = cells_top + (yard_tiers - t) * CELL_H
                            default_cell = self._create_cell_props(False, False, None, 0)
                            cell = cell_info.get((bay_num, r, t), default_cell)
                            self._draw_cell(cx, cy, CELL_W, CELL_H, cell, colors, fonts, False)

                # Bay dividers (vertical lines between and around bays in this display row)
                num_in_row = end_b - start_b
                for div in range(num_in_row + 1):
                    x_div = LEFT_MARGIN + div * yard_rows * CELL_W
                    pygame.draw.line(
                        self.screen,
                        colors["bay_grid"],
                        (x_div, cells_top),
                        (x_div, cells_top + yard_tiers * CELL_H),
                        2,
                    )

            result = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        finally:
            self.screen = old_screen

        return result

    def _draw_enhanced_container_box(
        self,
        top: int,
        section_h: int,
        colors: Dict[str, Any],
        fonts: Dict[str, Any],
    ) -> None:
        """
        Draw the current vessel container as a labeled colored box (enhanced render mode only).
        """
        import pygame

        box_size = min(section_h - 20, 70)
        box_y = top + (section_h - box_size) // 2
        font_info = pygame.font.Font(None, 30)

        # Estimate the total width of box + gap + text block so we can centre the whole unit
        GAP = 15
        INFO_LINE_W = 160  # rough max width of the info text block

        screen_w = self.screen.get_width()

        if self.current_vessel_container is None:
            all_done = (self.containers_retrieved >= self.num_containers)
            status_msg = "All containers unloaded" if all_done else "No container selected"
            status_lbl = font_info.render(status_msg, True, (80, 80, 80))
            total_w = box_size + GAP + status_lbl.get_width()
            box_x = (screen_w - total_w) // 2
            # Draw empty white box with grey border
            empty_rect = pygame.Rect(box_x, box_y, box_size, box_size)
            pygame.draw.rect(self.screen, (255, 255, 255), empty_rect)
            pygame.draw.rect(self.screen, (160, 160, 160), empty_rect, 2)
            # Status message beside the box, vertically centred
            text_x = box_x + box_size + GAP
            self.screen.blit(status_lbl, (text_x, box_y + (box_size - status_lbl.get_height()) // 2))
            # Progress stats right-aligned
            unloaded = self.containers_retrieved
            remaining = self.num_containers - self.containers_retrieved
            font_progress = pygame.font.Font(None, 30)
            stats_lines = [
                f"Unloaded: {unloaded}",
                f"Remaining: {remaining}",
            ]
            stats_right_x = screen_w - 15
            for j, stat in enumerate(stats_lines):
                stat_lbl = font_progress.render(stat, True, (30, 30, 30))
                self.screen.blit(
                    stat_lbl,
                    (stats_right_x - stat_lbl.get_width(), box_y + j * 28),
                )
            return

        idx = self.current_vessel_container
        state = self.vessel_state
        group = int(state[idx, StateIds.GROUP.value])
        size = int(state[idx, StateIds.SIZE.value]) if self.container_sizes else 0
        is_imo = bool(state[idx, self.imo_attr_idx]) if self.enable_imo else False

        group_colors = colors["group_colors"]
        group_idx = min(group, len(group_colors) - 1)
        fill_color = group_colors[group_idx][1]

        # Centre box + info block horizontally
        total_w = box_size + GAP + INFO_LINE_W
        box_x = (screen_w - total_w) // 2

        rect = pygame.Rect(box_x, box_y, box_size, box_size)
        pygame.draw.rect(self.screen, fill_color, rect)

        # Diagonal hash lines for 40ft containers
        if size == 1:
            hash_color = (0, 0, 0)
            spacing = 6
            for offset in range(-box_size, box_size, spacing):
                x1 = box_x + offset
                y1 = box_y
                x2 = box_x + offset + box_size
                y2 = box_y + box_size
                pygame.draw.line(
                    self.screen, hash_color,
                    (max(box_x, min(box_x + box_size, x1)), max(box_y, min(box_y + box_size, y1))),
                    (max(box_x, min(box_x + box_size, x2)), max(box_y, min(box_y + box_size, y2))),
                    1,
                )

        # IMO indicator circle for dangerous containers
        if is_imo:
            cx = int(box_x + box_size / 2)
            cy = int(box_y + box_size / 2)
            radius = int(box_size * 0.35)
            pygame.draw.circle(self.screen, fill_color, (cx, cy), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius, 2)

        pygame.draw.rect(self.screen, (255, 0, 0), rect, 3)  # red border = target/selected

        idx_lbl = pygame.font.Font(None, 38).render(f"{idx}", True, (255, 255, 255))
        self.screen.blit(idx_lbl, idx_lbl.get_rect(center=(box_x + box_size // 2, box_y + box_size // 2)))

        text_x = box_x + box_size + GAP
        info_lines = [
            f"Container #{idx}",
            f"Group: {group}",
            f"Size: {'40ft' if size == 1 else '20ft'}",
        ]
        if is_imo:
            info_lines.append("IMO / Dangerous")

        for i, line in enumerate(info_lines):
            lbl = font_info.render(line, True, (0, 0, 0))
            self.screen.blit(lbl, (text_x, box_y + i * 28))

        # Progress stats: right-aligned to the screen's right edge
        unloaded = self.containers_retrieved
        remaining = self.num_containers - self.containers_retrieved
        font_progress = pygame.font.Font(None, 30)
        stats_lines = [
            f"Unloaded: {unloaded}",
            f"Remaining: {remaining}",
        ]
        stats_right_x = screen_w - 15  # right-align to screen right edge
        for j, stat in enumerate(stats_lines):
            stat_lbl = font_progress.render(stat, True, (30, 30, 30))
            self.screen.blit(
                stat_lbl,
                (stats_right_x - stat_lbl.get_width(), box_y + j * 28),
            )

    def _draw_grid(
        self,
        state: np.ndarray,
        top: int,
        height: float,
        bays: int,
        rows: int,
        tiers: int,
        is_vessel: bool,
    ) -> None:
        """
        Draw a grid section (vessel or yard) with fixed cell size
        """
        import pygame

        cell_width, cell_height, left_margin, label_margin = (
            self._setup_grid_dimensions(bays, rows)
        )
        fonts = {
            "small": pygame.font.Font(None, 20),
            "tiny": pygame.font.Font(None, 18),
        }
        colors = self._setup_colors()
        self._draw_tier_labels(
            top, tiers, cell_height, left_margin, label_margin, fonts["small"]
        )
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
        self._draw_bay_dividers(
            left_margin,
            top,
            bays,
            rows,
            cell_width,
            tiers,
            cell_height,
            colors["bay_grid"],
        )

    def _setup_grid_dimensions(
        self, bays: int, rows: int
    ) -> Tuple[int, int, float, int]:
        cell_width, cell_height = 35, 35
        padding, label_margin = 30, 15
        left_margin = max(padding, (self.screen_width - bays * rows * cell_width) / 2)
        return cell_width, cell_height, left_margin, label_margin

    def _setup_colors(self) -> Dict[str, Any]:
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

    def _get_row_order(self, rows: int, is_vessel: bool) -> List[int]:
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

    def _draw_tier_labels(
        self,
        top: int,
        tiers: int,
        cell_height: int,
        left_margin: float,
        label_margin: int,
        font: Any,
    ) -> None:
        """
        Draw tier labels on the left side
        """
        for t in range(1, tiers + 1):
            y = top + (tiers - t) * cell_height + cell_height / 2
            tier_label = font.render(f"{t}", True, (0, 0, 0))
            self.screen.blit(
                tier_label,
                (left_margin - label_margin, y - tier_label.get_height() / 2),
            )

    def _build_cell_info(
        self, state: np.ndarray, bays: int, is_vessel: bool
    ) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
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
                size = int(state[i, StateIds.SIZE.value]) if self.container_sizes else 0
                is_imo = bool(state[i, self.imo_attr_idx]) if self.enable_imo else False

                if bay % 2 == 1:
                    cell_info[(bay, row, tier)] = self._create_cell_props(
                        is_occupied, is_target, i, group, size, is_imo
                    )

            # handle even bays
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])
                size = int(state[i, StateIds.SIZE.value]) if self.container_sizes else 0
                is_imo = bool(state[i, self.imo_attr_idx]) if self.enable_imo else False

                # Only apply even bay logic if it's occupied (maybe required later but not used currently)
                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            existing = cell_info.get(
                                (adj_bay, row, tier),
                                self._create_cell_props(False, False, None, group, size, is_imo),
                            )
                            existing["filled"] = True
                            existing["idx"] = i
                            existing["is_imo"] = is_imo
                            cell_info[(adj_bay, row, tier)] = existing
        else:
            # Handle yard containers
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])
                size = int(state[i, StateIds.SIZE.value]) if self.container_sizes else 0
                is_imo = bool(state[i, self.imo_attr_idx]) if self.enable_imo else False

                if bay % 2 == 1:
                    cell_info[(bay, row, tier)] = self._create_cell_props(
                        is_occupied, False, i, group, size, is_imo
                    )

            # handle even yard bays (should not be occupied but just in case)
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])
                size = int(state[i, StateIds.SIZE.value]) if self.container_sizes else 0
                is_imo = bool(state[i, self.imo_attr_idx]) if self.enable_imo else False

                # Only apply even bay logic if it's occupied (maybe required later but not used currently)
                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            cell_info[(adj_bay, row, tier)] = self._create_cell_props(
                                is_occupied, False, i, group, size, is_imo
                            )

        return cell_info

    def _is_target_cell(self, idx: int, is_vessel: bool) -> bool:
        if is_vessel:
            return (
                self.current_vessel_container == idx
                if self.current_vessel_container is not None
                else False
            )
        return False

    def _create_cell_props(
        self, filled: bool, target: bool, idx: Optional[int], group: int, size: int = 0, is_imo: bool = False
    ) -> Dict[str, Any]:
        # Used for inheritance
        return {"filled": filled, "target": target, "idx": idx, "group": group, "size": size, "is_imo": is_imo}

    def _get_cell_border_style(
        self, cell: Dict[str, Any]
    ) -> Tuple[Tuple[int, int, int], int]:
        if cell["target"]:
            return (
                (255, 0, 0),
                3,
            )  # Red border for curently selcted container in vessel for yard placement
        return (180, 180, 180), 1

    def _draw_grid_cells(
        self,
        cell_info: Dict[Tuple[int, int, int], Dict[str, Any]],
        top: int,
        left_margin: float,
        bays: int,
        rows: int,
        tiers: int,
        row_order: List[int],
        cell_width: int,
        cell_height: int,
        label_margin: int,
        colors: Dict[str, Any],
        fonts: Dict[str, Any],
        is_vessel: bool,
    ) -> None:
        """
        Draw grid cells with labels
        """
        for b in range(bays):
            bay_num = b * 2 + 1
            bay_x = left_margin + b * rows * cell_width

            bay_center_x = bay_x + (rows * cell_width) / 2
            bay_label = fonts["small"].render(f"Bay {bay_num}", True, (0, 0, 0))
            self.screen.blit(
                bay_label, bay_label.get_rect(center=(bay_center_x, top - 10))
            )

            for pos, r in enumerate(row_order):
                # Draw row labels
                x_label = bay_x + pos * cell_width + cell_width / 2
                row_label = fonts["small"].render(f"{r}", True, (0, 0, 0))
                self.screen.blit(
                    row_label,
                    row_label.get_rect(
                        center=(x_label, top + tiers * cell_height + label_margin / 2)
                    ),
                )

                # Draw cells for each tier
                for t in range(1, tiers + 1):
                    x = bay_x + pos * cell_width
                    y = top + (tiers - t) * cell_height

                    default_cell = self._create_cell_props(False, False, None, 0)
                    cell = cell_info.get((bay_num, r, t), default_cell)

                    self._draw_cell(
                        x, y, cell_width, cell_height, cell, colors, fonts, is_vessel
                    )

    def _draw_cell(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        cell: Dict[str, Any],
        colors: Dict[str, Any],
        fonts: Dict[str, Any],
        is_vessel: bool,
    ) -> None:
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

        # Draw diagonal hash lines for 40ft containers
        if cell["filled"] and cell.get("size", 0) == 1:
            hash_color = (0, 0, 0)
            spacing = 6
            for offset in range(-max(width, height), max(width, height), spacing):
                x1 = x + offset
                y1 = y
                x2 = x + offset + height
                y2 = y + height
                # Clip to cell rect
                pygame.draw.line(
                    self.screen, hash_color,
                    (max(x, min(x + width, x1)), max(y, min(y + height, y1))),
                    (max(x, min(x + width, x2)), max(y, min(y + height, y2))),
                    1,
                )

        # Draw IMO indicator circle for dangerous containers
        if cell["filled"] and cell.get("is_imo", False):
            center_x = int(x + width / 2)
            center_y = int(y + height / 2)
            radius = int(min(width, height) * 0.35)
            # Filled circle using the group's dark color
            circle_color = group_colors[group_idx][1]
            pygame.draw.circle(self.screen, circle_color, (center_x, center_y), radius)
            # Black border ring for visibility
            pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), radius, 2)

        line_color, line_width = self._get_cell_border_style(cell)
        pygame.draw.rect(self.screen, line_color, rect, line_width)
        if cell["idx"] is not None:
            if is_vessel:
                text_color = (255, 255, 255) if cell["filled"] else (50, 50, 50)
                label = fonts["tiny"].render(f"{cell['idx']}", True, text_color)
                self.screen.blit(
                    label, label.get_rect(center=(x + width / 2, y + height / 2))
                )
            else:
                text_color = (255, 255, 255) if cell["filled"] else (50, 50, 50)
                label = fonts["tiny"].render(f"{cell['idx']}", True, text_color)
                self.screen.blit(
                    label, label.get_rect(center=(x + width / 2, y + height / 2))
                )

    def _draw_bay_dividers(
        self,
        left_margin: float,
        top: int,
        bays: int,
        rows: int,
        cell_width: int,
        tiers: int,
        cell_height: int,
        color: Tuple[int, int, int],
    ) -> None:
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
