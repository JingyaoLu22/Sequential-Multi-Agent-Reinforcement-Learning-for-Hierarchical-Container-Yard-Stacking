from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from envs.stack_gym import StateIds


class AgentLevel(Enum):
    HIGH_LEVEL = 0
    LOW_LEVEL = 1


class BaseAgent(ABC):
    """
    Base class for hierarchical agents using abstract class
    
    Attributes:
        level: The hierarchical level of this agent (HIGH_LEVEL or LOW_LEVEL)
        verbosity: Verbosity level for debug output (0: silent, 1: verbose)
    """

    def __init__(self, level: AgentLevel, verbosity: int = 0) -> None:
        self.level = level
        self.verbosity = verbosity  # 0: silent, 1: verbose

    @abstractmethod
    def get_action(self, observation: dict | np.ndarray, valid_actions: np.ndarray) -> int | None:
        """
        Return action from agent based on observation
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            
        Returns:
            Selected action index or None if no valid action
        """
        pass

    @abstractmethod
    def update_agent(self, reward: float, info: dict | None = None) -> None:
        """
        Update agent based on feedback
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        pass


class HighLevelAgent(BaseAgent):
    """
    High-Level Agent in the Hierarchical learning framework
    This agent selects the bay for container placement
    Currently it receives as input the full observation from environment (maybe changed later)
    
    Attributes:
        level: AgentLevel.HIGH_LEVEL (inherited from BaseAgent)
        verbosity: Verbosity level for debug output (0: silent, 1: verbose)
        vessel_shape: Tuple of (num_bays, num_rows, num_tiers) for vessel
        yard_shape: Tuple of (num_bays, num_rows, num_tiers) for yard
        policy_type: Type of policy ("rule_based", "rule_based_grouped", or "random")
        num_slot_attrs: Number of attributes per slot in state representation
        episode_history: List of episode records with rewards and info
    """

    def __init__(
        self, vessel_shape: tuple, yard_shape: tuple, num_slot_attrs: int, 
        policy_type: str = "rule_based", verbosity: int = 0
    ) -> None:
        super().__init__(AgentLevel.HIGH_LEVEL, verbosity=verbosity)

        # Initialize parameters from environment
        self.vessel_shape = vessel_shape
        self.yard_shape = yard_shape
        self.policy_type = policy_type
        self.num_slot_attrs = num_slot_attrs
        self.episode_history = []

        # Getting total possible slots in vessel and yard (even bays would be ignored later using valid actions only)
        self.num_vessel_bay = vessel_shape[0] // 2 + vessel_shape[0]
        self.num_yard_bay = yard_shape[0] // 2 + yard_shape[0]

        self.total_vessel_coords = (
            self.num_vessel_bay * vessel_shape[1] * vessel_shape[2]
        )
        self.total_yard_coords = self.num_yard_bay * yard_shape[1] * yard_shape[2]

    def get_action(self, observation: dict | np.ndarray, valid_actions: np.ndarray) -> int | None:
        """
        Select a bay for container placement based on the selected policy
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            
        Returns:
            Selected bay number or None if no valid action
        """
        # Select policy based on type
        if self.policy_type == "rule_based":
            return self._rule_based_policy(observation, valid_actions)
        elif self.policy_type == "rule_based_grouped":
            return self._rule_based_grouped_policy(observation, valid_actions)
        elif self.policy_type == "random":
            return self._random_policy(observation, valid_actions)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

    def _random_policy(self, observation: dict | np.ndarray, valid_actions: np.ndarray) -> int | None:
        """
        Policy selects a random bay from valid actions
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            
        Returns:
            Randomly selected bay number or None if no valid actions
        """
        if len(valid_actions) == 0:
            return None

        # Get parsed yard state from observation
        yard_state = self._parse_yard_state(observation)

        # Extarct only valid bays using valid actions
        bays_in_valid_actions = yard_state[valid_actions, StateIds.BAY.value]

        # Fallback
        if len(bays_in_valid_actions) == 0:
            return None

        return np.random.choice(bays_in_valid_actions)

    def _rule_based_policy(self, observation: dict | np.ndarray, valid_actions: np.ndarray) -> int | None:
        """
        Select bay with heuristics for container grouping
        
        Implements three fallback rules:
        1. Select bay with stack having most same-group containers (not full)
        2. Select bay with most empty stacks
        3. Select any valid bay with available space
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            
        Returns:
            Selected bay number or None if no valid action
        """
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available for high-level agent. This should not happen during normal operation.")

        # Get parsed yard state and current container info from observation
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])

        # Get valid bays from valid actions
        valid_bays = set(yard_state[valid_actions, StateIds.BAY.value].astype(int))

        # Get max tier number possible
        max_tier = int(yard_state[:, StateIds.TIER.value].max())

        # Build bay,stack info: {(bay, row): {occupied_count, same_group_count}}
        # Build bay info: {bay: {occupied_count}}
        # Getting number of same group containers and total occupied containers in each stack (bay,row)
        stacks_info = {}
        bays_info = {}

        for idx in range(yard_state.shape[0]):
            bay = int(yard_state[idx, StateIds.BAY.value])
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])

            stack_key = (bay, row)

            if stack_key not in stacks_info:
                stacks_info[stack_key] = {"occupied": 0, "same_group": 0}

            if bay not in bays_info:
                bays_info[bay] = {"occupied": 0}

            if is_occupied:
                bays_info[bay]["occupied"] += 1
                stacks_info[stack_key]["occupied"] += 1

                if group == container_group:
                    stacks_info[stack_key]["same_group"] += 1

        best_bay = None
        best_bay_score = 0

        # Rule 1 : Select bay with stack having most same-group containers (not full)
        for (bay, row), stack_data in stacks_info.items():
            is_full = stack_data["occupied"] == max_tier

            if (
                not is_full
                and stack_data["same_group"] > best_bay_score
                and bay in valid_bays
            ):
                best_bay_score = stack_data["same_group"]
                best_bay = bay

        # Rule 2 : If no bay found in Rule 1, select first bay with available empty stack
        # This is triggered if no stack having same-group containers is found or if all stacks with same-group containers are full
        if best_bay is None:
            for (bay, row), stack_data in stacks_info.items():
                is_full = stack_data["occupied"] == max_tier
                is_empty = stack_data["occupied"] == 0
                if is_empty and bay in valid_bays:
                    best_bay = bay
                    break

        # Rule 3 : If no bay found in Rule 2, select any valid bay with available space
        if best_bay is None and len(valid_bays) > 0:
            for bay, bay_data in bays_info.items():
                if bay in valid_bays and bay_data["occupied"] < (
                    self.yard_shape[StateIds.ROW.value] * max_tier
                ):
                    best_bay = bay
                    break

        return best_bay

    def _rule_based_grouped_policy(self, observation: dict | np.ndarray, valid_actions: np.ndarray) -> int | None:
        """
        Select bay with stacks having most similar count of same-group containers (not full)
        Uses overall occupancy and grouping within bays to select best bay
        1. Select bay with stack having most same-group containers (not full)
        2. If no bay found in Rule 1, select bay with most same-group containers
        3. If no bay found in Rule 2, select bay with least containers overall
        """

        # Fallback for no valid actions
        if len(valid_actions) == 0:
            return None

        # Get parsed yard state and current container info from observation
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])

        # Get valid bays from valid actions
        valid_bays = set(yard_state[valid_actions, StateIds.BAY.value].astype(int))

        # Get max tier number possible
        max_tier = int(yard_state[:, StateIds.TIER.value].max())

        # Build stack info: {(bay, row): {occupied_count, same_group_count, different_group_count}}
        # Build bay info: {bay: {occupied_count, same_group_count, different_group_count}}
        # Getting number of same group containers, different group containers and total occupied containers in each stack (bay,row) and each bay

        stacks_info = {}
        bay_info = {}

        for idx in range(yard_state.shape[0]):
            bay = int(yard_state[idx, StateIds.BAY.value])
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])

            stack_key = (bay, row)

            if stack_key not in stacks_info:
                stacks_info[stack_key] = {
                    "occupied": 0,
                    "same_group": 0,
                    "different_group": 0,
                }
            if bay not in bay_info:
                bay_info[bay] = {"occupied": 0, "same_group": 0, "different_group": 0}

            if is_occupied:
                stacks_info[stack_key]["occupied"] += 1
                bay_info[bay]["occupied"] += 1
                if group == container_group:
                    stacks_info[stack_key]["same_group"] += 1
                    bay_info[bay]["same_group"] += 1
                else:
                    stacks_info[stack_key]["different_group"] += 1
                    bay_info[bay]["different_group"] += 1

        best_bay = None
        best_bay_score = 0
        # Rule 1 : Select bay with stack having most same-group containers (not full)
        for (bay, row), stack_data in stacks_info.items():
            is_full = stack_data["occupied"] == max_tier

            if (
                not is_full
                and stack_data["same_group"] > best_bay_score
                and bay in valid_bays
            ):
                best_bay_score = stack_data["same_group"]
                best_bay = bay

        # Rule 2 : If no bay found in Rule 1, select bay with most same-group containers overall
        # If no stack has same-group containers or if all stacks with same-group containers are full
        # then select bay with mostt empty available slots
        bay_with_most_similar_containers = None

        bay_with_least_containers = None
        least_containers_per_bay_count = np.inf

        if best_bay is None:
            for bay, bay_data in bay_info.items():
                # Getting bay with most same-group containers
                if bay in valid_bays and bay_data["same_group"] > best_bay_score:
                    best_bay_score = bay_data["same_group"]
                    bay_with_most_similar_containers = bay

                # Getting bay with least containers as fallback
                if (
                    bay_data["occupied"] < least_containers_per_bay_count
                    and bay in valid_bays
                ):
                    least_containers_per_bay_count = bay_data["occupied"]
                    bay_with_least_containers = bay

            best_bay = (
                bay_with_most_similar_containers
                if bay_with_most_similar_containers is not None
                else bay_with_least_containers
            )

        return best_bay

    def _parse_yard_state(self, observation: dict | np.ndarray) -> np.ndarray:
        """
        Extract yard state from observation.
        Handles both array and dictionary observation types.
        
        Args:
            observation: Environment observation (dict or flattened array)
            
        Returns:
            Yard state array in shape (total_yard_coords, num_slot_attrs)
        """
        if isinstance(observation, dict):
            # Dictionary observation type - yard_state is already provided
            return observation["yard_state"].astype(int)
        else:
            # Array observation type - need to parse from flattened array
            observation = observation.astype(int)
            vessel_end = self.total_vessel_coords * self.num_slot_attrs
            yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

            yard_state = observation[vessel_end:yard_end].reshape(
                self.total_yard_coords, self.num_slot_attrs
            )
            return yard_state

    def _parse_current_container(self, observation: dict | np.ndarray) -> np.ndarray:
        """
        Extract current container state from observation.
        Handles both array and dictionary observation types.
        
        Args:
            observation: Environment observation (dict or flattened array)
            
        Returns:
            Current container state array in shape (num_slot_attrs,)
        """
        if isinstance(observation, dict):
            # Dictionary observation type - current_container is already provided
            return observation["current_container"].astype(int)
        else:
            # Array observation type - need to parse from flattened array
            observation = observation.astype(int)
            vessel_end = self.total_vessel_coords * self.num_slot_attrs
            yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

            current_container = observation[yard_end : yard_end + self.num_slot_attrs]
            return current_container

    def update_agent(self, reward: float, info: dict | None = None) -> None:
        """
        Update agent based on feedback signal
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        self.episode_history.append({"reward": reward, "info": info})


class LowLevelAgent(BaseAgent):
    """
    Low-Level Agent in the Hierarchical learning framework
    This agent selects the specific slot (row) within the chosen bay for container placement
    Currently it receives as input the full observation from environment (maybe changed later)
    
    Attributes:
        level: AgentLevel.LOW_LEVEL (inherited from BaseAgent)
        verbosity: Verbosity level for debug output (0: silent, 1: verbose)
        vessel_shape: Tuple of (num_bays, num_rows, num_tiers) for vessel
        yard_shape: Tuple of (num_bays, num_rows, num_tiers) for yard
        policy_type: Type of policy ("rule_based", "rule_based_grouped", or "random")
        num_slot_attrs: Number of attributes per slot in state representation
        episode_history: List of episode records with rewards and info
    """

    def __init__(
        self, vessel_shape: tuple, yard_shape: tuple, num_slot_attrs: int, 
        policy_type: str = "rule_based", verbosity: int = 0
    ) -> None:
        super().__init__(AgentLevel.LOW_LEVEL, verbosity=verbosity)

        # Initialize parameters from environment
        self.vessel_shape = vessel_shape
        self.yard_shape = yard_shape
        self.policy_type = policy_type
        self.num_slot_attrs = num_slot_attrs
        self.episode_history = []

        # Getting total possible slots in vessel and yard (even bays would be ignored later using valid actions only)
        self.num_vessel_bay = vessel_shape[0] // 2 + vessel_shape[0]
        self.num_yard_bay = yard_shape[0] // 2 + yard_shape[0]

        self.total_vessel_coords = (
            self.num_vessel_bay * vessel_shape[1] * vessel_shape[2]
        )
        self.total_yard_coords = self.num_yard_bay * yard_shape[1] * yard_shape[2]

    def get_action(self, observation: dict | np.ndarray, valid_actions: np.ndarray, selected_bay: int) -> int | None:
        """
        Select a slot (row) within the selected bay
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            selected_bay: Bay number selected by high-level agent
            
        Returns:
            Selected slot index or None if no valid action
        """
        # Select policy based on type
        if self.policy_type == "rule_based":
            return self._rule_based_policy(observation, valid_actions, selected_bay)
        elif self.policy_type == "rule_based_grouped":
            return self._rule_based_grouped_policy(
                observation, valid_actions, selected_bay
            )
        elif self.policy_type == "random":
            return self._random_policy(observation, valid_actions, selected_bay)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

    def _random_policy(self, observation: dict | np.ndarray, valid_actions: np.ndarray, selected_bay: int) -> int | None:
        """
        Policy selects a random valid slot (row) within the selected bay
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            selected_bay: Bay number selected by high-level agent
            
        Returns:
            Selected slot index or None if no valid actions in bay
        """
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            return None

        # Get parsed yard state from observation
        yard_state = self._parse_yard_state(observation)

        # Selecting only rows that are valid and within selected bay
        bay_mask = yard_state[valid_actions, StateIds.BAY.value] == selected_bay
        bay_valid_actions = np.array(valid_actions)[bay_mask]

        # Fallback for no valid actions in selected bay
        if len(bay_valid_actions) == 0:
            return None
        else:
            return np.random.choice(bay_valid_actions)

    def _rule_based_policy(self, observation: dict | np.ndarray, valid_actions: np.ndarray, selected_bay: int) -> int | None:
        """
        Policy selects the best slot (row) within the selected bay based on count of same group containers
        
        Implements five fallback rules for optimal placement within selected bay.
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            selected_bay: Bay number selected by high-level agent
            
        Returns:
            Selected slot index or None if no valid action
        """

        # Fallback for no valid actions
        if len(valid_actions) == 0:
            return None

        # Get parsed yard state and current container info from observation
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])

        # Mask to get only slots in selected bay
        bay_mask = yard_state[:, StateIds.BAY.value] == selected_bay
        bay_indices = np.where(bay_mask)[0]

        # Get max number of tiers possible
        max_tier = int(yard_state[:, StateIds.TIER.value].max())

        # Build stack info: {(row): {occupied_count, same_group_count}}
        # Counting number of same-group containers and occupied slots in each stack (row) within the selected bay
        stacks_info = {}

        for idx in bay_indices:
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])

            if row not in stacks_info:
                stacks_info[row] = {"occupied": 0, "same_group": 0}

            if is_occupied:
                stacks_info[row]["occupied"] += 1
                if group == container_group:
                    stacks_info[row]["same_group"] += 1

        best_action = None
        fallback_best_action = None
        best_stack_score = 0

        # Rule 1 : Select stack (row) with most same-group containers (not full)
        for row, stack_data in stacks_info.items():
            is_full = stack_data["occupied"] == max_tier
            is_empty = stack_data["occupied"] == 0

            if not is_full and stack_data["same_group"] > best_stack_score:
                best_stack_score = stack_data["same_group"]
                best_action = row

        # Rule 2 : If no stack found in Rule 1, select first fully empty stack
        if best_action is None:
            for row, stack_data in stacks_info.items():
                is_full = stack_data["occupied"] == max_tier
                is_empty = stack_data["occupied"] == 0
                if is_empty:
                    best_action = row
                    break
                if not is_full:
                    fallback_best_action = row

        # Rule 3 : If no stack found in Rule 2, select any available stack (not full)
        if best_action is None and fallback_best_action is not None:
            best_action = fallback_best_action

        if self.verbosity >= 1:
            print("Best action (row):", best_action)

        # Get valid action indices (row number of yard_state) for the selected best action (row) within the selected bay
        valid_action_mask = (yard_state[:, StateIds.BAY.value] == selected_bay) & (
            yard_state[:, StateIds.ROW.value] == best_action
        )
        valid_action_indices = np.where(valid_action_mask)[0]

        valid_action_indices = [
            int(action) for action in valid_action_indices if action in valid_actions
        ]

        return valid_action_indices[0]

    def _rule_based_grouped_policy(self, observation: dict | np.ndarray, valid_actions: np.ndarray, selected_bay: int) -> int | None:
        """
        Policy selects the best slot (row) within the selected bay with improved grouping heuristics
        
        Objective is to group similar containers together while ensuring spacing between different groups.
        Implements multiple rules for optimal placement within selected bay.
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            selected_bay: Bay number selected by high-level agent
            
        Returns:
            Selected slot index or None if no valid action
        """
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            return None

        # Get parsed yard state and current container info from observation
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])

        # Mask to get only slots in selected bay
        bay_mask = yard_state[:, StateIds.BAY.value] == selected_bay
        bay_indices = np.where(bay_mask)[0]

        # Get max number of tiers possible
        max_tier = int(yard_state[:, StateIds.TIER.value].max())

        # Build stack info: {(row): {occupied_count, same_group_count, different_group_count}}
        # Counting number of same-group containers, different-group containers and occupied slots in each stack (row) within the selected bay
        # Also counting total occupied slots in the bay
        stacks_info = {}
        total_occupied_in_bay = 0

        for idx in bay_indices:
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])

            if row not in stacks_info:
                stacks_info[row] = {
                    "occupied": 0,
                    "same_group": 0,
                    "different_group": 0,
                }

            if is_occupied:
                stacks_info[row]["occupied"] += 1
                if group == container_group:
                    stacks_info[row]["same_group"] += 1
                else:
                    stacks_info[row]["different_group"] += 1
                total_occupied_in_bay += 1

        # Variables to store best action (row) within selected bay
        best_action = None
        best_stack_score = 0

        # Variables to store related metadata for further rules
        stack_with_most_similar_containers = None
        count_of_most_similar_containers_in_stack = 0
        empty_rows = []

        # Rule 1 : Select stack (row) with most same-group containers (not full)
        for row, stack_data in stacks_info.items():
            is_full = stack_data["occupied"] == max_tier
            is_empty = stack_data["occupied"] == 0

            # Getting stack with most same-group containers (not full)
            if stack_data["same_group"] > best_stack_score and not is_full:
                best_action = (selected_bay, row)
                best_stack_score = stack_data["same_group"]

            # Getting stack with most same-group containers (even if they are full)
            if stack_data["same_group"] > count_of_most_similar_containers_in_stack:
                count_of_most_similar_containers_in_stack = stack_data["same_group"]
                stack_with_most_similar_containers = row

            # Collect list of empty rows in the selected bay
            if is_empty:
                empty_rows.append(row)

        # Rule 2 : If no stack found in Rule 1 and selected bay is completely empty, select first empty stack
        if best_action is None:
            if total_occupied_in_bay == 0:
                best_action = (selected_bay, empty_rows[0])

        # Rule 3 : If no stack found in Rule 2, select first empty stack closest to stack with most same-group containers
        row_list = list(stacks_info.keys())

        if stack_with_most_similar_containers is not None:
            # Using absolute difference to sort rows based on proximity to stack with most similar containers
            row_list = sorted(
                row_list, key=lambda x: abs(x - stack_with_most_similar_containers)
            )

        if best_action is None and stack_with_most_similar_containers is not None:
            for row in row_list:
                stack_data = stacks_info[row]
                is_full = stack_data["occupied"] == max_tier
                is_empty = stack_data["occupied"] == 0

                # Selecting first empty stack closest to stack with most similar containers
                if not is_full and is_empty:
                    best_action = (selected_bay, row)
                    break

        # Rule 4 : If no stack found in Rule 3 (triggered when no same group containers present in selected bay
        # and multiple empty stacks available in selected bay)
        # Then select last empty stack among consecutive empty stacks (this ensures spacing between different groups)
        # Note other options for selceting empty stacks like first empty stack or random empty stack were also considered
        # but selecting last empty stack in consecutive sequence gave better spacing results in tests
        if best_action is None:
            if len(empty_rows) > 0:
                best_action = (selected_bay, self._pick_last_sorted(empty_rows))

        # Rule 5 : If no stack found in Rule 4 (triggered when no empty stacks available in selected bay)
        # select any available stack that is not full
        # This is the last fallback option
        if best_action is None:
            for row in row_list:
                if stacks_info[row]["occupied"] < max_tier:
                    best_action = (selected_bay, row)
                    break

        if self.verbosity >= 1:
            print("Best action (bay,row):", best_action)

        # Get valid action indices (row number of yard_state) for the selected best action (row) within the selected bay
        valid_action_mask = (yard_state[:, StateIds.BAY.value] == best_action[0]) & (
            yard_state[:, StateIds.ROW.value] == best_action[1]
        )
        valid_action_indices = np.where(valid_action_mask)[0]

        valid_action_indices = [
            int(action) for action in valid_action_indices if action in valid_actions
        ]

        return valid_action_indices[0]

    def _score_based_policy(self, observation, valid_actions, selected_bay):
        """
        Not being used currently but kept for future reference
        Policy selects the best slot (row) within the selected bay based on scoring function
        """

        if len(valid_actions) == 0:
            return None

        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)

        container_group = int(current_container[StateIds.GROUP.value])

        # Filter valid actions to only those in the selected bay
        valid_bay_actions = []
        for action in valid_actions:
            if yard_state[action, StateIds.BAY.value] == selected_bay:
                valid_bay_actions.append(action)

        # Get best action based on scoring function
        if len(valid_bay_actions) == 0:
            return None
        else:
            max_action_score = -np.inf
            best_action = None
            for action in valid_bay_actions:
                group_score = self._score_slots(
                    action, yard_state, current_container, container_group, selected_bay
                )

                if group_score > max_action_score:
                    max_action_score = group_score
                    best_action = action

            return best_action

    def _score_slots(
        self, action, yard_state, current_container, container_group, selected_bay
    ):
        """
        Scoring function to evaluate a slot (row) within a bay for container placement
        This is used by _score_based_policy() method (curently not in use)
        The scoring is based on:
        1. Occupying completely empty stacks is penalized (-0.5)
        2. Number of same-group containers in the same stack (bay,row) with weightage +/-1
        3. Number of same-group containers in other stacks in the same bay with weightage (+/-0.25)
        4. Number of same-group containers in adjacent stacks (bay, row+-1) with weightage (+/-0.5)
        """

        group_score = 0
        bay = yard_state[action, StateIds.BAY.value]
        row = yard_state[action, StateIds.ROW.value]
        tier = yard_state[action, StateIds.TIER.value]

        stack_mask = (yard_state[:, StateIds.BAY.value] == bay) & (
            yard_state[:, StateIds.ROW.value] == row
        )
        stack_indices = np.where(stack_mask)[0]

        occupied_slots_mask = yard_state[stack_indices, StateIds.IS_OCCUPIED.value] == 1
        occupied_slots = stack_indices[occupied_slots_mask]

        # Penalizing completely empty stacks being occupied by -0.5
        if len(occupied_slots) == 0:
            group_score -= 0.5

        # Scoring based on same-group and different-group containers in the same stack (bay,row) with weightage +/-1
        if len(occupied_slots) > 0:
            stack_container_groups = yard_state[occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(stack_container_groups == container_group)
            diff_group_count = np.sum(stack_container_groups != container_group)
            group_score += same_group_count - diff_group_count

        # Getting adjacent rows
        adjacent_rows = []
        if row - 1 > 0:
            adjacent_rows.append(row - 1)
        if row + 1 <= self.yard_shape[1]:
            adjacent_rows.append(row + 1)

        # Scoring based on same-group and different-group containers in other stacks in the same bay with weightage +/-0.25
        same_bay_mask = (
            (yard_state[:, StateIds.BAY.value] == bay)
            & (yard_state[:, StateIds.ROW.value] != row)
            & ~np.isin(yard_state[:, StateIds.ROW.value], adjacent_rows)
        )
        same_bay_indices = np.where(same_bay_mask)[0]

        same_bay_occupied_slots_mask = (
            yard_state[same_bay_indices, StateIds.IS_OCCUPIED.value] == 1
        )
        same_bay_occupied_slots = same_bay_indices[same_bay_occupied_slots_mask]

        if len(same_bay_occupied_slots) > 0:
            same_bay_container_groups = yard_state[
                same_bay_occupied_slots, StateIds.GROUP.value
            ]
            same_group_count = np.sum(same_bay_container_groups == container_group)
            diff_group_count = np.sum(same_bay_container_groups != container_group)
            group_score += (same_group_count - diff_group_count) * 0.25

        # Scoring based on same-group and different-group containers in adjacent stacks (bay, row+-1) with weightage +/-0.5
        adj_stack_mask = (yard_state[:, StateIds.BAY.value] == bay) & np.isin(
            yard_state[:, StateIds.ROW.value], adjacent_rows
        )
        adj_stack_indices = np.where(adj_stack_mask)[0]

        adj_occupied_slots_mask = (
            yard_state[adj_stack_indices, StateIds.IS_OCCUPIED.value] == 1
        )
        adj_occupied_slots = adj_stack_indices[adj_occupied_slots_mask]

        if len(adj_occupied_slots) > 0:
            adj_stack_container_groups = yard_state[
                adj_occupied_slots, StateIds.GROUP.value
            ]
            same_group_count = np.sum(adj_stack_container_groups == container_group)
            diff_group_count = np.sum(adj_stack_container_groups != container_group)
            group_score += (same_group_count - diff_group_count) * 0.5

        return group_score

    def _pick_last_sorted(self, nums: list[int]) -> int:
        """
        Helper function to pick the last number in the longest consecutive sequence of sorted integers
        
        Used to select the last row in empty rows that are consecutive.
        Used in _rule_based_grouped_policy() method to choose the last empty stack among consecutive empty stacks.
        Helps ensure spacing when placing containers in bays that have multiple empty stacks available.
        
        Args:
            nums: List of sorted integers (row indices)
            
        Returns:
            Last integer in the longest consecutive sequence
        """
        best_len = 1
        curr_len = 1
        best_last = nums[0]
        curr_last = nums[0]

        for i in range(1, len(nums)):
            # Check if current number is consecutive to previous to get consecutive rows in sequence
            if nums[i] == nums[i - 1] + 1:
                curr_len += 1
            else:
                curr_len = 1

            # Last number in connected consecutive sequence
            curr_last = nums[i]

            # Update longest sequence if current sequence is longer
            # and updates last number in that sequence as selected row
            if curr_len > best_len:
                best_len = curr_len
                best_last = curr_last

        return best_last

    def _parse_yard_state(self, observation: dict | np.ndarray) -> np.ndarray:
        """
        Extract yard state from observation.
        Handles both array and dictionary observation types.
        
        Args:
            observation: Environment observation (dict or flattened array)
            
        Returns:
            Yard state array in shape (total_yard_coords, num_slot_attrs)
        """
        if isinstance(observation, dict):
            # Dictionary observation type - yard_state is already provided
            return observation["yard_state"].astype(int)
        else:
            # Array observation type - need to parse from flattened array
            observation = observation.astype(int)
            vessel_end = self.total_vessel_coords * self.num_slot_attrs
            yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

            yard_state = observation[vessel_end:yard_end].reshape(
                self.total_yard_coords, self.num_slot_attrs
            )
            return yard_state

    def _parse_current_container(self, observation: dict | np.ndarray) -> np.ndarray:
        """
        Extract current container state from observation.
        Handles both array and dictionary observation types.
        
        Args:
            observation: Environment observation (dict or flattened array)
            
        Returns:
            Current container state array in shape (num_slot_attrs,)
        """
        if isinstance(observation, dict):
            # Dictionary observation type - current_container is already provided
            return observation["current_container"].astype(int)
        else:
            # Array observation type - need to parse from flattened array
            observation = observation.astype(int)
            vessel_end = self.total_vessel_coords * self.num_slot_attrs
            yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

            current_container = observation[yard_end : yard_end + self.num_slot_attrs]
            return current_container

    def update_agent(self, reward: float, info: dict | None = None) -> None:
        """
        Update agent based on feedback signal
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        self.episode_history.append({"reward": reward, "info": info})


class HierarchicalAgent:
    """
    Wrapper class for Hierarchical Agent combining High-Level and Low-Level agents
    
    The high-level agent selects a bay, and the low-level agent selects a slot
    within that bay for container placement.
    
    Attributes:
        high_level_agent: HighLevelAgent instance for bay selection
        low_level_agent: LowLevelAgent instance for slot selection
    """

    def __init__(
        self,
        vessel_shape: tuple,
        yard_shape: tuple,
        num_slot_attrs: int,
        high_level_policy_type: str = "rule_based",
        low_level_policy_type: str = "rule_based",
        verbosity: int = 0
    ):

        self.high_level_agent = HighLevelAgent(
            vessel_shape, yard_shape, num_slot_attrs, high_level_policy_type, verbosity=verbosity
        )

        self.low_level_agent = LowLevelAgent(
            vessel_shape, yard_shape, num_slot_attrs, low_level_policy_type, verbosity=verbosity
        )

    def get_action(self, observation: dict | np.ndarray, valid_actions: np.ndarray) -> tuple[int | None, dict]:
        """
        Get action from hierarchical agent (bay and slot)
        
        Args:
            observation: Environment observation (dict or flattened array)
            valid_actions: Array of valid action indices
            
        Returns:
            Tuple of (selected_slot, action_info) where action_info contains bay and slot details
        """

        if len(valid_actions) == 0:
            return None, {"error": "No valid actions available"}

        # High-Level Agent selects bay
        selected_bay = self.high_level_agent.get_action(observation, valid_actions)

        if selected_bay is None:
            return valid_actions[0], {
                "selected_bay": None,
                "selected_slot": valid_actions[0],
            }

        # Low-Level Agent selects slot within the chosen bay
        selected_slot = self.low_level_agent.get_action(
            observation, valid_actions, selected_bay
        )

        action_info = {"selected_bay": selected_bay, "selected_slot": selected_slot}

        return selected_slot, action_info

    def update_agent(self, reward: float, info: dict | None = None) -> None:
        """
        Update both high-level and low-level agents with reward and info
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        self.high_level_agent.update_agent(reward, info)
        self.low_level_agent.update_agent(reward, info)
