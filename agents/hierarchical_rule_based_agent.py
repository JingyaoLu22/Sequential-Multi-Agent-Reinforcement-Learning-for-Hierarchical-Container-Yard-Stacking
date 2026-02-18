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
        level: The hierarchical level of the agent (HIGH_LEVEL or LOW_LEVEL)
    """

    def __init__(self, level: AgentLevel) -> None:
        self.level = level

    @abstractmethod
    def get_action(self, observation: dict, valid_actions: np.ndarray) -> int:
        """
        Return action from agent based on observation
        """
        pass

    @abstractmethod
    def update_agent(self, reward: float) -> None:
        """
        Update agent based on feedback (may need later for RL agents)
        """
        pass

class HighLevelAgent(BaseAgent):
    """
    High-Level Agent in the Hierarchical learning framework
    This agent selects the bay for container placement
    Currently it receives as input the full observation from environment (maybe changed later)

    Attributes:
        vessel_shape: Tuple defining vessel dimensions (bays, rows, tiers)
        yard_shape: Tuple defining yard dimensions (bays, rows, tiers)
        policy_type: Type of policy for decision making ("rule_based", "rule_based_grouped", "random", or "rl_agent")
        num_slot_attrs: Number of attributes per slot (5: bay, row, tier, is_occupied, group)
        episode_history: List of episode experiences for learning
    """

    def __init__(self, vessel_shape: tuple[int, int, int], yard_shape: tuple[int, int, int], num_slot_attrs: int, policy_type: str = "rule_based") -> None:
        super().__init__(AgentLevel.HIGH_LEVEL)

        # Initialize parameters from environment
        self.vessel_shape = vessel_shape
        self.yard_shape = yard_shape
        self.policy_type = policy_type
        self.num_slot_attrs = num_slot_attrs
        self.episode_history = []

        # Getting total possible slots in vessel and yard (even bays would be ignored later using valid actions only)
        self.num_vessel_bay = vessel_shape[0]//2 + vessel_shape[0]
        self.num_yard_bay = yard_shape[0]//2 + yard_shape[0]

        self.total_vessel_coords = self.num_vessel_bay * vessel_shape[1] * vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * yard_shape[1] * yard_shape[2]

    def _action_to_bay_row(self, action):
        """
        Convert action index (stack) to bay and row numbering
        Action index x = ((b-1)/2)*n + (r-1)
        where b is odd bay number (1,3,5,...), r is row number (1,2,...,n), n is num_rows
        
        Returns:
            tuple: (bay, row) where bay is odd bay number (1,3,5,...) and row is 1-indexed
        """
        n = self.yard_shape[StateIds.ROW.value]  # num_rows
        bay_idx = action // n  # which physical bay (0-indexed)
        row_idx = action % n   # which row (0-indexed)
        
        # Convert to actual bay number (1,3,5,7,...)
        bay = 2 * bay_idx + 1
        row = row_idx + 1
        
        return bay, row
    
    def _bay_row_to_action(self, bay, row):
        """
        Convert (bay, row) numbering pair to action index (stack)
        Action index x = ((b-1)/2)*n + (r-1)
        
        Args:
            bay: odd bay number (1,3,5,7,...)
            row: row number (1,2,3,...)
        
        Returns:
            int: action index
        """
        n = self.yard_shape[StateIds.ROW.value]  # num_rows
        action = ((bay - 1) // 2) * n + (row - 1)
        return action
    
    def _get_valid_bays(self, valid_actions: list[int]) -> set:
        """
        Extract unique bay numbers from valid action indices
        
        Args:
            valid_actions: List or array of valid action indices
        
        Returns:
            Set of valid bay numbers
        """
        valid_bays = set()
        for action in valid_actions:
            bay, row = self._action_to_bay_row(action)
            valid_bays.add(bay)
        return valid_bays
    
    def _get_valid_bay_row_pairs(self, valid_actions: list[int]) -> list:
        """
        Extract all valid (bay, row) pairs from action indices
        
        Args:
            valid_actions: Array or list of valid action indices
        
        Returns:
            List of (bay, row) tuples
        """
        bay_row_pairs = []
        for action in valid_actions:
            bay, row = self._action_to_bay_row(action)
            bay_row_pairs.append((bay, row))
        return bay_row_pairs


    def get_action(self, observation: dict, valid_actions: list[int]) -> int:
        """
        Select an action (bay) based on the specified policy and observation
        
        Args:
            observation: Current environment observation containing yard state and current container
            valid_actions: List of valid action indices
        
        Returns:
            Selected bay number
        """
        # Select policy based on type
        if self.policy_type == "rule_based":
            return self._rule_based_policy(observation, valid_actions)
        elif self.policy_type == "rule_based_grouped":
            return self._rule_based_grouped_policy(observation, valid_actions)
        elif self.policy_type == "random":
            return self._random_policy(observation,valid_actions)
        elif self.policy_type == "rl_agent":
            # Placeholder for future RL-based high-level agent
            pass
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def _random_policy(self, observation: dict, valid_actions: list[int]) -> int:
        """
        Policy that selects a random bay from valid action bays
        
        Args:
            observation: Current environment observation
            valid_actions: List of valid action indices indicating each stack
        
        Returns:
            Randomly selected bay number
        """
        if len(valid_actions) == 0:
           raise RuntimeError("No valid actions available for the high level agent.")
        
        # Extract valid bays from action indices
        valid_bays = self._get_valid_bays(valid_actions)
        
        # Fallback
        if len(valid_bays) == 0:
            raise RuntimeError("No valid actions available for the high level agent.")
        
        return np.random.choice(list(valid_bays))
        
    def _rule_based_policy(self, observation: dict, valid_actions: list[int]) -> int:
        """
        Select bay with stacks having most similar count of same-group containers (not full)
        Does not account for grouping similar containers in nearby stacks (bay,row)
        
        Args:
            observation: Current environment observation containing yard state and current container
            valid_actions: Array of valid action indices
        
        Returns:
            Selected bay number
        """
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the high level agent.")
        
        # Get parsed yard state and current container info from observation
        yard_state = observation['yard_state']
        current_container = observation['current_container']
        container_group = int(current_container[StateIds.GROUP.value])
        
        # Get valid bays from action indices
        valid_bays = self._get_valid_bays(valid_actions)
        
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
                stacks_info[stack_key] = {'occupied': 0, 'same_group': 0}
            
            if bay not in bays_info:
                bays_info[bay] = {'occupied': 0}
            
            if is_occupied:
                bays_info[bay]['occupied'] += 1
                stacks_info[stack_key]['occupied'] += 1
                
                if group == container_group:
                    stacks_info[stack_key]['same_group'] += 1
        
        best_bay = None
        best_bay_score = 0
        
        # Rule 1 : Select bay with stack having most same-group containers (not full)
        for (bay, row), stack_data in stacks_info.items():
            is_full = stack_data['occupied'] == max_tier
            
            if not is_full and stack_data['same_group'] > best_bay_score and bay in valid_bays:
                best_bay_score = stack_data['same_group']
                best_bay = bay

        
        
        # Rule 2 : If no bay found in Rule 1, select first bay with available empty stack
        # This is triggered if no stack having same-group containers is found or if all stacks with same-group containers are full 
        if best_bay is None:
            for (bay,row), stack_data in stacks_info.items():
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                if is_empty and bay in valid_bays:
                    best_bay = bay
                    break

        # Rule 3 : If no bay found in Rule 2, select any valid bay with available space
        if best_bay is None and len(valid_bays) > 0:
            for bay, bay_data in bays_info.items():
                if bay in valid_bays and bay_data['occupied'] < (self.yard_shape[StateIds.ROW.value] * max_tier):
                    best_bay = bay
                    break 
        
        
        return best_bay
    
    def _rule_based_grouped_policy(self, observation: dict, valid_actions: list[int]) -> int:
        """
        Select bay with stacks having most similar count of same-group containers (not full)
        Uses overall occupancy and grouping within bays to select best bay
        1. Select bay with stack having most same-group containers (not full)
        2. If no bay found in Rule 1, select bay with most same-group containers
        3. If no bay found in Rule 2, select bay with least containers overall
        
        Args:
            observation: Current environment observation containing yard state and current container
            valid_actions: Array of valid action indices
        
        Returns:
            Selected bay number
        """

        # Fallback for no valid actions
        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the high level agent.")
        
        # Get parsed yard state and current container info from observation
        yard_state = observation['yard_state']
        current_container = observation['current_container']
        container_group = int(current_container[StateIds.GROUP.value])
        
        # Get valid bays from action indices
        valid_bays = self._get_valid_bays(valid_actions)
        
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
                stacks_info[stack_key] = {'occupied': 0, 'same_group': 0, 'different_group': 0}
            if bay not in bay_info:
                bay_info[bay] = {'occupied': 0, 'same_group': 0, 'different_group': 0}
            
            
            if is_occupied:
                stacks_info[stack_key]['occupied'] += 1
                bay_info[bay]['occupied'] += 1
                if group == container_group:
                    stacks_info[stack_key]['same_group'] += 1
                    bay_info[bay]['same_group'] += 1
                else:
                    stacks_info[stack_key]['different_group'] += 1
                    bay_info[bay]['different_group'] += 1
        
        best_bay = None
        best_bay_score = 0
        # Rule 1 : Select bay with stack having most same-group containers (not full)
        for (bay, row), stack_data in stacks_info.items():
            
            is_full = stack_data['occupied'] == max_tier
            
            if not is_full and stack_data['same_group'] > best_bay_score and bay in valid_bays:
                best_bay_score = stack_data['same_group']
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
                if bay in valid_bays and bay_data['same_group'] > best_bay_score:
                    best_bay_score = bay_data['same_group']
                    bay_with_most_similar_containers = bay
                
                # Getting bay with least containers as fallback
                if bay_data['occupied'] < least_containers_per_bay_count and bay in valid_bays:
                    least_containers_per_bay_count = bay_data['occupied']
                    bay_with_least_containers = bay
            
            best_bay = bay_with_most_similar_containers if bay_with_most_similar_containers is not None else bay_with_least_containers
        
        return best_bay
    
    def update_agent(self, reward: float, info: dict) -> None:
        """
        Update agent by storing episode experience
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        self.episode_history.append({
            "reward": reward,
            "info": info
        })

class LowLevelAgent(BaseAgent):
    """
    Low-Level Agent in the Hierarchical learning framework
    This agent selects the specific slot (row) within the chosen bay for container placement
    Currently it receives as input the full observation from environment (maybe changed later)
    
    Attributes:
        vessel_shape: Tuple defining vessel dimensions (bays, rows, tiers)
        yard_shape: Tuple defining yard dimensions (bays, rows, tiers)
        policy_type: Type of policy for decision making ("rule_based", "rule_based_grouped", "random", or "rl_agent")
        num_slot_attrs: Number of attributes per slot (5: bay, row, tier, is_occupied, group)
        episode_history: List of episode experiences for learning
    """

    def __init__(self, vessel_shape: tuple[int, int, int], yard_shape: tuple[int, int, int], num_slot_attrs: int, policy_type: str = "rule_based") -> None:
        super().__init__(AgentLevel.LOW_LEVEL)

        # Initialize parameters from environment
        self.vessel_shape = vessel_shape
        self.yard_shape = yard_shape
        self.policy_type = policy_type
        self.num_slot_attrs = num_slot_attrs
        self.episode_history = []

        # Getting total possible slots in vessel and yard (even bays would be ignored later using valid actions only)
        self.num_vessel_bay = vessel_shape[0]//2 + vessel_shape[0]
        self.num_yard_bay = yard_shape[0]//2 + yard_shape[0]

        self.total_vessel_coords = self.num_vessel_bay * vessel_shape[1] * vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * yard_shape[1] * yard_shape[2]

    def _action_to_bay_row(self, action):
        """
        Convert action index (stack) to bay and row numbering
        Action index x = ((b-1)/2)*n + (r-1)
        where b is odd bay number (1,3,5,...), r is row number (1,2,...,n), n is num_rows
        
        Returns:
            tuple: (bay, row) where bay is odd bay number (1,3,5,...) and row is 1-indexed
        """
        n = self.yard_shape[StateIds.ROW.value]  # num_rows
        bay_idx = action // n  # which physical bay (0-indexed)
        row_idx = action % n   # which row (0-indexed)
        
        # Convert to actual bay number (1,3,5,7,...)
        bay = 2 * bay_idx + 1
        row = row_idx + 1
        
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
        n = self.yard_shape[StateIds.ROW.value]  # num_rows
        action = ((bay - 1) // 2) * n + (row - 1)
        return action

    def get_action(self, observation: dict, valid_actions: list[int], selected_bay: int) -> int:
        """
        Select an action (row) within the selected bay based on the specified policy
        
        Args:
            observation: Current environment observation containing yard state and current container
            valid_actions: List of valid action indices
            selected_bay: Bay selected by the high-level agent
        
        Returns:
            Selected action (row) index
        """
        # Select policy based on type
        if self.policy_type == "rule_based":
            return self._rule_based_policy(observation, valid_actions, selected_bay)
        elif self.policy_type == "rule_based_grouped":
            return self._rule_based_grouped_policy(observation, valid_actions, selected_bay)
        elif self.policy_type == "random":
            return self._random_policy(observation,valid_actions, selected_bay)
        elif self.policy_type == "rl_agent":
            # Placeholder for future RL-based low-level agent
            pass
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
    def _random_policy(self, observation: dict, valid_actions: list[int], selected_bay: int) -> int:
        """
        Policy that selects a random valid action (stack) within the selected bay
        
        Args:
            observation: Current environment observation
            valid_actions: List of valid action indices
            selected_bay: Bay selected by the high-level agent
        
        Returns:
            Randomly selected action within the bay
        """
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the low level agent.")
        
        # Filter valid actions to only those in selected bay
        bay_valid_actions = []
        for action in valid_actions:
            bay, row = self._action_to_bay_row(action)
            if bay == selected_bay:
                bay_valid_actions.append(action)

        # Fallback for no valid actions in selected bay
        if len(bay_valid_actions) == 0:
            raise RuntimeError("No valid actions available in selected bay for the low level agent.")
        
        return np.random.choice(bay_valid_actions)

    def _rule_based_policy(self, observation: dict, valid_actions: list[int], selected_bay: int) -> int:
        """
        Policy that selects the best action (stack) within the selected bay based on count of same-group containers
        
        Args:
            observation: Current environment observation containing yard state and current container
            valid_actions: List of valid action indices
            selected_bay: Bay selected by the high-level agent
        
        Returns:
            Selected action (row) within the bay
        """
        
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the current state in low level agent.")
        
        # Get parsed yard state and current container info from observation
        yard_state = observation['yard_state']
        current_container = observation['current_container']
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
                stacks_info[row] = {'occupied': 0, 'same_group': 0}
            
            if is_occupied:
                stacks_info[row]['occupied'] += 1
                if group == container_group:
                    stacks_info[row]['same_group'] += 1
        
        best_row = None
        fallback_best_row = None
        best_stack_score = 0
        
        # Rule 1 : Select stack (row) with most same-group containers (not full)
        for row, stack_data in stacks_info.items():
            is_full = stack_data['occupied'] == max_tier
            is_empty = stack_data['occupied'] == 0

            
            if not is_full and stack_data['same_group'] > best_stack_score:
                best_stack_score = stack_data['same_group']
                best_row = row
        

        # Rule 2 : If no stack found in Rule 1, select first fully empty stack
        if best_row is None:
            for row, stack_data in stacks_info.items():
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                if is_empty:
                    best_row = row
                    break
                if not is_full :
                    fallback_best_row = row
        

        # Rule 3 : If no stack found in Rule 2, select any available stack (not full)
        if best_row is None and fallback_best_row is not None:
            best_row = fallback_best_row
        
        
        # Convert selected (bay, row) to action index
        if best_row is not None:
            return self._bay_row_to_action(selected_bay, best_row)
        
        # Fallback: return first valid action in selected bay
        for action in valid_actions:
            bay, row = self._action_to_bay_row(action)
            if bay == selected_bay:
                return action
        
        raise RuntimeError("No valid action found in selected bay")
    
    def _rule_based_grouped_policy(self, observation: dict, valid_actions: list[int], selected_bay: int) -> int:
        """
        Policy that selects the best slot (row) within the selected bay based on multiple rules
        Objective is to group similar containers together while ensuring spacing between different groups
        
        Rules applied in order:
        1. Select stack with most same-group containers (not full)
        2. If no stack found in Rule 1 and bay is completely empty, select first empty stack
        3. If no stack found in Rule 2, select first empty stack closest to stack with most same-group containers
        4. If no stack found in Rule 3, select last empty stack among consecutive empty stacks for spacing
        5. If no stack found in Rule 4, select any available stack that is not full
        
        Args:
            observation: Current environment observation containing yard state and current container
            valid_actions: List of valid action indices
            selected_bay: Bay selected by the high-level agent
        
        Returns:
            Selected action (row) within the bay
        """
        # Fallback for no valid actions
        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the low level agent.")
        
        # Get parsed yard state and current container info from observation
        yard_state = observation['yard_state']
        current_container = observation['current_container']
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
                stacks_info[row] = {'occupied': 0, 'same_group': 0, 'different_group': 0}
            
            if is_occupied:
                stacks_info[row]['occupied'] += 1
                if group == container_group:
                    stacks_info[row]['same_group'] += 1
                else:
                    stacks_info[row]['different_group'] += 1
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
            
            is_full = stack_data['occupied'] == max_tier
            is_empty = stack_data['occupied'] == 0

            # Getting stack with most same-group containers (not full)
            if stack_data['same_group'] > best_stack_score and not is_full:
                best_action = (selected_bay,row)
                best_stack_score = stack_data['same_group']
            
            # Getting stack with most same-group containers (even if they are full)
            if stack_data['same_group'] > count_of_most_similar_containers_in_stack:
                count_of_most_similar_containers_in_stack = stack_data['same_group']
                stack_with_most_similar_containers = row
            
            # Collect list of empty rows in the selected bay
            if is_empty:
                empty_rows.append(row)


        # Rule 2 : If no stack found in Rule 1 and selected bay is completely empty, select first empty stack
        if best_action is None :
            if total_occupied_in_bay == 0:
                best_action = (selected_bay, empty_rows[0])
                
        # Rule 3 : If no stack found in Rule 2, select first empty stack closest to stack with most same-group containers
        row_list = list(stacks_info.keys())
        
        if stack_with_most_similar_containers is not None:
                # Using absolute difference to sort rows based on proximity to stack with most similar containers
                row_list = sorted(row_list, key=lambda x: abs(x - stack_with_most_similar_containers))
        
        if best_action is None and stack_with_most_similar_containers is not None :
            
            for row in row_list:
                
                stack_data = stacks_info[row]
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                
                # Selecting first empty stack closest to stack with most similar containers
                if not is_full and is_empty:
                    best_action = (selected_bay,row)
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
                if stacks_info[row]['occupied'] < max_tier:
                    best_action = (selected_bay, row)
                    break

        if best_action is None:
            # Fallback: return first valid action in selected bay
            for action in valid_actions:
                bay, row = self._action_to_bay_row(action)
                if bay == selected_bay:
                    return action
            raise RuntimeError("No valid action found in selected bay")

        # Convert (bay, row) to action index
        return self._bay_row_to_action(best_action[0], best_action[1])


    
    def _score_based_policy(self, observation: dict, valid_actions: list[int], selected_bay: int) -> int:
        """
        Policy that selects the best action (stack) within the selected bay based on scoring function
        
        Note: Currently not being used but kept for future reference
        
        Args:
            observation: Current environment observation
            valid_actions: List of valid action indices
            selected_bay: Bay selected by the high-level agent
        
        Returns:
            Selected action
        """

        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the low level agent.")
        
        yard_state = observation['yard_state']
        current_container = observation['current_container']

        container_group = int(current_container[StateIds.GROUP.value])

        # Filter valid actions to only those in the selected bay
        valid_bay_actions = []
        for action in valid_actions:
            bay, row = self._action_to_bay_row(action)
            if bay == selected_bay:
                valid_bay_actions.append(action)
        


        # Get best action based on scoring function
        if len(valid_bay_actions) == 0:
            raise RuntimeError("No valid actions available for the low level agent.")
        else:
            max_action_score = -np.inf
            best_action = None
            for action in valid_bay_actions:
                bay, row = self._action_to_bay_row(action)
                group_score = self._score_slots(bay, row, yard_state, current_container, container_group, selected_bay)
                    
                if group_score > max_action_score:
                    max_action_score = group_score
                    best_action = action
            
            
            return best_action
        
    def _score_slots(self, bay: int, row: int, yard_state: np.ndarray, current_container: np.ndarray, container_group: int, selected_bay: int) -> float:
        """
        Scoring function to evaluate a stack (bay, row) within a bay for container placement
        Currently not in use but kept for future reference
        
        Scoring basis:
        1. Occupying completely empty stacks: -0.5
        2. Same-group containers in same stack: +/-1
        3. Same-group containers in other stacks in same bay: +/-0.25
        4. Same-group containers in adjacent stacks: +/-0.5
        
        Args:
            bay: Bay number
            row: Row number
            yard_state: Current yard state array
            current_container: Current container being placed
            container_group: Group of current container
            selected_bay: Selected bay from high-level agent
        
        Returns:
            Score value
        """

        group_score = 0
        
        stack_mask = (yard_state[:,StateIds.BAY.value] == bay) & (yard_state[:,StateIds.ROW.value] == row)
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
        if row - 1 >  0:
            adjacent_rows.append(row - 1)
        if row + 1 <= self.yard_shape[1]:
            adjacent_rows.append(row + 1)
        
        
        # Scoring based on same-group and different-group containers in other stacks in the same bay with weightage +/-0.25
        same_bay_mask = (yard_state[:,StateIds.BAY.value] == bay) & (yard_state[:,StateIds.ROW.value] != row) & ~np.isin(yard_state[:,StateIds.ROW.value],adjacent_rows)
        same_bay_indices = np.where(same_bay_mask)[0]

        same_bay_occupied_slots_mask = yard_state[same_bay_indices, StateIds.IS_OCCUPIED.value] == 1
        same_bay_occupied_slots = same_bay_indices[same_bay_occupied_slots_mask]

        if len(same_bay_occupied_slots) > 0:
            same_bay_container_groups = yard_state[same_bay_occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(same_bay_container_groups == container_group)
            diff_group_count = np.sum(same_bay_container_groups != container_group)
            group_score += (same_group_count - diff_group_count) * 0.25

        # Scoring based on same-group and different-group containers in adjacent stacks (bay, row+-1) with weightage +/-0.5
        adj_stack_mask = (yard_state[:,StateIds.BAY.value] == bay) & np.isin(yard_state[:, StateIds.ROW.value], adjacent_rows)
        adj_stack_indices = np.where(adj_stack_mask)[0]

        adj_occupied_slots_mask = yard_state[adj_stack_indices, StateIds.IS_OCCUPIED.value] == 1
        adj_occupied_slots = adj_stack_indices[adj_occupied_slots_mask]

        if len(adj_occupied_slots) > 0:
            adj_stack_container_groups = yard_state[adj_occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(adj_stack_container_groups == container_group)
            diff_group_count = np.sum(adj_stack_container_groups != container_group)
            group_score += (same_group_count - diff_group_count) * 0.5

        return group_score
    
    def _pick_last_sorted(self, nums: list) -> int:
        """
        Helper function to pick the last number in the longest consecutive sequence of sorted integers
        Used to select the last row in empty rows that are consecutive
        Used in _rule_based_grouped_policy() method to choose the last empty stack among consecutive empty stacks
        Helps ensure spacing when placing containers in bays that have multiple empty stacks available
        
        Args:
            nums: List of sorted integers
        
        Returns:
            Last number in the longest consecutive sequence
        """
        best_len = 1
        curr_len = 1
        best_last = nums[0]
        curr_last = nums[0]

        for i in range(1, len(nums)):
            # Check if current number is consecutive to previous to get consecutive rows in sequence
            if nums[i] == nums[i-1] + 1:
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
    
    def update_agent(self, reward: float, info: dict) -> None:
        """
        Update agent by storing episode experience
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        self.episode_history.append({
            "reward": reward,
            "info": info
        })

class HierarchicalAgent:
    """
    Wrapper class for Hierarchical Agent combining High-Level and Low-Level agents
    Coordinates decision-making between bay selection (high-level) and row selection (low-level)
    
    Attributes:
        high_level_agent: HighLevelAgent instance for bay selection
        low_level_agent: LowLevelAgent instance for row/stack selection within selected bay
    """

    def __init__(self, vessel_shape: tuple[int, int, int], yard_shape: tuple[int, int, int], num_slot_attrs: int,
                 high_level_policy_type: str = "rule_based",
                 low_level_policy_type: str = "rule_based") -> None:
        """
        Initialize the hierarchical agent with high-level and low-level components
        
        Args:
            vessel_shape: Tuple defining vessel dimensions (bays, rows, tiers)
            yard_shape: Tuple defining yard dimensions (bays, rows, tiers)
            num_slot_attrs: Number of attributes per slot (5: bay, row, tier, is_occupied, group)
            high_level_policy_type: Policy type for high-level agent
            low_level_policy_type: Policy type for low-level agent
        """
        
        self.high_level_agent = HighLevelAgent(vessel_shape,
                     yard_shape, num_slot_attrs, high_level_policy_type)
        
        self.low_level_agent = LowLevelAgent(vessel_shape,
                     yard_shape, num_slot_attrs, low_level_policy_type)
        
    def get_action(self, observation: dict, valid_actions: list[int]) -> tuple[int, dict]:
        """
        Get action from hierarchical agent by combining high-level and low-level decisions
        
        Args:
            observation: Current environment observation
            valid_actions: Array of valid action indices
        
        Returns:
            Tuple of (selected_slot_action, action_info_dict) where action_info_dict contains
            selected_bay and selected_slot information
        """

        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available for the low level agent.")
        
        # High-Level Agent selects bay
        selected_bay = self.high_level_agent.get_action(observation, valid_actions)

        if selected_bay is None:
            return valid_actions[0], {"selected_bay": None,
                                        "selected_slot": valid_actions[0]}
        
        # Low-Level Agent selects slot within the chosen bay
        selected_slot = self.low_level_agent.get_action(observation, valid_actions, 
                            selected_bay)
        
        action_info = {
            "selected_bay": selected_bay,
            "selected_slot": selected_slot
        }

        return selected_slot, action_info
    
    def update_agent(self, reward: float, info: dict) -> None:
        """
        Update both high-level and low-level agents with reward signal
        
        Args:
            reward: Reward signal from environment
            info: Additional information dictionary
        """
        self.high_level_agent.update_agent(reward, info)
        self.low_level_agent.update_agent(reward, info)

        
            


