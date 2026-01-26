from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class StateIds(Enum):
    BAY = 0
    ROW = 1
    TIER = 2
    IS_OCCUPIED = 3
    GROUP = 4

class AgentLevel(Enum):
    HIGH_LEVEL = 0
    LOW_LEVEL = 1

class BaseAgent(ABC):

    def __init__(self, level: AgentLevel):
        self.level = level

    @abstractmethod
    def get_action(self, observation, valid_actions):
        """
        Return action from agent based on observation
        """
        pass

    @abstractmethod
    def update_agent(self,reward):
        """
        Update agent based on feedback (may need later for RL agents)
        """
        pass

class HighLevelAgent(BaseAgent):

    def __init__(self, vessel_shape, yard_shape, num_slot_attrs, policy_type ="rule_based"):
        super().__init__(AgentLevel.HIGH_LEVEL)

        self.vessel_shape = vessel_shape
        self.yard_shape = yard_shape
        self.policy_type = policy_type
        self.num_slot_attrs = num_slot_attrs
        self.episode_history = []

        self.num_vessel_bay = vessel_shape[0]//2 + vessel_shape[0]
        self.num_yard_bay = yard_shape[0]//2 + yard_shape[0]

        self.total_vessel_coords = self.num_vessel_bay * vessel_shape[1] * vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * yard_shape[1] * yard_shape[2]


    def get_action(self, observation, valid_actions):
        if self.policy_type == "rule_based":
            return self._rule_based_policy(observation, valid_actions)
        elif self.policy_type == "rule_based_grouped":
            return self._rule_based_grouped_policy(observation, valid_actions)
        elif self.policy_type == "random":
            return self._random_policy(observation,valid_actions)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def _random_policy(self, observation, valid_actions):
        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        # current_container = self._parse_current_container(observation)

        # max_bay_num = yard_state[:, StateIds.BAY.value].max()

        bays_in_valid_actions = yard_state[valid_actions, StateIds.BAY.value]
        """
        nearest_bay = current_container[StateIds.BAY.value]
        allowed_bays = [nearest_bay]
        if nearest_bay -2 >=1:
            allowed_bays.append(nearest_bay -2)
        if nearest_bay +2 <= max_bay_num:
            allowed_bays.append(nearest_bay +2)
        

        bays_in_valid_actions = [bay for bay in allowed_bays if bay in bays_in_valid_actions]
        """
        
        if len(bays_in_valid_actions) == 0:
            return None
        
        return np.random.choice(bays_in_valid_actions)
        
    def _rule_based_policy(self, observation, valid_actions):
        """
        Select bay with stacks having most similar count of same-group containers (not full)
        """
        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])
        
        # Get valid bays from valid actions
        valid_bays = set(yard_state[valid_actions, StateIds.BAY.value].astype(int))
        
        # Get max tier to check if stack is full
        max_tier = int(yard_state[:, StateIds.TIER.value].max())
        
        # Build stack info: {(bay, row): {occupied_count, same_group_count}}
        stacks_info = {}
        
        for idx in range(yard_state.shape[0]):
            bay = int(yard_state[idx, StateIds.BAY.value])
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])
            
            stack_key = (bay, row)
            if stack_key not in stacks_info:
                stacks_info[stack_key] = {'occupied': 0, 'same_group': 0}
            
            if is_occupied:
                stacks_info[stack_key]['occupied'] += 1
                if group == container_group:
                    stacks_info[stack_key]['same_group'] += 1
        
        best_bay = None
        best_bay_score = -1
        
        for (bay, row), stack_data in stacks_info.items():
            is_full = stack_data['occupied'] == max_tier
            
            if not is_full and stack_data['same_group'] > best_bay_score and bay in valid_bays:
                best_bay_score = stack_data['same_group']
                best_bay = bay
        
        
        if best_bay_score == 0:
            for (bay,row), stack_data in stacks_info.items():
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                if is_empty and bay in valid_bays:
                    best_bay = bay
                    break
        
        
        return best_bay
    
    def _rule_based_grouped_policy(self, observation, valid_actions):

        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])
        
        # Get valid bays from valid actions
        valid_bays = set(yard_state[valid_actions, StateIds.BAY.value].astype(int))
        
        # Get max tier to check if stack is full
        max_tier = int(yard_state[:, StateIds.TIER.value].max())
        
        # Build stack info: {(bay, row): {occupied_count, same_group_count}}
        stacks_info = {}
        bay_info = {}
        
        for idx in range(yard_state.shape[0]):
            bay = int(yard_state[idx, StateIds.BAY.value])
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])
            
            stack_key = (bay, row)
            bay_key = bay
            if stack_key not in stacks_info:
                stacks_info[stack_key] = {'occupied': 0, 'same_group': 0, 'different_group': 0}
            if bay_key not in bay_info:
                bay_info[bay_key] = {'occupied': 0, 'same_group': 0, 'different_group': 0}
            
            
            if is_occupied:
                stacks_info[stack_key]['occupied'] += 1
                bay_info[bay_key]['occupied'] += 1
                if group == container_group:
                    stacks_info[stack_key]['same_group'] += 1
                    bay_info[bay_key]['same_group'] += 1
                else:
                    stacks_info[stack_key]['different_group'] += 1
                    bay_info[bay_key]['different_group'] += 1
        
        best_bay = None
        best_bay_score = 0
        
        for (bay, row), stack_data in stacks_info.items():
            is_full = stack_data['occupied'] == max_tier
            
            if not is_full and stack_data['same_group'] > best_bay_score and bay in valid_bays:
                best_bay_score = stack_data['same_group']
                best_bay = bay
        
        """
        if best_bay_score == 0:
            for (bay,row), stack_data in stacks_info.items():
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                if is_empty and bay in valid_bays:
                    best_bay = bay
                    break
        """
        #print(f"best_bay before fallback: {best_bay}")
        
        bay_with_most_similar_containers = None
        bay_with_least_containers = None
        least_containers_per_bay_count = np.inf
        empty_bay = None
        
        if best_bay is None:
            for bay, bay_data in bay_info.items():
                if bay_data['same_group'] > 0 and bay in valid_bays:
                    if bay_data['same_group'] > best_bay_score:
                        best_bay_score = bay_data['same_group']
                        bay_with_most_similar_containers = bay
                if bay_data['occupied'] < least_containers_per_bay_count and bay in valid_bays:
                    least_containers_per_bay_count = bay_data['occupied']
                    bay_with_least_containers = bay
            best_bay = bay_with_most_similar_containers if bay_with_most_similar_containers is not None else bay_with_least_containers
        
        #print(f"best_bay after fallback: {best_bay}")
        
        return best_bay
    
    def _parse_yard_state(self, observation):
        observation = observation.astype(int)
        vessel_end = self.total_vessel_coords * self.num_slot_attrs
        yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

        yard_state = observation[vessel_end:yard_end].reshape(self.total_yard_coords, self.num_slot_attrs)
        return yard_state
    
    def _parse_current_container(self, observation):
        observation = observation.astype(int)
        vessel_end = self.total_vessel_coords * self.num_slot_attrs
        yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

        current_container = observation[yard_end:yard_end + self.num_slot_attrs]
        return current_container
    
    def update_agent(self, reward, info):
        self.episode_history.append({
            "reward": reward,
            "info": info
        })

class LowLevelAgent(BaseAgent):

    def __init__(self, vessel_shape, yard_shape, num_slot_attrs, policy_type ="rule_based"):
        super().__init__(AgentLevel.LOW_LEVEL)

        self.vessel_shape = vessel_shape
        self.yard_shape = yard_shape
        self.policy_type = policy_type
        self.num_slot_attrs = num_slot_attrs
        self.episode_history = []

        self.num_vessel_bay = vessel_shape[0]//2 + vessel_shape[0]
        self.num_yard_bay = yard_shape[0]//2 + yard_shape[0]

        self.total_vessel_coords = self.num_vessel_bay * vessel_shape[1] * vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * yard_shape[1] * yard_shape[2]

        
    def get_action(self, observation, valid_actions, selected_bay):
        if self.policy_type == "rule_based":
            return self._rule_based_policy(observation, valid_actions, selected_bay)
        elif self.policy_type == "rule_based_grouped":
            return self._rule_based_grouped_policy(observation, valid_actions, selected_bay)
        elif self.policy_type == "random":
            return self._random_policy(observation,valid_actions, selected_bay)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
    def _random_policy(self, observation, valid_actions, selected_bay):
        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)

        bay_mask = yard_state[valid_actions, StateIds.BAY.value] == selected_bay
        bay_valid_actions = np.array(valid_actions)[bay_mask]

        if len(bay_valid_actions) == 0:
            return None
        else:
            return np.random.choice(bay_valid_actions)

    def _rule_based_policy(self, observation, valid_actions, selected_bay):
        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])

        bay_mask = yard_state[:, StateIds.BAY.value] == selected_bay
        bay_indices = np.where(bay_mask)[0]

        #print(valid_actions)

        #valid_actions = [int(action) for action in valid_actions if action in bay_indices]
        

        #print(valid_actions)
        #print(bay_indices)

        
        # Get valid bays from valid actions
        # valid_bays = set(yard_state[valid_actions, StateIds.BAY.value].astype(int))
        
        # Get max tier to check if stack is full
        max_tier = int(yard_state[:, StateIds.TIER.value].max())
        
        # Build stack info: {(bay, row): {occupied_count, same_group_count}}
        stacks_info = {}
        
        for idx in bay_indices:
            #print("Evaluating yard slot:", idx, yard_state[idx])
            bay = int(yard_state[idx, StateIds.BAY.value])
            row = int(yard_state[idx, StateIds.ROW.value])
            is_occupied = int(yard_state[idx, StateIds.IS_OCCUPIED.value])
            group = int(yard_state[idx, StateIds.GROUP.value])
            
            stack_key = (bay,row)
            if stack_key not in stacks_info:
                stacks_info[stack_key] = {'occupied': 0, 'same_group': 0}
            
            if is_occupied:
                stacks_info[stack_key]['occupied'] += 1
                if group == container_group:
                    stacks_info[stack_key]['same_group'] += 1
        
        best_action = None
        best_stack_score = -1
        
        for (bay,row), stack_data in stacks_info.items():
            #print(f"Evaluating action:", (bay,row), "Stack data:", stack_data)
            is_full = stack_data['occupied'] == max_tier
            is_empty = stack_data['occupied'] == 0

            
            if not is_full and stack_data['same_group'] >= best_stack_score:
                best_stack_score = stack_data['same_group']
                best_action = (bay,row)

        if best_stack_score == 0:
            for (bay,row), stack_data in stacks_info.items():
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                if is_empty:
                    best_action = (bay,row)
                    break
        
        print("Best action (bay,row):", best_action)
        valid_action_mask = (yard_state[:,StateIds.BAY.value] == best_action[0]) & (yard_state[:,StateIds.ROW.value] == best_action[1])
        valid_action_indices = np.where(valid_action_mask)[0]

        #print("Valid action indices for best action:", valid_action_indices)
        #print("Valid actions:", valid_actions)

        valid_action_indices = [int(action) for action in valid_action_indices if action in valid_actions]
        #print(valid_action_indices)
        return valid_action_indices[0]
    
    def _rule_based_grouped_policy(self, observation, valid_actions, selected_bay):
        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)
        container_group = int(current_container[StateIds.GROUP.value])

        bay_mask = yard_state[:, StateIds.BAY.value] == selected_bay
        bay_indices = np.where(bay_mask)[0]
        
        # Get max tier to check if stack is full
        max_tier = int(yard_state[:, StateIds.TIER.value].max())
        
        # Build stack info: {(row): {occupied_count, same_group_count}}
        stacks_info = {}
        total_occupied_in_bay = 0
        
        for idx in bay_indices:
            # print("Evaluating yard slot:", idx, yard_state[idx])
            # bay = int(yard_state[idx, StateIds.BAY.value])
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
        
        best_action = None
        best_stack_score = 0
        stack_with_most_similar_containers = None
        count_of_most_similar_containers_in_stack = 0
        empty_rows = []
        
        for row, stack_data in stacks_info.items():
            #print(f"Evaluating action:", (selected_bay,row), "Stack data:", stack_data)
            is_full = stack_data['occupied'] == max_tier
            is_empty = stack_data['occupied'] == 0

            
            if stack_data['same_group'] > best_stack_score:
                if not is_full:
                    best_action = (selected_bay,row)
                    best_stack_score = stack_data['same_group']
            if stack_data['same_group'] > count_of_most_similar_containers_in_stack:
                count_of_most_similar_containers_in_stack = stack_data['same_group']
                stack_with_most_similar_containers = row
            if is_empty:
                empty_rows.append(row)

        #print(f"Best action at 1st stage (bay,row):", best_action)
        #print("Empty rows:", empty_rows)

        if best_action is None :
            if total_occupied_in_bay == 0:
                best_action = (selected_bay, empty_rows[0])
                
        row_list = list(stacks_info.keys())
        if stack_with_most_similar_containers is not None:
                row_list = sorted(row_list, key=lambda x: abs(x - stack_with_most_similar_containers))
        
        if best_action is None and stack_with_most_similar_containers is not None :
            for row in row_list:
                stack_data = stacks_info[row]
                is_full = stack_data['occupied'] == max_tier
                is_empty = stack_data['occupied'] == 0
                if not is_full and is_empty:
                    best_action = (selected_bay,row)
                    break
        #print(f"Best action at 2nd stage (bay,row):", best_action)
        
        if best_action is None:
            if len(empty_rows) > 0:
                best_action = (selected_bay, self._pick_last_sorted(empty_rows))

        #print(f"Best action at 3rd stage (bay,row):", best_action)

        if best_action is None:
            for row in row_list:
                if stacks_info[row]['occupied'] < max_tier:
                    best_action = (selected_bay, row)
                    break
        #print(f"Best action at final stage (bay,row):", best_action)
            
        
        print("Best action (bay,row):", best_action)
        valid_action_mask = (yard_state[:,StateIds.BAY.value] == best_action[0]) & (yard_state[:,StateIds.ROW.value] == best_action[1])
        valid_action_indices = np.where(valid_action_mask)[0]

        #print("Valid action indices for best action:", valid_action_indices)
        #print("Valid actions:", valid_actions)

        valid_action_indices = [int(action) for action in valid_action_indices if action in valid_actions]
        #print(valid_action_indices)
        return valid_action_indices[0]


    
    def _score_based_policy(self, observation, valid_actions, selected_bay):

        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)

        container_group = int(current_container[StateIds.GROUP.value])

        valid_bay_actions = []
        for action in valid_actions:
            if yard_state[action,StateIds.BAY.value] == selected_bay:
                valid_bay_actions.append(action)
        


        if len(valid_bay_actions) == 0:
            return None
        else:
            max_action_score = -np.inf
            best_action = None
            for action in valid_bay_actions:
                group_score = self._score_slots(action, yard_state, current_container, container_group, selected_bay)
                    
                if group_score > max_action_score:
                    max_action_score = group_score
                    best_action = action
            
            print(f"Low-Level Agent selected action {best_action} in bay {selected_bay} with score {max_action_score}")
            return best_action
        
    def _score_slots(self, action, yard_state, current_container, container_group, selected_bay):

        group_score = 0
        bay = yard_state[action,StateIds.BAY.value]
        row = yard_state[action,StateIds.ROW.value]
        tier = yard_state[action,StateIds.TIER.value]
        
        # Get all occupied slots in same stack
        stack_mask = (yard_state[:,StateIds.BAY.value] == bay) & (yard_state[:,StateIds.ROW.value] == row)
        stack_indices = np.where(stack_mask)[0]

        occupied_slots_mask = yard_state[stack_indices, StateIds.IS_OCCUPIED.value] == 1
        occupied_slots = stack_indices[occupied_slots_mask]

        if len(occupied_slots) == 0:
            group_score -= 0.5
        
        if len(occupied_slots) > 0:
            stack_container_groups = yard_state[occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(stack_container_groups == container_group)
            diff_group_count = np.sum(stack_container_groups != container_group)
            group_score += same_group_count - diff_group_count
        
        adjacent_rows = []
        if row - 1 >  0:
            adjacent_rows.append(row - 1)
        if row + 1 <= self.yard_shape[1]:
            adjacent_rows.append(row + 1)
        
        
        # Check other stacks in same bay (but bot same row or adjacent rows)
        same_bay_mask = (yard_state[:,StateIds.BAY.value] == bay) & (yard_state[:,StateIds.ROW.value] != row) & ~np.isin(yard_state[:,StateIds.ROW.value],adjacent_rows)
        same_bay_indices = np.where(same_bay_mask)[0]

        same_bay_occupied_slots_mask = yard_state[same_bay_indices, StateIds.IS_OCCUPIED.value] == 1
        same_bay_occupied_slots = same_bay_indices[same_bay_occupied_slots_mask]

        if len(same_bay_occupied_slots) > 0:
            same_bay_container_groups = yard_state[same_bay_occupied_slots, StateIds.GROUP.value]
            same_group_count = np.sum(same_bay_container_groups == container_group)
            diff_group_count = np.sum(same_bay_container_groups != container_group)
            group_score += (same_group_count - diff_group_count) * 0.25

        # Check adjacent stacks
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
    
    def _pick_last_sorted(self, nums):
        best_len = curr_len = 1
        best_last = curr_last = nums[0]

        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                curr_len += 1
            else:
                curr_len = 1

            curr_last = nums[i]

            if curr_len > best_len:
                best_len = curr_len
                best_last = curr_last

        return best_last




        
    def _parse_yard_state(self, observation):
        observation = observation.astype(int)
        vessel_end = self.total_vessel_coords * self.num_slot_attrs
        yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

        yard_state = observation[vessel_end:yard_end].reshape(self.total_yard_coords, self.num_slot_attrs)
        return yard_state
    
    def _parse_current_container(self, observation):
        observation = observation.astype(int)
        vessel_end = self.total_vessel_coords * self.num_slot_attrs
        yard_end = vessel_end + (self.total_yard_coords * self.num_slot_attrs)

        current_container = observation[yard_end:yard_end + self.num_slot_attrs]
        return current_container
    
    def update_agent(self, reward, info):
        self.episode_history.append({
            "reward": reward,
            "info": info
        })

class HierarchicalAgent:

    def __init__(self, vessel_shape, yard_shape, num_slot_attrs,
                 high_level_policy_type="rule_based",
                 low_level_policy_type="rule_based"):
        
        self.high_level_agent = HighLevelAgent(vessel_shape,
                     yard_shape, num_slot_attrs, high_level_policy_type)
        
        self.low_level_agent = LowLevelAgent(vessel_shape,
                     yard_shape, num_slot_attrs, low_level_policy_type)
        
    def get_action(self, observation, valid_actions):

        if len(valid_actions) == 0:
            return None, {"error": "No valid actions available"}
        
        selected_bay = self.high_level_agent.get_action(observation, valid_actions)

        if selected_bay is None:
            return valid_actions[0], {"selected_bay": None,
                                        "selected_slot": valid_actions[0]}
        
        selected_slot = self.low_level_agent.get_action(observation, valid_actions, 
                            selected_bay)
        
        action_info = {
            "selected_bay": selected_bay,
            "selected_slot": selected_slot
        }

        return selected_slot, action_info
    
    def update_agent(self, reward, info):
        self.high_level_agent.update_agent(reward, info)
        self.low_level_agent.update_agent(reward, info)

        
            


