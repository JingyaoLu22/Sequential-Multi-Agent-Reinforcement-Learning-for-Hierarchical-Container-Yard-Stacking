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
        elif self.policy_type == "random":
            return self._random_policy(observation,valid_actions)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def _random_policy(self, observation, valid_actions):
        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)

        bays_in_valid_actions = yard_state[valid_actions, StateIds.BAY.value]
        unique_bays = np.unique(bays_in_valid_actions)

        selected_bay = np.random.choice(unique_bays)
        return selected_bay
        
    def _rule_based_policy(self, observation, valid_actions):

        if len(valid_actions) == 0:
            return None
        
        yard_state = self._parse_yard_state(observation)
        current_container = self._parse_current_container(observation)

        nearest_bay = current_container[StateIds.BAY.value]
        allowed_bays = [nearest_bay, nearest_bay - 2, nearest_bay + 2]


        valid_bays = set(yard_state[valid_actions,StateIds.BAY.value])
        candidate_bays = [bay for bay in allowed_bays if bay in valid_bays]

        if len(candidate_bays) != 0:
            return candidate_bays[0]
        else:
            return valid_actions[0, StateIds.BAY.value]
    
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

        
            


