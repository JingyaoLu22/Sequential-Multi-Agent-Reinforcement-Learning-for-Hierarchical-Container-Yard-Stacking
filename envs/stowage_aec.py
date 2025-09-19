import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from typing import Dict, Optional
from .stowage_gym import StowageEnv, StateIds
from .stowage_crane_gym import MultiCraneStowageEnv


class StowageAEC(AECEnv, MultiCraneStowageEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "stowage_aec"}

    def __init__(self, config: Dict = None, render_mode: Optional[str] = None):
        MultiCraneStowageEnv.__init__(self, config, render_mode)
        AECEnv.__init__(self)

        self.possible_agents = [f"crane_{i}" for i in range(self.num_cranes)]
        self.agents = self.possible_agents.copy()

        self._agent_action_spaces = {
            agent: gym.spaces.Discrete(self.total_yard_coords) for agent in self.possible_agents
        }

        obs_size = self.obs_coords * self.num_slot_attrs + self.num_cranes * 2 + self.total_yard_coords
        self._agent_observation_spaces = {
            agent: gym.spaces.Box(
                low=0,
                high=max(
                    max(self.num_vessel_bay, self.num_yard_bay),
                    max(self.vessel_shape[1], self.yard_shape[1]),
                    max(self.vessel_shape[2], self.yard_shape[2]),
                    1,  # occupied max limit
                    self.group_num,  # group max limit
                    1000,  # time max limit
                ),
                shape=(obs_size,),
                dtype=np.int64,
            )
            for agent in self.possible_agents
        }
        self.observation_space = self._agent_observation_spaces[self.possible_agents[0]]
        self.action_space = self._agent_action_spaces[self.possible_agents[0]]
        self.time_arr = self._get_randomized_time_array()

    def observation_space(self, agent: str) -> gym.spaces.Space:
        """Return observation space for an agent (AEC interface)"""
        return self._agent_observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        """Return action space for an agent (AEC interface)"""
        return self._agent_action_spaces[agent]

    def reset(self, seed=None, options=None):
        _, info = MultiCraneStowageEnv.reset(self, seed=seed, **({} if options is None else options))

        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: info for agent in self.agents}

        self._agent_selector = agent_selector.AgentSelector(self.agents)
        self._determine_next_agent()

        return self.observe(self.agent_selection), info

    def observe(self, agent):
        # Match pettingzoo interface
        return self._create_observation()

    def action_masks(self):
        if self.agent_selection is None:
            return np.zeros(self.total_yard_coords, dtype=bool)
        crane_idx = int(self.agent_selection.split("_")[1])
        return self._get_valid_actions_for_crane(crane_idx)

    def step(self, action):
        """Execute action for the current agent."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        crane_idx = int(self.agent_selection.split("_")[1])
        # Check if action is valid
        valid_actions = self._get_valid_actions_for_crane(crane_idx)
        if not valid_actions[action]:
            reward = -100.0
            self.rewards[self.agent_selection] = reward
            self._cumulative_rewards[self.agent_selection] += reward
            # Get next agent
            prev_agent = self.agent_selection
            self._determine_next_agent()
            return (
                self._create_observation(),
                reward,
                self.terminations[prev_agent],
                self.truncations[prev_agent],
                self.infos[prev_agent],
            )

        reward = self._execute_crane_action(crane_idx, action)
        self.rewards[self.agent_selection] = reward
        self._cumulative_rewards[self.agent_selection] += reward
        self._check_termination()
        prev_agent = self.agent_selection
        MultiCraneStowageEnv._advance_time(self)
        self._determine_next_agent()

        return (
            self._create_observation(),
            reward,
            self.terminations[prev_agent],
            self.truncations[prev_agent],
            self.infos[prev_agent],
        )

    def _create_observation(self):
        state = StowageEnv._create_observation(self)
        state = np.append(state, self.crane_positions)
        busy_relative = self.crane_busy_until - self.current_time
        state = np.append(state, busy_relative)
        state = np.append(state, self.time_arr)

        return state

    def _get_valid_actions_for_crane(self, crane_idx):
        """Get action mask for a specific crane"""
        if self.current_vessel_slots[crane_idx] is None or self.crane_busy_until[crane_idx] > self.current_time:
            return np.zeros(self.total_yard_coords, dtype=bool)

        vessel_slot = self.current_vessel_slots[crane_idx]
        vessel_group = self.vessel_state[vessel_slot, StateIds.GROUP.value]

        occupied_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1
        bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 == 1
        group_mask = self.yard_state[:, StateIds.GROUP.value] == vessel_group

        valid_actions = np.zeros(self.total_yard_coords, dtype=bool)
        valid_indices = np.where(occupied_mask & bay_mask & group_mask)[0]
        valid_actions[valid_indices] = True

        return valid_actions

    def _execute_crane_action(self, crane_idx, yard_slot):
        """Execute action for a specific crane and return reward"""
        vessel_slot = self.current_vessel_slots[crane_idx]
        yard_bay = self.yard_state[yard_slot, StateIds.BAY.value]

        shifters = MultiCraneStowageEnv._process_shifters(self, yard_slot, vessel_slot)
        # operation_time = self.time_arr[yard_slot]
        operation_time = self.time_arr[yard_slot] + shifters * 50
        self.crane_positions[crane_idx] = yard_bay
        self.crane_busy_until[crane_idx] = self.current_time + operation_time
        crane_idle_time = max(0, np.sum(self.current_time - self.crane_busy_until))
        reward = -shifters - crane_idle_time * self.time_penalty_coef * 0.5

        self.current_vessel_slots[crane_idx] = MultiCraneStowageEnv._get_next_vessel_slot_for_crane(self, crane_idx)
        MultiCraneStowageEnv._update_crane_vessel_slots(self)

        self.total_shifters += shifters
        self._update_info(crane_idx, shifters, operation_time)

        return reward

    def _determine_next_agent(self):
        """Select the next agent to act based on availability"""
        if not self.agents or all(self.terminations.values()):
            self.agent_selection = None
            return

        available_agents = []
        for agent in self.agents:
            crane_idx = int(agent.split("_")[1])
            if (
                self.current_vessel_slots[crane_idx] is not None
                and self.crane_busy_until[crane_idx] <= self.current_time
            ):
                available_agents.append(agent)

        if available_agents:
            self.agent_selection = min(available_agents, key=lambda a: self.crane_busy_until[int(a.split("_")[1])])
        else:
            # If no agents available, advance time and try again
            MultiCraneStowageEnv._advance_time(self)
            self._determine_next_agent()

    def _check_termination(self, return_bool=False):
        """Check if environment is terminated"""
        all_slots_done = all(slot is None for slot in self.current_vessel_slots)
        no_available_groups = self.available_groups.size == 0
        terminated = all_slots_done or no_available_groups
        if terminated:
            # Apply time penalty at termination
            time_penalty = self.current_time * self.time_penalty_coef
            for agent in self.agents:
                self.terminations[agent] = True
                # Only apply penalty once at termination
                if self.rewards[agent] != -time_penalty:
                    self.rewards[agent] -= time_penalty
                    self._cumulative_rewards[agent] -= time_penalty

        return terminated if return_bool else None

    def _update_info(self, crane_idx, shifters, operation_time):
        """Update info for all agents"""
        for agent in self.agents:
            self.infos[agent] = {
                "shifters": shifters,
                "total_shifters": self.total_shifters,
                "operation_time": operation_time if agent == f"crane_{crane_idx}" else 0,
                "current_time": self.current_time,
                "vessel_slots_filled": self.vessel_slots_filled,
                "cranes": {
                    "positions": self.crane_positions.copy(),
                    "busy_until": self.crane_busy_until.copy(),
                    "vessel_slots": self.current_vessel_slots.copy(),
                },
            }

    def render(self):
        return MultiCraneStowageEnv.render(self)
