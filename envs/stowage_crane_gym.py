import numpy as np
import gymnasium as gym
import colorsys
from typing import Dict, Optional

from .stowage_gym import StowageEnv, StateIds


class MultiCraneStowageEnv(StowageEnv):
    def __init__(self, config: Dict = None, render_mode: Optional[str] = None):
        if config is None:
            config = {}

        super().__init__(config, render_mode)
        self.num_cranes = min(config.get("num_cranes", 2), self.vessel_shape[0])
        self.time_penalty_coef = config.get("time_penalty_coef", 0.01)

        self.crane_positions = None
        self.crane_busy_until = np.zeros(self.num_cranes)
        self.current_time = 0
        self.total_shifters = 0
        # Sequencers for all cranes
        self.current_vessel_slots = [None for _ in range(self.num_cranes)]  # current vessel slots for each crane
        self.crane_bay_ranges = None  # bay ranges for each cranes

        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(
                max(self.num_vessel_bay, self.num_yard_bay),  # max number of bays
                max(self.vessel_shape[1], self.yard_shape[1]),  # max number of rows
                max(self.vessel_shape[2], self.yard_shape[2]),  # max number of tiers
                1,  # occupied max limit
                self.group_num,  # group max limit
                1000,  # time max limit
            ),
            shape=(
                self.obs_coords * 5 + self.num_cranes * 2 + 1,
            ),  # original state + crane positions + busy time + global time
            dtype=np.int32,
        )
        self.action_space = gym.spaces.Discrete(self.total_yard_coords * self.num_cranes)

    def reset(self, seed=None, **kwargs):
        _, info = super().reset(seed, **kwargs)

        self.time_arr = self._get_randomized_time_array()
        self.current_time = 0
        self.total_shifters = 0

        self.crane_positions = np.array(
            [1 + i * max(1, self.vessel_shape[0] // self.num_cranes) for i in range(self.num_cranes)]
        )
        self.crane_busy_until = np.zeros(self.num_cranes)

        all_bays = np.unique(self.vessel_state[:, StateIds.BAY.value])
        odd_bays = all_bays[all_bays % 2 == 1]
        even_bays = all_bays[all_bays % 2 == 0]
        odd_ranges = np.array_split(np.sort(odd_bays), self.num_cranes)
        even_ranges = np.array_split(np.sort(even_bays), self.num_cranes)
        self.crane_bay_ranges = [
            np.sort(np.concatenate((odd_ranges[i], even_ranges[i]))) for i in range(self.num_cranes)
        ]

        self.current_vessel_slots = [self._get_next_vessel_slot_for_crane(i) for i in range(self.num_cranes)]
        observation = self._create_observation()
        info["cranes"] = {
            "positions": self.crane_positions.copy(),
            "busy_until": self.crane_busy_until.copy(),
            "current_time": self.current_time,
            "vessel_slots": self.current_vessel_slots.copy(),
        }
        info["total_shifters"] = self.total_shifters

        return observation, info

    def _create_observation(self):
        state = super()._create_observation()
        state = np.append(state, self.crane_positions)
        busy_relative = self.crane_busy_until - self.current_time
        state = np.append(state, busy_relative)
        state = np.append(state, self.current_time)

        return state

    def _decode_action(self, action):
        """Decodes the action into yard slot and crane index"""
        yard_slot = action // self.num_cranes
        crane_idx = action % self.num_cranes
        return yard_slot, crane_idx

    def _get_next_vessel_slot_for_crane(self, crane_idx):
        """Get next vessel slot for a crane"""
        crane_bays = self.crane_bay_ranges[crane_idx]

        valid_slots = np.where(
            (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
            & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
            & np.isin(self.vessel_state[:, StateIds.BAY.value], crane_bays)
            & np.isin(self.vessel_state[:, StateIds.GROUP.value], self.available_groups)
        )[0]

        if len(valid_slots) == 0:
            return self._try_steal_work(crane_idx)

        bays = self.vessel_state[valid_slots, StateIds.BAY.value]
        rows = self.vessel_state[valid_slots, StateIds.ROW.value]
        tiers = self.vessel_state[valid_slots, StateIds.TIER.value]

        sort_order = np.lexsort((rows, tiers, bays))
        return valid_slots[sort_order[0]]

        return None

    def _try_steal_work(self, crane_idx):
        """Try to steal work from other cranes"""
        busy_bays = [
            self.vessel_state[self.current_vessel_slots[i], StateIds.BAY.value]
            for i in range(self.num_cranes)
            if i != crane_idx and self.current_vessel_slots[i] is not None
        ]

        # Try to steal from each crane
        for other_crane_idx in range(self.num_cranes):
            if other_crane_idx == crane_idx:
                continue

            if self.current_vessel_slots[other_crane_idx] is None:
                continue

            other_crane_bays = self.crane_bay_ranges[other_crane_idx]
            other_valid_slots = np.where(
                (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
                & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
                & ~np.isin(self.vessel_state[:, StateIds.BAY.value], busy_bays)
                & np.isin(self.vessel_state[:, StateIds.BAY.value], other_crane_bays)
                & np.isin(self.vessel_state[:, StateIds.GROUP.value], self.available_groups)
            )[0]

            if len(other_valid_slots) > 0:
                bays = self.vessel_state[other_valid_slots, StateIds.BAY.value]
                rows = self.vessel_state[other_valid_slots, StateIds.ROW.value]
                tiers = self.vessel_state[other_valid_slots, StateIds.TIER.value]

                sort_order = np.lexsort((rows, tiers, -bays))
                chosen_slot = other_valid_slots[sort_order[0]]

                bay_to_add = self.vessel_state[chosen_slot, StateIds.BAY.value]
                if bay_to_add not in self.crane_bay_ranges[crane_idx]:
                    self.crane_bay_ranges[crane_idx] = np.append(self.crane_bay_ranges[crane_idx], bay_to_add)

                self.crane_bay_ranges[other_crane_idx] = np.array(
                    [b for b in self.crane_bay_ranges[other_crane_idx] if b != bay_to_add]
                )

                return chosen_slot

        return None

    def action_masks(self):
        masks = np.zeros(self.action_space.n, dtype=bool)
        available_cranes, vessel_groups = self._get_available_cranes()
        if not available_cranes:
            return masks
        valid_yard_slots, yard_groups = self._get_valid_yard_slots()

        for i, yard_slot in enumerate(valid_yard_slots):
            yard_group = yard_groups[i]

            for j, crane_idx in enumerate(available_cranes):
                if yard_group == vessel_groups[j]:
                    action = yard_slot * self.num_cranes + crane_idx
                    masks[action] = True

        return masks

    def _get_available_cranes(self):
        """Identify available cranes that can take actions"""
        available_cranes = []
        vessel_groups = []

        for crane_idx in range(self.num_cranes):
            if (
                self.current_vessel_slots[crane_idx] is not None
                and self.crane_busy_until[crane_idx] <= self.current_time
            ):
                available_cranes.append(crane_idx)
                vessel_slot = self.current_vessel_slots[crane_idx]
                vessel_groups.append(self.vessel_state[vessel_slot, StateIds.GROUP.value])

        return available_cranes, vessel_groups

    def _get_valid_yard_slots(self):
        """Identify valid yard slots with containers"""
        yard_valid_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1
        yard_valid_mask &= self.yard_state[:, StateIds.BAY.value] % 2 == 1

        valid_yard_slots = np.where(yard_valid_mask)[0]
        yard_groups = self.yard_state[valid_yard_slots, StateIds.GROUP.value]

        return valid_yard_slots, yard_groups

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        yard_slot, crane_idx = self._decode_action(action)

        valid_actions = self.action_masks()
        if not valid_actions[action]:
            reward = -100.0
            observation = self._create_observation()
            info["action_masks"] = valid_actions
            return observation, reward, terminated, truncated, info
        # Container movement
        shifters = StowageEnv._process_shifters(self, yard_slot, self.current_vessel_slots[crane_idx])
        operation_time = self.time_arr[yard_slot]

        self.crane_positions[crane_idx] = self.yard_state[yard_slot, StateIds.BAY.value]
        self.crane_busy_until[crane_idx] = self.current_time + operation_time

        self.current_vessel_slots[crane_idx] = self._get_next_vessel_slot_for_crane(crane_idx)
        self._update_crane_vessel_slots()
        self.total_shifters += shifters
        reward = -shifters
        terminated = self._check_termination()
        self._advance_time()
        observation = self._create_observation()
        info.update(
            {
                "shifters": shifters,
                "total_shifters": self.total_shifters,
                "operation_time": operation_time,
                "current_time": self.current_time,
                "vessel_slots_filled": self.vessel_slots_filled,
                "cranes": {
                    "positions": self.crane_positions.copy(),
                    "busy_until": self.crane_busy_until.copy(),
                    "vessel_slots": self.current_vessel_slots.copy(),
                },
            }
        )
        if np.any(valid_actions) is False and not terminated:
            print("No valid actions available")
        if terminated:
            reward -= self.current_time * self.time_penalty_coef

        return observation, reward, terminated, truncated, info

    def _update_crane_vessel_slots(self):
        """Update vessel slots for all cranes"""
        for cdx in range(self.num_cranes):
            if self.current_vessel_slots[cdx] is None:
                continue

            group_val = self.vessel_state[self.current_vessel_slots[cdx], StateIds.GROUP.value]
            if self.available_groups.size == 0 or not np.any(self.available_groups == group_val):
                self.current_vessel_slots[cdx] = self._get_next_vessel_slot_for_crane(cdx)

    def _advance_time(self):
        """Advance time to next event"""
        active_cranes = [i for i in range(self.num_cranes) if self.current_vessel_slots[i] is not None]
        if active_cranes:
            active_busy_times = self.crane_busy_until[active_cranes]
            self.current_time = np.min(active_busy_times)
        else:
            self.current_time = np.max(self.crane_busy_until)

    def _check_termination(self):
        all_sequencers_done = all(slot is None for slot in self.current_vessel_slots)
        no_available_groups = self.available_groups.size == 0
        return all_sequencers_done or no_available_groups

    def _get_randomized_time_array(self):
        rng = np.random.RandomState(self.seed)
        return rng.randint(1, 100, self.total_yard_coords)

    # render part
    def _is_target_cell(self, idx, is_vessel):
        if is_vessel:
            return idx in self.current_vessel_slots
        return False

    def _create_cell_props(self, filled, target, idx, group):
        cell_props = {"filled": filled, "target": target, "idx": idx, "group": group, "waiting": False}

        if target and idx is not None and idx in self.current_vessel_slots:
            crane_idx = self.current_vessel_slots.index(idx)
            cell_props["waiting"] = (self.crane_busy_until[crane_idx] - self.current_time) > 0

        return cell_props

    def _get_cell_border_style(self, cell):
        if cell.get("waiting", False):
            return (94, 73, 73), 3
        elif cell["target"]:
            return (255, 0, 0), 3
        return (180, 180, 180), 1
