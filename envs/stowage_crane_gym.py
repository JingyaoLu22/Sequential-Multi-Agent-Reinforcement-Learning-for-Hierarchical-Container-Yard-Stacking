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

        if self.has_sequencer:
            self.action_space = gym.spaces.Discrete(self.total_yard_coords * self.num_cranes)
        
        self.crane_positions = None
        self.crane_busy_until = 0
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

    def reset(self, seed=None, **kwargs):
        observation, info = super().reset(seed, **kwargs)
        self.time_arr = self._get_randomized_time_array()

        self.current_time = 0
        self.crane_positions = np.array(
            [1 + i * max(1, self.vessel_shape[0] // self.num_cranes) for i in range(self.num_cranes)]
        )
        self.crane_busy_until = np.zeros(self.num_cranes)

        all_bays = np.unique(self.vessel_state[:, StateIds.BAY.value])
        odd_bays = all_bays[all_bays % 2 == 1]
        even_bays = all_bays[all_bays % 2 == 0]
        odd_ranges = np.array_split(np.sort(odd_bays), self.num_cranes)
        even_ranges = np.array_split(np.sort(even_bays), self.num_cranes)
        self.crane_bay_ranges = [np.sort(np.concatenate((odd_ranges[i], even_ranges[i]))) 
                                for i in range(self.num_cranes)]

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
        """Get the next vessel slot for the given crane index"""
        if self.container_type == "one":
            crane_bays = self.crane_bay_ranges[crane_idx]

            valid_slots = np.where(
                (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
                & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
                & np.isin(self.vessel_state[:, StateIds.BAY.value], crane_bays)
                & np.isin(self.vessel_state[:, StateIds.GROUP.value], self.available_groups)
            )[0]

            if len(valid_slots) == 0:
                for other_crane_idx in range(self.num_cranes):
                    if other_crane_idx == crane_idx:
                        continue

                    if self.current_vessel_slots[other_crane_idx] is None:
                        continue

                    busy_bays = []
                    for i in range(self.num_cranes):
                        if i == crane_idx or self.current_vessel_slots[i] is None:
                            continue

                        vessel_slot = self.current_vessel_slots[i]
                        busy_bay = self.vessel_state[vessel_slot, StateIds.BAY.value]
                        busy_bays.append(busy_bay)

                    other_crane_bays = self.crane_bay_ranges[other_crane_idx]
                    other_valid_slots = np.where(
                        (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
                        & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
                        & ~np.isin(self.vessel_state[:, StateIds.BAY.value], busy_bays)
                        & np.isin(self.vessel_state[:, StateIds.BAY.value], other_crane_bays)
                        & np.isin(self.vessel_state[:, StateIds.GROUP.value], self.available_groups)
                    )[0]

                    # If the other crane has multiple slots, steal one
                    if len(other_valid_slots) > 1:
                        bays = self.vessel_state[other_valid_slots, StateIds.BAY.value]
                        rows = self.vessel_state[other_valid_slots, StateIds.ROW.value]
                        tiers = self.vessel_state[other_valid_slots, StateIds.TIER.value]

                        sort_order = np.lexsort((rows, tiers, -bays))

                        # Update the crane bay ranges to include this new bay
                        bay_to_add = self.vessel_state[other_valid_slots[sort_order[0]], StateIds.BAY.value]
                        if bay_to_add not in self.crane_bay_ranges[crane_idx]:
                            self.crane_bay_ranges[crane_idx] = np.append(self.crane_bay_ranges[crane_idx], bay_to_add)

                        # Remove this bay from the other crane's responsibility
                        self.crane_bay_ranges[other_crane_idx] = np.array(
                            [b for b in self.crane_bay_ranges[other_crane_idx] if b != bay_to_add]
                        )

                        return other_valid_slots[sort_order[0]]

                return None

            bays = self.vessel_state[valid_slots, StateIds.BAY.value]
            rows = self.vessel_state[valid_slots, StateIds.ROW.value]
            tiers = self.vessel_state[valid_slots, StateIds.TIER.value]

            sort_order = np.lexsort((rows, tiers, bays))
            return valid_slots[sort_order[0]]
        else:
            pass

    def action_masks(self):
        masks = np.zeros(self.action_space.n, dtype=bool)

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

        # If no available cranes, return empty mask
        if not available_cranes:
            return masks

        # Find valid yard slots with containers
        yard_valid_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1
        if self.container_type == "one":
            yard_valid_mask &= self.yard_state[:, StateIds.BAY.value] % 2 == 1

        valid_yard_slots = np.where(yard_valid_mask)[0]
        yard_groups = self.yard_state[valid_yard_slots, StateIds.GROUP.value]

        # For each valid yard slot and available crane combination
        for i, yard_slot in enumerate(valid_yard_slots):
            yard_group = yard_groups[i]

            for j, crane_idx in enumerate(available_cranes):
                if yard_group == vessel_groups[j]:
                    action = yard_slot * self.num_cranes + crane_idx
                    masks[action] = True

        return masks

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        if self.has_sequencer:
            yard_slot, crane_idx = self._decode_action(action)

            valid_actions = self.action_masks()
            if not valid_actions[action]:
                reward = -100.0
                observation = self._create_observation()
                info["action_masks"] = valid_actions
                return observation, reward, terminated, truncated, info

            # Container movement
            vessel_slot = self.current_vessel_slots[crane_idx]
            yard_bay = self.yard_state[yard_slot, StateIds.BAY.value]
            yard_row = self.yard_state[yard_slot, StateIds.ROW.value]
            yard_tier = self.yard_state[yard_slot, StateIds.TIER.value]

            same_bay_row_mask = (
                (self.yard_state[:, StateIds.BAY.value] == yard_bay)
                & (self.yard_state[:, StateIds.ROW.value] == yard_row)
                & (self.yard_state[:, StateIds.TIER.value] > yard_tier)
                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
            )

            upper_slots = np.where(same_bay_row_mask)[0]
            sorted_upper_slots = []

            if len(upper_slots) > 0:
                sorted_upper_slots = sorted(upper_slots, key=lambda x: self.yard_state[x, StateIds.TIER.value])
                container_group = self.yard_state[yard_slot, StateIds.GROUP.value]
                self.yard_state[yard_slot, StateIds.IS_OCCUPIED.value] = 0

                for slot in sorted_upper_slots:
                    current_tier = self.yard_state[slot, StateIds.TIER.value]
                    new_tier = current_tier - 1
                    self.yard_state[slot, StateIds.TIER.value] = new_tier
            else:
                container_group = self.yard_state[yard_slot, StateIds.GROUP.value]
                self.yard_state[yard_slot, StateIds.IS_OCCUPIED.value] = 0

            self.vessel_state[vessel_slot, StateIds.IS_OCCUPIED.value] = 1
            self.vessel_state[vessel_slot, StateIds.GROUP.value] = container_group

            self.available_groups = np.unique(
                self.yard_state[self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1, StateIds.GROUP.value]
            )

                
            self.vessel_slots_filled += 1
            operation_time = self.time_arr[action]

            self.crane_positions[crane_idx] = yard_bay
            self.crane_busy_until[crane_idx] = self.current_time + operation_time

            self.current_vessel_slots[crane_idx] = self._get_next_vessel_slot_for_crane(crane_idx)
            for cdx in range(self.num_cranes):
                if self.current_vessel_slots[cdx] is None: # Skip finished cranes
                    continue
                group_val = self.vessel_state[self.current_vessel_slots[cdx], StateIds.GROUP.value]
                if (self.available_groups.size == 0) or (not np.any(self.available_groups == group_val)):
                    self.current_vessel_slots[cdx] = self._get_next_vessel_slot_for_crane(cdx)

            shifters = len(sorted_upper_slots)
            self.total_shifters += shifters
            reward = -shifters

            all_sequencers_done = all(slot is None for slot in self.current_vessel_slots)
            no_available_groups = self.available_groups.size == 0
            terminated = all_sequencers_done or no_available_groups

            active_cranes = [i for i in range(self.num_cranes) if self.current_vessel_slots[i] is not None]

            if active_cranes:
                active_busy_times = self.crane_busy_until[active_cranes]
                self.current_time = np.min(active_busy_times)
            else:
                self.current_time = np.max(self.crane_busy_until)

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

    def _get_randomized_time_array(self):
        rng = np.random.RandomState(self.seed)

        return rng.randint(low=1, high=100, size=self.action_space.n, dtype=np.int32)

    def _draw_grid(self, state, top, height, bays, rows, tiers, is_vessel):
        """Draw a grid section (vessel or yard) with fixed cell size"""
        import pygame

        cell_width, cell_height, padding, label_margin = 35, 35, 30, 15
        left_margin = max(padding, (self.screen_width - bays * rows * cell_width) / 2)
        small_font = pygame.font.Font(None, 20)
        tiny_font = pygame.font.Font(None, 18)

        # Each tuple is (light_color, dark_color)
        group_colors = []
        for i in range(self.group_num):
            hue = i / self.group_num
            r, g, b = colorsys.hsv_to_rgb(hue, 0.2, 0.95)
            light_color = (int(r * 255), int(g * 255), int(b * 255))
            
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.74)
            dark_color = (int(r * 255), int(g * 255), int(b * 255))
            
            group_colors.append((light_color, dark_color))

        empty_color = (255, 255, 255)  

        for t in range(1, tiers + 1):
            y = top + (tiers - t) * cell_height + cell_height / 2
            tier_label = small_font.render(f"{t}", True, (0, 0, 0))
            self.screen.blit(tier_label, (left_margin - label_margin, y - tier_label.get_height() / 2))

        if is_vessel:
            if rows % 2 == 0: 
                left = list(range(rows - 1, 0, -2))
                right = list(range(2, rows + 1, 2))
                row_order = left + right
            else: 
                left = list(range(rows, 0, -2))
                right = list(range(2, rows, 2))
                row_order = left + right
        else:
            row_order = list(range(1, rows + 1))

        cell_info = {}

        if is_vessel:
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                # is_target = self.current_vessel_slot == i if self.current_vessel_slot is not None else False
                is_target = i in self.current_vessel_slots
                is_waiting = False
                if is_target:
                    is_waiting = (self.crane_busy_until[self.current_vessel_slots.index(i)] - self.current_time) > 0
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1: 
                    cell_info[(bay, row, tier)] = {"filled": is_occupied, "target": is_target, "idx": i, "group": group, "waiting": is_waiting}

            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                # Only apply even bay influence if it's occupied
                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            existing = cell_info.get(
                                (adj_bay, row, tier), {"filled": False, "target": False, "idx": None, "group": group, "waiting": False}
                            )
                            existing["filled"] = True  # Mark as filled due to adjacent even bay
                            existing["idx"] = i  # Use the even bay's index
                            cell_info[(adj_bay, row, tier)] = existing
        else:
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1 and is_occupied:
                    cell_info[(bay, row, tier)] = {"filled": True, "target": False, "idx": i, "group": group, "waiting": False}

            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            cell_info[(adj_bay, row, tier)] = {
                                "filled": True,
                                "target": False,
                                "idx": i,
                                "group": group,
                                "waiting": False,
                            }
        thin_grid_color = (180, 180, 180)
        bay_grid_color = (0, 0, 0)

        for b in range(bays):
            bay_num = b * 2 + 1
            bay_x = left_margin + b * rows * cell_width

            bay_center_x = bay_x + (rows * cell_width) / 2
            bay_label = small_font.render(f"Bay {bay_num}", True, (0, 0, 0))
            self.screen.blit(bay_label, bay_label.get_rect(center=(bay_center_x, top - 10)))

            for pos, r in enumerate(row_order):
                x_label = bay_x + pos * cell_width + cell_width / 2
                row_label = small_font.render(f"{r}", True, (0, 0, 0))
                self.screen.blit(
                    row_label, row_label.get_rect(center=(x_label, top + tiers * cell_height + label_margin / 2))
                )

                for t in range(1, tiers + 1):
                    x = bay_x + pos * cell_width
                    y = top + (tiers - t) * cell_height

                    cell = cell_info.get((bay_num, r, t), {"filled": False, "target": False, "idx": None, "group": 0, "waiting": False})

                    group_idx = min(cell["group"], len(group_colors) - 1)

                    if cell["filled"]:
                        color = group_colors[group_idx][1]
                    elif is_vessel:
                        color = group_colors[group_idx][0]
                    else:
                        color = empty_color

                    rect = pygame.Rect(x, y, cell_width, cell_height)
                    pygame.draw.rect(self.screen, color, rect)

                    line_color = (255, 0, 0) if cell["target"] else thin_grid_color
                    line_color = (218, 68, 68) if cell["waiting"] else line_color
                    line_width = 3 if cell["target"] else 1
                    pygame.draw.rect(self.screen, line_color, rect, line_width)

                    if cell["idx"] is not None:
                        if is_vessel:
                            text_color = (255, 255, 255) if cell["filled"] else (50, 50, 50)
                            label = tiny_font.render(f"{cell['idx']}", True, text_color)
                            self.screen.blit(label, label.get_rect(center=(x + cell_width / 2, y + cell_height / 2)))
                        elif cell["filled"]:
                            label = tiny_font.render(f"{cell['idx']}", True, (255, 255, 255))
                            self.screen.blit(label, label.get_rect(center=(x + cell_width / 2, y + cell_height / 2)))

        for b in range(bays + 1):
            x = left_margin + b * rows * cell_width
            pygame.draw.line(
                self.screen,
                bay_grid_color,
                (x, top),
                (x, top + tiers * cell_height),
                2,  # Bay divider line width
            )
