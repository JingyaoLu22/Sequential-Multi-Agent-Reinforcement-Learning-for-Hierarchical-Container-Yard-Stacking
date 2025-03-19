import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional
from enum import Enum


class StateIds(Enum):
    BAY = 0
    ROW = 1
    TIER = 2
    IS_OCCUPIED = 3
    GROUP = 4


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
        if config is None:
            config = {}

        self.vessel_shape = config.get("vessel_shape", (1, 2, 2))
        self.yard_shape = config.get("yard_shape", (2, 2, 2))
        self.num_containers = config.get("num_containers", 4)
        self.container_type = config.get("container_type", "one")
        self.group_num = config.get("group_num", 1)
        self.group_placement = config.get("group_placement", "fixed")
        self.seed = config.get("seed", 0)

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
                self.group_num,  # group upper limit
            ),
            shape=(self.obs_coords, 5),
            dtype=np.int32,
        )
        self.action_space = gym.spaces.Discrete(self.yard_shape[0] * self.yard_shape[1] * self.yard_shape[2])
        # Each slot stores 5 values: bay, row, tier, occupied(0/1), group number of the container

        # Render part
        self.render_mode = render_mode
        self.screen_width = (
            35 * max(self.vessel_shape[1]*self.vessel_shape[0], self.yard_shape[1]*self.yard_shape[0]) + 60
        )
        print(self.screen_width)
        self.screen_height = 60*max(self.vessel_shape[2],self.yard_shape[2]) + 150
        self.screen = None
        self.isopen = True

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        valid_actions = self._get_valid_yard_actions().tolist()

        if action not in valid_actions:
            reward = -100.0
            observation = self._create_observation()
            info["yard_mask"] = valid_actions
            return observation, reward, terminated, truncated, info

        original_bay = self.yard_state[action, StateIds.BAY.value]
        original_row = self.yard_state[action, StateIds.ROW.value]
        original_tier = self.yard_state[action, StateIds.TIER.value]
        same_bay_row_mask = (
            (self.yard_state[:, StateIds.BAY.value] == original_bay)
            & (self.yard_state[:, StateIds.ROW.value] == original_row)
            & (self.yard_state[:, StateIds.TIER.value] > original_tier)
            & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
        )

        upper_slots = np.where(same_bay_row_mask)[0]  # get containers above the action container
        sorted_upper_slots = []
        if len(upper_slots) > 0:
            # Sort containers from bottom to top (ascending tier order)
            sorted_upper_slots = sorted(upper_slots, key=lambda x: self.yard_state[x, StateIds.TIER.value])
            container_group = self.yard_state[action, StateIds.GROUP.value]
            self.yard_state[action, StateIds.IS_OCCUPIED.value] = 0

            for slot in sorted_upper_slots:
                current_tier = self.yard_state[slot, StateIds.TIER.value]
                new_tier = current_tier - 1
                self.yard_state[slot, StateIds.TIER.value] = new_tier
        else:
            # No containers above, just remove the container and get its group
            container_group = self.yard_state[action, StateIds.GROUP.value]
            self.yard_state[action, StateIds.IS_OCCUPIED.value] = 0

        self.vessel_state[self.current_vessel_slot, StateIds.IS_OCCUPIED.value] = 1
        self.vessel_state[self.current_vessel_slot, StateIds.GROUP.value] = container_group
        self.available_groups = np.unique(
            self.yard_state[self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1, StateIds.GROUP.value]
        )

        self.vessel_slots_filled += 1
        self.current_vessel_slot = self._get_next_vessel_slot()

        shifters = len(sorted_upper_slots)
        reward = -shifters
        terminated = (self.current_vessel_slot is None) or (self.available_groups.size == 0)
        valid_actions = self._get_valid_yard_actions() if not terminated else []

        observation = self._create_observation()
        info.update({"yard_mask": valid_actions, "shifters": shifters, "vessel_slots_filled": self.vessel_slots_filled})

        return observation, reward, terminated, truncated, info

    def reset(self):
        yard_bays = self._generate_bay_coords(self.yard_shape[0])
        _, R, T = self.yard_shape
        self.yard_state = np.zeros((self.total_yard_coords, 5), dtype=int)
        # Generate coords in bay-row-tier order
        self.yard_state[:, StateIds.BAY.value] = np.repeat(yard_bays, R * T)
        self.yard_state[:, StateIds.ROW.value] = np.tile(np.arange(1, R + 1).repeat(T), len(yard_bays))
        self.yard_state[:, StateIds.TIER.value] = np.tile(np.arange(1, T + 1), len(yard_bays) * R)

        physical_vessel_bays = self.vessel_shape[0]
        vessel_bays = self._generate_bay_coords(physical_vessel_bays)
        _, Rv, Tv = self.vessel_shape
        self.vessel_state = np.zeros((self.total_vessel_coords, 5), dtype=int)

        self.vessel_state[:, StateIds.BAY.value] = np.repeat(vessel_bays, Rv * Tv)
        self.vessel_state[:, StateIds.ROW.value] = np.tile(np.arange(1, Rv + 1).repeat(Tv), len(vessel_bays))
        self.vessel_state[:, StateIds.TIER.value] = np.tile(np.arange(1, Tv + 1), len(vessel_bays) * Rv)

        if self.group_num > 1:
            if self.container_type == "one":
                # For 20-ft containers, only assign groups to odd-bay slots
                odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 == 1
                odd_bay_indices = np.where(odd_bay_mask)[0]
                valid_slots = odd_bay_indices
            else:
                # For other container types, use all slots
                valid_slots = np.arange(self.total_vessel_coords)

            slots_per_group = len(valid_slots) // self.group_num
            for group in range(self.group_num):
                start_pos = group * slots_per_group
                end_pos = (group + 1) * slots_per_group if group < self.group_num - 1 else len(valid_slots)
                group_indices = valid_slots[start_pos:end_pos]
                if len(group_indices) > 0:
                    self.vessel_state[group_indices, StateIds.GROUP.value] = group

        self.vessel_slots_filled = 0

        self._initialize_containers()
        self.available_groups = np.unique(
            self.yard_state[self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1, StateIds.GROUP.value]
        )

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

    def _get_next_vessel_slot(self):
        if self.container_type == "one":
            valid_slots = np.where(
                (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
                & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
                & np.isin(self.vessel_state[:, StateIds.GROUP.value], self.available_groups)
            )[0]
            if len(valid_slots) == 0:
                return None

            # Retrieve the bay, row, and tier coordinates for the valid slots.
            bays = self.vessel_state[valid_slots, StateIds.BAY.value]
            rows = self.vessel_state[valid_slots, StateIds.ROW.value]
            tiers = self.vessel_state[valid_slots, StateIds.TIER.value]

            sort_order = np.lexsort((rows, tiers, bays))
            return valid_slots[sort_order[0]]

    def _create_observation(self):
        state = np.concatenate((self.vessel_state, self.yard_state), axis=0)
        if self.current_vessel_slot is not None:
            state = np.concatenate((state, self.vessel_state[self.current_vessel_slot].reshape(1, -1)), axis=0)
        else:
            state = np.concatenate((state, np.zeros((1, 5), dtype=int)), axis=0)
        return state

    def _get_valid_yard_actions(self) -> np.ndarray:
        occupied_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1
        if self.container_type == "one":
            bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 == 1
            group_mask = (
                self.yard_state[:, StateIds.GROUP.value]
                == self.vessel_state[self.current_vessel_slot, StateIds.GROUP.value]
            )
            valid_actions = np.where(occupied_mask & bay_mask & group_mask)[0]
        return valid_actions

    def render(self):
        """Render the environment as an RGB array"""
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
            3*padding + 2*title_height + vessel_height + section_gap,
            yard_height,
            self.yard_shape[0],
            self.yard_shape[1],
            self.yard_shape[2],
            False,
        )

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _draw_grid(self, state, top, height, bays, rows, tiers, is_vessel):
        """Draw a grid section (vessel or yard) with fixed cell size"""
        import pygame

        cell_width, cell_height, padding, label_margin = 35, 35, 30, 15
        left_margin = max(padding, (self.screen_width - bays * rows * cell_width) / 2)
        small_font = pygame.font.Font(None, 20)
        tiny_font = pygame.font.Font(None, 18)

        # Each tuple is (light_color, dark_color)
        group_colors = [
            ((230, 230, 255), (100, 100, 220)),  # Blue group
            ((230, 255, 230), (100, 220, 100)),  # Green group
            ((255, 230, 230), (220, 100, 100)),  # Red group
            ((255, 255, 230), (220, 220, 100)),  # Yellow group
            ((230, 255, 255), (100, 220, 220)),  # Cyan group
            ((255, 230, 255), (220, 100, 220)),  # Magenta group
        ]

        empty_color = (255, 255, 255)  # White for empty slots

        for t in range(1, tiers + 1):
            y = top + (tiers - t) * cell_height + cell_height / 2
            tier_label = small_font.render(f"{t}", True, (0, 0, 0))
            self.screen.blit(tier_label, (left_margin - label_margin, y - tier_label.get_height() / 2))

        if is_vessel:
            if rows % 2 == 0:  # Even number of rows
                left = list(range(rows - 1, 0, -2))
                right = list(range(2, rows + 1, 2))
                row_order = left + right
            else:  # Odd number of rows
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
                is_target = self.current_vessel_slot == i
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1:  # Odd bays map directly
                    cell_info[(bay, row, tier)] = {"filled": is_occupied, "target": is_target, "idx": i, "group": group}

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
                                (adj_bay, row, tier), {"filled": False, "target": False, "idx": None, "group": group}
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
                    cell_info[(bay, row, tier)] = {"filled": True, "target": False, "idx": i, "group": group}

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

                    cell = cell_info.get((bay_num, r, t), {"filled": False, "target": False, "idx": None, "group": 0})

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
                    line_width = 3 if cell["target"] else 1
                    pygame.draw.rect(self.screen, line_color, rect, line_width)

                    if cell["filled"] and cell["idx"] is not None:
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

    def _get_bay_groups(self, type="yard") -> dict:
        """generate bay groups based on the yard shape
        Returns:
            dict: dictionary containing bay groups, e.g. {1: [1, 2, 3], 2: [5, 6, 7]}
        """
        state = self.yard_state if type == "yard" else self.vessel_state
        bays = sorted(np.unique(state[:, StateIds.BAY.value]))

        groups, current_group, group_id = {}, [], 1

        for bay in bays:
            if current_group and bay - current_group[-1] > 1:
                groups[group_id] = current_group
                current_group, group_id = [], group_id + 1
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
                selected_slots = available_slots[:num_to_set]
                self.yard_state[selected_slots, StateIds.IS_OCCUPIED.value] = 1

                if self.group_num > 1:
                    containers_per_group = num_to_set // self.group_num

                    if self.group_placement == "fixed":
                        for group in range(self.group_num):
                            start_idx = group * containers_per_group
                            end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                            if start_idx < end_idx:
                                self.yard_state[selected_slots[start_idx:end_idx], StateIds.GROUP.value] = group
                    else:
                        rng = np.random.RandomState(self.seed)

                        shuffled_indices = rng.permutation(num_to_set)
                        shuffled_slots = selected_slots[shuffled_indices]
                        for group in range(self.group_num):
                            start_idx = group * containers_per_group
                            end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                            if start_idx < end_idx:
                                self.yard_state[shuffled_slots[start_idx:end_idx], StateIds.GROUP.value] = group
