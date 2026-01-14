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
    Unstacking/Retrieval Environment
    
    Containers start fully loaded on the vessel. Agent removes containers sequentially
    from the vessel (top containers first) and places them in the yard. The goal is to:
    1. Place similar-group containers close together (bonus if same bay)
    2. Avoid placing near dissimilar containers (penalty)
    3. Consolidate space by reusing bays (penalty for using new bay/row cells)
    
    Agent has flexible group retrieval order - can choose which group to retrieve next.
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
        
        # Reward coefficients
        self.base_placement_reward = config.get("base_placement_reward", 10.0)
        self.same_bay_bonus = config.get("same_bay_bonus", 5.0)
        self.same_group_nearby_bonus = config.get("same_group_nearby_bonus", 2.0)
        self.dissimilar_penalty = config.get("dissimilar_penalty", 3.0)
        self.new_cell_penalty = config.get("new_cell_penalty", 1.0)
        self.time_penalty_coef = config.get("time_penalty_coef", 0.01)
        
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

        # Calculate total coordinates
        self.num_vessel_bay = self.vessel_shape[0] // 2 + self.vessel_shape[0]
        self.num_yard_bay = self.yard_shape[0] // 2 + self.yard_shape[0]
        self.total_vessel_coords = self.num_vessel_bay * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * self.yard_shape[1] * self.yard_shape[2]
        self.obs_coords = self.total_vessel_coords + self.total_yard_coords + 1  # +1 for current target

        # Sequencer
        self.current_vessel_container = None  # Index of container currently being retrieved
        self.current_retrieval_group = None   # Group being retrieved (agent-chosen, flexible)
        self.containers_retrieved = 0

        # Track which bay/row cells have been occupied in yard
        self.yard_bay_row_occupied = set()

        # Observation and action spaces
        observation_space = gym.spaces.Box(
            low=0,
            high=max(
                max(self.num_vessel_bay, self.num_yard_bay),
                max(self.vessel_shape[1], self.yard_shape[1]),
                max(self.vessel_shape[2], self.yard_shape[2]),
                1,
                self.group_num,
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

        self.action_space = gym.spaces.Discrete(self.total_yard_coords)

        # Render part
        self.render_mode = render_mode
        self.screen_width = (
            35 * max(self.vessel_shape[1] * self.vessel_shape[0], self.yard_shape[1] * self.yard_shape[0]) + 60
        )
        self.screen_height = 60 * max(self.vessel_shape[2], self.yard_shape[2]) + 150
        self.screen = None

    def step(self, action):
        """Execute one step: place current vessel container in yard at specified action (yard slot)"""
        self.total_timesteps += 1
        truncated = False
        info = {}
        
        # Get valid actions for current state
        valid_actions = self._get_valid_yard_actions()
        if type(valid_actions) is not list:
            valid_actions = valid_actions.tolist()

        # Check if action is valid
        if action not in valid_actions:
            reward = -100.0
            observation = self._create_observation()
            info["yard_mask"] = valid_actions
            terminated = False
            return observation, reward, terminated, truncated, info

        # Calculate reward for this placement
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
        valid_actions = self._get_valid_yard_actions() if not terminated else []

        observation = self._create_observation()
        info.update({
            "yard_mask": valid_actions,
            "containers_retrieved": self.containers_retrieved,
            "containers_remaining": self.total_containers_remaining,
        })

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        """Reset environment"""
        self._reset()
        observation = self._create_observation()
        info = {}
        return observation, info

    def _reset(self):
        """Internal reset logic"""
        self.total_timesteps = 0
        self.containers_retrieved = 0
        self.yard_bay_row_occupied = set()

        # Initialize yard (empty)
        yard_bays = self._generate_bay_coords(self.yard_shape[0])
        _, R, T = self.yard_shape
        self.yard_state = np.zeros((self.total_yard_coords, self.num_slot_attrs), dtype=int)
        self.yard_state[:, StateIds.BAY.value] = np.repeat(yard_bays, R * T)
        self.yard_state[:, StateIds.ROW.value] = np.tile(np.arange(1, R + 1).repeat(T), len(yard_bays))
        self.yard_state[:, StateIds.TIER.value] = np.tile(np.arange(1, T + 1), len(yard_bays) * R)

        # Initialize vessel (fully loaded)
        vessel_bays = self._generate_bay_coords(self.vessel_shape[0])
        _, Rv, Tv = self.vessel_shape
        self.vessel_state = np.zeros((self.total_vessel_coords, self.num_slot_attrs), dtype=int)
        self.vessel_state[:, StateIds.BAY.value] = np.repeat(vessel_bays, Rv * Tv)
        self.vessel_state[:, StateIds.ROW.value] = np.tile(np.arange(1, Rv + 1).repeat(Tv), len(vessel_bays))
        self.vessel_state[:, StateIds.TIER.value] = np.tile(np.arange(1, Tv + 1), len(vessel_bays) * Rv)

        # Assign groups to vessel containers
        if self.group_num > 1:
            odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 == 1
            odd_bay_indices = np.where(odd_bay_mask)[0]
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
        """Load vessel with containers - mirrors stowage but loads vessel instead of yard"""
        odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 != 0
        available_slots = np.where(odd_bay_mask)[0]
        num_to_set = min(self.num_containers, len(available_slots))

        if num_to_set > 0:
            selected_slots = available_slots[:num_to_set]
            self.vessel_state[selected_slots, StateIds.IS_OCCUPIED.value] = 1

            if self.group_num > 1:
                containers_per_group = num_to_set // self.group_num

                if self.group_placement == "fixed":
                    for group in range(self.group_num):
                        start_idx = group * containers_per_group
                        end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                        if start_idx < end_idx:
                            self.vessel_state[selected_slots[start_idx:end_idx], StateIds.GROUP.value] = group
                else:
                    rng = np.random.RandomState(self.seed)
                    shuffled_indices = rng.permutation(num_to_set)
                    shuffled_slots = selected_slots[shuffled_indices]
                    for group in range(self.group_num):
                        start_idx = group * containers_per_group
                        end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                        if start_idx < end_idx:
                            self.vessel_state[shuffled_slots[start_idx:end_idx], StateIds.GROUP.value] = group

    def _generate_bay_coords(self, physical_bays: int) -> list:
        """Generate bay coordinates based on the number of physical bays"""
        bay_groups = []
        group_start = 1

        for _ in range(physical_bays // 2):
            bay_groups.extend([group_start, group_start + 1, group_start + 2])
            group_start += 4

        if physical_bays % 2 == 1:
            bay_groups.append(group_start)

        return bay_groups

    def _get_next_vessel_container(self):
        """Get next accessible top container from vessel - randomly selected from all topmost containers"""
        occupied_mask = self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 1
        occupied_indices = np.where(occupied_mask)[0]
        
        if len(occupied_indices) == 0:
            return None
        
        # Find topmost (highest tier) container for each bay-row combination
        bays = self.vessel_state[occupied_indices, StateIds.BAY.value]
        rows = self.vessel_state[occupied_indices, StateIds.ROW.value]
        tiers = self.vessel_state[occupied_indices, StateIds.TIER.value]
        
        topmost_containers = []
        unique_bay_rows = set(zip(bays, rows))
        
        # For each bay-row combo, find the highest tier container
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
        """Place vessel container in yard at specified slot, respecting gravity (tier 1 fill first)"""
        # Get yard slot coordinates
        bay = self.yard_state[yard_slot, StateIds.BAY.value]
        row = self.yard_state[yard_slot, StateIds.ROW.value]
        group = self.vessel_state[vessel_container_idx, StateIds.GROUP.value]
        
        # Find lowest unoccupied tier in this bay-row
        bay_row_mask = (self.yard_state[:, StateIds.BAY.value] == bay) & (self.yard_state[:, StateIds.ROW.value] == row)
        bay_row_indices = np.where(bay_row_mask)[0]
        
        # Find lowest unoccupied slot in this bay-row
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
            
            # Track bay-row occupation
            self.yard_bay_row_occupied.add((bay, row))
            self.containers_retrieved += 1

    def _calculate_reward(self, yard_action):
        """Calculate reward for placing current container at yard_action slot"""
        placement_bay = self.yard_state[yard_action, StateIds.BAY.value]
        placement_row = self.yard_state[yard_action, StateIds.ROW.value]
        container_group = self.vessel_state[self.current_vessel_container, StateIds.GROUP.value]
        
        reward = self.base_placement_reward
        
        # Check if placing in new bay-row cell
        if (placement_bay, placement_row) not in self.yard_bay_row_occupied:
            reward -= self.new_cell_penalty
        
        # Analyze neighbors for cohesion rewards
        nearby_indices = self._get_nearby_slots(yard_action)
        
        same_group_nearby = 0
        dissimilar_nearby = 0
        same_bay_same_group = 0
        
        for idx in nearby_indices:
            if self.yard_state[idx, StateIds.IS_OCCUPIED.value] == 1:
                neighbor_group = self.yard_state[idx, StateIds.GROUP.value]
                neighbor_bay = self.yard_state[idx, StateIds.BAY.value]
                
                if neighbor_group == container_group:
                    same_group_nearby += 1
                    if neighbor_bay == placement_bay:
                        same_bay_same_group += 1
                else:
                    dissimilar_nearby += 1
        
        # Add cohesion bonuses
        if same_bay_same_group > 0:
            reward += self.same_bay_bonus
        else:
            reward += same_group_nearby * self.same_group_nearby_bonus
        
        # Add dissimilar penalty
        reward -= dissimilar_nearby * self.dissimilar_penalty
        
        # Add time penalty
        reward -= self.time_penalty_coef * self.total_timesteps
        
        return reward

    def _get_nearby_slots(self, slot_idx, radius=1):
        """Get nearby slot indices (within radius in bay/row/tier space)"""
        bay = self.yard_state[slot_idx, StateIds.BAY.value]
        row = self.yard_state[slot_idx, StateIds.ROW.value]
        tier = self.yard_state[slot_idx, StateIds.TIER.value]
        
        nearby = []
        for idx in range(len(self.yard_state)):
            if idx == slot_idx:
                continue
            other_bay = self.yard_state[idx, StateIds.BAY.value]
            other_row = self.yard_state[idx, StateIds.ROW.value]
            other_tier = self.yard_state[idx, StateIds.TIER.value]
            
            # Check if within radius
            bay_diff = abs(other_bay - bay)
            row_diff = abs(other_row - row)
            tier_diff = abs(other_tier - tier)
            
            if bay_diff <= radius and row_diff <= radius and tier_diff <= radius:
                nearby.append(idx)
        
        return nearby

    def _get_valid_yard_actions(self) -> np.ndarray:
        """Get valid yard placement slots"""
        if self.current_vessel_container is None:
            return np.array([])
        
        # Valid slots: unoccupied, odd bay
        unoccupied_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 0
        bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 == 1
        
        valid_actions = np.where(unoccupied_mask & bay_mask)[0]
        return valid_actions

    def action_masks(self):
        """For compatibility with SB3"""
        if self.current_vessel_container is None:
            return [False] * self.action_space.n

        valid_actions = self._get_valid_yard_actions()
        return [action in valid_actions for action in range(self.action_space.n)]

    def _create_observation(self):
        """Create flattened observation"""
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
        self.screen.blit(font.render("Vessel (Depleting)", True, (0, 0, 0)), (padding, padding))
        self.screen.blit(font.render("Yard (Growing)", True, (0, 0, 0)), (padding, padding + vessel_height + section_gap))

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
        """Draw a grid section (vessel or yard) with fixed cell size"""
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
        """Determine row ordering based on vessel or yard"""
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
        """Draw tier labels on the left side"""
        for t in range(1, tiers + 1):
            y = top + (tiers - t) * cell_height + cell_height / 2
            tier_label = font.render(f"{t}", True, (0, 0, 0))
            self.screen.blit(tier_label, (left_margin - label_margin, y - tier_label.get_height() / 2))

    def _build_cell_info(self, state, bays, is_vessel):
        """Build cell information dictionary for rendering"""
        cell_info = {}

        if is_vessel:
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                is_target = self._is_target_cell(i, is_vessel)
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1:
                    cell_info[(bay, row, tier)] = self._create_cell_props(is_occupied, is_target, i, group)

            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

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

                if bay % 2 == 1 and is_occupied:
                    cell_info[(bay, row, tier)] = self._create_cell_props(True, False, i, group)

            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            cell_info[(adj_bay, row, tier)] = self._create_cell_props(True, False, i, group)

        return cell_info

    def _is_target_cell(self, idx, is_vessel):
        if is_vessel:
            return self.current_vessel_container == idx if self.current_vessel_container is not None else False
        return False

    def _create_cell_props(self, filled, target, idx, group):
        return {"filled": filled, "target": target, "idx": idx, "group": group}

    def _get_cell_border_style(self, cell):
        if cell["target"]:
            return (255, 0, 0), 3
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
        """Draw grid cells with labels"""
        for b in range(bays):
            bay_num = b * 2 + 1
            bay_x = left_margin + b * rows * cell_width

            bay_center_x = bay_x + (rows * cell_width) / 2
            bay_label = fonts["small"].render(f"Bay {bay_num}", True, (0, 0, 0))
            self.screen.blit(bay_label, bay_label.get_rect(center=(bay_center_x, top - 10)))

            for pos, r in enumerate(row_order):
                x_label = bay_x + pos * cell_width + cell_width / 2
                row_label = fonts["small"].render(f"{r}", True, (0, 0, 0))
                self.screen.blit(
                    row_label, row_label.get_rect(center=(x_label, top + tiers * cell_height + label_margin / 2))
                )

                for t in range(1, tiers + 1):
                    x = bay_x + pos * cell_width
                    y = top + (tiers - t) * cell_height

                    default_cell = self._create_cell_props(False, False, None, 0)
                    cell = cell_info.get((bay_num, r, t), default_cell)

                    self._draw_cell(x, y, cell_width, cell_height, cell, colors, fonts, is_vessel)

    def _draw_cell(self, x, y, width, height, cell, colors, fonts, is_vessel):
        """Draw an individual cell"""
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
            elif cell["filled"]:
                label = fonts["tiny"].render(f"{cell['idx']}", True, (255, 255, 255))
                self.screen.blit(label, label.get_rect(center=(x + width / 2, y + height / 2)))

    def _draw_bay_dividers(self, left_margin, top, bays, rows, cell_width, tiers, cell_height, color):
        """Draw vertical divider lines between bays"""
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
