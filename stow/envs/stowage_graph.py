from .stowage_gym import StowageEnv, StateIds
from gymnasium.spaces import Graph, Box, Discrete, GraphInstance
import numpy as np


class StowageGraphEnv(StowageEnv):
    def __init__(self, config):
        super().__init__(config)
        # Additional initialization for graph-based environment can be added here
        self.observation_space = Graph(
            node_space=Box(
                low=0,
                high=max(
                    max(self.num_vessel_bay, self.num_yard_bay),  # bay upper limit
                    max(self.vessel_shape[1], self.yard_shape[1]),  # row upper limit
                    max(self.vessel_shape[2], self.yard_shape[2]),  # tier upper limit
                    1,  # occupied upper limit
                    self.group_num,  # group upper limit
                ),
                shape=(5,),
                dtype=np.int64,
            ),
            edge_space=Discrete(self.yard_shape[2]),
            seed=123,
        )

    def reset(self):
        super()._reset()
        return self._update_graph_info(), {}

    def step(self, action):
        self.total_timesteps += 1
        truncated = False if self.total_timesteps < self.num_containers * 10 else True  # Handle timeout
        truncated = False
        info = {}
        valid_actions = self._get_valid_yard_actions()
        if type(valid_actions) is not list:
            valid_actions = valid_actions.tolist()

        if action not in valid_actions:
            reward = -100.0
            observation = self.obs
            info["yard_mask"] = valid_actions
            terminated = False
            return observation, reward, terminated, truncated, info
        shifters = self._process_shifters(action, self.current_vessel_slot)
        # Sequencer select next vessel slot
        self.current_vessel_slot = self._get_next_vessel_slot()
        reward = -shifters
        terminated = (self.current_vessel_slot is None) or (self.available_groups.size == 0)
        valid_actions = self._get_valid_yard_actions() if not terminated else []

        observation = self._update_graph_info()
        self.total_shifters += shifters
        info.update({"yard_mask": valid_actions, "shifters": shifters, "vessel_slots_filled": self.vessel_slots_filled})
        info["total_shifters"] = self.total_shifters

        return observation, reward, terminated, truncated, info

    def _update_graph_info(self):
        self.unfilled_vessel_indices = np.where(self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)[0]
        self.filled_yard_indices = np.where(self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)[0]

        unfilled_vessel_slots = self.vessel_state[self.unfilled_vessel_indices]
        filled_yard_slots = self.yard_state[self.filled_yard_indices]

        nodes = np.concatenate((unfilled_vessel_slots, filled_yard_slots), axis=0)

        edges, edge_links = self.build_edges(unfilled_vessel_slots, filled_yard_slots)
        self.obs = GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)
        return self.obs

    def build_edges(self, vessel_state, yard_state):
        vessel_edges, vessel_edge_links = self._connect_adjacent(vessel_state, offset=0)
        yard_edges, yard_edge_links = self._connect_adjacent(yard_state, offset=len(vessel_state))

        vessel_pos = np.where(self.unfilled_vessel_indices == self.current_vessel_slot)[0]
        valid_yard_actions = self._get_valid_yard_actions()
        valid_positions = np.isin(self.filled_yard_indices, valid_yard_actions)
        valid_node_indices = np.where(valid_positions)[0] + len(self.unfilled_vessel_indices)
        shifter_edges = []
        shifter_edge_links = []
        for i, vp in enumerate(valid_node_indices):
            shifter_edges.append(len(self._get_shifters(valid_yard_actions[i])))
            shifter_edge_links.append([vp, vessel_pos[0]])

        edges = vessel_edges + yard_edges + shifter_edges
        edge_links = vessel_edge_links + yard_edge_links + shifter_edge_links

        return np.array(edges), np.array(edge_links)

    def _connect_adjacent(self, coords, offset=0):
        """
        coords: np.ndarray of shape (N, num_slot_attrs)
        offset: Node index offset for the current set of coords
        """
        edges = []
        edge_links = []
        coord_to_idx = {
            tuple(c[:3]): i + offset  # (bay, row, tier) -> node index
            for i, c in enumerate(coords)
        }
        for i, c in enumerate(coords):
            b, r, t = c[:3]
            idx = i + offset
            # Neighbors in 6 directions
            neighbors = [
                (b + 1, r, t),
                (b - 1, r, t),
                (b, r + 1, t),
                (b, r - 1, t),
                (b, r, t + 1),
                (b, r, t - 1),
            ]
            for nb in neighbors:
                if nb in coord_to_idx:  # Valid neighbor exists
                    j = coord_to_idx[nb]
                    edge_links.append([idx, j])
                    edges.append(0)  # Can use 0/1/2 to represent edge types, here we use 0

        return edges, edge_links
