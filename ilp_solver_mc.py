import numpy as np
import copy
from typing import List, Tuple, Optional
import random
from envs.stowage_gym import StowageEnv

import pulp
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from envs.stowage_crane_gym import MultiCraneStowageEnv
from envs.stowage_aec import StowageAEC
from envs.stowage_gym import StowageEnv, StateIds
from ilp_solver import ILPStowageSolver


class ILPStowageSolverMC(ILPStowageSolver):
    def __init__(self, env, lookahead_steps: int = 3, time_limit: int = 10):
        super().__init__(env, lookahead_steps, time_limit)

    def _solve_multistep_ilp(self, valid_actions: List[int]) -> Optional[int]:
        import time

        start_time = time.time()

        prob = pulp.LpProblem("Stowage_Multistep", pulp.LpMinimize)

        # 获取当前状态快照
        current_yard_state = self.env.yard_state.copy()
        remaining_containers = np.sum(current_yard_state[:, 3])  # IS_OCCUPIED

        actual_lookahead = min(self.lookahead_steps, remaining_containers)

        if actual_lookahead <= 1:
            return self._solve_single_step_ilp(valid_actions)

        print(f"using lookahead of {actual_lookahead} steps for ILP")

        step_action_vars = {}
        for step in range(actual_lookahead):
            step_action_vars[step] = {}
            if step == 0:
                action_candidates = valid_actions
            else:
                action_candidates = self._estimate_future_valid_actions(step, current_yard_state)

            for action in action_candidates:
                step_action_vars[step][action] = pulp.LpVariable(f"step_{step}_action_{action}", cat="Binary")

        for step in range(actual_lookahead):
            if step_action_vars[step]:
                prob += pulp.lpSum([step_action_vars[step][action] for action in step_action_vars[step]]) <= 1

        if step_action_vars[0]:
            prob += pulp.lpSum([step_action_vars[0][action] for action in step_action_vars[0]]) == 1

        state_consistency_vars = {}
        for step in range(actual_lookahead):
            for action in step_action_vars.get(step, {}):
                yard_choice, crane_idx = self.env._decode_action(action)
                shifters_needed = self._calculate_shifters_needed(yard_choice, current_yard_state, step)
                state_consistency_vars[f"step_{step}_action_{action}_shifters"] = shifters_needed

        objective_terms = []
        for step in range(actual_lookahead):
            step_weight = 1.0 / (step + 1)
            for action in step_action_vars.get(step, {}):
                shifters_cost = state_consistency_vars.get(f"step_{step}_action_{action}_shifters", 0)
                time_penalty = step * self.env.time_penalty_coef
                total_cost = shifters_cost + time_penalty

                objective_terms.append(total_cost * step_weight * step_action_vars[step][action])

        if objective_terms:
            prob += pulp.lpSum(objective_terms)
        else:
            return self._simple_heuristic_selection(valid_actions)

        extended_time_limit = self.time_limit * actual_lookahead
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=extended_time_limit, threads=4)
        prob.solve(solver)

        solve_time = time.time() - start_time
        print(f"ILP Solver ran in: {solve_time:.3f} seconds, Status: {pulp.LpStatus[prob.status]}")

        if prob.status == pulp.LpStatusOptimal or prob.status == pulp.LpStatusNotSolved:
            for action in valid_actions:
                if (
                    action in step_action_vars[0]
                    and step_action_vars[0][action].value()
                    and step_action_vars[0][action].value() > 0.5
                ):
                    print(f"ILP Choose: {action}")
                    return action

    def _solve_single_step_ilp(self, valid_actions: List[int]) -> Optional[int]:
        """
        Single step ILP solver with enhanced objective function
        """
        prob = pulp.LpProblem("Stowage_SingleStep", pulp.LpMinimize)

        # Decision variables: action selection
        action_vars = {}
        for action in valid_actions:
            action_vars[action] = pulp.LpVariable(f"action_{action}", cat="Binary")  # Valid actions as binary variables

        # Constraint: only one action can be chosen
        prob += pulp.lpSum([action_vars[action] for action in valid_actions]) == 1

        objective_terms = []
        action_costs = {}

        for action in valid_actions:
            yard_choice, crane_idx = self.env._decode_action(action)
            immediate_shifters = self._calculate_shifters_needed(yard_choice, self.env.yard_state, 0)
            total_cost = immediate_shifters
            action_costs[action] = total_cost

            objective_terms.append(total_cost * action_vars[action])

        prob += pulp.lpSum(objective_terms)

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))

        if prob.status == pulp.LpStatusOptimal:
            for action in valid_actions:
                if action_vars[action].value() and action_vars[action].value() > 0.5:
                    return action

        return None

    def _estimate_future_valid_actions(self, step: int, initial_state: np.ndarray) -> List[int]:
        # occupied_slots = np.where(initial_state[:, 3] == 1)[0]  # IS_OCCUPIED
        # return occupied_slots.tolist()

        # Get all occupied slots (containers that can be moved)
        occupied_slots = np.where(initial_state[:, 3] == 1)[0]  # IS_OCCUPIED

        valid_actions = []

        # Generate all combinations of occupied slots and available cranes
        for yard_slot in occupied_slots:
            for crane_idx in range(self.env.num_cranes):
                # Encode the action: yard_slot * num_cranes + crane_idx
                action = yard_slot * self.env.num_cranes + crane_idx
                valid_actions.append(action)

        return valid_actions


def run_ilp_solver_demo(env_config: Dict = None):
    if env_config is None:
        env_config = {
            "vessel_shape": (8, 5, 8),
            "yard_shape": (8, 5, 8),
            "num_containers": 200,
            "group_num": 8,
            "num_cranes": 3,
            "group_placement": "random",
            "seed": 4307,
            "time_penalty_coef": 0.002,
        }

    env = MultiCraneStowageEnv(config=env_config)

    solver = ILPStowageSolverMC(env, lookahead_steps=200, time_limit=10000)

    total_reward, actions, info = solver.solve_episode()

    print("\n=== Results ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    print(f"Total Shifters: {info['total_shifters']}")
    print(f"Filled Vessel Slots: {info['vessel_slots_filled']}")
    print(f"Action Sequence: {actions}")

    return total_reward, actions, info


if __name__ == "__main__":
    total_reward, actions, info = run_ilp_solver_demo()
    print("ILP Solver ran successfully!")
