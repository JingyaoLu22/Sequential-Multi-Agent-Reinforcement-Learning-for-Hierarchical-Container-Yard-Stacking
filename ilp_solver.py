import pulp
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from envs.stowage_crane_gym import MultiCraneStowageEnv
from envs.stowage_aec import StowageAEC
from envs.stowage_gym import StowageEnv, StateIds


class ILPStowageSolver:
    def __init__(self, env, lookahead_steps: int = 3, time_limit: int = 10):
        """
        Initialize the ILP-based stowage solver.
        """
        self.env = env
        self.lookahead_steps = lookahead_steps
        self.time_limit = time_limit

    def solve_episode(self) -> Tuple[float, List[int], Dict]:
        observation = self.env.reset()

        total_reward = 0.0
        actions = []
        episode_info = {"total_shifters": 0, "vessel_slots_filled": 0, "steps": 0, "ilp_solve_times": []}

        terminated = False
        truncated = False

        while not (terminated or truncated):
            valid_actions = self._get_current_valid_actions()

            if not valid_actions:
                print("No valid actions available, ending episode.")
                break

            best_action = valid_actions[0] if len(valid_actions) == 1 else self._solve_multistep_ilp(valid_actions)
            observation, reward, terminated, truncated, info = self.env.step(best_action)

            total_reward += reward
            actions.append(best_action)
            episode_info["steps"] += 1
            episode_info["total_shifters"] = info.get("total_shifters", 0)
            episode_info["vessel_slots_filled"] = info.get("vessel_slots_filled", 0)

            print(
                f"Step {episode_info['steps']}: Action {best_action}, Reward {reward}, Total Reward {total_reward}, Terminated: {terminated}"
            )

        return total_reward, actions, episode_info

    def _get_current_valid_actions(self) -> List[int]:
        """Get the list of valid actions for the current state."""
        mask = self.env.action_masks()
        return np.nonzero(mask)[0].tolist()

    def _solve_step_ilp(self, valid_actions: List[int]) -> Optional[int]:
        """
        Solve ILP for the current step to select the best action.
        """
        if len(valid_actions) == 1:
            return valid_actions[0]
        return self._solve_multistep_ilp(valid_actions)

    def _solve_multistep_ilp(self, valid_actions: List[int]) -> Optional[int]:
        """
        Solve ILP for multiple steps ahead.
        """
        import time

        start_time = time.time()

        prob = pulp.LpProblem("Stowage_Multistep", pulp.LpMinimize)

        current_yard_state = self.env.yard_state.copy()
        remaining_containers = np.sum(current_yard_state[:, 3])  # IS_OCCUPIED

        actual_lookahead = min(self.lookahead_steps, remaining_containers)

        if actual_lookahead <= 1:
            return self._solve_single_step_ilp(valid_actions)

        print(f"Using {actual_lookahead} steps lookahead ILP to solve, valid actions count: {len(valid_actions)}")

        # Decision variables: action selection for each step
        step_action_vars = {}
        for step in range(actual_lookahead):
            step_action_vars[step] = {}
            # The first step can only choose from the current valid actions
            if step == 0:
                action_candidates = valid_actions
            else:
                # Subsequent steps need to estimate possible valid actions
                action_candidates = self._estimate_future_valid_actions(step, current_yard_state)

            for action in action_candidates:
                step_action_vars[step][action] = pulp.LpVariable(f"step_{step}_action_{action}", cat="Binary")

        for step in range(actual_lookahead):
            if step_action_vars[step]:  # Ensure there are selectable actions
                prob += pulp.lpSum([step_action_vars[step][action] for action in step_action_vars[step]]) <= 1

        # The first step must choose one action
        if step_action_vars[0]:
            prob += pulp.lpSum([step_action_vars[0][action] for action in step_action_vars[0]]) == 1

        # State consistency constraints: ensure the feasibility of the action sequence
        state_consistency_vars = {}
        for step in range(actual_lookahead):
            for action in step_action_vars.get(step, {}):
                # Create state impact variables for each action
                shifters_needed = self._calculate_shifters_needed(action, current_yard_state, step)
                state_consistency_vars[f"step_{step}_action_{action}_shifters"] = shifters_needed

        # Objective function: minimize total shifters + time penalty
        objective_terms = []
        for step in range(actual_lookahead):
            step_weight = 1.0 / (step + 1)  # Recent steps have higher weight
            for action in step_action_vars.get(step, {}):
                shifters_cost = state_consistency_vars.get(f"step_{step}_action_{action}_shifters", 0)
                time_penalty = step * self.env.time_penalty_coef
                total_cost = shifters_cost + time_penalty

                objective_terms.append(total_cost * step_weight * step_action_vars[step][action])

        if objective_terms:
            prob += pulp.lpSum(objective_terms)

        # Solve with an extended time limit for multi-step planning
        extended_time_limit = self.time_limit * actual_lookahead
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=extended_time_limit, threads=4)
        prob.solve(solver)

        solve_time = time.time() - start_time
        print(f"ILP solve time: {solve_time:.3f} seconds, status: {pulp.LpStatus[prob.status]}")

        # Check the solve status and return the optimal action for the first step
        if prob.status == pulp.LpStatusOptimal or prob.status == pulp.LpStatusNotSolved:
            for action in valid_actions:
                if (
                    action in step_action_vars[0]
                    and step_action_vars[0][action].value()
                    and step_action_vars[0][action].value() > 0.5
                ):
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
            immediate_shifters = self._calculate_shifters_needed(action, self.env.yard_state, 0)
            # future_impact = self._calculate_detailed_future_impact(action)

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
        """
        Estimate possible valid actions for future steps
        """
        # Simply return all occupied slots as potential actions
        occupied_slots = np.where(initial_state[:, 3] == 1)[0]  # IS_OCCUPIED
        return occupied_slots.tolist()

    def _calculate_shifters_needed(self, action: int, yard_state: np.ndarray, step: int) -> int:
        """
        Calculate the number of shifters needed to access the container at the given action index.
        """
        if action >= len(yard_state) or yard_state[action, 3] != 1:
            return 999

        bay = yard_state[action, 0]
        row = yard_state[action, 1]
        tier = yard_state[action, 2]

        higher_tier_mask = (
            (yard_state[:, 0] == bay)  # same bay
            & (yard_state[:, 1] == row)  # same row
            & (yard_state[:, 2] > tier)  # higher tier
            & (yard_state[:, 3] == 1)  # occupied
        )

        return int(np.sum(higher_tier_mask))

    def _calculate_action_cost(self, yard_idx: int) -> float:
        """
        Calculate a cost metric for the given yard index action.
        """
        yard_state = self.env.yard_state.copy()

        if yard_idx >= len(yard_state) or yard_state[yard_idx, 3] != 1:  # IS_OCCUPIED
            return float("inf")  # invalid action

        original_bay = yard_state[yard_idx, 0]  # BAY
        original_row = yard_state[yard_idx, 1]  # ROW
        original_tier = yard_state[yard_idx, 2]  # TIER

        same_bay_row_mask = (
            (yard_state[:, 0] == original_bay)  # same bay
            & (yard_state[:, 1] == original_row)  # same row
            & (yard_state[:, 2] > original_tier)  # higher tier
            & (yard_state[:, 3] == 1)  # occupied
        )

        shifters_needed = np.sum(same_bay_row_mask)
        total_cost = shifters_needed

        return total_cost


def run_ilp_solver_demo(env_config: Dict = None):
    if env_config is None:
        # env_config = {
        #     "vessel_shape": (8, 5, 5),
        #     "yard_shape": (8, 5, 5),
        #     "num_containers": 200,
        #     "group_num": 8,
        #     "seed": 4307,
        #     "group_placement": "random",
        # }
        env_config = {
            "vessel_shape": (8, 5, 8),
            "yard_shape": (8, 5, 8),
            "num_containers": 200,
            "group_num": 5,
            "num_cranes": 3,
            "group_placement": "random",
            "seed": 4307,
            "time_penalty_coef": 0.002,
        }

    # env = StowageEnv(config=env_config)
    env = StowageAEC(config=env_config)

    solver = ILPStowageSolver(env, lookahead_steps=200, time_limit=10000)

    total_reward, actions, info = solver.solve_episode()

    print("\n=== ILP Solver Episode Summary ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    print(f"Total Shifters: {info['total_shifters']}")
    print(f"Filled Vessel Slots: {info['vessel_slots_filled']}")
    print(f"Action Sequence: {actions}")

    return total_reward, actions, info


if __name__ == "__main__":
    total_reward, actions, info = run_ilp_solver_demo()
    print("ILP Solver ran successfully!")
