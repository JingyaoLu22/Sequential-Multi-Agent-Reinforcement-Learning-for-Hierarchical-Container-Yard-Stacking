import copy


def enumeration_solver(env):
    best_reward = -float("inf")
    best_actions = []

    def dfs(current_env, current_actions, current_reward):
        nonlocal best_reward, best_actions
        valid_actions = current_env._get_valid_yard_actions().tolist()

        if not valid_actions:
            if current_reward > best_reward:
                best_reward = current_reward
                best_actions = current_actions.copy()
            return

        for action in valid_actions:
            new_env = copy.deepcopy(current_env)
            _, reward, terminated, _, _ = new_env.step(action)
            new_total_reward = current_reward + reward

            if terminated:
                if new_total_reward > best_reward:
                    best_reward = new_total_reward
                    best_actions = current_actions + [action]
            else:
                dfs(new_env, current_actions + [action], new_total_reward)

    initial_env = copy.deepcopy(env)
    initial_env.reset()
    dfs(initial_env, [], 0)

    return best_actions, best_reward
