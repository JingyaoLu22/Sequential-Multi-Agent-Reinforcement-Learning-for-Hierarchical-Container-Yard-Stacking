from envs.stowage_gym import StowageEnv
from algorithms.enumeration import enumeration_solver
if __name__ == "__main__":

    config_dict = {
        "vessel_shape": (1, 2, 2),
        "yard_shape": (2, 2, 2),
        "num_containers": 8,
        "group_num": 3,
        "group_placement": "random",
        "seed": 4307,
    }
    env = StowageEnv(config_dict)
    actions, reward = enumeration_solver(env)
    print(actions)
    print(reward)