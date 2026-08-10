"""
`python train_spge.py --config spge.ppo.yaml`
"""
import numpy as np
import argparse
from torch.nn import Tanh, ReLU
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from hpo_spge import NAME_TO_ALGO
# from stable_baselines3.common.monitor import Monitor
from hpo_spge import NAME_TO_ENV
from hpo_spge import load_config
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}


class TrialEvalCallback(MaskableEvalCallback, WandbCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        n_eval_episodes: int = 5,
        eval_freq: int = 500,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        MaskableEvalCallback.__init__(
            self,
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        WandbCallback.__init__(self)

    def _on_step(self) -> bool:
        WandbCallback._on_step(self)
        MaskableEvalCallback._on_step(self)
        return True


class TestCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        # Call the parent class's method to log the metrics
        super()._on_step()
        if self.num_timesteps % n_eval_freq == 0 or self.num_timesteps == 1:
            # Evaluate the policy
            rew_lst = []
            ep_len_lst = []
            time_lst = []
            for i in range(n_evaluations):
                test_env = NAME_TO_ENV[env_name](config["environment"])
                obs, _ = test_env.reset()
                rewards = []
                total_shifters_lst = []
                ep_len = 0
                while True:
                    action, _ = self.model.predict(
                        obs, deterministic=True, action_masks=np.expand_dims(test_env.action_masks(), axis=0)
                    )
                    ep_len += 1
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    rewards.append(reward)
                    if terminated or truncated:
                        total_shifters_lst.append(info["total_shifters"])
                        current_time=info.get("current_time")
                        break
                mean_r = np.sum(rewards)
                rew_lst.append(mean_r)
                time_lst.append(current_time)
                ep_len_lst.append(ep_len)
            # Log to tensorboard
            self.logger.record("eval/mean_reward", np.mean(rew_lst))
            self.logger.record("eval/mean_shifters", np.mean(total_shifters_lst))
            self.logger.record("eval/mean_time", np.mean(time_lst))
            self.logger.record("eval/ep_len", np.mean(ep_len_lst))
            self.logger.dump(self.num_timesteps)
        return True


def get_argparser():
    """Define all the Argparse arguments for the training script."""
    parser = argparse.ArgumentParser(description="Training arguments for PPO with configurable YAML setup.")

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="train.sb3hpo.yaml",
        help="Path to the configuration YAML file. Example: train.default.yaml (placed in StowAI/train_configs/).",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="environment name to be used in the training. Example: SPGE.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Algorithm to be used in the training. Example: ppo.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes for the study. Example: 'First trial with PPO'.",
    )

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    config_str = args.config
    config = load_config(config_str)
    env_name = args.env_name if args.env_name else config["general"]['env_name']
    config["general"]["algorithm"] = args.algo if args.algo else config["general"]["algorithm"]

    n_evaluations = config["general"]["n_evaluations"]
    n_eval_freq = config["general"]["n_eval_freq"]

    # Load Hyperparameters
    kwargs = config["hyperparameters"]
    if kwargs:
        kwargs["policy_kwargs"]["activation_fn"] = Tanh if kwargs["policy_kwargs"]["activation_fn"] == "tanh" else ReLU
    else:
        kwargs = {}

    # Load the environment
    environment_config = config["environment"]
    environment = NAME_TO_ENV[env_name]

    for _ in range(config["general"]["n_repetition"]):
        train_env = NAME_TO_ENV[env_name](config["environment"])
        obs, info = train_env.reset()
        # Load the algorithm
        algorithm = NAME_TO_ALGO[config["general"]["algorithm"]]
        print("Algorithm: ", config["general"]["algorithm"], "Environment: ", env_name)

        run = wandb.init(
            project=config["logging"]["wandb_project"],
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            dir="../data1",
        )
        model = algorithm(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=f"../data1/runs/{run.id}",
            **kwargs,
        )
        model.learn(
            total_timesteps=config["general"]["n_timesteps"],
            # callback=TrialEvalCallback(
            #     eval_env=NAME_TO_ENV[env_name](config["environment"]),
            #     n_eval_episodes=n_evaluations,
            #     eval_freq=n_eval_freq,
            #     deterministic=True,
            # ),
            callback=TestCallback(),
        )
        wandb.finish()
