from __future__ import annotations
from typing import Callable, Type
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from envs.stack_gym import StackEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
import torch

import numpy as np

class MaskedEvalCallback(BaseCallback):
    """Evaluation callback with action masking support and best-model saving."""

    def __init__(
        self,
        eval_env: ActionMasker,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 100,
        verbose: int = 1,
        save_best_model: bool = False,
        save_dir: str | None = None,
        save_filename: str | None = None,
        max_reward_threshold: float | None = None,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_best_model = save_best_model
        self.save_dir = save_dir
        self.save_filename = save_filename
        self.max_reward_threshold = max_reward_threshold
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            all_rewards = []

            # Evaluate using current model for n_eval_episodes
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                terminated = False
                truncated = False
                total_reward = 0.0

                while not (terminated or truncated):
                    action, _ = self.model.predict(
                        obs,
                        deterministic=True,
                        action_masks=get_action_masks(self.eval_env)
                    )
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    total_reward += reward

                all_rewards.append(total_reward)

            all_rewards = np.array(all_rewards)
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            min_reward = np.min(all_rewards)
            max_reward = np.max(all_rewards)

            print("\n" + "=" * 70)
            print("EVALUATION RESULTS")
            print("=" * 70)
            print(f"Step: {self.num_timesteps} | Episodes: {self.n_eval_episodes}")
            print(f"Mean Reward: {mean_reward:.4f}")
            print(f"Std Reward:  {std_reward:.4f}")
            print(f"Min Reward:  {min_reward:.4f}")
            print(f"Max Reward:  {max_reward:.4f}")

            if self.max_reward_threshold is not None:
                pct = np.sum(all_rewards >= self.max_reward_threshold) / self.n_eval_episodes * 100
                print(f"% episodes >= threshold ({self.max_reward_threshold}): {pct:.2f}%")

            print("=" * 70)

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/min_reward", min_reward)
            self.logger.record("eval/max_reward", max_reward)


            if self.max_reward_threshold is not None:
                self.logger.record("eval/pct_above_threshold", pct)
            self.logger.dump(self.num_timesteps)

            # Save best model based on best mean eval reward
            if self.save_best_model and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                path = f"{self.save_dir}/{self.save_filename}_best"
                self.model.save(path)
                if self.verbose:
                    print(f"New best model saved to {path} (mean reward: {mean_reward:.4f})")

        return True
    
class SaveModelCallback(BaseCallback):
    """Callback to periodically save the model, overwriting the previous save.
    Not being used currently (currently modle is saved withtin MaskedEvalCallback).
    But will be needed in the future."""

    def __init__(
        self,
        save_freq: int,
        save_dir: str,
        save_filename: str,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.save_filename = save_filename

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.save_dir}/{self.save_filename}"
            self.model.save(path)
            if self.verbose:
                print(f"Model saved to {path} at step {self.num_timesteps}")
        return True
    
    
def mask_fn(env: ActionMasker) -> np.ndarray:
    """Extract action mask from environment for MaskablePPO.
        Needed for sb3_contrib action masking to work"""
    return env.action_masks()


def make_env(config: dict, rank: int = 0) -> Callable[[], ActionMasker]:
    """
    Utility function to create a single environment instance.

    Args:
        config: Environment configuration dict
        rank: Environment ID needed for properly seeding (for parallel envs)

    Returns:
        Wrapped environment.
    """
    def _init() -> ActionMasker:
        env_config = config.copy()
        if env_config.get('seed') is not None:
            env_config['seed'] = config['seed'] + rank

        env = StackEnv(config=env_config, render_mode=None)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=env_config.get('seed'))  # Explicitly seed first reset with rank offset

        return env

    return _init

def create_parallel_envs(
    config: dict,
    n_envs: int = 4,
    vec_env_cls: Type[VecEnv] = SubprocVecEnv,
) -> VecEnv:
    """
    Create parallel environments using SB3's VecEnv wrappers.

    Args:
        config: Environment configuration dict
        n_envs: Number of parallel environments
        vec_env_cls: VecEnv class (SubprocVecEnv for multiprocessing, DummyVecEnv for single process)

    Returns:
        Vectorized environment
    """
    env_fns = [make_env(config, i) for i in range(n_envs)]
    vec_env = vec_env_cls(env_fns)

    return vec_env





def save_model(model: MaskablePPO, dir: str, filename: str) -> None:
    """Utility function to save the trained model"""
    model.save(f"{dir}/{filename}")
    print(f"Model saved to {dir}/{filename}")


def set_config(size: str = "small", seed: int = 42) -> dict:
    """Utility function to get environment configurations"""
    if size == "small":
        config = {
            "vessel_shape": (3, 3, 3),
            "yard_shape": (3, 3, 3),
            "num_containers": 27,
            "group_num": 3,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features",
            "reward_norm": True,
            "reward_clip": True
        }
    elif size == "medium":
        config = {
            "vessel_shape": (4, 4, 4),
            "yard_shape": (4, 4, 4),
            "num_containers": 64,
            "group_num": 4,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features",
            "reward_norm": True,
            "reward_clip": True
        }
    elif size == "large":
        config = {
            "vessel_shape": (6, 6, 5),
            "yard_shape": (6, 6, 5),
            "num_containers": 180,
            "group_num": 6,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features",
            "reward_norm": True,
            "reward_clip": True
        }
    else:
        raise ValueError("Invalid size. Choose 'small', 'medium', or 'large'.")
    
    return config

def get_device() -> str:
    """Utility function to get the available device (GPU or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
