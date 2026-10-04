from __future__ import annotations
from typing import Callable, Type
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from envs.stack_gym import StackEnv
from envs.hierarchical_envs.hierarchical_low_level_env import HierarchicalLowLevelEnv
from envs.hierarchical_envs.hierarchical_high_level_env import HierarchicalHighLevelEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
import torch
import os

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
        if save_best_model and save_filename is None:
            self.save_filename = "best_model"
        else:
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
                        action_masks=get_action_masks(self.eval_env),
                    )
                    obs, reward, terminated, truncated, info = self.eval_env.step(
                        action
                    )
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
                pct = (
                    np.sum(all_rewards >= self.max_reward_threshold)
                    / self.n_eval_episodes
                    * 100
                )
                print(
                    f"% episodes >= threshold ({self.max_reward_threshold}): {pct:.2f}%"
                )

            print("=" * 70)

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/min_reward", min_reward)
            self.logger.record("eval/max_reward", max_reward)

            if self.max_reward_threshold is not None:
                self.logger.record("eval/pct_above_threshold", pct)
            self.logger.dump(self.num_timesteps)

            # Save best model based on best mean eval reward
            if self.save_best_model and mean_reward >= self.best_mean_reward:
                self.best_mean_reward = mean_reward
                path = f"{self.save_dir}/{self.save_filename}"
                self.model.save(path)
                if self.verbose:
                    print(
                        f"New best model saved to {path} (mean reward: {mean_reward:.4f})"
                    )

        return True


class SaveModelCallback(BaseCallback):
    """Callback to periodically save model checkpoints at fixed step intervals.
    Creates a subfolder for each training run and saves checkpoints with step-based naming.
    E.g., {save_dir}/{save_filename}/checkpoint_1000000_steps.zip"""

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
        self.last_saved_step = 0

    def _on_step(self) -> bool:
        # Check if we've reached a new milestone (every save_freq steps)
        current_milestone = (self.num_timesteps // self.save_freq) * self.save_freq
        
        # Only save if we've crossed a new milestone and it's not step 0
        if current_milestone > self.last_saved_step and current_milestone > 0:
            self.last_saved_step = current_milestone
            
            # Create subfolder: save_dir/save_filename/
            base_dir = os.path.join(self.save_dir, self.save_filename)
            os.makedirs(base_dir, exist_ok=True)
            
            # Save with step count in filename
            checkpoint_name = f"checkpoint_{current_milestone}_steps"
            
            path = os.path.join(base_dir, checkpoint_name)
            self.model.save(path)
            if self.verbose:
                print(f"Checkpoint saved to {path} at step {self.num_timesteps}")
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
        if env_config.get("seed") is not None:
            env_config["seed"] = config["seed"] + rank

        env = StackEnv(config=env_config, render_mode=None)
        env = ActionMasker(env, mask_fn)
        env.reset(
            seed=env_config.get("seed")
        )  # Explicitly seed first reset with rank offset

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


def make_hierarchical_env(
    config: dict,
    high_level_policy_type: str = "rule_based_grouped",
    rank: int = 0,
) -> Callable[[], ActionMasker]:
    """
    Factory for a single HierarchicalLowLevelEnv instance.

    Args:
        config: Environment configuration dict
        high_level_policy_type: Policy for the embedded high-level agent
        rank: Environment ID for seeding in parallel setups

    Returns:
        Thunk that creates a wrapped HierarchicalLowLevelEnv.
    """

    def _init() -> ActionMasker:
        env_config = config.copy()
        if env_config.get("seed") is not None:
            env_config["seed"] = config["seed"] + rank

        env = HierarchicalLowLevelEnv(
            config=env_config,
            high_level_policy_type=high_level_policy_type,
            render_mode=None,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=env_config.get("seed"))
        return env

    return _init


def create_parallel_hierarchical_envs(
    config: dict,
    high_level_policy_type: str = "rule_based_grouped",
    n_envs: int = 4,
    vec_env_cls: Type[VecEnv] = SubprocVecEnv,
) -> VecEnv:
    """
    Create parallel HierarchicalLowLevelEnv environments.

    Args:
        config: Environment configuration dict
        high_level_policy_type: Policy for the embedded high-level agent
        n_envs: Number of parallel environments
        vec_env_cls: VecEnv class

    Returns:
        Vectorized environment
    """
    env_fns = [
        make_hierarchical_env(config, high_level_policy_type, i)
        for i in range(n_envs)
    ]
    return vec_env_cls(env_fns)


def make_high_level_env(
    config: dict,
    low_level_policy_type: str = "rule_based_grouped",
    low_level_model_path: str | None = None,
    rank: int = 0,
) -> Callable[[], ActionMasker]:
    """
    Factory for a single HierarchicalHighLevelEnv instance (parallel-safe).

    When low_level_model_path is provided the frozen RL agent is created
    inside the thunk so each subprocess gets its own model copy.

    Args:
        config: Environment configuration dict
        low_level_policy_type: Policy for the embedded low-level agent
        low_level_model_path: Path to a saved MaskablePPO low-level checkpoint
        rank: Environment ID for seeding in parallel setups

    Returns:
        Thunk that creates a wrapped HierarchicalHighLevelEnv.
    """

    def _init() -> ActionMasker:
        env_config = config.copy()
        if env_config.get("seed") is not None:
            env_config["seed"] = config["seed"] + rank

        env = HierarchicalHighLevelEnv(
            config=env_config,
            low_level_policy_type=low_level_policy_type,
            low_level_model_path=low_level_model_path,
            render_mode=None,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=env_config.get("seed"))
        return env

    return _init


def create_parallel_high_level_envs(
    config: dict,
    low_level_policy_type: str = "rule_based_grouped",
    low_level_model_path: str | None = None,
    n_envs: int = 4,
    vec_env_cls: Type[VecEnv] = SubprocVecEnv,
) -> VecEnv:
    """
    Create parallel HierarchicalHighLevelEnv environments.

    Args:
        config: Environment configuration dict
        low_level_policy_type: Policy for the embedded low-level agent
        low_level_model_path: Path to a saved MaskablePPO low-level checkpoint
        n_envs: Number of parallel environments
        vec_env_cls: VecEnv class

    Returns:
        Vectorized environment
    """
    env_fns = [
        make_high_level_env(config, low_level_policy_type, low_level_model_path, i)
        for i in range(n_envs)
    ]
    return vec_env_cls(env_fns)


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
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": False,
            "enable_imo": False
        }
    elif size == "small_with_margin":
        config = {
            "vessel_shape": (3, 3, 3),
            "yard_shape": (3, 4, 3),
            "num_containers": 27,
            "group_num": 3,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": True,
            "enable_imo": True
        }
    elif size == "medium":
        config = {
            "vessel_shape": (4, 4, 4),
            "yard_shape": (4, 4, 4),
            "num_containers": 64,
            "group_num": 4,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": False,
            "enable_imo": False
        }
    elif size == "large":
        config = {
            "vessel_shape": (6, 6, 5),
            "yard_shape": (6, 6, 5),
            "num_containers": 180,
            "group_num": 6,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": False,
            "enable_imo": False
        }
    elif size == "large_v2":
        config = {
            "vessel_shape": (8, 5, 5),
            "yard_shape": (8, 5, 5),
            "num_containers": 200,
            "group_num": 8,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": False,
            "enable_imo": False
        }
    elif size == "large_v3":
        config = {
            "vessel_shape": (10, 6, 5),
            "yard_shape": (10, 6, 5),
            "num_containers": 300,
            "group_num": 10,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": False,
            "enable_imo": False
        }
    elif size == "large_v4":
        config = {
            "vessel_shape": (10, 8, 5),
            "yard_shape": (10, 8, 5),
            "num_containers": 400,
            "group_num": 10,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": False,
            "enable_imo": False
        }
    elif size == "medium_with_margin":
        config = {
            "vessel_shape": (4, 4, 4),
            "yard_shape": (4, 5, 4),
            "num_containers": 64,
            "group_num": 4,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": True,
            "enable_imo": True
        }
    elif size == "large_with_margin":
        config = {
            "vessel_shape": (6, 6, 5),
            "yard_shape": (6, 7, 5),
            "num_containers": 180,
            "group_num": 6,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": True,
            "enable_imo": True
        }
    elif size == "large_v2_with_margin":
        config = {
            "vessel_shape": (8, 5, 5),
            "yard_shape": (8, 7, 5),
            "num_containers": 200,
            "group_num": 8,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": True,
            "enable_imo": True
        }
    elif size == "large_v3_with_margin":
        config = {
            "vessel_shape": (10, 6, 5),
            "yard_shape": (10, 7, 5),
            "num_containers": 300,
            "group_num": 10,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": True,
            "enable_imo": True
        }
    elif size == "large_v4_with_margin":
        config = {
            "vessel_shape": (10, 8, 5),
            "yard_shape": (10, 9, 5),
            "num_containers": 400,
            "group_num": 10,
            "group_placement": "random",
            "seed": seed,
            "observation_type": "stack_features_v3",
            "reward_norm": True,
            "reward_clip": True,
            "stack_fill_penalty": True,
            "container_sizes": True,
            "enable_imo": True
        }
    else:
        raise ValueError("Invalid size. Choose 'small', 'small_with_margin', 'medium', 'medium_with_margin', 'large', 'large_with_margin', 'large_v2', 'large_v2_with_margin', 'large_v3', 'large_v3_with_margin', 'large_v4', or 'large_v4_with_margin'.")

    return config


def get_device() -> str:
    """Utility function to get the available device (GPU or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
