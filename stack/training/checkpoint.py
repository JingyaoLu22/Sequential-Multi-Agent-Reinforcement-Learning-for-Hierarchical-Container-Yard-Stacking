"""
What Sequential HPPO writes to and restores from a run's save directory:

    {prefix}_bay_actor.pt / _row_actor.pt / _critic.pt / _config.json
        "best" and "final" model exports for evaluation.
    latest_training_state.pt
        The restartable state: networks, optimizers, counters, best
        evaluation reward, RNG states and each StackEnv copy's episode seed.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from ..configs.hierarchical_config import HierarchicalConfig
from .sequential_trainer import SequentialPPOTrainer

LATEST_TRAINING_STATE_FILENAME = "latest_training_state.pt"
TRAINING_CHECKPOINT_FORMAT_VERSION = 2

# Settings a resumed run may change (buffer_size is rounded up to a
# multiple of --num_envs). The schedules and target_kl only shape future
# updates, so they may be switched on for a run that started without them.
_RESUMABLE_ALGORITHM_KEYS = ("total_timesteps", "eval_freq", "n_eval_episodes", "checkpoint_freq", "buffer_size",
                             "final_learning_rate", "final_ent_coef", "target_kl")


def save_models(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    prefix: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.bay_actor.state_dict(), save_dir / f"{prefix}_bay_actor.pt")
    torch.save(trainer.row_actor.state_dict(), save_dir / f"{prefix}_row_actor.pt")
    torch.save(trainer.critic.state_dict(), save_dir / f"{prefix}_critic.pt")
    config = {"environment": environment_config, "algorithm": algorithm_config.to_dict()}
    (save_dir / f"{prefix}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to: {save_dir}")


def save_training_state(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    env: VecEnv,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    best_eval_reward: float,
) -> None:
    """Atomically overwrite latest_training_state.pt. Saved only after a
    complete PPO update, so a resumed run starts with env.reset()."""

    rng_state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    payload = {
        "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "environment_config": environment_config,
        "algorithm_config": algorithm_config.to_dict(),
        "trainer_state": {
            "bay_actor_state_dict": trainer.bay_actor.state_dict(),
            "row_actor_state_dict": trainer.row_actor.state_dict(),
            "critic_state_dict": trainer.critic.state_dict(),
            "bay_optimizer_state_dict": trainer.bay_optimizer.state_dict(),
            "row_optimizer_state_dict": trainer.row_optimizer.state_dict(),
            "critic_optimizer_state_dict": trainer.critic_optimizer.state_dict(),
            "total_environment_steps": trainer.total_environment_steps,
            "total_episodes": trainer.total_episodes,
        },
        "best_eval_reward": float(best_eval_reward),
        "rng_state": rng_state,
        # StackEnv draws each episode from its seed, incremented on every
        # reset: this is the environments' whole random state.
        "env_seeds": [int(seed) for seed in env.get_attr("seed")],
    }

    path = save_dir / LATEST_TRAINING_STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Latest training checkpoint saved (overwriting previous): {path} "
          f"at step {trainer.total_environment_steps:,}", flush=True)


def resume_training_state(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    env: VecEnv,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
) -> float:
    """Restore latest_training_state.pt into ``trainer``, ``env``'s episode
    seeds and the process RNGs, and return its best evaluation reward.
    Refuses a checkpoint with other environment or model/training settings."""

    path = save_dir / LATEST_TRAINING_STATE_FILENAME
    # Our own trusted file: it holds Python RNG state, not only tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported training checkpoint format in {path}.")
    _check_compatible(checkpoint["environment_config"], checkpoint["algorithm_config"],
                      environment_config, algorithm_config)

    state = checkpoint["trainer_state"]
    trainer.bay_actor.load_state_dict(state["bay_actor_state_dict"])
    trainer.row_actor.load_state_dict(state["row_actor_state_dict"])
    trainer.critic.load_state_dict(state["critic_state_dict"])
    trainer.bay_optimizer.load_state_dict(state["bay_optimizer_state_dict"])
    trainer.row_optimizer.load_state_dict(state["row_optimizer_state_dict"])
    trainer.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
    trainer.total_environment_steps = int(state["total_environment_steps"])
    trainer.total_episodes = int(state["total_episodes"])
    trainer.reset_rollout_state()

    if trainer.total_environment_steps > algorithm_config.total_timesteps:
        raise ValueError(f"Checkpoint step exceeds the requested total timesteps: "
                         f"{trainer.total_environment_steps:,} > {algorithm_config.total_timesteps:,}.")

    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    # The next env.reset() moves every copy to a new episode. With another
    # --num_envs, every copy continues past all saved episodes.
    env_seeds = checkpoint["env_seeds"]
    if len(env_seeds) != env.num_envs:
        env_seeds = [max(env_seeds) + 1 + rank for rank in range(env.num_envs)]
    for rank, seed in enumerate(env_seeds):
        env.set_attr("seed", seed, indices=rank)

    print(f"Resuming training from: {path}")
    print(f"Restored training step: {trainer.total_environment_steps:,} / {algorithm_config.total_timesteps:,}")
    return float(checkpoint["best_eval_reward"])


def _check_compatible(saved_environment: Dict[str, Any], saved_algorithm: Dict[str, Any],
                      environment_config: Dict[str, Any], algorithm_config: HierarchicalConfig) -> None:

    def normalized(value):
        return json.loads(json.dumps(value, sort_keys=True))  # tuples vs lists, key order

    saved, current = normalized(saved_environment), normalized(environment_config)
    differences = [f"{key}: checkpoint={saved.get(key)!r}, now={current.get(key)!r}"
                   for key in sorted(saved.keys() | current.keys()) if saved.get(key) != current.get(key)]
    if differences:
        raise ValueError(f"Latest checkpoint uses a different environment configuration ({', '.join(differences)}). "
                         "Use the original --size/--seed/--[no-]reward_norm/--[no-]reward_clip settings or pass --fresh.")

    def signature(algorithm: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in normalized(algorithm).items() if k not in _RESUMABLE_ALGORITHM_KEYS}

    # A checkpoint from before a HierarchicalConfig field existed has that
    # field's default.
    saved_algorithm = HierarchicalConfig(**saved_algorithm).to_dict()
    if signature(saved_algorithm) != signature(algorithm_config.to_dict()):
        raise ValueError("Latest checkpoint uses incompatible model/training settings. "
                         "Use the original settings or pass --fresh.")
