"""
Everything Sequential HPPO writes to or restores from disk.

Two kinds of files live in a run's save directory:

    {prefix}_agent_b.pt / _agent_r.pt / _critic.pt / _config.json
        Model exports ("best" and "final") for evaluation.

    latest_training_state.pt
        The one restartable training state: network and optimizer states,
        step/episode counters, the best evaluation reward and every RNG
        state, overwritten atomically at each checkpoint.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ..configs.hierarchical_config import HierarchicalConfig
from ..models.pointer_actor import load_actor_state_dict
from .sequential_trainer import SequentialPPOTrainer

LATEST_TRAINING_STATE_FILENAME = "latest_training_state.pt"
TRAINING_CHECKPOINT_FORMAT_VERSION = 1

# Settings a resumed run may change: horizons, evaluation cadence, and
# buffer_size, which is rounded up to a multiple of --num_envs.
_RESUMABLE_ALGORITHM_KEYS = (
    "total_timesteps",
    "eval_freq",
    "n_eval_episodes",
    "checkpoint_freq",
    "buffer_size",
)


# ======================================================================
# Model exports
# ======================================================================


def save_models(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    prefix: str,
) -> None:
    """Write {prefix}_agent_b/_agent_r/_critic.pt and {prefix}_config.json.

    The actor file names predate removing AgentB/AgentR and are kept so
    existing tooling keeps working.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(trainer.bay_actor.state_dict(), save_dir / f"{prefix}_agent_b.pt")
    torch.save(trainer.row_actor.state_dict(), save_dir / f"{prefix}_agent_r.pt")
    torch.save(trainer.critic.state_dict(), save_dir / f"{prefix}_critic.pt")

    config = {"environment": environment_config, "algorithm": algorithm_config.to_dict()}
    with open(save_dir / f"{prefix}_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Saved checkpoint to: {save_dir}")


# ======================================================================
# Restartable training state
# ======================================================================


def save_training_state(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    best_eval_reward: float,
) -> None:
    """Atomically overwrite latest_training_state.pt.

    Written only after a complete PPO update, so the in-progress episode
    and rollout buffer are not needed: a restored run starts with
    env.reset() and the exact updated networks and optimizers.
    """

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    payload = {
        "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "environment_config": environment_config,
        "algorithm_config": algorithm_config.to_dict(),
        "trainer_state": {
            # Key names predate removing AgentB/AgentR; kept so existing
            # checkpoints stay loadable.
            "agent_b_state_dict": trainer.bay_actor.state_dict(),
            "agent_r_state_dict": trainer.row_actor.state_dict(),
            "critic_state_dict": trainer.critic.state_dict(),
            "bay_optimizer_state_dict": trainer.bay_optimizer.state_dict(),
            "row_optimizer_state_dict": trainer.row_optimizer.state_dict(),
            "critic_optimizer_state_dict": trainer.critic_optimizer.state_dict(),
            "total_environment_steps": trainer.total_environment_steps,
            "total_episodes": trainer.total_episodes,
        },
        "best_eval_reward": float(best_eval_reward),
        "rng_state": rng_state,
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

    print(
        "Latest training checkpoint saved (overwriting previous): "
        f"{path} at step {trainer.total_environment_steps:,}",
        flush=True,
    )


def resume_training_state(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
) -> float:
    """Restore latest_training_state.pt into ``trainer`` and the process
    RNGs; return the checkpoint's best evaluation reward.

    Refuses a checkpoint whose environment or model/training settings
    differ from this run's.
    """

    path = save_dir / LATEST_TRAINING_STATE_FILENAME
    # Our own trusted file: it contains Python RNG state, not only tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if checkpoint.get("format_version") != TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported training checkpoint format in {path}.")

    _check_compatible(checkpoint["environment_config"], checkpoint["algorithm_config"],
                      environment_config, algorithm_config)

    state = checkpoint["trainer_state"]
    load_actor_state_dict(trainer.bay_actor, state["agent_b_state_dict"])
    load_actor_state_dict(trainer.row_actor, state["agent_r_state_dict"])
    trainer.critic.load_state_dict(state["critic_state_dict"])
    trainer.bay_optimizer.load_state_dict(state["bay_optimizer_state_dict"])
    trainer.row_optimizer.load_state_dict(state["row_optimizer_state_dict"])
    trainer.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
    trainer.total_environment_steps = int(state["total_environment_steps"])
    trainer.total_episodes = int(state["total_episodes"])
    trainer.reset_rollout_state()

    if trainer.total_environment_steps > algorithm_config.total_timesteps:
        raise ValueError(
            "Checkpoint step exceeds the requested total timesteps: "
            f"{trainer.total_environment_steps:,} > {algorithm_config.total_timesteps:,}."
        )

    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    print(f"Resuming training from: {path}")
    print(
        "Restored training step: "
        f"{trainer.total_environment_steps:,} / {algorithm_config.total_timesteps:,}"
    )
    return float(checkpoint["best_eval_reward"])


def _check_compatible(
    saved_environment: Dict[str, Any],
    saved_algorithm: Dict[str, Any],
    environment_config: Dict[str, Any],
    algorithm_config: HierarchicalConfig,
) -> None:

    def normalized(value):
        # JSON round trip: tuples vs lists, key order.
        return json.loads(json.dumps(value, sort_keys=True))

    saved, current = normalized(saved_environment), normalized(environment_config)
    if saved != current:
        differences = ", ".join(
            f"{key}: checkpoint={saved.get(key)!r}, now={current.get(key)!r}"
            for key in sorted(saved.keys() | current.keys())
            if saved.get(key) != current.get(key)
        )
        raise ValueError(
            "Latest checkpoint uses a different environment configuration "
            f"({differences}). Use the original --size/--seed/--[no-]reward_norm/"
            "--[no-]reward_clip settings or pass --fresh."
        )

    def signature(algorithm: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in normalized(algorithm).items() if k not in _RESUMABLE_ALGORITHM_KEYS}

    if signature(saved_algorithm) != signature(algorithm_config.to_dict()):
        raise ValueError(
            "Latest checkpoint uses incompatible model/training settings. "
            "Use the original settings or pass --fresh."
        )
