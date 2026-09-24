"""
run.py

Entry point for the separated hierarchical multi-agent PPO pipeline.

New architecture
----------------

    StackEnv
        |
        v
    HierarchicalEnv
        |
        +------------------------+
        |                        |
        v                        v
    AgentB                   AgentR
    BayPolicy                RowPolicy
        |                        |
        +-----------+------------+
                    |
                    v
          CentralizedCritic
                    |
                    v
          JointRolloutBuffer
                    |
                    v
         SequentialPPOTrainer

Decision sequence
-----------------

    global state s_t
        |
        v
    Agent B chooses bay
        |
        | NO environment step
        v
    Agent R receives selected-bay observation
        |
        v
    Agent R chooses row
        |
        v
    HierarchicalEnv.step(bay, row)
        |
        v
    ONE StackEnv.step(global_action)

Training sequence
-----------------

    rollout
        |
        v
    GAE
        |
        v
    centralized critic update
        |
        v
    Agent B PPO update
        |
        v
    detached Bay sequence ratio
        |
        v
    Agent R PPO update
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .agents.agent_b import AgentB
from .agents.agent_r import AgentR

from .configs.hierarchical_config import (
    HierarchicalConfig,
    TRAINING_PROFILES,
    get_hierarchical_config,
)

from .envs.hierarchical_envs.hierarchical_env import (
    HierarchicalEnv,
)
from .envs.hierarchical_envs.vec_hierarchical_env import (
    VecHierarchicalEnv,
)
from .envs.hierarchical_envs.subproc_vec_hierarchical_env import (
    SubprocVecHierarchicalEnv,
)

from .models.centralized_critic import (
    CentralizedCritic,
)

from .training.rollout_buffer import (
    JointRolloutBuffer,
)

from .training.sequential_trainer import (
    SequentialPPOTrainer,
)

from .training.profiling import (
    Profiler,
)

from .training.training_monitor import (
    TrainingMonitor,
    evaluate_for_training,
    evaluation_base_seed,
)
from .evaluation.evaluate import make_evaluation_config

# stack/utils.py uses script-style imports (``from envs...``), so the
# stack/ directory has to be importable as a top-level path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import set_config  # noqa: E402


LATEST_TRAINING_STATE_FILENAME = "latest_training_state.pt"
TRAINING_CHECKPOINT_FORMAT_VERSION = 1


# ======================================================================
# Original StackEnv configurations
# ======================================================================


def get_environment_config(
    size: str,
    seed: int,
) -> Dict:
    """
    Return the StackEnv configuration from the existing stack/utils.py.

    small_with_margin is trained with reward normalization and clipping
    enabled, unlike the default in set_config().
    """

    config = set_config(
        size=size,
        seed=seed,
    )

    if size == "small_with_margin":

        config["reward_norm"] = True
        config["reward_clip"] = True

    return config


# ======================================================================
# Reproducibility
# ======================================================================


def set_global_seed(
    seed: int,
) -> None:
    """
    Seed Python, NumPy, and PyTorch.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ======================================================================
# Device
# ======================================================================


def get_device(
    requested_device: str,
) -> torch.device:
    """
    Resolve training device.

    'auto' keeps the original project's behaviour:
    CUDA when available, otherwise CPU.
    """

    if requested_device == "auto":

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    return torch.device(
        requested_device
    )


def round_up_to_multiple(
    value: int,
    multiple: int,
) -> int:
    """
    Round ``value`` up to the nearest positive multiple of ``multiple``.

    Used to keep rollout buffer sizes divisible by num_envs, which
    compute_and_store_gae() requires when num_envs > 1.
    """

    if multiple <= 1:
        return value

    return (
        (value + multiple - 1)
        // multiple
    ) * multiple


# ======================================================================
# Build complete system
# ======================================================================


def build_training_system(
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    device: torch.device,
    num_envs: int = 1,
    vec_backend: str = "sync",
    profiler: Optional[Profiler] = None,
):
    """
    Construct:

        VecHierarchicalEnv/SubprocVecHierarchicalEnv (num_envs
            parallel HierarchicalEnv copies)
        AgentB
        AgentR
        CentralizedCritic
        SequentialPPOTrainer
    """

    # ==============================================================
    # Environment
    # ==============================================================
    #
    # Each of the num_envs copies is seeded from environment_config's
    # base seed, offset by its index, so they diverge instead of
    # producing identical trajectories.
    #
    # vec_backend="sync" (default) steps every copy in a Python
    # for-loop inside this process. vec_backend="subproc" steps every
    # copy in its own OS process (see SubprocVecHierarchicalEnv's
    # docstring for why this matters/doesn't matter depending on how
    # expensive StackEnv.step() is relative to IPC overhead).
    # ==============================================================

    if vec_backend not in ("sync", "subproc"):
        raise ValueError(
            "vec_backend must be 'sync' or 'subproc', got "
            f"{vec_backend!r}."
        )

    vec_env_cls = (
        SubprocVecHierarchicalEnv
        if vec_backend == "subproc"
        else VecHierarchicalEnv
    )

    env = vec_env_cls(
        config=environment_config,
        num_envs=num_envs,
        base_seed=environment_config.get("seed"),
    )

    # ==============================================================
    # Dimensions come directly from environment
    # ==============================================================

    n_bays = env.bay_action_space.n

    n_rows = env.row_action_space.n

    n_global_stacks = (
        n_bays
        * n_rows
    )

    # ==============================================================
    # Agent B
    # ==============================================================

    agent_b = AgentB(

        observation_space=(
            env.bay_observation_space
        ),

        n_bays=n_bays,

        n_rows_per_bay=n_rows,

        device=device,

        **algorithm_config.agent_kwargs(),
    )

    # ==============================================================
    # Agent R
    # ==============================================================

    agent_r = AgentR(

        observation_space=(
            env.row_observation_space
        ),

        n_rows=n_rows,

        device=device,

        **algorithm_config.agent_kwargs(),
    )

    # ==============================================================
    # Centralized critic
    # ==============================================================

    critic = CentralizedCritic(

        observation_space=(
            env.global_observation_space
        ),

        n_stacks=(
            n_global_stacks
        ),

        **algorithm_config.critic_kwargs(),
    )

    # ==============================================================
    # Sequential trainer
    # ==============================================================

    trainer = SequentialPPOTrainer(

        agent_b=agent_b,

        agent_r=agent_r,

        critic=critic,

        device=device,

        num_envs=num_envs,

        profiler=profiler,

        **algorithm_config.trainer_kwargs(),
    )

    return (
        env,
        agent_b,
        agent_r,
        critic,
        trainer,
    )


# ======================================================================
# Buffer factory
# ======================================================================


def create_rollout_buffer(
    env: HierarchicalEnv | VecHierarchicalEnv | SubprocVecHierarchicalEnv,
    buffer_size: int,
    device: torch.device,
) -> JointRolloutBuffer:
    """
    Create a joint Bay/Row rollout buffer.
    """

    return JointRolloutBuffer(

        buffer_size=buffer_size,

        global_observation_space=(
            env.global_observation_space
        ),

        bay_observation_space=(
            env.bay_observation_space
        ),

        row_observation_space=(
            env.row_observation_space
        ),

        n_bays=(
            env.bay_action_space.n
        ),

        n_rows=(
            env.row_action_space.n
        ),

        device=device,
    )


# ======================================================================
# Saving
# ======================================================================


def save_models(
    save_dir: Path,
    agent_b: AgentB,
    agent_r: AgentR,
    critic: CentralizedCritic,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    prefix: str = "final",
) -> None:
    """
    Save the two actors, centralized critic and configurations.
    """

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # Agent B
    # ==============================================================

    agent_b.save(
        save_dir
        / f"{prefix}_agent_b.pt"
    )

    # ==============================================================
    # Agent R
    # ==============================================================

    agent_r.save(
        save_dir
        / f"{prefix}_agent_r.pt"
    )

    # ==============================================================
    # Critic
    # ==============================================================

    torch.save(

        critic.state_dict(),

        save_dir
        / f"{prefix}_critic.pt",
    )

    # ==============================================================
    # Configurations
    # ==============================================================

    config_data = {

        "environment":
            environment_config,

        "algorithm":
            algorithm_config.to_dict(),
    }

    with open(
        save_dir
        / f"{prefix}_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_data,
            file,
            indent=2,
        )

    print(
        f"Saved checkpoint to: "
        f"{save_dir}"
    )


# ======================================================================
# Restartable latest checkpoint
# ======================================================================


def _atomic_torch_save(
    payload: Dict,
    path: Path,
) -> None:
    """Write a torch checkpoint atomically on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _torch_load(
    path: Path,
):
    """Load trusted local training state across PyTorch versions."""

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # PyTorch versions before ``weights_only`` was introduced.
        return torch.load(
            path,
            map_location="cpu",
        )


def _normalized_config(
    value,
):
    """Normalize tuples/lists before comparing JSON-compatible configs."""

    return json.loads(
        json.dumps(value, sort_keys=True)
    )


def _algorithm_resume_signature(
    value: Dict,
) -> Dict:
    """Ignore runtime horizons that may safely change when resuming."""

    signature = dict(
        _normalized_config(value)
    )
    for key in (
        "total_timesteps",
        "eval_freq",
        "n_eval_episodes",
        "checkpoint_freq",
        # buffer_size is rounded up to a multiple of --num_envs at
        # construction time (see round_up_to_multiple() call sites
        # below), so it can differ across otherwise-identical resumes
        # that just pass a different --num_envs. It is a rollout-
        # collection runtime setting, not a model/training-algorithm
        # property, so it belongs in this "safe to change" list too.
        "buffer_size",
    ):
        signature.pop(key, None)
    return signature


def _validate_resume_config(
    checkpoint_environment_config: Dict,
    checkpoint_algorithm_config: Dict,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
) -> None:
    """Prevent a checkpoint from being loaded into an incompatible run."""

    if _normalized_config(checkpoint_environment_config) != _normalized_config(
        environment_config
    ):
        raise ValueError(
            "Latest checkpoint uses a different environment configuration. "
            "Use the original --size/--seed settings or pass --fresh."
        )

    if _algorithm_resume_signature(
        checkpoint_algorithm_config
    ) != _algorithm_resume_signature(
        algorithm_config.to_dict()
    ):
        raise ValueError(
            "Latest checkpoint uses incompatible model/training settings. "
            "Use the original settings or pass --fresh."
        )


def _capture_rng_state() -> Dict:
    """Capture process RNG state used by sampling and minibatch shuffling."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: Optional[Dict],
) -> None:
    """Restore RNG state when it is available in a full checkpoint."""

    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_latest_training_state(
    save_dir: Path,
    trainer: SequentialPPOTrainer,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    best_eval_reward: float,
) -> Path:
    """Atomically overwrite the one restartable training checkpoint."""

    checkpoint_path = save_dir / LATEST_TRAINING_STATE_FILENAME
    payload = {
        "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "environment_config": environment_config,
        "algorithm_config": algorithm_config.to_dict(),
        "trainer_state": trainer.training_state_dict(),
        "best_eval_reward": float(best_eval_reward),
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(payload, checkpoint_path)
    print(
        "Latest training checkpoint saved (overwriting previous): "
        f"{checkpoint_path} at step {trainer.total_environment_steps:,}",
        flush=True,
    )
    return checkpoint_path


def load_latest_training_state(
    checkpoint_path: Path,
    trainer: SequentialPPOTrainer,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
) -> float:
    """Restore a full checkpoint and return its best evaluation reward."""

    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    if checkpoint.get("format_version") != TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported training checkpoint format in "
            f"{checkpoint_path}."
        )

    _validate_resume_config(
        checkpoint["environment_config"],
        checkpoint["algorithm_config"],
        environment_config,
        algorithm_config,
    )
    trainer.load_training_state_dict(checkpoint["trainer_state"])

    if trainer.total_environment_steps > algorithm_config.total_timesteps:
        raise ValueError(
            "Checkpoint step exceeds the requested total timesteps: "
            f"{trainer.total_environment_steps:,} > "
            f"{algorithm_config.total_timesteps:,}."
        )

    _restore_rng_state(checkpoint.get("rng_state"))
    print(f"Resuming training from: {checkpoint_path}")
    print(
        "Restored training step: "
        f"{trainer.total_environment_steps:,} / "
        f"{algorithm_config.total_timesteps:,}"
    )
    return float(checkpoint.get("best_eval_reward", float("-inf")))

# ======================================================================
# Console logging
# ======================================================================


def print_training_stats(
    stats: Dict[str, float],
    total_timesteps: int,
) -> None:
    """
    Print the most useful training quantities.

    These are training statistics, not final evaluation metrics.
    """

    step = int(
        stats[
            "total_environment_steps"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Training step: "
        f"{step:,} / "
        f"{total_timesteps:,}"
    )

    print(
        "-" * 70
    )

    print(
        "Rollout reward sum: "
        f"{stats['rollout_reward_sum']:.4f}"
    )

    mean_reward = stats[
        "mean_episode_reward"
    ]

    if np.isnan(
        mean_reward
    ):

        print(
            "Mean episode reward: "
            "N/A (no complete episode in rollout)"
        )

    else:

        print(
            "Mean episode reward: "
            f"{mean_reward:.4f}"
        )

    print(
        "Advantage mean: "
        f"{stats['advantage_mean']:.4f}"
    )

    print(
        "Critic loss: "
        f"{stats['critic_loss']:.6f}"
    )

    print(
        "Bay policy loss: "
        f"{stats['bay_policy_loss']:.6f}"
    )

    print(
        "Row policy loss: "
        f"{stats['row_policy_loss']:.6f}"
    )

    print(
        "Bay entropy: "
        f"{stats['bay_entropy']:.4f}"
    )

    print(
        "Row entropy: "
        f"{stats['row_entropy']:.4f}"
    )

    print(
        "Bay sequence ratio mean: "
        f"{stats['bay_sequence_ratio_mean']:.4f}"
    )

    print(
        "Bay sequence ratio std: "
        f"{stats['bay_sequence_ratio_std']:.4f}"
    )


# ======================================================================
# Main training
# ======================================================================


def main(
    args: argparse.Namespace,
) -> None:

    # ==============================================================
    # Seed
    # ==============================================================

    set_global_seed(
        args.seed
    )

    # ==============================================================
    # Environment configuration
    # ==============================================================

    environment_config = (
        get_environment_config(
            size=args.size,
            seed=args.seed,
        )
    )

    if args.random_group_sizes:

        environment_config[
            "random_group_sizes"
        ] = True

    # ==============================================================
    # Algorithm configuration
    # ==============================================================

    algorithm_config = (
        get_hierarchical_config(
            size=args.size,
            training_profile=args.training_profile,
        )
    )

    # CLI overrides are applied using dataclasses.replace because
    # HierarchicalConfig is frozen.

    resolved_buffer_size = (
        args.buffer_size
        if args.buffer_size is not None
        else algorithm_config.buffer_size
    )

    # compute_and_store_gae() requires the rollout buffer length to be an
    # exact multiple of num_envs (it recovers each environment's own
    # chronological sequence via strided slicing). Round up so this always
    # holds, for both the configured buffer_size and the final, possibly
    # shorter, rollout of a training run.
    resolved_buffer_size = round_up_to_multiple(
        resolved_buffer_size,
        args.num_envs,
    )

    algorithm_config = replace(

        algorithm_config,

        total_timesteps=(
            args.timesteps
            if args.timesteps is not None
            else algorithm_config.total_timesteps
        ),

        buffer_size=(
            resolved_buffer_size
        ),

        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else algorithm_config.batch_size
        ),

        n_epochs=(
            args.n_epochs
            if args.n_epochs is not None
            else algorithm_config.n_epochs
        ),

        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else algorithm_config.learning_rate
        ),

        clip_range=(
            args.clip_range
            if args.clip_range is not None
            else algorithm_config.clip_range
        ),

        ent_coef=(
            args.ent_coef
            if args.ent_coef is not None
            else algorithm_config.ent_coef
        ),

        checkpoint_freq=(
            args.checkpoint_freq
            if args.checkpoint_freq is not None
            # By default, use the size tier's checkpoint_freq (see
            # TRAINING_PROFILES in hierarchical_config.py) rather than
            # resolved_buffer_size - saving the full training state
            # after every single rollout is wasteful for long "massive"
            # runs (thousands of writes over 28M steps).
            else algorithm_config.checkpoint_freq
        ),

        eval_freq=(
            args.eval_freq
            if args.eval_freq is not None
            else algorithm_config.eval_freq
        ),

        n_eval_episodes=(
            args.n_eval_episodes
            if args.n_eval_episodes is not None
            else algorithm_config.n_eval_episodes
        ),
    )

    if not 0 < args.ema_alpha <= 1:
        raise ValueError("--ema_alpha must satisfy 0 < alpha <= 1.")
    if args.plot_points < 2:
        raise ValueError("--plot_points must be at least 2.")
    if args.num_envs < 1:
        raise ValueError("--num_envs must be at least 1.")

    # ==============================================================
    # Device
    # ==============================================================

    device = get_device(
        args.device
    )

    print(
        f"Using device: {device}"
    )

    profiler = Profiler(
        enabled=not args.no_profiling,
        device=device,
    )

    # ==============================================================
    # Build entire architecture
    # ==============================================================

    (
        env,
        agent_b,
        agent_r,
        critic,
        trainer,
    ) = build_training_system(

        environment_config=(
            environment_config
        ),

        algorithm_config=(
            algorithm_config
        ),

        device=device,

        profiler=profiler,

        num_envs=args.num_envs,

        vec_backend=args.vec_backend,
    )

    # ==============================================================
    # Automatic restart
    # ==============================================================

    save_dir = Path(args.save_dir)
    resume_kind: Optional[str] = None
    best_eval_reward = float("-inf")

    if args.fresh:
        print(
            "Fresh training explicitly requested; existing checkpoints "
            "will not be loaded."
        )
    else:
        latest_checkpoint = save_dir / LATEST_TRAINING_STATE_FILENAME

        if latest_checkpoint.exists():
            best_eval_reward = load_latest_training_state(
                latest_checkpoint,
                trainer,
                environment_config,
                algorithm_config,
            )
            resume_kind = "full"
        else:
            print(
                "No latest training checkpoint found; "
                "starting training from scratch."
            )

    # ==============================================================
    # Print architecture sanity information
    # ==============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEQUENTIAL MULTI-AGENT PPO"
    )

    print(
        "=" * 70
    )

    print(
        f"Environment size: "
        f"{args.size}"
    )

    print(
        f"Training profile: "
        f"{algorithm_config.training_profile}"
    )

    print(
        f"Yard shape: "
        f"{env.inner_env.yard_shape}"
    )

    print(
        f"Bay observation: "
        f"{env.bay_observation_space.shape}"
    )

    print(
        f"Row observation: "
        f"{env.row_observation_space.shape}"
    )

    print(
        f"Number of bays: "
        f"{env.bay_action_space.n}"
    )

    print(
        f"Rows per bay: "
        f"{env.row_action_space.n}"
    )

    print(
        f"Learning rate: "
        f"{algorithm_config.learning_rate}"
    )

    print(
        f"Buffer size: "
        f"{algorithm_config.buffer_size}"
    )

    print(
        f"Batch size: "
        f"{algorithm_config.batch_size}"
    )

    print(
        f"PPO epochs: "
        f"{algorithm_config.n_epochs}"
    )

    print(
        f"Clip range: "
        f"{algorithm_config.clip_range}"
    )

    print(
        f"Entropy coefficient: "
        f"{algorithm_config.ent_coef}"
    )

    print(
        f"Total timesteps: "
        f"{algorithm_config.total_timesteps:,}"
    )

    print(
        f"Latest-checkpoint frequency: "
        f"{algorithm_config.checkpoint_freq:,} steps"
    )

    evaluation_config = dict(environment_config)
    if not args.eval_use_training_rewards:
        evaluation_config = make_evaluation_config(environment_config)
    evaluation_config["seed"] = args.eval_seed
    yard_label = "x".join(str(x) for x in environment_config["yard_shape"])
    display_name = f"{args.size.replace('_', ' ').title()} (yard {yard_label})"
    monitor = TrainingMonitor(
        log_dir=save_dir / "training_logs",
        metadata={
            "seed": args.seed,
            "environment_size": args.size,
            "display_name": display_name,
            "environment": environment_config,
            "algorithm": algorithm_config.to_dict(),
            "evaluation": {
                "enabled": not args.no_eval,
                "environment": evaluation_config,
                "initial_base_seed": args.eval_seed,
                "seed_mode": args.eval_seed_mode,
                "n_episodes": algorithm_config.n_eval_episodes,
                "frequency": algorithm_config.eval_freq,
                "deterministic": True,
                "evaluate_at_step_zero": False,
            },
            "plot": {"ema_alpha": args.ema_alpha, "n_points": args.plot_points},
            "step_unit": "physical environment transitions used for training",
        },
        ema_alpha=args.ema_alpha, n_points=args.plot_points,
        plot=not args.no_training_plots,
        resume_step=(
            trainer.total_environment_steps
            if resume_kind is not None
            else None
        ),
    )
    print(f"Training history: {monitor.log_dir}")
    if not args.no_eval:
        print(f"Evaluation: every {algorithm_config.eval_freq:,} exact training steps, "
              f"{algorithm_config.n_eval_episodes} episodes; no step-0 evaluation.")
        print(f"Evaluation seed mode={args.eval_seed_mode}; first seed="
              f"{args.eval_seed}.")
        print(f"Evaluation reward_norm={evaluation_config['reward_norm']}, "
              f"reward_clip={evaluation_config['reward_clip']}")

    def run_evaluation():
        nonlocal best_eval_reward
        step = int(trainer.total_environment_steps)
        evaluation_index = len(monitor.evaluations)
        current_eval_base_seed = evaluation_base_seed(
            initial_seed=args.eval_seed,
            evaluation_index=evaluation_index,
            n_episodes=algorithm_config.n_eval_episodes,
            mode=args.eval_seed_mode,
        )
        summary = evaluate_for_training(
            evaluation_config, agent_b, agent_r,
            n_episodes=algorithm_config.n_eval_episodes,
            base_seed=current_eval_base_seed,
            eval_batch_size=(
                args.eval_batch_size
                if args.eval_batch_size is not None
                else min(algorithm_config.n_eval_episodes, 8)
            ),
        )
        monitor.record_evaluation(
            step,
            summary,
            evaluation_index=evaluation_index,
            base_seed=current_eval_base_seed,
        )
        last_seed = current_eval_base_seed + summary.n_episodes - 1
        print(f"Evaluation at {step:,} steps | Seeds: "
              f"{current_eval_base_seed}-{last_seed} "
              f"| Mean reward: {summary.mean_reward:.4f} "
              f"| Std: {summary.std_reward:.4f} "
              f"| Complete: {summary.successful_episodes}/{summary.n_episodes}", flush=True)
        if summary.mean_reward > best_eval_reward:
            best_eval_reward = summary.mean_reward
            if args.save_model:
                save_models(
                    save_dir, agent_b, agent_r, critic,
                    environment_config, algorithm_config, prefix="best",
                )
                (save_dir / "best_evaluation.json").write_text(json.dumps({
                    "timesteps": step,
                    "evaluation_environment": dict(
                        evaluation_config,
                        seed=current_eval_base_seed,
                    ),
                    "evaluation_index": evaluation_index,
                    "base_seed": current_eval_base_seed,
                    "last_seed": last_seed,
                    "seed_mode": args.eval_seed_mode,
                    **summary.to_dict(),
                }, indent=2), encoding="utf-8")
        monitor.refresh_plot()
        return {
            "eval/mean_reward": summary.mean_reward,
            "eval/std_reward": summary.std_reward,
            "eval/mean_episode_length": summary.mean_episode_length,
            "eval/completion_rate": summary.completion_rate,
            "eval/evaluation_index": evaluation_index,
            "eval/base_seed": current_eval_base_seed,
        }

    # ==============================================================
    # Optional W&B
    # ==============================================================

    wandb_run = None

    if not args.no_wandb:

        try:

            import wandb

        except ImportError as error:

            raise ImportError(
                "wandb is not installed. "
                "Use --no_wandb or install wandb."
            ) from error

        wandb_run = wandb.init(

            project=args.wandb_project,

            name=args.wandb_run_name,

            config={
                **environment_config,
                **algorithm_config.to_dict(),
                "environment_size":
                    args.size,
                "seed":
                    args.seed,
                "architecture":
                    "sequential_multi_agent_ppo",
                "evaluation_initial_base_seed":
                    args.eval_seed,
                "evaluation_seed_mode":
                    args.eval_seed_mode,
                "evaluation_at_step_zero":
                    False,
            },
        )

    # ==============================================================
    # Training loop
    # ==============================================================

    next_checkpoint = (
        trainer.total_environment_steps
        // algorithm_config.checkpoint_freq
        + 1
    ) * algorithm_config.checkpoint_freq

    next_eval = (trainer.total_environment_steps // algorithm_config.eval_freq + 1) * algorithm_config.eval_freq

    def evaluate_if_due(completed_steps: int) -> None:
        """Evaluate at exact scheduled steps without changing the rollout."""
        nonlocal next_eval
        if args.no_eval or completed_steps < next_eval:
            return
        if completed_steps != next_eval:
            raise RuntimeError(
                "Periodic evaluation threshold was skipped: "
                f"expected {next_eval:,}, reached {completed_steps:,}."
            )
        # Same shared `profiler` instance the trainer uses (see
        # build_training_system() above), so this accumulates into the
        # same per-iteration timing dict train_iteration() later pops -
        # even though this runs nested inside collect_rollout(), via
        # this step_callback. sync_cuda=True since evaluation does
        # real GPU inference (agent_b.act()/agent_r.act()).
        with profiler.region(
            "evaluation_seconds",
            sync_cuda=True,
        ):
            evaluation_stats = run_evaluation()
        if wandb_run is not None:
            wandb_run.log(
                {
                    "total_environment_steps": completed_steps,
                    **evaluation_stats,
                },
                step=completed_steps,
            )
        next_eval += algorithm_config.eval_freq

    try:

        # Convert an old best-only export into the new complete format before
        # collecting more data.  This one-time migration cannot recover the
        # old optimizer moments, but every later save can.
        if args.save_model and resume_kind == "legacy_best":
            save_latest_training_state(
                save_dir,
                trainer,
                environment_config,
                algorithm_config,
                best_eval_reward,
            )

        while (
            trainer.total_environment_steps
            <
            algorithm_config.total_timesteps
        ):

            remaining_steps = (
                algorithm_config.total_timesteps
                -
                trainer.total_environment_steps
            )

            # ------------------------------------------------------
            # Final rollout may be smaller than the default buffer.
            # ------------------------------------------------------

            current_buffer_size = min(
                algorithm_config.buffer_size,
                remaining_steps,
            )

            # Keep the buffer length divisible by num_envs (required by
            # compute_and_store_gae()) even for the final, possibly
            # shorter, rollout of a training run. This means the very
            # last rollout may overshoot total_timesteps by up to
            # num_envs - 1 steps, which is standard for vectorized PPO.
            current_buffer_size = round_up_to_multiple(
                current_buffer_size,
                args.num_envs,
            )

            buffer = create_rollout_buffer(

                env=env,

                buffer_size=(
                    current_buffer_size
                ),

                device=device,
            )

            # ======================================================
            # ONE complete sequential PPO iteration
            # ======================================================

            iteration_start = time.perf_counter()

            stats = trainer.train_iteration(
                env=env,
                buffer=buffer,
                step_callback=evaluate_if_due,
            )

            # ======================================================
            # Console
            # ======================================================

            print_training_stats(

                stats=stats,

                total_timesteps=(
                    algorithm_config.total_timesteps
                ),
            )

            # ======================================================
            # Restartable latest checkpoint
            # ======================================================

            if (
                args.save_model
                and
                trainer.total_environment_steps
                >= next_checkpoint
            ):

                with profiler.region("checkpoint_seconds"):
                    save_latest_training_state(
                        save_dir,
                        trainer,
                        environment_config,
                        algorithm_config,
                        best_eval_reward,
                    )

                while (
                    next_checkpoint
                    <=
                    trainer.total_environment_steps
                ):

                    next_checkpoint += (
                        algorithm_config.checkpoint_freq
                    )

            # ======================================================
            # Remaining profiling columns
            # ======================================================
            #
            # train_iteration() already merged its own phase timings
            # (rollout_seconds, critic_update_seconds, etc.) into
            # `stats`. checkpoint_seconds is only known now, since the
            # checkpoint save above happens after train_iteration()
            # returns - default it to 0.0 on iterations that didn't
            # checkpoint, for the same fixed-column-set reason
            # train_iteration() defaults its own profiling keys.
            #
            # steps_per_second uses plain wall-clock spanning
            # train_iteration() + the checkpoint save above, not the
            # Profiler's named regions, so it reflects true end-to-end
            # iteration throughput regardless of whether profiling
            # itself is enabled.
            # ======================================================

            stats["checkpoint_seconds"] = 0.0
            stats.update(profiler.pop_stats())

            iteration_wall_seconds = (
                time.perf_counter() - iteration_start
            )

            stats["steps_per_second"] = (
                stats["rollout_steps"] / iteration_wall_seconds
                if iteration_wall_seconds > 0
                else 0.0
            )

            monitor.record_rollout(stats)

            # ======================================================
            # W&B
            # ======================================================

            if wandb_run is not None:

                wandb_run.log(
                    stats,
                    step=int(
                        trainer.total_environment_steps
                    ),
                )

        # ==========================================================
        # Final model
        # ==========================================================

        if args.save_model:

            save_models(

                save_dir=save_dir,

                agent_b=agent_b,

                agent_r=agent_r,

                critic=critic,

                environment_config=(
                    environment_config
                ),

                algorithm_config=(
                    algorithm_config
                ),

                prefix="final",
            )

            save_latest_training_state(
                save_dir,
                trainer,
                environment_config,
                algorithm_config,
                best_eval_reward,
            )

    finally:

        env.close()

        if wandb_run is not None:

            wandb_run.finish()

    print(
        "\nTraining finished successfully."
    )


# ======================================================================
# CLI
# ======================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(

        description=(
            "Train separated Bay/Row agents "
            "with sequential multi-agent PPO."
        )
    )

    # ==============================================================
    # Environment
    # ==============================================================

    parser.add_argument(
        "--size",
        type=str,
        default="small_with_margin",
        choices=[
            "small",
            "small_with_margin",
            "medium",
            "medium_with_margin",
            "large",
            "large_with_margin",
            "large_v2",
            "large_v2_with_margin",
            "large_v3",
            "large_v3_with_margin",
            "large_v4",
            "large_v4_with_margin",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--random_group_sizes",
        action="store_true",
    )

    # ==============================================================
    # Runtime
    # ==============================================================

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help=(
            "Override total training timesteps. "
            "Default comes from the size-dependent training profile."
        ),
    )

    parser.add_argument(
        "--training_profile",
        "--profile",
        type=str,
        default="auto",
        choices=["auto", *sorted(TRAINING_PROFILES)],
        help=(
            "Training profile. 'auto' maps --size to small, medium, "
            "large or massive."
        ),
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=None,
        help=(
            "Override rollout buffer size."
        ),
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help=(
            "Number of environments stepped in parallel during "
            "rollout collection (see --vec_backend for in-process vs. "
            "multiprocess). Increasing this batches every Agent "
            "B/Agent R/critic forward pass across more environments, "
            "which reduces GPU sync overhead per collected "
            "transition."
        ),
    )

    parser.add_argument(
        "--vec_backend",
        type=str,
        choices=["sync", "subproc"],
        default="sync",
        help=(
            "'sync' (default) steps every --num_envs copy in a Python "
            "for-loop inside this process. 'subproc' steps every copy "
            "in its own OS process (true multiprocessing), which can "
            "help when StackEnv.step() is expensive enough per call "
            "to outweigh the IPC overhead of one extra process per "
            "environment."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override the profile learning rate.",
    )

    parser.add_argument(
        "--clip_range",
        type=float,
        default=None,
        help="Override the profile PPO clip range.",
    )

    parser.add_argument(
        "--ent_coef",
        type=float,
        default=None,
        help="Override the profile entropy coefficient.",
    )

    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help=(
            "Environment steps between overwrites of "
            "latest_training_state.pt. Default: the size tier's "
            "checkpoint_freq (100k for small/medium, 250k for "
            "large/massive; see TRAINING_PROFILES)."
        ),
    )

    parser.add_argument("--eval_freq", type=int, default=None,
                        help="Evaluate at exact multiples of this training-step interval (default 25000).")
    parser.add_argument("--n_eval_episodes", type=int, default=None,
                        help="Episodes averaged at each evaluation point (default 10).")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Episodes' environments to evaluate simultaneously at each evaluation "
                             "point (default: min(n_eval_episodes, 8)). Does not change results, "
                             "only throughput - see evaluate_policy_batched().")
    parser.add_argument("--eval_seed", type=int, default=100_000,
                        help="First evaluation episode seed; independent of the training seed.")
    parser.add_argument(
        "--eval_seed_mode",
        choices=["rolling", "fixed"],
        default="rolling",
        help=(
            "Seed schedule across evaluation points. 'rolling' (default) "
            "uses a new non-overlapping seed block each time, matching "
        ),
    )
    parser.add_argument("--ema_alpha", type=float, default=0.1,
                        help="Current-value weight in plot EMA; smaller values are smoother.")
    parser.add_argument("--plot_points", type=int, default=500,
                        help="Interpolation grid size before EMA")
    parser.add_argument("--no_training_plots", action="store_true",
                        help="Keep evaluation CSV records but skip live PNG/PDF generation.")
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable periodic training evaluation; rollout CSV is still recorded.")
    parser.add_argument("--no_profiling", action="store_true",
                        help="Disable per-phase wall-clock timing columns in rollouts.csv "
                             "(rollout_seconds, critic_update_seconds, etc.). Enabled by default; "
                             "adds a coarse-grained torch.cuda.synchronize() per phase per rollout "
                             "step, never per PPO minibatch.")
    parser.add_argument("--eval_use_training_rewards", action="store_true",
                        help="Use training reward normalization/clipping during evaluation instead of raw rewards.")

    # ==============================================================
    # Saving
    # ==============================================================

    parser.add_argument(
        "--save_model",
        action="store_true",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models/sequential_hppo",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Start from newly initialized networks even when "
            "latest_training_state.pt or legacy best weights exist."
        ),
    )

    # ==============================================================
    # W&B
    # ==============================================================

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stack-sequential-hppo",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
    )

    return parser


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":

    parser = build_parser()

    args = parser.parse_args()

    main(
        args
    )
