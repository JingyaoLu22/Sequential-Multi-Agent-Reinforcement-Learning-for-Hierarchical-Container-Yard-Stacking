"""
Train Sequential HPPO on StackEnv.

Agent B picks a bay, Agent R picks a row inside it, and StackEnv takes
ONE step with action bay * n_rows + row; one centralized critic
(training/sequential_trainer.py has the algorithm).

This entry point only wires the pieces together:

    arguments -> environment + algorithm config
              -> StackEnv copies, actors, critic, trainer
              -> resume (training/checkpoint.py)
              -> training loop with periodic evaluation and logging

    python -m stack.run_sequential_hppo --size small_with_margin --save_model
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .configs.device import get_device
from .configs.environments import ENVIRONMENT_SIZES, add_reward_arguments, set_config
from .configs.hierarchical_config import (
    TRAINING_PROFILES,
    HierarchicalConfig,
    get_hierarchical_config,
)
from .envs.stack_gym import StackEnv
from .evaluation.evaluate import (
    add_evaluation_reward_argument,
    evaluate_policy,
    make_evaluation_config,
)
from .models.centralized_critic import CentralizedCritic
from .training.bay_row_layout import BayRowLayout
from .training.checkpoint import (
    LATEST_TRAINING_STATE_FILENAME,
    resume_training_state,
    save_models,
    save_training_state,
)
from .training.csv_logger import CsvLogger
from .training.profiling import Profiler
from .training.rollout_buffer import JointRolloutBuffer
from .training.sequential_trainer import PHASE_TIMINGS, SequentialPPOTrainer
from .training_plots import plot_csv_training_curves

# Local CSV columns; W&B receives every statistic.
ROLLOUT_CSV_COLUMNS = [
    "timesteps",
    "mean_episode_reward",
    "mean_episode_length",
    "critic_loss",
    "bay_policy_loss",
    "row_policy_loss",
    "steps_per_second",
]
EVALUATION_CSV_COLUMNS = [
    "timesteps",
    "mean_reward",
    "std_reward",
    "mean_episode_length",
    "completion_rate",
    "base_seed",
]

# CLI flag -> HierarchicalConfig field it overrides when given.
_CONFIG_OVERRIDES = {
    "timesteps": "total_timesteps",
    "buffer_size": "buffer_size",
    "batch_size": "batch_size",
    "n_epochs": "n_epochs",
    "learning_rate": "learning_rate",
    "clip_range": "clip_range",
    "ent_coef": "ent_coef",
    "checkpoint_freq": "checkpoint_freq",
    "eval_freq": "eval_freq",
    "n_eval_episodes": "n_eval_episodes",
}


def _round_up(value: int, multiple: int) -> int:
    """JointRolloutBuffer stores (steps, envs): sizes must be a multiple
    of num_envs, so the last rollout may overshoot by < num_envs steps."""

    return -(-value // multiple) * multiple


def build_algorithm_config(args: argparse.Namespace) -> HierarchicalConfig:
    """The size's training profile with the CLI overrides applied."""

    config = get_hierarchical_config(args.size, args.training_profile)
    config = replace(config, **{
        field: getattr(args, flag)
        for flag, field in _CONFIG_OVERRIDES.items()
        if getattr(args, flag) is not None
    })
    return replace(config, buffer_size=_round_up(config.buffer_size, args.num_envs))


def build_training_system(
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    device: torch.device,
    num_envs: int = 1,
    vec_backend: str = "sync",
    profiler: Optional[Profiler] = None,
):
    """SB3 VecEnv of StackEnv copies, the two actors, the critic and the
    trainer. Copy i is seeded with the config's seed + i so the copies
    diverge."""

    def make_env(rank: int):
        def factory() -> StackEnv:
            config = dict(environment_config)
            if config.get("seed") is not None:
                config["seed"] = int(config["seed"]) + rank
            return StackEnv(config=config)

        return factory

    env_fns = [make_env(rank) for rank in range(num_envs)]
    env = SubprocVecEnv(env_fns, start_method="spawn") if vec_backend == "subproc" else DummyVecEnv(env_fns)

    layout = BayRowLayout.from_env(env)
    bay_actor, row_actor = layout.build_actors(**algorithm_config.actor_kwargs())
    critic = CentralizedCritic(
        observation_space=layout.observation_space,
        n_stacks=layout.n_bays * layout.n_rows,
        **algorithm_config.critic_kwargs(),
    )
    trainer = SequentialPPOTrainer(
        bay_actor,
        row_actor,
        critic,
        layout,
        device=device,
        num_envs=num_envs,
        profiler=profiler,
        **algorithm_config.trainer_kwargs(),
    )
    return env, bay_actor, row_actor, critic, trainer


def print_training_stats(stats: Dict[str, float], total_timesteps: int) -> None:
    mean_reward = stats["mean_episode_reward"]
    print(
        f"\nStep {int(stats['total_environment_steps']):,} / {total_timesteps:,} | "
        f"rollout reward sum {stats['rollout_reward_sum']:.4f} | mean episode reward "
        + ("N/A (no complete episode)" if math.isnan(mean_reward) else f"{mean_reward:.4f}")
    )
    print(
        f"  advantage mean {stats['advantage_mean']:.4f} | critic loss {stats['critic_loss']:.6f} | "
        f"policy loss bay {stats['bay_policy_loss']:.6f} / row {stats['row_policy_loss']:.6f} | "
        f"entropy bay {stats['bay_entropy']:.4f} / row {stats['row_entropy']:.4f} | "
        f"M_B {stats['bay_sequence_ratio_mean']:.4f} ± {stats['bay_sequence_ratio_std']:.4f}"
    )


def main(args: argparse.Namespace) -> None:

    if not 0 < args.ema_alpha <= 1:
        raise ValueError("--ema_alpha must satisfy 0 < alpha <= 1.")
    if args.plot_points < 2:
        raise ValueError("--plot_points must be at least 2.")
    if args.num_envs < 1:
        raise ValueError("--num_envs must be at least 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    # The same set_config() as every baseline: reward settings change only
    # through the explicit --[no-]reward_norm / --[no-]reward_clip flags.
    environment_config = set_config(args.size, args.seed, args.reward_norm, args.reward_clip)
    if args.random_group_sizes:
        environment_config["random_group_sizes"] = True
    algorithm_config = build_algorithm_config(args)

    device = torch.device(get_device(args.device))
    # Off unless --profile: every timed phase synchronizes CUDA once.
    profiler = Profiler(enabled=args.profile, device=device)

    # ------------------------------------------------------------------
    # Environments, models, trainer, resume
    # ------------------------------------------------------------------

    env, bay_actor, row_actor, critic, trainer = build_training_system(
        environment_config, algorithm_config, device,
        num_envs=args.num_envs, vec_backend=args.vec_backend, profiler=profiler,
    )

    save_dir = Path(args.save_dir)
    best_eval_reward = float("-inf")
    resumed = not args.fresh and (save_dir / LATEST_TRAINING_STATE_FILENAME).exists()
    if resumed:
        best_eval_reward = resume_training_state(save_dir, trainer, environment_config, algorithm_config)
    elif args.fresh:
        print("Fresh training explicitly requested; existing checkpoints will not be loaded.")
    else:
        print("No latest training checkpoint found; starting training from scratch.")

    layout = trainer.layout
    print(
        f"\nSequential HPPO | {args.size}, yard {environment_config['yard_shape']}, "
        f"{layout.n_bays} bays x {layout.n_rows} rows | device {device} | "
        f"profile {algorithm_config.training_profile}: lr {algorithm_config.learning_rate}, "
        f"buffer {algorithm_config.buffer_size}, batch {algorithm_config.batch_size}, "
        f"epochs {algorithm_config.n_epochs}, clip {algorithm_config.clip_range}, "
        f"ent {algorithm_config.ent_coef} | {algorithm_config.total_timesteps:,} steps, "
        f"checkpoint every {algorithm_config.checkpoint_freq:,}"
    )

    # ------------------------------------------------------------------
    # Periodic evaluation and logs
    # ------------------------------------------------------------------

    evaluation_config = make_evaluation_config(environment_config, args.eval_use_training_rewards)
    evaluation_config["seed"] = args.eval_seed
    yard_label = "x".join(str(x) for x in environment_config["yard_shape"])
    display_name = f"{args.size.replace('_', ' ').title()} (yard {yard_label})"

    log_dir = save_dir / "training_logs"
    resume_step = trainer.total_environment_steps if resumed else None
    rollout_columns = ROLLOUT_CSV_COLUMNS
    if profiler.enabled:
        rollout_columns = rollout_columns + [*PHASE_TIMINGS, "checkpoint_seconds"]
    rollout_log = CsvLogger(log_dir / "rollouts.csv", rollout_columns, resume_step)
    evaluation_log = CsvLogger(log_dir / "evaluations.csv", EVALUATION_CSV_COLUMNS, resume_step)

    print(f"Training history: {log_dir}")
    if not args.no_eval:
        print(
            f"Evaluation: every {algorithm_config.eval_freq:,} training steps, "
            f"{algorithm_config.n_eval_episodes} episodes, seed mode {args.eval_seed_mode} from "
            f"{args.eval_seed}, reward_norm={evaluation_config['reward_norm']}, "
            f"reward_clip={evaluation_config['reward_clip']}"
        )

    def run_evaluation() -> Dict[str, float]:
        nonlocal best_eval_reward
        step = trainer.total_environment_steps
        evaluation_index = evaluation_log.n_rows
        # "rolling": every evaluation gets a new block of episode seeds.
        base_seed = args.eval_seed + (
            evaluation_index * algorithm_config.n_eval_episodes if args.eval_seed_mode == "rolling" else 0
        )
        # Deterministic evaluation uses its own environments and leaves
        # every global RNG untouched, so it cannot change training.
        _episodes, summary = evaluate_policy(
            environment_config=evaluation_config,
            bay_actor=bay_actor,
            row_actor=row_actor,
            n_episodes=algorithm_config.n_eval_episodes,
            base_seed=base_seed,
            eval_batch_size=args.eval_batch_size or 8,
            deterministic=True,
            verbose=False,
        )
        last_seed = base_seed + summary.n_episodes - 1
        evaluation_log.log({
            "timesteps": step,
            "mean_reward": summary.mean_reward,
            "std_reward": summary.std_reward,
            "mean_episode_length": summary.mean_episode_length,
            "completion_rate": summary.completion_rate,
            "base_seed": base_seed,
        })
        print(
            f"Evaluation at {step:,} steps | Seeds: {base_seed}-{last_seed} "
            f"| Mean reward: {summary.mean_reward:.4f} | Std: {summary.std_reward:.4f} "
            f"| Complete: {summary.successful_episodes}/{summary.n_episodes}",
            flush=True,
        )

        if summary.mean_reward > best_eval_reward:
            best_eval_reward = summary.mean_reward
            if args.save_model:
                save_models(save_dir, trainer, environment_config, algorithm_config, prefix="best")
                (save_dir / "best_evaluation.json").write_text(json.dumps({
                    "timesteps": step,
                    "evaluation_environment": dict(evaluation_config, seed=base_seed),
                    "evaluation_index": evaluation_index,
                    "base_seed": base_seed,
                    "last_seed": last_seed,
                    "seed_mode": args.eval_seed_mode,
                    **summary.to_dict(),
                }, indent=2), encoding="utf-8")

        if not args.no_training_plots:
            try:
                fig, _ = plot_csv_training_curves(
                    [evaluation_log.path],
                    display_name=display_name,
                    ema_alpha=args.ema_alpha,
                    n_points=args.plot_points,
                    save_path=log_dir / "training_curves",
                )
                fig.clear()
            except (ImportError, OSError, ValueError) as error:
                print(f"Training plot could not be refreshed: {error}")

        return {
            "eval/mean_reward": summary.mean_reward,
            "eval/std_reward": summary.std_reward,
            "eval/mean_episode_length": summary.mean_episode_length,
            "eval/completion_rate": summary.completion_rate,
            "eval/evaluation_index": evaluation_index,
            "eval/base_seed": base_seed,
        }

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **environment_config,
                **algorithm_config.to_dict(),
                "environment_size": args.size,
                "seed": args.seed,
                "architecture": "sequential_multi_agent_ppo",
                "evaluation_initial_base_seed": args.eval_seed,
                "evaluation_seed_mode": args.eval_seed_mode,
                "evaluation_at_step_zero": False,
            },
        )

    # Evaluation and checkpoints happen at exact multiples of their
    # frequencies; evaluation may fall inside a rollout (step_callback).
    def next_multiple(frequency: int) -> int:
        return (trainer.total_environment_steps // frequency + 1) * frequency

    next_checkpoint = next_multiple(algorithm_config.checkpoint_freq)
    next_eval = next_multiple(algorithm_config.eval_freq)

    def evaluate_if_due(completed_steps: int) -> None:
        nonlocal next_eval
        if args.no_eval or completed_steps < next_eval:
            return
        if completed_steps != next_eval:
            raise RuntimeError(
                f"Periodic evaluation threshold was skipped: expected {next_eval:,}, "
                f"reached {completed_steps:,}."
            )
        # Counted inside this iteration's rollout_seconds as well.
        with profiler.region("evaluation_seconds", sync_cuda=True):
            evaluation_stats = run_evaluation()
        if wandb_run is not None:
            wandb_run.log({"total_environment_steps": completed_steps, **evaluation_stats}, step=completed_steps)
        next_eval += algorithm_config.eval_freq

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    try:
        while trainer.total_environment_steps < algorithm_config.total_timesteps:

            # The final rollout may be shorter than a full buffer.
            remaining_steps = algorithm_config.total_timesteps - trainer.total_environment_steps
            buffer = JointRolloutBuffer(
                layout,
                _round_up(min(algorithm_config.buffer_size, remaining_steps), args.num_envs),
                args.num_envs,
                device,
            )

            iteration_start = time.perf_counter()
            stats = trainer.train_iteration(env, buffer, step_callback=evaluate_if_due)
            print_training_stats(stats, algorithm_config.total_timesteps)

            if args.save_model and trainer.total_environment_steps >= next_checkpoint:
                with profiler.region("checkpoint_seconds"):
                    save_training_state(save_dir, trainer, environment_config, algorithm_config, best_eval_reward)
                next_checkpoint = next_multiple(algorithm_config.checkpoint_freq)

            # With --profile, the same columns on every row whether or not
            # this iteration checkpointed. Wall-clock throughput is always
            # logged and includes the checkpoint.
            if profiler.enabled:
                stats["checkpoint_seconds"] = 0.0
                stats.update(profiler.pop_stats())
            iteration_seconds = time.perf_counter() - iteration_start
            stats["steps_per_second"] = stats["rollout_steps"] / iteration_seconds if iteration_seconds > 0 else 0.0

            rollout_log.log({"timesteps": trainer.total_environment_steps, **stats})
            if wandb_run is not None:
                wandb_run.log(stats, step=trainer.total_environment_steps)

        if args.save_model:
            save_models(save_dir, trainer, environment_config, algorithm_config, prefix="final")
            save_training_state(save_dir, trainer, environment_config, algorithm_config, best_eval_reward)

    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print("\nTraining finished successfully.")


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Train separated Bay/Row agents with sequential multi-agent PPO."
    )

    # Environment
    parser.add_argument("--size", default="small_with_margin", choices=ENVIRONMENT_SIZES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_group_sizes", action="store_true")
    add_reward_arguments(parser)

    # Runtime
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Environments stepped together; every actor/critic forward pass "
                             "is batched across them.")
    parser.add_argument("--vec_backend", choices=["sync", "subproc"], default="sync",
                        help="'sync': SB3 DummyVecEnv (one process). 'subproc': SB3 SubprocVecEnv "
                             "(one process per environment).")

    # Training profile and overrides (default: the size's profile)
    parser.add_argument("--training_profile", default="auto",
                        choices=["auto", *sorted(TRAINING_PROFILES)],
                        help="'auto' maps --size to small, medium, large or massive.")
    parser.add_argument("--timesteps", type=int, help="Override total training timesteps.")
    parser.add_argument("--buffer_size", type=int, help="Override the rollout buffer size.")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--clip_range", type=float)
    parser.add_argument("--ent_coef", type=float)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="Environment steps between overwrites of latest_training_state.pt.")

    # Periodic evaluation
    parser.add_argument("--eval_freq", type=int,
                        help="Evaluate at exact multiples of this many training steps (default 25000).")
    parser.add_argument("--n_eval_episodes", type=int, help="Episodes per evaluation (default 10).")
    parser.add_argument("--eval_batch_size", type=int,
                        help="Episodes evaluated simultaneously (default 8); does not change results.")
    parser.add_argument("--eval_seed", type=int, default=100_000,
                        help="First evaluation episode seed; independent of the training seed.")
    parser.add_argument("--eval_seed_mode", choices=["rolling", "fixed"], default="rolling",
                        help="'rolling': a new seed block per evaluation; 'fixed': always the first.")
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable periodic evaluation; rollouts.csv is still written.")
    add_evaluation_reward_argument(parser)

    # Logging
    parser.add_argument("--ema_alpha", type=float, default=0.1,
                        help="Current-value weight of the training-curve EMA.")
    parser.add_argument("--plot_points", type=int, default=500,
                        help="Interpolation grid size before the EMA.")
    parser.add_argument("--no_training_plots", action="store_true",
                        help="Keep the evaluation CSV but skip the live PNG/PDF curves.")
    parser.add_argument("--profile", action="store_true",
                        help="Record per-phase wall-clock timings (rollout, critic/Bay/Row updates, "
                             "evaluation, checkpoint) in rollouts.csv. Off by default: each timed "
                             "phase synchronizes CUDA once, never per environment step.")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="stack-sequential-hppo")
    parser.add_argument("--wandb_run_name")

    # Saving
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_dir", default="./models/sequential_hppo")
    parser.add_argument("--fresh", action="store_true",
                        help="Start from new networks even when latest_training_state.pt exists.")

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
