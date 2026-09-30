"""
Evaluation of trained Sequential HPPO actors.

Every step makes the same Bay -> Row decision as training rollouts
(select_hierarchical_action) and takes ONE StackEnv.step(bay * n_rows
+ row). The centralized critic is not used at inference time.

    python -m stack.evaluation.evaluate --model_dir <run dir> --prefix best

Writes evaluation_summary.json, episode_metrics.json (per-step rewards,
Bay/Row/StackEnv actions) and, unless --no_plots, cumulative_rewards.png
and episode_rewards.png to --output_dir.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from ..configs.device import get_device
from ..configs.hierarchical_config import HierarchicalConfig
from ..envs.stack_gym import StackEnv
from ..models.pointer_actor import PointerActor
from ..training.bay_row_layout import BayRowLayout, select_hierarchical_action
from .metrics import (
    EpisodeMetrics,
    EvaluationSummary,
    build_episode_metrics,
    summarize_episodes,
)


def load_checkpoint_config(config_path: Path) -> Tuple[Dict, HierarchicalConfig]:
    """Environment and algorithm config from a {prefix}_config.json
    written by run_sequential_hppo (training/checkpoint.py)."""

    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config does not exist: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    environment_config = dict(data["environment"])
    # JSON turns the shape tuples into lists.
    for key in ("vessel_shape", "yard_shape"):
        if key in environment_config:
            environment_config[key] = tuple(environment_config[key])

    return environment_config, HierarchicalConfig(**data["algorithm"])


def make_evaluation_config(environment_config: Dict) -> Dict:
    """Copy of an environment config with raw rewards.

    reward_norm / reward_clip only shape the training signal. Every
    evaluation - periodic during training, this script and plots.py -
    reports raw rewards, the scale the baselines' evaluation uses too.
    """
    return {**environment_config, "reward_norm": False, "reward_clip": False}


def build_evaluation_system(
    *,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    bay_actor_path: Path,
    row_actor_path: Path,
    device: torch.device,
) -> Tuple[PointerActor, PointerActor]:
    """The trained Bay and Row actors in eval mode (no critic, no trainer)."""

    probe_env = StackEnv(config=environment_config)
    try:
        layout = BayRowLayout.from_env(probe_env)
    finally:
        probe_env.close()

    bay_actor, row_actor = layout.build_actors(**algorithm_config.actor_kwargs())

    for actor, path, name in ((bay_actor, bay_actor_path, "Bay actor"), (row_actor, row_actor_path, "Row actor")):
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
        actor.to(device)
        actor.load_state_dict(torch.load(path, map_location=device))
        actor.eval()

    return bay_actor, row_actor


@torch.no_grad()
def evaluate_policy(
    *,
    environment_config: Dict,
    bay_actor: PointerActor,
    row_actor: PointerActor,
    n_episodes: int = 10,
    base_seed: int = 42,
    eval_batch_size: int = 8,
    deterministic: bool = True,
    verbose_steps: bool = False,
    verbose: bool = True,
) -> Tuple[List[EpisodeMetrics], EvaluationSummary]:
    """
    Run n_episodes, up to eval_batch_size at a time in an SB3 DummyVecEnv.

    Episode i always uses seed base_seed + i in its own StackEnv, so
    deterministic results do not depend on eval_batch_size. A slot stops
    recording once its episode ends (DummyVecEnv auto-resets it); the
    padding slots of a partial last round never record.

    verbose prints one line per episode, verbose_steps one per decision.
    Rewards are always raw (make_evaluation_config), whatever the config says.
    """

    environment_config = make_evaluation_config(environment_config)
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    eval_batch_size = min(eval_batch_size, n_episodes)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")

    bay_actor.eval()
    row_actor.eval()
    device = next(bay_actor.parameters()).device

    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION (batched)")
        print("=" * 70)

    vec_env = DummyVecEnv([lambda: StackEnv(config=environment_config)] * eval_batch_size)
    layout = BayRowLayout.from_env(vec_env)
    episodes: List[EpisodeMetrics] = []

    try:
        for round_start in range(0, n_episodes, eval_batch_size):
            n_active = min(eval_batch_size, n_episodes - round_start)

            # VecEnv.seed(s) resets slot j with seed s + j.
            vec_env.seed(base_seed + round_start)
            observations = vec_env.reset()

            active = np.arange(eval_batch_size) < n_active
            traces = [defaultdict(list) for _ in range(eval_batch_size)]
            final = [(False, False, {})] * eval_batch_size  # (terminated, truncated, info)

            while active.any():
                action = select_hierarchical_action(
                    layout,
                    bay_actor,
                    row_actor,
                    torch.as_tensor(observations, dtype=torch.float32, device=device),
                    get_action_masks(vec_env),
                    deterministic=deterministic,
                )
                observations, rewards, dones, infos = vec_env.step(action.stack_actions)

                for i in np.flatnonzero(active):
                    global_action = int(action.stack_actions[i])
                    bay_idx, row_idx = divmod(global_action, layout.n_rows)

                    trace = traces[i]
                    trace["step_rewards"].append(float(rewards[i]))
                    trace["bay_actions"].append(bay_idx)
                    trace["row_actions"].append(row_idx)
                    trace["global_actions"].append(global_action)

                    if verbose_steps:
                        print(
                            f"Episode {round_start + i + 1:03d} | Step {len(trace['step_rewards']):03d} | "
                            f"Bay {bay_idx} | Row {row_idx} | Reward {rewards[i]:8.3f} | "
                            f"Cumulative {sum(trace['step_rewards']):9.3f}"
                        )

                    if dones[i]:
                        # SB3 folds (terminated, truncated) into done plus
                        # this flag, set only when not terminated.
                        truncated = bool(infos[i].get("TimeLimit.truncated", False))
                        final[i] = (not truncated, truncated, dict(infos[i]))
                        active[i] = False

            for i in range(n_active):
                terminated, truncated, info = final[i]
                episode = build_episode_metrics(
                    episode_index=round_start + i,
                    terminated=terminated,
                    truncated=truncated,
                    final_info=info,
                    **traces[i],
                )
                episodes.append(episode)

                if verbose:
                    status = "SUCCESS" if episode.completed_successfully else "INCOMPLETE"
                    print(
                        f"Episode {round_start + i + 1:03d} | Seed {base_seed + round_start + i} | "
                        f"Reward {episode.total_reward:10.3f} | Steps {episode.episode_length:4d} | {status}"
                    )
    finally:
        vec_env.close()

    return episodes, summarize_episodes(episodes)


def print_summary(summary: EvaluationSummary) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Episodes: {summary.n_episodes}")
    print(f"Mean reward: {summary.mean_reward:.4f}")
    print(f"Reward std: {summary.std_reward:.4f}")
    print(f"Min reward: {summary.min_reward:.4f}")
    print(f"Max reward: {summary.max_reward:.4f}")
    print(f"Mean episode length: {summary.mean_episode_length:.2f}")
    print(f"Successful episodes: {summary.successful_episodes}/{summary.n_episodes}")
    print(f"Completion rate: {summary.completion_rate:.2%}")
    print(f"Truncated episodes: {summary.truncated_episodes}")
    print(f"IMO violation episodes: {summary.imo_violation_episodes}")
    print(f"All-actions-masked episodes: {summary.all_actions_masked_episodes}")


def save_metrics(*, output_dir: Path, episodes: List[EpisodeMetrics], summary: EvaluationSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    episode_path = output_dir / "episode_metrics.json"

    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    episode_path.write_text(json.dumps([episode.to_dict() for episode in episodes], indent=2), encoding="utf-8")

    print("\nSaved evaluation metrics:")
    print(f"  {summary_path}")
    print(f"  {episode_path}")


def plot_rewards(
    *,
    episodes: List[EpisodeMetrics],
    summary: EvaluationSummary,
    output_dir: Path,
    show_plot: bool = False,
) -> None:
    """cumulative_rewards.png (one line per episode) and
    episode_rewards.png (total reward per episode)."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    def finish(title: str, xlabel: str, ylabel: str, filename: str, legend: bool) -> None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        if show_plot:
            plt.show()
        plt.close()

    plt.figure(figsize=(10, 6))
    for episode in episodes:
        steps = np.arange(1, episode.episode_length + 1)
        plt.plot(steps, np.cumsum(episode.step_rewards), label=f"Episode {episode.episode_index + 1}")
    # Avoid a huge legend when evaluating many episodes.
    finish("Evaluation Cumulative Reward", "Container placement step", "Cumulative reward",
           "cumulative_rewards.png", legend=len(episodes) <= 10)
    print(f"Saved cumulative reward plot: {output_dir / 'cumulative_rewards.png'}")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, summary.n_episodes + 1), summary.episode_rewards, marker="o")
    plt.axhline(summary.mean_reward, linestyle="--", label=f"Mean = {summary.mean_reward:.3f}")
    finish("Evaluation Episode Rewards", "Evaluation episode", "Total episode reward",
           "episode_rewards.png", legend=True)
    print(f"Saved episode reward plot: {output_dir / 'episode_rewards.png'}")


def main(args: argparse.Namespace) -> None:

    model_dir = Path(args.model_dir)
    bay_actor_path = model_dir / f"{args.prefix}_bay_actor.pt"
    row_actor_path = model_dir / f"{args.prefix}_row_actor.pt"
    output_dir = Path(args.output_dir)

    environment_config, algorithm_config = load_checkpoint_config(model_dir / f"{args.prefix}_config.json")
    environment_config = make_evaluation_config(environment_config)
    print(
        f"Evaluation reward_norm={environment_config['reward_norm']}, "
        f"reward_clip={environment_config['reward_clip']}"
    )

    device = torch.device(get_device(args.device))
    print(f"Using device: {device}")
    print(f"Bay actor checkpoint: {bay_actor_path}")
    print(f"Row actor checkpoint: {row_actor_path}")

    bay_actor, row_actor = build_evaluation_system(
        environment_config=environment_config,
        algorithm_config=algorithm_config,
        bay_actor_path=bay_actor_path,
        row_actor_path=row_actor_path,
        device=device,
    )

    episodes, summary = evaluate_policy(
        environment_config=environment_config,
        bay_actor=bay_actor,
        row_actor=row_actor,
        n_episodes=args.episodes,
        base_seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        deterministic=not args.stochastic,
        verbose_steps=args.verbose_steps,
    )

    print_summary(summary)
    save_metrics(output_dir=output_dir, episodes=episodes, summary=summary)
    if not args.no_plots:
        plot_rewards(episodes=episodes, summary=summary, output_dir=output_dir, show_plot=args.show_plot)

    print("\nEvaluation finished successfully.")


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Evaluate trained sequential Bay/Row multi-agent PPO.")

    # Checkpoint
    parser.add_argument("--model_dir", type=str, default="./models/sequential_hppo",
                        help="Directory containing the saved Bay actor, Row actor and config files.")
    parser.add_argument("--prefix", type=str, default="final",
                        help="Checkpoint prefix: 'best' or 'final'.")

    # Evaluation
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Number of episodes' environments to run simultaneously (batched Agent B/"
                             "Agent R inference + DummyVecEnv stepping). Does not change results, only "
                             "throughput - see evaluate_policy().")
    parser.add_argument("--seed", type=int, default=1000, help="Base evaluation seed. Episode i uses seed + i.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of using deterministic argmax inference.")

    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--verbose_steps", action="store_true",
                        help="Print Bay, Row, reward and cumulative reward at every environment step.")
    parser.add_argument("--show_plot", action="store_true", help="Display plots in addition to saving them.")
    parser.add_argument("--no_plots", action="store_true", help="Skip matplotlib visualization.")

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
