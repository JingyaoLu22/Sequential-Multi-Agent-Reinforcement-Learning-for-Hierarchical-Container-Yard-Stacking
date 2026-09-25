"""
Evaluation for the separated hierarchical multi-agent PPO pipeline.

Evaluation architecture
-----------------------

    global observation
          |
          v
       Agent B
          |
          v
       bay_idx
          |
          | NO environment step
          v
    selected-bay observation
          |
          v
       Agent R
          |
          v
       row_idx
          |
          v
    ONE StackEnv.step(bay_idx * n_rows + row_idx)
          |
          v
       reward

The centralized critic is NOT used during inference.

Evaluation records:

    - reward at every placement step
    - cumulative reward
    - total episode reward
    - episode length
    - Bay decisions
    - Row decisions
    - original StackEnv global actions
    - completion rate
    - truncation rate
    - IMO violations
    - all-actions-masked failures

It can also save:

    cumulative_rewards.png
    episode_rewards.png
    evaluation_summary.json
    episode_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from ..models.pointer_actor import (
    PointerActor,
    load_actor_state_dict,
)

from ..configs.device import (
    get_device,
)

from ..configs.hierarchical_config import (
    HierarchicalConfig,
)

from ..envs.stack_gym import StackEnv

from ..training.bay_row_layout import (
    BayRowLayout,
    select_hierarchical_action,
)

from .metrics import (
    EpisodeMetrics,
    EvaluationSummary,
    build_episode_metrics,
    summarize_episodes,
)


# ======================================================================
# Load saved configuration
# ======================================================================


def load_checkpoint_config(
    config_path: Path,
) -> Tuple[
    Dict,
    HierarchicalConfig,
]:
    """
    Load environment + algorithm configuration saved by run.py.

    run.py stores:

        {
            "environment": {...},
            "algorithm": {...}
        }

    The environment configuration is reused so evaluation runs with
    the same physical environment definition as training.
    """

    if not config_path.exists():

        raise FileNotFoundError(
            f"Checkpoint config does not exist: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if "environment" not in data:

        raise ValueError(
            "Checkpoint configuration does not contain "
            "'environment'."
        )

    if "algorithm" not in data:

        raise ValueError(
            "Checkpoint configuration does not contain "
            "'algorithm'."
        )

    # ==============================================================
    # Environment configuration
    # ==============================================================

    environment_config = dict(
        data["environment"]
    )

    # JSON converts tuples to lists.
    #
    # Restore the source configuration representation.
    if "vessel_shape" in environment_config:

        environment_config[
            "vessel_shape"
        ] = tuple(
            environment_config[
                "vessel_shape"
            ]
        )

    if "yard_shape" in environment_config:

        environment_config[
            "yard_shape"
        ] = tuple(
            environment_config[
                "yard_shape"
            ]
        )

    # ==============================================================
    # Algorithm configuration
    # ==============================================================

    algorithm_config = (
        HierarchicalConfig(
            **data["algorithm"]
        )
    )

    return (
        environment_config,
        algorithm_config,
    )


# ======================================================================
# Evaluation reward configuration
# ======================================================================
#
# The ONE place both periodic training evaluation (run_sequential_hppo)
# and this standalone script decide the evaluation reward scale, so the
# same checkpoint reports the same numbers in both.


def make_evaluation_config(
    environment_config: Dict,
    use_training_rewards: bool = False,
) -> Dict:
    """Copy of the training environment config for evaluation.

    By default rewards are raw (no normalization, no clipping), the scale
    plots.py and the baselines report. use_training_rewards keeps the
    training run's reward_norm / reward_clip instead.
    """
    evaluation_config = dict(environment_config)
    if not use_training_rewards:
        evaluation_config["reward_norm"] = False
        evaluation_config["reward_clip"] = False
    return evaluation_config


def add_evaluation_reward_argument(parser: argparse.ArgumentParser) -> None:
    """--eval_use_training_rewards, shared by training and this script."""
    parser.add_argument(
        "--eval_use_training_rewards",
        action="store_true",
        help=(
            "Evaluate with the training run's reward normalization/clipping "
            "instead of raw rewards (default: raw, identical for training-time "
            "and standalone evaluation)."
        ),
    )


# ======================================================================
# Build evaluation actors
# ======================================================================


def build_evaluation_system(
    *,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    agent_b_path: Path,
    agent_r_path: Path,
    device: torch.device,
) -> Tuple[
    PointerActor,
    PointerActor,
]:
    """
    Build the inference-only architecture.

    Notice that there is NO centralized critic and NO trainer here.

    Execution requires only:

        bay actor (Agent B)
        row actor (Agent R)

    The environments themselves are created by evaluate_policy().
    """

    # ==============================================================
    # Dimensions come directly from the environment
    # ==============================================================

    probe_env = StackEnv(
        config=environment_config,
    )

    try:
        layout = BayRowLayout.from_env(
            probe_env
        )
    finally:
        probe_env.close()

    # ==============================================================
    # Agent B / Agent R actors
    # ==============================================================

    bay_actor, row_actor = layout.build_actors(
        **algorithm_config.actor_kwargs(),
    )

    bay_actor.to(device)
    row_actor.to(device)

    # ==============================================================
    # Load trained actor parameters
    # ==============================================================

    if not agent_b_path.exists():

        raise FileNotFoundError(
            f"Agent B checkpoint not found: "
            f"{agent_b_path}"
        )

    if not agent_r_path.exists():

        raise FileNotFoundError(
            f"Agent R checkpoint not found: "
            f"{agent_r_path}"
        )

    load_actor_state_dict(
        bay_actor,
        torch.load(agent_b_path, map_location=device),
    )

    load_actor_state_dict(
        row_actor,
        torch.load(agent_r_path, map_location=device),
    )

    # ==============================================================
    # Evaluation mode
    # ==============================================================

    bay_actor.eval()
    row_actor.eval()

    return (
        bay_actor,
        row_actor,
    )


# ======================================================================
# Evaluate multiple episodes - batched across environments
# ======================================================================


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
) -> Tuple[
    List[EpisodeMetrics],
    EvaluationSummary,
]:
    """
    Evaluate the two-agent policy over multiple episodes, running up
    to eval_batch_size environments simultaneously (Agent B/Agent R
    inference batched, StackEnv copies stepped through an SB3
    DummyVecEnv) instead of one environment/episode at a time.

    Episode i always uses seed = base_seed + i, regardless of
    eval_batch_size, round count, or partial final rounds, and each
    environment slot owns its own independent StackEnv instance (own
    RNG). With deterministic=True every episode therefore has exactly
    the same action sequence for any eval_batch_size, including 1.

    Episodes run in rounds of at most eval_batch_size. Within a round,
    every environment slot keeps stepping every iteration (Agent B/
    Agent R inference and StackEnv.step() are always called for the
    full batch), but an `active` mask stops this function from
    recording anything for a slot once its episode has terminated or
    truncated - DummyVecEnv auto-resets that slot into a new
    (uninteresting) episode internally, which is harmless precisely
    because nothing reads that slot's output again after `active`
    flips False for it. On the last, possibly partial round (fewer
    than eval_batch_size episodes remaining), the unused slots are
    seeded but marked inactive from the start, so they contribute
    nothing and are simply wasted compute for that round.

    verbose controls only the "EVALUATION (batched)" header and the
    one-line-per-episode summary; periodic in-training evaluation sets
    it False to stay quiet. verbose_steps (per-decision printing) is independent and
    still defaults to False either way.
    """

    if n_episodes <= 0:

        raise ValueError(
            "n_episodes must be positive."
        )

    eval_batch_size = min(
        eval_batch_size,
        n_episodes,
    )

    if eval_batch_size <= 0:

        raise ValueError(
            "eval_batch_size must be positive."
        )

    bay_actor.eval()
    row_actor.eval()

    device = next(bay_actor.parameters()).device

    episodes: List[
        Optional[EpisodeMetrics]
    ] = [None] * n_episodes

    if verbose:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "EVALUATION (batched)"
        )

        print(
            "=" * 70
        )

    vec_env = DummyVecEnv(
        [lambda: StackEnv(config=environment_config)]
        * eval_batch_size
    )

    layout = BayRowLayout.from_env(
        vec_env
    )

    try:

        with torch.no_grad():

            # ==========================================================
            # Rounds of at most eval_batch_size episodes
            # ==========================================================

            for round_start in range(
                0,
                n_episodes,
                eval_batch_size,
            ):

                round_indices = list(
                    range(
                        round_start,
                        min(
                            round_start + eval_batch_size,
                            n_episodes,
                        ),
                    )
                )

                k = len(round_indices)

                # VecEnv.seed(s) resets slot j with seed s + j, so
                # slot j of this round gets base_seed + round_start + j,
                # i.e. base_seed + (its episode index). Slots j >= k
                # are padding for a partial final round and are marked
                # inactive below so their results are never read.
                vec_env.seed(
                    base_seed + round_start
                )

                observations = vec_env.reset()

                active = np.zeros(
                    eval_batch_size,
                    dtype=bool,
                )
                active[:k] = True

                step_rewards = [[] for _ in range(eval_batch_size)]
                bay_actions_acc = [[] for _ in range(eval_batch_size)]
                row_actions_acc = [[] for _ in range(eval_batch_size)]
                selected_bays_acc = [[] for _ in range(eval_batch_size)]
                selected_rows_acc = [[] for _ in range(eval_batch_size)]
                global_actions_acc = [[] for _ in range(eval_batch_size)]
                final_terminated = [False] * eval_batch_size
                final_truncated = [False] * eval_batch_size
                final_info: List[Dict] = [{} for _ in range(eval_batch_size)]

                while active.any():

                    # Same Bay -> Row decision as training rollouts.
                    action = select_hierarchical_action(
                        layout,
                        bay_actor,
                        row_actor,
                        torch.as_tensor(observations, dtype=torch.float32, device=device),
                        get_action_masks(vec_env),
                        deterministic=deterministic,
                    )

                    (
                        observations,
                        rewards,
                        dones,
                        infos,
                    ) = vec_env.step(
                        action.stack_actions
                    )

                    for i in range(eval_batch_size):

                        if not active[i]:
                            continue

                        global_action = int(action.stack_actions[i])
                        bay_idx, row_idx = divmod(global_action, layout.n_rows)

                        # StackEnv's own (odd bay, 1-based row) numbering.
                        (
                            selected_bay,
                            selected_row,
                        ) = vec_env.envs[i]._action_to_bay_row(
                            global_action
                        )

                        step_rewards[i].append(float(rewards[i]))
                        bay_actions_acc[i].append(bay_idx)
                        row_actions_acc[i].append(row_idx)
                        selected_bays_acc[i].append(int(selected_bay))
                        selected_rows_acc[i].append(int(selected_row))
                        global_actions_acc[i].append(global_action)

                        if verbose_steps:

                            print(
                                f"Episode "
                                f"{round_indices[i] + 1:03d} | "
                                f"Step "
                                f"{len(step_rewards[i]):03d} | "
                                f"Bay "
                                f"{selected_bay} | "
                                f"Row "
                                f"{selected_row} | "
                                f"Reward "
                                f"{rewards[i]:8.3f} | "
                                f"Cumulative "
                                f"{sum(step_rewards[i]):9.3f}"
                            )

                        if dones[i]:
                            # SB3 folds StackEnv's (terminated,
                            # truncated) into done plus this flag,
                            # which is set only when not terminated.
                            truncated = bool(
                                infos[i].get(
                                    "TimeLimit.truncated",
                                    False,
                                )
                            )
                            final_terminated[i] = not truncated
                            final_truncated[i] = truncated
                            final_info[i] = dict(infos[i])
                            active[i] = False

                for local_i, global_i in enumerate(round_indices):

                    episode = build_episode_metrics(
                        episode_index=global_i,
                        step_rewards=step_rewards[local_i],
                        bay_actions=bay_actions_acc[local_i],
                        row_actions=row_actions_acc[local_i],
                        selected_bays=selected_bays_acc[local_i],
                        selected_rows=selected_rows_acc[local_i],
                        global_actions=global_actions_acc[local_i],
                        terminated=final_terminated[local_i],
                        truncated=final_truncated[local_i],
                        final_info=final_info[local_i],
                    )

                    episodes[global_i] = episode

                    if verbose:

                        episode_seed = base_seed + global_i

                        status = (
                            "SUCCESS"
                            if episode.completed_successfully
                            else "INCOMPLETE"
                        )

                        print(
                            f"Episode "
                            f"{global_i + 1:03d} | "
                            f"Seed "
                            f"{episode_seed} | "
                            f"Reward "
                            f"{episode.total_reward:10.3f} | "
                            f"Steps "
                            f"{episode.episode_length:4d} | "
                            f"{status}"
                        )

    finally:

        vec_env.close()

    # ==============================================================
    # Aggregate metrics
    # ==============================================================

    complete_episodes: List[EpisodeMetrics] = episodes  # type: ignore[assignment]

    summary = summarize_episodes(
        complete_episodes
    )

    return (
        complete_episodes,
        summary,
    )


# ======================================================================
# Console summary
# ======================================================================


def print_summary(
    summary: EvaluationSummary,
) -> None:
    """
    Print final evaluation statistics.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Episodes: "
        f"{summary.n_episodes}"
    )

    print(
        f"Mean reward: "
        f"{summary.mean_reward:.4f}"
    )

    print(
        f"Reward std: "
        f"{summary.std_reward:.4f}"
    )

    print(
        f"Min reward: "
        f"{summary.min_reward:.4f}"
    )

    print(
        f"Max reward: "
        f"{summary.max_reward:.4f}"
    )

    print(
        f"Mean episode length: "
        f"{summary.mean_episode_length:.2f}"
    )

    print(
        f"Successful episodes: "
        f"{summary.successful_episodes}"
        f"/"
        f"{summary.n_episodes}"
    )

    print(
        f"Completion rate: "
        f"{summary.completion_rate:.2%}"
    )

    print(
        f"Truncated episodes: "
        f"{summary.truncated_episodes}"
    )

    print(
        f"IMO violation episodes: "
        f"{summary.imo_violation_episodes}"
    )

    print(
        f"All-actions-masked episodes: "
        f"{summary.all_actions_masked_episodes}"
    )


# ======================================================================
# Save JSON metrics
# ======================================================================


def save_metrics(
    *,
    output_dir: Path,
    episodes: List[
        EpisodeMetrics
    ],
    summary: EvaluationSummary,
) -> None:
    """
    Save complete evaluation metrics.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # Summary
    # ==============================================================

    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary.to_dict(),
            file,
            indent=2,
        )

    # ==============================================================
    # Full episode traces
    # ==============================================================

    episode_path = (
        output_dir
        / "episode_metrics.json"
    )

    with open(
        episode_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                episode.to_dict()
                for episode in episodes
            ],
            file,
            indent=2,
        )

    print(
        "\nSaved evaluation metrics:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {episode_path}"
    )


# ======================================================================
# Plot cumulative rewards
# ======================================================================


def plot_cumulative_rewards(
    *,
    episodes: List[
        EpisodeMetrics
    ],
    output_dir: Path,
    show_plot: bool = False,
) -> None:
    """
    Plot cumulative reward as a function of placement step.

    Each line represents one evaluation episode:

        C_t =
            sum_{k=0}^{t} r_k

    This is the inference-time reward visualization discussed for the
    project.
    """

    try:

        import matplotlib.pyplot as plt

    except ImportError as error:

        raise ImportError(
            "matplotlib is required for evaluation plots."
        ) from error

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # Separate figure: cumulative reward
    # ==============================================================

    plt.figure(
        figsize=(10, 6)
    )

    for episode in episodes:

        steps = np.arange(
            1,
            episode.episode_length + 1,
        )

        plt.plot(
            steps,
            episode.cumulative_rewards,
            label=(
                f"Episode "
                f"{episode.episode_index + 1}"
            ),
        )

    plt.xlabel(
        "Container placement step"
    )

    plt.ylabel(
        "Cumulative reward"
    )

    plt.title(
        "Evaluation Cumulative Reward"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    # Avoid a huge legend when evaluating many episodes.
    if len(episodes) <= 10:

        plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "cumulative_rewards.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    if show_plot:

        plt.show()

    plt.close()

    print(
        f"Saved cumulative reward plot: "
        f"{output_path}"
    )


# ======================================================================
# Plot episode total rewards
# ======================================================================


def plot_episode_rewards(
    *,
    summary: EvaluationSummary,
    output_dir: Path,
    show_plot: bool = False,
) -> None:
    """
    Plot final reward obtained in each evaluation episode.
    """

    try:

        import matplotlib.pyplot as plt

    except ImportError as error:

        raise ImportError(
            "matplotlib is required for evaluation plots."
        ) from error

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_indices = np.arange(
        1,
        summary.n_episodes + 1,
    )

    # ==============================================================
    # Separate figure: episode rewards
    # ==============================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        episode_indices,
        summary.episode_rewards,
        marker="o",
    )

    plt.axhline(
        summary.mean_reward,
        linestyle="--",
        label=(
            f"Mean = "
            f"{summary.mean_reward:.3f}"
        ),
    )

    plt.xlabel(
        "Evaluation episode"
    )

    plt.ylabel(
        "Total episode reward"
    )

    plt.title(
        "Evaluation Episode Rewards"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "episode_rewards.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    if show_plot:

        plt.show()

    plt.close()

    print(
        f"Saved episode reward plot: "
        f"{output_path}"
    )


# ======================================================================
# Main
# ======================================================================


def main(
    args: argparse.Namespace,
) -> None:

    # ==============================================================
    # Paths
    # ==============================================================

    model_dir = Path(
        args.model_dir
    )

    prefix = args.prefix

    agent_b_path = (
        model_dir
        / f"{prefix}_agent_b.pt"
    )

    agent_r_path = (
        model_dir
        / f"{prefix}_agent_r.pt"
    )

    config_path = (
        model_dir
        / f"{prefix}_config.json"
    )

    output_dir = Path(
        args.output_dir
    )

    # ==============================================================
    # Load saved training configuration
    # ==============================================================

    (
        environment_config,
        algorithm_config,
    ) = load_checkpoint_config(
        config_path
    )

    # Same reward scale as the periodic evaluation logged during training.
    environment_config = make_evaluation_config(
        environment_config,
        use_training_rewards=args.eval_use_training_rewards,
    )

    print(
        f"Evaluation reward_norm={environment_config['reward_norm']}, "
        f"reward_clip={environment_config['reward_clip']}"
    )

    # ==============================================================
    # Device
    # ==============================================================

    device = torch.device(
        get_device(args.device)
    )

    print(
        f"Using device: {device}"
    )

    print(
        f"Agent B checkpoint: "
        f"{agent_b_path}"
    )

    print(
        f"Agent R checkpoint: "
        f"{agent_r_path}"
    )

    # ==============================================================
    # Build inference architecture
    # ==============================================================

    (
        bay_actor,
        row_actor,
    ) = build_evaluation_system(

        environment_config=(
            environment_config
        ),

        algorithm_config=(
            algorithm_config
        ),

        agent_b_path=(
            agent_b_path
        ),

        agent_r_path=(
            agent_r_path
        ),

        device=device,
    )

    # ==============================================================
    # Evaluation
    # ==============================================================

    (
        episodes,
        summary,
    ) = evaluate_policy(

        environment_config=environment_config,

        bay_actor=bay_actor,

        row_actor=row_actor,

        n_episodes=(
            args.episodes
        ),

        base_seed=(
            args.seed
        ),

        eval_batch_size=(
            args.eval_batch_size
        ),

        deterministic=(
            not args.stochastic
        ),

        verbose_steps=(
            args.verbose_steps
        ),
    )

    # ==============================================================
    # Print metrics
    # ==============================================================

    print_summary(
        summary
    )

    # ==============================================================
    # Save JSON
    # ==============================================================

    save_metrics(

        output_dir=(
            output_dir
        ),

        episodes=(
            episodes
        ),

        summary=(
            summary
        ),
    )

    # ==============================================================
    # Save plots
    # ==============================================================

    if not args.no_plots:

        plot_cumulative_rewards(

            episodes=(
                episodes
            ),

            output_dir=(
                output_dir
            ),

            show_plot=(
                args.show_plot
            ),
        )

        plot_episode_rewards(

            summary=(
                summary
            ),

            output_dir=(
                output_dir
            ),

            show_plot=(
                args.show_plot
            ),
        )

    print(
        "\nEvaluation finished successfully."
    )


# ======================================================================
# CLI
# ======================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(

        description=(
            "Evaluate trained sequential "
            "Bay/Row multi-agent PPO."
        )
    )

    # ==============================================================
    # Checkpoint
    # ==============================================================

    parser.add_argument(
        "--model_dir",
        type=str,
        default=(
            "./models/sequential_hppo"
        ),
        help=(
            "Directory containing saved Agent B, "
            "Agent R and config files."
        ),
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="final",
        help=(
            "Checkpoint prefix. For example: "
            "'final' or 'step_1000000'."
        ),
    )

    # ==============================================================
    # Evaluation
    # ==============================================================

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help=(
            "Number of episodes' environments to run simultaneously "
            "(batched Agent B/Agent R inference + DummyVecEnv "
            "stepping). Does not change results, only throughput - "
            "see evaluate_policy()."
        ),
    )

    add_evaluation_reward_argument(parser)

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help=(
            "Base evaluation seed. "
            "Episode i uses seed + i."
        ),
    )

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

    # Deterministic by default.
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Sample actions instead of using "
            "deterministic argmax inference."
        ),
    )

    # ==============================================================
    # Output
    # ==============================================================

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
    )

    parser.add_argument(
        "--verbose_steps",
        action="store_true",
        help=(
            "Print Bay, Row, reward and cumulative "
            "reward at every environment step."
        ),
    )

    parser.add_argument(
        "--show_plot",
        action="store_true",
        help=(
            "Display plots in addition to saving them."
        ),
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help=(
            "Skip matplotlib visualization."
        ),
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
