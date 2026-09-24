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
    HierarchicalEnv.step(...)
          |
          v
    ONE StackEnv.step(...)
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

from ..agents.agent_b import AgentB
from ..agents.agent_r import AgentR

from ..configs.hierarchical_config import (
    HierarchicalConfig,
)

from ..envs.hierarchical_envs.hierarchical_env import (
    HierarchicalEnv,
)
from ..envs.hierarchical_envs.vec_hierarchical_env import (
    VecHierarchicalEnv,
)

from .metrics import (
    EpisodeMetrics,
    EvaluationSummary,
    build_episode_metrics,
    summarize_episodes,
)


# ======================================================================
# Device
# ======================================================================


def get_device(
    requested_device: str,
) -> torch.device:
    """
    Resolve evaluation device.

    This follows the same behaviour as training:

        auto:
            CUDA if available,
            otherwise CPU.
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
# Evaluation reward configuration (also imported by the Notebook)
# ======================================================================


def make_evaluation_config(environment_config: Dict) -> Dict:
    """Return a copy configured for unnormalized, unclipped evaluation.

    Preserve the public helper used by the step-by-step Notebook viewer.
    The checkpoint's original training configuration is not modified.
    """
    evaluation_config = dict(environment_config)
    evaluation_config["reward_norm"] = False
    evaluation_config["reward_clip"] = False
    return evaluation_config


# ======================================================================
# Build evaluation environment + actors
# ======================================================================


def build_evaluation_system(
    *,
    environment_config: Dict,
    algorithm_config: HierarchicalConfig,
    agent_b_path: Path,
    agent_r_path: Path,
    device: torch.device,
) -> Tuple[
    HierarchicalEnv,
    AgentB,
    AgentR,
]:
    """
    Build the inference-only architecture.

    Notice that there is NO centralized critic and NO trainer here.

    Execution requires only:

        AgentB
        AgentR
        HierarchicalEnv
    """

    # ==============================================================
    # Environment
    # ==============================================================

    env = HierarchicalEnv(
        config=environment_config,
    )

    n_bays = int(
        env.bay_action_space.n
    )

    n_rows = int(
        env.row_action_space.n
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

    agent_b.load(
        agent_b_path
    )

    agent_r.load(
        agent_r_path
    )

    # ==============================================================
    # Evaluation mode
    # ==============================================================

    agent_b.eval()
    agent_r.eval()

    return (
        env,
        agent_b,
        agent_r,
    )


# ======================================================================
# Evaluate ONE episode
# ======================================================================


@torch.no_grad()
def evaluate_episode(
    *,
    env: HierarchicalEnv,
    agent_b: AgentB,
    agent_r: AgentR,
    episode_index: int,
    seed: int,
    deterministic: bool = True,
    verbose_steps: bool = False,
) -> EpisodeMetrics:
    """
    Run one complete evaluation episode.

    Decision sequence
    -----------------

        bay_observation
              |
              v
           Agent B
              |
              v
           bay_idx
              |
              | NO STEP
              v
        local row observation
              |
              v
           Agent R
              |
              v
           row_idx
              |
              v
        env.step(bay_idx, row_idx)
              |
              v
           reward

    No training or gradient computation occurs.
    """

    # ==============================================================
    # Reset environment
    # ==============================================================

    bay_observation, _reset_info = (
        env.reset(
            seed=seed
        )
    )

    bay_observation = np.asarray(
        bay_observation,
        dtype=np.float32,
    )

    # ==============================================================
    # Episode traces
    # ==============================================================

    step_rewards: List[
        float
    ] = []

    bay_actions: List[
        int
    ] = []

    row_actions: List[
        int
    ] = []

    selected_bays: List[
        int
    ] = []

    selected_rows: List[
        int
    ] = []

    global_actions: List[
        int
    ] = []

    terminated = False
    truncated = False

    final_info: Dict = {}

    step_index = 0
    cumulative_reward = 0.0

    # ==============================================================
    # Episode loop
    # ==============================================================

    while not (
        terminated
        or truncated
    ):

        # ==========================================================
        # Agent B mask
        # ==========================================================

        bay_action_mask = np.asarray(
            env.get_bay_action_mask(),
            dtype=np.bool_,
        )

        if not bay_action_mask.any():

            raise RuntimeError(
                "Evaluation reached a state with no valid Bay action "
                "before StackEnv reported episode termination or "
                "truncation."
            )

        # ==========================================================
        # Agent B selects BAY
        # ==========================================================

        bay_idx = agent_b.predict(

            observation=(
                bay_observation
            ),

            action_mask=(
                bay_action_mask
            ),

            deterministic=(
                deterministic
            ),
        )

        # ==========================================================
        # Agent R observation + mask
        # ==========================================================
        #
        # No physical environment transition happens here.
        # ==========================================================

        (
            row_observation,
            row_action_mask,
        ) = env.get_row_decision_input(
            bay_idx
        )

        row_observation = np.asarray(
            row_observation,
            dtype=np.float32,
        )

        row_action_mask = np.asarray(
            row_action_mask,
            dtype=np.bool_,
        )

        if not row_action_mask.any():

            raise RuntimeError(
                "Agent B selected a bay containing no valid Row "
                "action. Bay/Row masks are inconsistent."
            )

        # ==========================================================
        # Agent R selects ROW
        # ==========================================================

        row_idx = agent_r.predict(

            observation=(
                row_observation
            ),

            action_mask=(
                row_action_mask
            ),

            deterministic=(
                deterministic
            ),
        )

        # ==========================================================
        # ONE physical environment step
        # ==========================================================

        (
            next_bay_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            bay_idx,
            row_idx,
        )

        final_info = dict(
            info
        )

        # ==========================================================
        # Save decision trace
        # ==========================================================

        reward = float(
            reward
        )

        step_rewards.append(
            reward
        )

        bay_actions.append(
            int(
                bay_idx
            )
        )

        row_actions.append(
            int(
                row_idx
            )
        )

        selected_bays.append(
            int(
                info[
                    "selected_bay"
                ]
            )
        )

        selected_rows.append(
            int(
                info[
                    "selected_row"
                ]
            )
        )

        global_actions.append(
            int(
                info[
                    "global_action"
                ]
            )
        )

        # ==========================================================
        # Running cumulative reward
        # ==========================================================

        cumulative_reward += (
            reward
        )

        step_index += 1

        # ==========================================================
        # Optional step-by-step inference output
        # ==========================================================

        if verbose_steps:

            print(
                f"Episode "
                f"{episode_index + 1} | "
                f"Step "
                f"{step_index:03d} | "
                f"Bay "
                f"{info['selected_bay']} | "
                f"Row "
                f"{info['selected_row']} | "
                f"Reward "
                f"{reward:8.3f} | "
                f"Cumulative "
                f"{cumulative_reward:9.3f}"
            )

        # ==========================================================
        # Next Agent B observation
        # ==========================================================

        bay_observation = np.asarray(
            next_bay_observation,
            dtype=np.float32,
        )

    # ==============================================================
    # Episode metrics
    # ==============================================================

    episode_metrics = build_episode_metrics(

        episode_index=(
            episode_index
        ),

        step_rewards=(
            step_rewards
        ),

        bay_actions=(
            bay_actions
        ),

        row_actions=(
            row_actions
        ),

        selected_bays=(
            selected_bays
        ),

        selected_rows=(
            selected_rows
        ),

        global_actions=(
            global_actions
        ),

        terminated=(
            terminated
        ),

        truncated=(
            truncated
        ),

        final_info=(
            final_info
        ),
    )

    return episode_metrics


# ======================================================================
# Evaluate multiple episodes - batched across environments
# ======================================================================


def evaluate_policy_batched(
    *,
    environment_config: Dict,
    agent_b: AgentB,
    agent_r: AgentR,
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
    inference batched, StackEnv stepped through VecHierarchicalEnv)
    instead of one environment/episode at a time.

    Same seed mapping as evaluate_policy()/evaluate_episode() - episode
    i always uses seed = base_seed + i, regardless of eval_batch_size,
    round count, or partial final rounds - and the SAME per-episode
    action sequence, since each of the eval_batch_size environment
    slots owns its own independent HierarchicalEnv/StackEnv instance
    (own RNG), exactly like running eval_batch_size single-environment
    evaluations concurrently rather than one at a time.

    Episodes run in rounds of at most eval_batch_size. Within a round,
    every environment slot keeps stepping every iteration (Agent B/
    Agent R inference and StackEnv.step() are always called for the
    full batch), but an `active` mask stops this function from
    recording anything for a slot once its episode has terminated or
    truncated - VecHierarchicalEnv auto-resets that slot into a new
    (uninteresting) episode internally, which is harmless precisely
    because nothing reads that slot's output again after `active`
    flips False for it. On the last, possibly partial round (fewer
    than eval_batch_size episodes remaining), the unused slots are
    seeded but marked inactive from the start, so they contribute
    nothing and are simply wasted compute for that round.

    verbose controls only the "EVALUATION (batched)" header and the
    one-line-per-episode summary (matching evaluate_policy()'s
    always-on console output); it is set False by
    evaluate_for_training() so periodic in-training evaluation stays
    as quiet as it was before this function existed. verbose_steps
    (per-decision printing) is independent and still defaults to
    False either way.
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

    agent_b.eval()
    agent_r.eval()

    device = agent_b.device

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

    vec_env = VecHierarchicalEnv(
        config=environment_config,
        num_envs=eval_batch_size,
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

                # Every slot needs a seed (VecHierarchicalEnv.reset()
                # always resets all eval_batch_size envs), but only
                # the first k are real for this round - pad with a
                # valid, arbitrary seed for the rest, marked inactive
                # below so their results are never read.
                round_seeds = [
                    base_seed + i
                    for i in round_indices
                ] + [
                    base_seed + round_indices[0]
                ] * (eval_batch_size - k)

                bay_observation = vec_env.reset(
                    seeds=round_seeds
                )

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

                    bay_action_mask = np.asarray(
                        vec_env.get_bay_action_mask(),
                        dtype=np.bool_,
                    )

                    if not bay_action_mask[active].any(axis=-1).all():

                        raise RuntimeError(
                            "Evaluation reached a state with no valid "
                            "Bay action before StackEnv reported "
                            "episode termination or truncation."
                        )

                    bay_action_tensor, _ = agent_b.act(
                        observations=torch.as_tensor(
                            bay_observation,
                            dtype=torch.float32,
                            device=device,
                        ),
                        action_masks=torch.as_tensor(
                            bay_action_mask,
                            dtype=torch.bool,
                            device=device,
                        ),
                        deterministic=deterministic,
                    )

                    bay_idx_batch = (
                        bay_action_tensor
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )

                    (
                        row_observation,
                        row_action_mask,
                    ) = vec_env.get_row_decision_input(
                        bay_idx_batch
                    )

                    row_action_mask = np.asarray(
                        row_action_mask,
                        dtype=np.bool_,
                    )

                    if not row_action_mask[active].any(axis=-1).all():

                        raise RuntimeError(
                            "Agent B selected a bay containing no "
                            "valid Row action. Bay/Row masks are "
                            "inconsistent."
                        )

                    row_action_tensor, _ = agent_r.act(
                        observations=torch.as_tensor(
                            row_observation,
                            dtype=torch.float32,
                            device=device,
                        ),
                        action_masks=torch.as_tensor(
                            row_action_mask,
                            dtype=torch.bool,
                            device=device,
                        ),
                        deterministic=deterministic,
                    )

                    row_idx_batch = (
                        row_action_tensor
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )

                    (
                        next_bay_observation,
                        rewards,
                        terminated,
                        truncated,
                        infos,
                    ) = vec_env.step(
                        bay_idx_batch,
                        row_idx_batch,
                    )

                    for i in range(eval_batch_size):

                        if not active[i]:
                            continue

                        step_rewards[i].append(float(rewards[i]))
                        bay_actions_acc[i].append(int(bay_idx_batch[i]))
                        row_actions_acc[i].append(int(row_idx_batch[i]))
                        selected_bays_acc[i].append(
                            int(infos[i]["selected_bay"])
                        )
                        selected_rows_acc[i].append(
                            int(infos[i]["selected_row"])
                        )
                        global_actions_acc[i].append(
                            int(infos[i]["global_action"])
                        )

                        if verbose_steps:

                            print(
                                f"Episode "
                                f"{round_indices[i] + 1:03d} | "
                                f"Step "
                                f"{len(step_rewards[i]):03d} | "
                                f"Bay "
                                f"{infos[i]['selected_bay']} | "
                                f"Row "
                                f"{infos[i]['selected_row']} | "
                                f"Reward "
                                f"{rewards[i]:8.3f} | "
                                f"Cumulative "
                                f"{sum(step_rewards[i]):9.3f}"
                            )

                        if terminated[i] or truncated[i]:
                            final_terminated[i] = bool(terminated[i])
                            final_truncated[i] = bool(truncated[i])
                            final_info[i] = dict(infos[i])
                            active[i] = False

                    bay_observation = next_bay_observation

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
# Evaluate multiple episodes
# ======================================================================


def evaluate_policy(
    *,
    environment_config: Dict,
    agent_b: AgentB,
    agent_r: AgentR,
    n_episodes: int = 10,
    base_seed: int = 42,
    eval_batch_size: int = 8,
    deterministic: bool = True,
    verbose_steps: bool = False,
) -> Tuple[
    List[EpisodeMetrics],
    EvaluationSummary,
]:
    """
    Evaluate the two-agent policy over multiple episodes.

    A different deterministic seed is used for each episode:

        episode 0 -> base_seed
        episode 1 -> base_seed + 1
        episode 2 -> base_seed + 2
        ...

    This avoids evaluating the deterministic policy repeatedly on
    exactly the same environment instance.

    Delegates to evaluate_policy_batched(), which produces identical
    per-episode results (see its docstring) while running up to
    eval_batch_size episodes' environments simultaneously instead of
    one at a time. evaluate_episode()/the single-environment path
    remains available separately and is what evaluate_policy_batched()
    is verified against (see stack/tests/test_evaluate_batched.py).
    """

    return evaluate_policy_batched(
        environment_config=environment_config,
        agent_b=agent_b,
        agent_r=agent_r,
        n_episodes=n_episodes,
        base_seed=base_seed,
        eval_batch_size=eval_batch_size,
        deterministic=deterministic,
        verbose_steps=verbose_steps,
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

    # Match training_plots' Aritra-style raw evaluation reward when requested.
    if getattr(args, "raw_rewards", False):
        environment_config = make_evaluation_config(environment_config)

    # ==============================================================
    # Device
    # ==============================================================

    device = get_device(
        args.device
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
        env,
        agent_b,
        agent_r,
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

    try:

        # ==========================================================
        # Evaluation
        # ==========================================================

        (
            episodes,
            summary,
        ) = evaluate_policy(

            environment_config=environment_config,

            agent_b=agent_b,

            agent_r=agent_r,

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

        # ==========================================================
        # Print metrics
        # ==========================================================

        print_summary(
            summary
        )

        # ==========================================================
        # Save JSON
        # ==========================================================

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

        # ==========================================================
        # Save plots
        # ==========================================================

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

    finally:

        env.close()

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
            "(batched Agent B/Agent R inference + VecHierarchicalEnv "
            "stepping). Does not change results, only throughput - "
            "see evaluate_policy_batched()."
        ),
    )

    parser.add_argument(
        "--raw_rewards",
        action="store_true",
        help="Disable reward normalization and clipping, matching the default training evaluation curve.",
    )

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
