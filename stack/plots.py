import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from envs.stack_gym import StackEnv
from agents.hierarchical_rule_based_agent import HierarchicalAgent

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from utils import mask_fn
from models.joint_hierarchical_policy import MaskableJointTransformerPolicy


# ============================================================
# HELPERS
# ============================================================

def _evaluate_rule_based(config_dict, policy_type, num_seeds, max_steps):
    all_rewards = []
    for seed in range(num_seeds):
        env = StackEnv(config=config_dict, render_mode=None)
        agent = HierarchicalAgent(
            vessel_shape=config_dict["vessel_shape"],
            yard_shape=config_dict["yard_shape"],
            num_slot_attrs=5,
            high_level_policy_type=policy_type,
            low_level_policy_type=policy_type,
        )
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        episode_terminated = False
        for _ in range(max_steps):
            valid_actions = env._get_valid_yard_actions()
            if len(valid_actions) == 0:
                break
            try:
                action, _ = agent.get_action(observation, valid_actions)
                if action is None:
                    break
            except Exception:
                break
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                episode_terminated = True
                break
        if episode_terminated:
            all_rewards.append(total_reward)
        env.close()
        if (seed + 1) % 20 == 0:
            print(f"  Completed {seed + 1}/{num_seeds} seeds...")
    return all_rewards


def _evaluate_rl_model(config_dict, model, num_seeds, max_steps):
    all_rewards = []
    for seed in range(num_seeds):
        eval_env = StackEnv(config=config_dict, render_mode=None)
        eval_env = ActionMasker(eval_env, mask_fn)
        obs, info = eval_env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0
        while not (terminated or truncated) and step_count < max_steps:
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=get_action_masks(eval_env),
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            step_count += 1
        all_rewards.append(total_reward)
        eval_env.close()
        if (seed + 1) % 20 == 0:
            print(f"  Completed {seed + 1}/{num_seeds} seeds...")
    return all_rewards


def _compute_stats(name, rewards, num_seeds, threshold):
    rewards = np.array(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    min_r = np.min(rewards)
    max_r = np.max(rewards)
    percent_above = np.sum(rewards > threshold) / num_seeds * 100

    lines = [
        f"Statistics for {name}:",
        f"  Seed with highest reward: {np.argmax(rewards)} with reward {max_r:.2f}",
        f"  Seed with lowest reward:  {np.argmin(rewards)} with reward {min_r:.2f}",
        "  " + "=" * 50,
        f"  Mean Reward: {mean_r:.4f}",
        f"  Std Reward:  {std_r:.4f}",
        f"  Min Reward:  {min_r:.4f}",
        f"  Max Reward:  {max_r:.4f}",
        f"  % seeds > {threshold}: {percent_above:.2f}%",
        "  " + "=" * 50,
    ]
    return "\n".join(lines)


# ============================================================
# MAIN EVALUATE FUNCTION — 9 models
#   3 rule-based (random, rule_based, rule_based_grouped)
#   3 flat RL    (seed1, seed2, seed3)
#   3 HRL        (seed1, seed2, seed3)
# ============================================================

def evaluate(config_dict, flat_model_dir, hrl_model_dir,
             num_seeds=100, max_steps=500, threshold=90):
    """
    Parameters
    ----------
    config_dict : dict
        Environment config (observation_type should be "flat_parsed" for
        rule-based agents; RL models will get "stack_features_v3" automatically).
    flat_model_dir : str
        Base path containing seed1/, seed2/, seed3/ for flat RL models,
        e.g. "stack/models/flat/pointer/small".
    hrl_model_dir : str
        Base path containing seed1/, seed2/, seed3/ for hierarchical RL models,
        e.g. "stack/models/hierarchical/pointer/small".
    num_seeds : int
        Number of environment seeds to evaluate per model.
    max_steps : int
        Max steps per episode.
    threshold : float
        Reward threshold for computing "% seeds > threshold".
    """
    # Derive size label from model dir (e.g. "small", "medium", "large1")
    size_label = os.path.basename(os.path.normpath(flat_model_dir))
    results_dir = os.path.join("results", size_label)
    os.makedirs(results_dir, exist_ok=True)
    all_stats = []
    model_num = 0

    print(f"\n{'#' * 70}")
    print(f"STARTING EVALUATION — 9 models x {num_seeds} seeds each")
    print(f"{'#' * 70}")

    # ── 1. Rule-based agents (random, rule_based, rule_based_grouped) ──
    rule_based_agents = [
        ("random", "random"),
        ("rule_based", "rule_based"),
        ("rule_based_grouped", "rule_based_grouped"),
    ]

    for name, policy_type in rule_based_agents:
        model_num += 1
        npy_path = os.path.join(results_dir, f"{name}_rewards.npy")
        if os.path.exists(npy_path):
            print(f"\n[{model_num}/9] LOADING {name.upper()} from existing {npy_path}")
            rewards = np.load(npy_path)
        else:
            print(f"\n{'=' * 70}")
            print(f"[{model_num}/9] EVALUATING {name.upper()} AGENT ACROSS {num_seeds} SEEDS")
            print(f"{'=' * 70}")
            rewards = _evaluate_rule_based(config_dict, policy_type, num_seeds, max_steps)
            np.save(npy_path, np.array(rewards))
            print(f"  ✓ Saved {npy_path}")
        stats = _compute_stats(name, rewards, num_seeds, threshold)
        print(stats)
        all_stats.append(stats)

    # ── RL config: same env settings, observation_type for RL models ──
    rl_config = {k: v for k, v in config_dict.items()}
    rl_config["observation_type"] = "stack_features_v3"

    # ── 2. Flat RL models (seed1, seed2, seed3) ──
    flat_dir_name = os.path.basename(os.path.normpath(flat_model_dir))
    for seed_idx in range(1, 4):
        seed_dir = f"seed{seed_idx}"
        run_name = f"flat_pointer_{flat_dir_name}_{seed_dir}"
        model_path = os.path.join(flat_model_dir, seed_dir, run_name, "best_model")
        name = f"flat_{seed_dir}"
        model_num += 1
        npy_path = os.path.join(results_dir, f"{name}_rewards.npy")
        if os.path.exists(npy_path):
            print(f"\n[{model_num}/9] LOADING FLAT RL ({seed_dir.upper()}) from existing {npy_path}")
            rewards = np.load(npy_path)
        else:
            if not os.path.exists(model_path + ".zip"):
                print(f"\n[{model_num}/9] SKIPPING FLAT RL ({seed_dir.upper()}) — model not found: {model_path}")
                continue
            print(f"\n{'=' * 70}")
            print(f"[{model_num}/9] EVALUATING FLAT RL ({seed_dir.upper()}) ACROSS {num_seeds} SEEDS")
            print(f"  Loading model: {model_path}")
            print(f"{'=' * 70}")
            model = MaskablePPO.load(model_path)
            print(f"  Model loaded successfully.")
            rewards = _evaluate_rl_model(rl_config, model, num_seeds, max_steps)
            np.save(npy_path, np.array(rewards))
            print(f"  ✓ Saved {npy_path}")
        stats = _compute_stats(name, rewards, num_seeds, threshold)
        print(stats)
        all_stats.append(stats)

    # ── 3. Hierarchical RL models (seed1, seed2, seed3) ──
    hrl_dir_name = os.path.basename(os.path.normpath(hrl_model_dir))
    for seed_idx in range(1, 4):
        seed_dir = f"seed{seed_idx}"
        run_name = f"hrl_pointer_{hrl_dir_name}_{seed_dir}"
        model_path = os.path.join(hrl_model_dir, seed_dir, run_name, "best_model")
        name = f"hrl_{seed_dir}"
        model_num += 1
        npy_path = os.path.join(results_dir, f"{name}_rewards.npy")
        if os.path.exists(npy_path):
            print(f"\n[{model_num}/9] LOADING HRL ({seed_dir.upper()}) from existing {npy_path}")
            rewards = np.load(npy_path)
        else:
            if not os.path.exists(model_path + ".zip"):
                print(f"\n[{model_num}/9] SKIPPING HRL ({seed_dir.upper()}) — model not found: {model_path}")
                continue
            print(f"\n{'=' * 70}")
            print(f"[{model_num}/9] EVALUATING HRL ({seed_dir.upper()}) ACROSS {num_seeds} SEEDS")
            print(f"  Loading model: {model_path}")
            print(f"{'=' * 70}")
            model = MaskablePPO.load(
                model_path,
                custom_objects={"policy_class": MaskableJointTransformerPolicy},
            )
            print(f"  Model loaded successfully.")
            rewards = _evaluate_rl_model(rl_config, model, num_seeds, max_steps)
            np.save(npy_path, np.array(rewards))
            print(f"  ✓ Saved {npy_path}")
        stats = _compute_stats(name, rewards, num_seeds, threshold)
        print(stats)
        all_stats.append(stats)

    # ── Save all stats to a single log file ──
    with open(os.path.join(results_dir, "evaluation_stats.txt"), "w") as f:
        f.write("\n\n".join(all_stats))
    print(f"\n{'#' * 70}")
    print(f"EVALUATION COMPLETE — 9/9 models done")
    print(f"All rewards (.npy) and stats saved to {results_dir}/")
    print(f"{'#' * 70}")


# ============================================================
# KDE PLOT — picks best seed for each RL model type
# ============================================================

def plot_kde(num_seeds, title_env_label="Environment", results_dir="results"):
    """Load saved .npy rewards, pick best-mean seed for flat/hrl, and save KDE plot."""

    print(f"\n{'=' * 70}")
    print(f"GENERATING KDE PLOT (from {results_dir}/)")
    print(f"{'=' * 70}")

    # ── Load rule-based rewards ──
    print("  Loading rule_based_grouped rewards...")
    rule_based_grouped = np.load(os.path.join(results_dir, "rule_based_grouped_rewards.npy"))

    # ── Pick best flat RL seed by mean reward ──
    print("  Selecting best Flat RL seed...")
    best_flat_mean, best_flat_rewards, best_flat_idx = -np.inf, None, None
    for seed_idx in range(1, 4):
        path = os.path.join(results_dir, f"flat_seed{seed_idx}_rewards.npy")
        if not os.path.exists(path):
            continue
        r = np.load(path)
        m = np.mean(r)
        print(f"    seed{seed_idx}: mean={m:.4f}")
        if m > best_flat_mean:
            best_flat_mean, best_flat_rewards, best_flat_idx = m, r, seed_idx
    if best_flat_idx is not None:
        print(f"    → Best: seed{best_flat_idx} (mean={best_flat_mean:.4f})")

    # ── Pick best HRL seed by mean reward ──
    print("  Selecting best HRL seed...")
    best_hrl_mean, best_hrl_rewards, best_hrl_idx = -np.inf, None, None
    for seed_idx in range(1, 4):
        path = os.path.join(results_dir, f"hrl_seed{seed_idx}_rewards.npy")
        if not os.path.exists(path):
            continue
        r = np.load(path)
        m = np.mean(r)
        print(f"    seed{seed_idx}: mean={m:.4f}")
        if m > best_hrl_mean:
            best_hrl_mean, best_hrl_rewards, best_hrl_idx = m, r, seed_idx
    if best_hrl_idx is not None:
        print(f"    → Best: seed{best_hrl_idx} (mean={best_hrl_mean:.4f})")

    # ── Build agents dict ──
    print("  Building KDE plot...")
    agents = {
        "Heuristic Bay Grouping Agent": rule_based_grouped,
    }
    if best_flat_rewards is not None:
        agents["Flat RL"] = best_flat_rewards
    if best_hrl_rewards is not None:
        agents["Hierarchical RL"] = best_hrl_rewards

    colors = {
        "Heuristic Bay Grouping Agent": "#4C78A8",
        "Flat RL": "#59A14F",
        "Hierarchical RL": "#F28E2B",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 18,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # shared x-axis range
    all_rewards_combined = np.concatenate(list(agents.values()))
    x_min, x_max = all_rewards_combined.min(), all_rewards_combined.max()
    x_pad = 0.05 * (x_max - x_min)
    xs = np.linspace(x_min - x_pad, x_max + x_pad, 400)

    # First pass: draw KDE curves and median lines, collect medians
    medians = []
    for name, rewards in agents.items():
        kde = gaussian_kde(rewards)
        ys = kde(xs)
        median = np.median(rewards)
        medians.append((median, name))

        ax.fill_between(xs, ys, alpha=0.18, color=colors[name])
        ax.plot(xs, ys, linewidth=2.2, color=colors[name], label=name)

        ax.axvline(
            median,
            color=colors[name],
            linestyle="--",
            linewidth=1.7,
            alpha=0.95,
            zorder=5,
        )

    ax.set_xlabel("Final episode reward")
    ax.set_ylabel("Density")
    ax.set_title(title_env_label, fontsize=18, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    env_slug = title_env_label.split("(")[0].strip().lower().replace(" ", "_").replace("/", "-")
    base_name = f"reward_distribution_kde_{env_slug}"
    fig.savefig(os.path.join(results_dir, f"{base_name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, f"{base_name}.png"), dpi=300, bbox_inches="tight")
    print(f"  KDE plot saved to {results_dir}/{base_name}.{{pdf,png}}")
    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    config_dict = {
        "vessel_shape": (10, 8, 5),
        "yard_shape": (10, 9, 5),
        "num_containers": 400,
        "group_num": 10,
        "group_placement": "random",
        "observation_type": "flat_parsed",
        "reward_norm": False,
        "reward_clip": False,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
        "random_group_sizes": True,
    }

    FLAT_MODEL_DIR = "stack/models/flat/pointer/large4"
    HRL_MODEL_DIR = "stack/models/hierarchical/pointer/large4"
    NUM_SEEDS = 100
    MAX_STEPS = 500
    THRESHOLD = 3800
    env_label = "Massive (10x8x5)"

    evaluate(
        config_dict,
        flat_model_dir=FLAT_MODEL_DIR,
        hrl_model_dir=HRL_MODEL_DIR,
        num_seeds=NUM_SEEDS,
        max_steps=MAX_STEPS,
        threshold=THRESHOLD,
    )

    plot_kde(num_seeds=NUM_SEEDS, title_env_label=env_label,
            results_dir="results/large4")