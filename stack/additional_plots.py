import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from envs.stack_gym import StackEnv, StateIds
from agents.hierarchical_rule_based_agent import HierarchicalAgent

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from utils import mask_fn
from models.joint_hierarchical_policy import MaskableJointTransformerPolicy
from models.sequential_hierarchical_policy import MaskableSequentialTransformerPolicy


# ============================================================
# PLOT STYLE
# ============================================================

COLORS = {
    "Heuristic Bay Grouping Agent": "#4C78A8",
    "Flat RL": "#59A14F",
    "Hierarchical RL": "#F28E2B",
    "Sequential HPPO": "#E15759",
}

def _apply_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================
# METRIC HELPERS
# ============================================================

def _compute_bay_group_dominance(yard_state, yard_shape):
    """
    Per-bay max-group proportion: max_group_count / total_occupied for each bay.
    Returns np.ndarray of shape (num_physical_bays,).
    0.0 if a bay is empty; 1.0 if all occupied containers share one group.
    """
    bay_col = yard_state[:, StateIds.BAY.value]
    occ_col = yard_state[:, StateIds.IS_OCCUPIED.value]
    grp_col = yard_state[:, StateIds.GROUP.value]

    num_physical_bays = yard_shape[0]
    odd_bays = [2 * i + 1 for i in range(num_physical_bays)]

    result = np.zeros(num_physical_bays, dtype=np.float64)
    for i, b in enumerate(odd_bays):
        mask = (bay_col == b) & (occ_col == 1)
        total = int(np.sum(mask))
        if total == 0:
            result[i] = 0.0
        else:
            _, counts = np.unique(grp_col[mask], return_counts=True)
            result[i] = counts.max() / total
    return result


def _compute_partial_stacks(yard_state, yard_shape):
    """Count stacks with 0 < occupied < max_tiers."""
    bay_col = yard_state[:, StateIds.BAY.value]
    row_col = yard_state[:, StateIds.ROW.value]
    occ_col = yard_state[:, StateIds.IS_OCCUPIED.value]
    max_tiers = yard_shape[2]

    num_physical_bays = yard_shape[0]
    num_rows = yard_shape[1]
    odd_bays = [2 * i + 1 for i in range(num_physical_bays)]

    partial = 0
    for b in odd_bays:
        for r in range(1, num_rows + 1):
            mask = (bay_col == b) & (row_col == r)
            occupied = int(np.sum(occ_col[mask]))
            if 0 < occupied < max_tiers:
                partial += 1
    return partial


def _compute_perturbation_steps(max_steps, num_perturbations=4):
    """Return sorted list of step indices at fixed intervals."""
    return [max_steps * (i + 1) // (num_perturbations + 1)
            for i in range(num_perturbations)]


# ============================================================
# EVALUATION LOOPS
# ============================================================

def _run_episode_robustness(env, base_env, get_action_fn, seed, max_steps,
                            perturbation_steps, perturb):
    """
    Run one episode. If perturb=True, inject random valid actions at
    perturbation_steps. Returns total_reward.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    perturb_rng = np.random.RandomState(seed)
    perturb_set = set(perturbation_steps) if perturb else set()

    for step in range(max_steps):
        valid_actions = base_env._get_valid_yard_actions()
        if len(valid_actions) == 0:
            break

        if step in perturb_set:
            action = int(perturb_rng.choice(valid_actions))
        else:
            action = get_action_fn(obs, env)
            if action is None:
                break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


def _run_episode_metrics(env, base_env, get_action_fn, seed, max_steps,
                         yard_shape):
    """
    Run one clean episode, recording per-step per-bay group dominance and
    partial stacks.
    Returns (total_reward, bay_dom_list, partial_stacks_list).
    bay_dom_list: list of np.ndarray of shape (num_physical_bays,), one per step.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    bay_dom_list = []
    partial_stacks_list = []

    for step in range(max_steps):
        valid_actions = base_env._get_valid_yard_actions()
        if len(valid_actions) == 0:
            break

        action = get_action_fn(obs, env)
        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        bay_dom_list.append(_compute_bay_group_dominance(base_env.yard_state, yard_shape))
        partial_stacks_list.append(_compute_partial_stacks(base_env.yard_state, yard_shape))

        if terminated or truncated:
            break

    return total_reward, bay_dom_list, partial_stacks_list


# ============================================================
# ACTION FUNCTION FACTORIES
# ============================================================

def _make_rule_based_action_fn(agent):
    """Return get_action_fn(obs, env) for a rule-based agent (unwrapped env)."""
    def fn(obs, env):
        valid_actions = env._get_valid_yard_actions()
        if len(valid_actions) == 0:
            return None
        try:
            action, _ = agent.get_action(obs, valid_actions)
            return action
        except Exception:
            return None
    return fn


def _make_rl_action_fn(model):
    """Return get_action_fn(obs, env) for an RL model (ActionMasker-wrapped env)."""
    def fn(obs, env):
        action, _ = model.predict(
            obs, deterministic=True,
            action_masks=get_action_masks(env),
        )
        return int(action)
    return fn


# ============================================================
# PER-MODEL EVALUATION DRIVER
# ============================================================

def _evaluate_model(name, env_factory, get_action_fn, base_env_getter,
                    num_seeds, max_steps, yard_shape, mode, results_dir,
                    perturbation_steps):
    """
    Evaluate a single model across seeds. Skips sub-modes whose output
    files already exist.
    env_factory() -> (env, base_env)
    """
    # Determine which sub-modes still need running
    run_robust = mode in ("robust", "both")
    run_metrics = mode in ("metrics", "both")

    if run_robust:
        if (os.path.exists(os.path.join(results_dir, f"{name}_normal_rewards.npy")) and
                os.path.exists(os.path.join(results_dir, f"{name}_perturbed_rewards.npy"))):
            print(f"  [SKIP] {name} robust — files already exist.")
            run_robust = False

    if run_metrics:
        if (os.path.exists(os.path.join(results_dir, f"{name}_bay_dominance.npy")) and
                os.path.exists(os.path.join(results_dir, f"{name}_partial_stacks.npy"))):
            print(f"  [SKIP] {name} metrics — files already exist.")
            run_metrics = False

    if not run_robust and not run_metrics:
        return

    normal_rewards = []
    perturbed_rewards = []
    all_entropy = []
    all_partial = []

    for seed in range(num_seeds):
        if run_robust:
            env, base_env = env_factory()
            r_normal = _run_episode_robustness(
                env, base_env, get_action_fn, seed, max_steps,
                perturbation_steps, perturb=False)
            normal_rewards.append(r_normal)
            env.close()

            env, base_env = env_factory()
            r_pert = _run_episode_robustness(
                env, base_env, get_action_fn, seed, max_steps,
                perturbation_steps, perturb=True)
            perturbed_rewards.append(r_pert)
            env.close()

        if run_metrics:
            env, base_env = env_factory()
            _, bay_doms, partials = _run_episode_metrics(
                env, base_env, get_action_fn, seed, max_steps, yard_shape)
            # bay_doms: list of (num_bays,) arrays; pad to (num_bays, max_steps)
            num_bays = yard_shape[0]
            dom_padded = np.full((num_bays, max_steps), np.nan)
            for t, vec in enumerate(bay_doms):
                dom_padded[:, t] = vec
            par_padded = np.full(max_steps, np.nan)
            par_padded[:len(partials)] = partials
            all_entropy.append(dom_padded)
            all_partial.append(par_padded)
            env.close()

        if (seed + 1) % 20 == 0:
            print(f"  Completed {seed + 1}/{num_seeds} seeds...")

    # Save arrays
    if run_robust:
        np.save(os.path.join(results_dir, f"{name}_normal_rewards.npy"),
                np.array(normal_rewards))
        np.save(os.path.join(results_dir, f"{name}_perturbed_rewards.npy"),
                np.array(perturbed_rewards))
        print(f"  Saved {name}_normal_rewards.npy, {name}_perturbed_rewards.npy")

    if run_metrics:
        np.save(os.path.join(results_dir, f"{name}_bay_dominance.npy"),
                np.array(all_entropy))  # shape: (num_seeds, num_bays, max_steps)
        np.save(os.path.join(results_dir, f"{name}_partial_stacks.npy"),
                np.array(all_partial))
        print(f"  Saved {name}_bay_dominance.npy, {name}_partial_stacks.npy")


# ============================================================
# MAIN EVALUATE FUNCTION
# ============================================================

def evaluate_all(config_dict, flat_model_dir, hrl_model_dir, sequential_model_dir,
                 num_seeds=100, max_steps=500, mode="both",
                 flat_seed=1, hrl_seed=1, sequential_seed=1):
    """
    Evaluate 6 models (3 rule-based + 1 flat RL + 1 HRL + 1 Sequential HPPO).

    Parameters
    ----------
    flat_seed : int
        Seed index (1–3) to use for the flat RL model.
    hrl_seed : int
        Seed index (1–3) to use for the hierarchical RL model.
    sequential_seed : int
        Seed index (1–3) to use for the Sequential HPPO model.
    mode : str
        "robust"  — robustness analysis only (normal + perturbed rewards)
        "metrics" — bay uniformity + stacking strategy only
        "both"    — all analyses
    """
    size_label = os.path.basename(os.path.normpath(flat_model_dir))
    results_dir = os.path.join("results_extra", size_label)
    os.makedirs(results_dir, exist_ok=True)
    yard_shape = config_dict["yard_shape"]
    perturbation_steps = _compute_perturbation_steps(max_steps)

    print(f"\n{'#' * 70}")
    print(f"ADDITIONAL ANALYSIS — mode={mode}, 6 models x {num_seeds} seeds")
    print(f"Flat RL seed: {flat_seed} | HRL seed: {hrl_seed} | Sequential HPPO seed: {sequential_seed}")
    print(f"Perturbation steps: {perturbation_steps}")
    print(f"Results dir: {results_dir}/")
    print(f"{'#' * 70}")

    model_num = 0

    # ── 1. Rule-based agents ──
    rule_based_agents = [
        ("random", "random"),
        ("rule_based", "rule_based"),
        ("rule_based_grouped", "rule_based_grouped"),
    ]

    for name, policy_type in rule_based_agents:
        model_num += 1
        print(f"\n{'=' * 70}")
        print(f"[{model_num}/6] {name.upper()} — {mode}")
        print(f"{'=' * 70}")

        agent = HierarchicalAgent(
            vessel_shape=config_dict["vessel_shape"],
            yard_shape=config_dict["yard_shape"],
            num_slot_attrs=5,
            high_level_policy_type=policy_type,
            low_level_policy_type=policy_type,
        )
        action_fn = _make_rule_based_action_fn(agent)

        def env_factory(_cfg=config_dict):
            env = StackEnv(config=_cfg, render_mode=None)
            return env, env  # base_env is the env itself

        _evaluate_model(name, env_factory, action_fn, None,
                        num_seeds, max_steps, yard_shape, mode,
                        results_dir, perturbation_steps)

    # ── RL config ──
    rl_config = {k: v for k, v in config_dict.items()}
    rl_config["observation_type"] = "stack_features_v3"

    # ── 2. Flat RL model ──
    flat_dir_name = os.path.basename(os.path.normpath(flat_model_dir))
    for seed_idx in [flat_seed]:
        seed_dir = f"seed{seed_idx}"
        run_name = f"flat_pointer_{flat_dir_name}_{seed_dir}"
        model_path = os.path.join(flat_model_dir, seed_dir, run_name, "best_model")
        name = f"flat_{seed_dir}"
        model_num += 1

        if not os.path.exists(model_path + ".zip"):
            print(f"\n[{model_num}/6] SKIPPING FLAT RL ({seed_dir.upper()}) — not found: {model_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{model_num}/6] FLAT RL ({seed_dir.upper()}) — {mode}")
        print(f"  Loading: {model_path}")
        print(f"{'=' * 70}")
        model = MaskablePPO.load(model_path)
        action_fn = _make_rl_action_fn(model)

        def env_factory(_cfg=rl_config):
            base = StackEnv(config=_cfg, render_mode=None)
            wrapped = ActionMasker(base, mask_fn)
            return wrapped, base

        _evaluate_model(name, env_factory, action_fn, None,
                        num_seeds, max_steps, yard_shape, mode,
                        results_dir, perturbation_steps)

    # ── 3. Hierarchical RL model ──
    hrl_dir_name = os.path.basename(os.path.normpath(hrl_model_dir))
    for seed_idx in [hrl_seed]:
        seed_dir = f"seed{seed_idx}"
        run_name = f"hrl_pointer_{hrl_dir_name}_{seed_dir}"
        model_path = os.path.join(hrl_model_dir, seed_dir, run_name, "best_model")
        name = f"hrl_{seed_dir}"
        model_num += 1

        if not os.path.exists(model_path + ".zip"):
            print(f"\n[{model_num}/6] SKIPPING HRL ({seed_dir.upper()}) — not found: {model_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{model_num}/6] HRL ({seed_dir.upper()}) — {mode}")
        print(f"  Loading: {model_path}")
        print(f"{'=' * 70}")
        model = MaskablePPO.load(
            model_path,
            custom_objects={"policy_class": MaskableJointTransformerPolicy},
        )
        action_fn = _make_rl_action_fn(model)

        def env_factory(_cfg=rl_config):
            base = StackEnv(config=_cfg, render_mode=None)
            wrapped = ActionMasker(base, mask_fn)
            return wrapped, base

        _evaluate_model(name, env_factory, action_fn, None,
                        num_seeds, max_steps, yard_shape, mode,
                        results_dir, perturbation_steps)

    # ── 4. Sequential HPPO model ──
    seq_dir_name = os.path.basename(os.path.normpath(sequential_model_dir))
    for seed_idx in [sequential_seed]:
        seed_dir = f"seed{seed_idx}"
        run_name = f"sequential_hppo_pointer_{seq_dir_name}_{seed_dir}"
        model_path = os.path.join(sequential_model_dir, seed_dir, run_name, "best_model")
        name = f"sequential_hppo_{seed_dir}"
        model_num += 1

        if not os.path.exists(model_path + ".zip"):
            print(f"\n[{model_num}/6] SKIPPING SEQUENTIAL HPPO ({seed_dir.upper()}) — not found: {model_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{model_num}/6] SEQUENTIAL HPPO ({seed_dir.upper()}) — {mode}")
        print(f"  Loading: {model_path}")
        print(f"{'=' * 70}")
        model = MaskablePPO.load(
            model_path,
            custom_objects={"policy_class": MaskableSequentialTransformerPolicy},
        )
        action_fn = _make_rl_action_fn(model)

        def env_factory(_cfg=rl_config):
            base = StackEnv(config=_cfg, render_mode=None)
            wrapped = ActionMasker(base, mask_fn)
            return wrapped, base

        _evaluate_model(name, env_factory, action_fn, None,
                        num_seeds, max_steps, yard_shape, mode,
                        results_dir, perturbation_steps)

    print(f"\n{'#' * 70}")
    print(f"EVALUATION COMPLETE — results saved to {results_dir}/")
    print(f"{'#' * 70}")
    return results_dir


# ============================================================
# HELPER: pick best RL seed by mean normal reward (robust) or
#         mean total reward from entropy arrays (metrics)
# ============================================================

def _pick_best_seeds(results_dir, mode):
    """Return (best_flat_name, best_hrl_name) or None if not found."""
    best = {}
    for prefix, label in [("flat", "Flat RL"), ("hrl", "Hierarchical RL")]:
        best_mean, best_name = -np.inf, None
        for seed_idx in range(1, 4):
            name = f"{prefix}_seed{seed_idx}"
            if mode in ("robust", "both"):
                path = os.path.join(results_dir, f"{name}_normal_rewards.npy")
            else:
                path = os.path.join(results_dir, f"{name}_bay_dominance.npy")
            if not os.path.exists(path):
                continue
            if mode in ("robust", "both"):
                m = np.mean(np.load(path))
            else:
                # In metrics-only mode, try normal_rewards first; fall back
                # to avg episode length from bay_dominance (use bay 0 as proxy)
                nr_path = os.path.join(results_dir, f"{name}_normal_rewards.npy")
                if os.path.exists(nr_path):
                    m = np.mean(np.load(nr_path))
                else:
                    data = np.load(path)  # bay_dominance (num_seeds, num_bays, max_steps)
                    m = np.nanmean(np.nansum(~np.isnan(data[:, 0, :]), axis=1))  # avg ep len
            print(f"    {name}: score={m:.4f}")
            if m > best_mean:
                best_mean, best_name = m, name
        if best_name is not None:
            print(f"    -> Best {prefix}: {best_name} (mean={best_mean:.4f})")
        best[label] = best_name
    return best


# ============================================================
# PLOT 1 & 2: ROBUSTNESS — KDE + BAR
# ============================================================

def plot_robustness_kde(results_dir, num_seeds, title_env_label="Environment",
                        flat_seed=1, hrl_seed=1, sequential_seed=1):
    """KDE overlay: normal (solid) vs perturbed (dashed) per model."""
    _apply_plot_style()
    print(f"\n{'=' * 70}")
    print(f"GENERATING ROBUSTNESS KDE PLOT")
    print(f"{'=' * 70}")

    # Heuristic Bay Grouping Agent vs Joint Hierarchical RL vs Sequential HPPO
    agents = {}
    nr = os.path.join(results_dir, "rule_based_grouped_normal_rewards.npy")
    pr = os.path.join(results_dir, "rule_based_grouped_perturbed_rewards.npy")
    if os.path.exists(nr) and os.path.exists(pr):
        agents["Heuristic Bay Grouping Agent"] = (np.load(nr), np.load(pr))

    hrl_name = f"hrl_seed{hrl_seed}"
    nr = os.path.join(results_dir, f"{hrl_name}_normal_rewards.npy")
    pr = os.path.join(results_dir, f"{hrl_name}_perturbed_rewards.npy")
    if os.path.exists(nr) and os.path.exists(pr):
        agents["Hierarchical RL"] = (np.load(nr), np.load(pr))

    seq_name = f"sequential_hppo_seed{sequential_seed}"
    nr = os.path.join(results_dir, f"{seq_name}_normal_rewards.npy")
    pr = os.path.join(results_dir, f"{seq_name}_perturbed_rewards.npy")
    if os.path.exists(nr) and os.path.exists(pr):
        agents["Sequential HPPO"] = (np.load(nr), np.load(pr))

    if not agents:
        print("  No robustness data found, skipping KDE plot.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # Shared x range
    all_vals = np.concatenate([np.concatenate(v) for v in agents.values()])
    x_min, x_max = all_vals.min(), all_vals.max()
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    xs = np.linspace(x_min - x_pad, x_max + x_pad, 400)

    for label, (normal, perturbed) in agents.items():
        color = COLORS[label]
        # Normal — solid
        kde_n = gaussian_kde(normal)
        ys_n = kde_n(xs)
        ax.fill_between(xs, ys_n, alpha=0.12, color=color)
        ax.plot(xs, ys_n, linewidth=2.2, color=color, label=f"{label} (normal)")
        # Perturbed — dashed
        kde_p = gaussian_kde(perturbed)
        ys_p = kde_p(xs)
        ax.fill_between(xs, ys_p, alpha=0.08, color=color)
        ax.plot(xs, ys_p, linewidth=2.2, color=color, linestyle="--",
                label=f"{label} (perturbed)")

    ax.set_xlabel("Episode reward")
    ax.set_ylabel("Density")
    ax.set_title(f"Robustness Analysis - {title_env_label}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "robustness_kde.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, "robustness_kde.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved robustness_kde.{{pdf,png}}")
    plt.close(fig)


def plot_robustness_degradation(results_dir, num_seeds, title_env_label="Environment",
                                flat_seed=1, hrl_seed=1, sequential_seed=1):
    """
    Degradation bar chart: ΔR = mean(normal) − mean(perturbed) per model.
    Heuristic Bay Grouping Agent vs Joint Hierarchical RL vs Sequential HPPO.
    """
    _apply_plot_style()
    print(f"\n{'=' * 70}")
    print(f"GENERATING ROBUSTNESS DEGRADATION PLOT")
    print(f"{'=' * 70}")

    labels_data = []
    nr = os.path.join(results_dir, "rule_based_grouped_normal_rewards.npy")
    pr = os.path.join(results_dir, "rule_based_grouped_perturbed_rewards.npy")
    if os.path.exists(nr) and os.path.exists(pr):
        labels_data.append(("Heuristic Bay Grouping Agent", np.load(nr), np.load(pr)))

    hrl_name = f"hrl_seed{hrl_seed}"
    nr = os.path.join(results_dir, f"{hrl_name}_normal_rewards.npy")
    pr = os.path.join(results_dir, f"{hrl_name}_perturbed_rewards.npy")
    if os.path.exists(nr) and os.path.exists(pr):
        labels_data.append(("Hierarchical RL", np.load(nr), np.load(pr)))

    seq_name = f"sequential_hppo_seed{sequential_seed}"
    nr = os.path.join(results_dir, f"{seq_name}_normal_rewards.npy")
    pr = os.path.join(results_dir, f"{seq_name}_perturbed_rewards.npy")
    if os.path.exists(nr) and os.path.exists(pr):
        labels_data.append(("Sequential HPPO", np.load(nr), np.load(pr)))

    if not labels_data:
        print("  No robustness data found, skipping degradation plot.")
        return

    model_names = [d[0] for d in labels_data]
    mean_normal = np.array([np.mean(d[1]) for d in labels_data])
    mean_perturbed = np.array([np.mean(d[2]) for d in labels_data])
    delta_r = mean_perturbed - mean_normal  # negative: bars go downward
    pct_drop = np.where(mean_normal != 0, 100.0 * np.abs(delta_r) / np.abs(mean_normal), 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels_data))
    colors = [COLORS[lbl] for lbl in model_names]
    bars = ax.bar(x, delta_r, width=0.5, color=colors, alpha=0.85)

    y_offset = max(abs(delta_r)) * 0.06 if delta_r.any() else 0.05
    for bar, dr, pct in zip(bars, delta_r, pct_drop):
        # Place text inside the bar, just above the bottom tip (dr is negative)
        text_y = dr + y_offset
        ax.text(bar.get_x() + bar.get_width() / 2, text_y,
                f"ΔR = {dr:.2f}\n({pct:.1f}% drop)",
                ha="center", va="bottom", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Reward drop  ΔR = R(perturbed) − R(normal)")
    ax.set_title(f"Robustness Reward Degradation under Random Perturbations - {title_env_label}", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "robustness_degradation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, "robustness_degradation.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved robustness_degradation.{{pdf,png}}")
    plt.close(fig)


# ============================================================
# PLOT 3: BAY UNIFORMITY (ENTROPY OVER TIME)
# ============================================================

def plot_bay_uniformity(results_dir, num_seeds, title_env_label="Environment",
                        flat_seed=1, hrl_seed=1, sequential_seed=1):
    """
    3-pane per-bay plot: Flat RL (left) | Joint Hierarchical RL (middle) | Sequential HPPO (right).
    Bay 1 (first physical bay): alpha=1.0, linewidth=3.0.
    Other bays: alpha=0.45, linewidth=3.0.
    Handles 6 and 10 bays.
    """
    import matplotlib.lines as mlines
    _apply_plot_style()
    print(f"\n{'=' * 70}")
    print(f"GENERATING BAY UNIFORMITY PLOT")
    print(f"{'=' * 70}")

    # Dash styles — enough for up to 10 bays
    DASH_STYLES = [
        "-", "--", "-.", ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 2)),
        (0, (1, 1)),
        (0, (3, 5, 1, 5)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    panel_specs = [
        ("Flat RL",          flat_seed, "flat"),
        ("Hierarchical RL",  hrl_seed,  "hrl"),
        ("Sequential HPPO",  sequential_seed, "sequential_hppo"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21.0, 5.0), sharey=True)

    num_bays_ref = None
    odd_bays_ref = None

    for ax, (label, seed_idx, prefix) in zip(axes, panel_specs):
        name = f"{prefix}_seed{seed_idx}"
        path = os.path.join(results_dir, f"{name}_bay_dominance.npy")
        if not os.path.exists(path):
            ax.set_title(f"{label}\n(no data)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Episode step")
            continue

        data = np.load(path)  # (num_seeds, num_bays, max_steps)
        num_bays = data.shape[1]
        odd_bays = [2 * i + 1 for i in range(num_bays)]
        if num_bays_ref is None:
            num_bays_ref = num_bays
            odd_bays_ref = odd_bays

        # Trim trailing all-NaN columns using bay 0 as episode-length proxy
        has_data = ~np.isnan(data[:, 0, :])
        last_valid = 0
        for col in range(data.shape[2]):
            if has_data[:, col].any():
                last_valid = col
        data_trimmed = data[:, :, :last_valid + 1]

        color = COLORS[label]
        for b_idx in range(num_bays):
            bay_data = data_trimmed[:, b_idx, :]
            with np.errstate(all="ignore"):
                mean = np.nanmean(bay_data, axis=0)
            valid = ~np.isnan(mean)
            steps = np.arange(len(mean))[valid]
            mean_v = mean[valid]
            dash = DASH_STYLES[b_idx % len(DASH_STYLES)]
            alpha = 1.0 if b_idx == 0 else 0.45
            ax.plot(steps, mean_v, linewidth=3.0, color=color,
                    linestyle=dash, alpha=alpha, label="_nolegend_")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode step")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

        # Per-pane legend showing bay dash styles
        n_ref = num_bays_ref if num_bays_ref is not None else num_bays
        o_ref = odd_bays_ref if odd_bays_ref is not None else odd_bays
        bay_handles = [
            mlines.Line2D([], [], color="gray",
                          linewidth=2.2 if i == 0 else 1.4,
                          linestyle=DASH_STYLES[i % len(DASH_STYLES)],
                          alpha=1.0 if i == 0 else 0.65,
                          label=f"Bay {o_ref[i]}")
            for i in range(n_ref)
        ]
        ax.legend(handles=bay_handles, frameon=False, loc="best",
                  fontsize=9, ncol=2 if n_ref > 5 else 1)

    axes[0].set_ylabel("Max-group proportion per bay")
    fig.suptitle(
        f"Bay Uniformity Over Time - {title_env_label}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "bay_uniformity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, "bay_uniformity.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved bay_uniformity.{{pdf,png}}")
    plt.close(fig)


# ============================================================
# PLOT 4: PARTIAL STACKS OVER TIME
# ============================================================

def plot_partial_stacks(results_dir, num_seeds, title_env_label="Environment",
                        flat_seed=1, hrl_seed=1, sequential_seed=1):
    """Line plot: step vs mean partial-stack count ± 1 std."""
    _apply_plot_style()
    print(f"\n{'=' * 70}")
    print(f"GENERATING PARTIAL STACKS PLOT")
    print(f"{'=' * 70}")

    agents = {}
    path = os.path.join(results_dir, "rule_based_grouped_partial_stacks.npy")
    if os.path.exists(path):
        agents["Heuristic Bay Grouping Agent"] = np.load(path)

    for label, seed_idx in [("Flat RL", flat_seed), ("Hierarchical RL", hrl_seed)]:
        prefix = "flat" if label == "Flat RL" else "hrl"
        name = f"{prefix}_seed{seed_idx}"
        path = os.path.join(results_dir, f"{name}_partial_stacks.npy")
        if os.path.exists(path):
            agents[label] = np.load(path)

    path = os.path.join(results_dir, f"sequential_hppo_seed{sequential_seed}_partial_stacks.npy")
    if os.path.exists(path):
        agents["Sequential HPPO"] = np.load(path)

    if not agents:
        print("  No metrics data found, skipping partial stacks plot.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for label, data in agents.items():
        has_data = ~np.isnan(data)
        last_valid = 0
        for col in range(data.shape[1]):
            if has_data[:, col].any():
                last_valid = col
        data = data[:, :last_valid + 1]
        with np.errstate(all="ignore"):
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
        valid = ~np.isnan(mean)
        steps = np.arange(len(mean))[valid]
        mean = mean[valid]
        std = std[valid]

        color = COLORS[label]
        ax.plot(steps, mean, linewidth=2.0, color=color, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Episode step")
    ax.set_ylabel("Partially open stacks")
    ax.set_title(f"Stacking Strategy Over Time - {title_env_label}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "partial_stacks.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, "partial_stacks.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved partial_stacks.{{pdf,png}}")
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Additional analysis plots: robustness, bay uniformity, stacking strategy")
    parser.add_argument("--mode", choices=["robust", "metrics", "both"],
                        default="both",
                        help="Which analyses to run (default: both)")
    args = parser.parse_args()

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
    SEQUENTIAL_MODEL_DIR = "stack/models/sequential_hppo/pointer/large4"
    NUM_SEEDS = 100
    MAX_STEPS = 500
    FLAT_SEED = 2   # which seed (1-3) to use for flat RL
    HRL_SEED = 3    # which seed (1-3) to use for hierarchical RL
    SEQUENTIAL_SEED = 1   # which seed (1-3) to use for Sequential HPPO

    # Derive env label from model directory: large1 → "Large", large4 → "Massive"
    _dir_lower = FLAT_MODEL_DIR.lower()
    if "large1" in _dir_lower:
        TITLE_LABEL = "Large"
    elif "large4" in _dir_lower:
        TITLE_LABEL = "Massive"
    else:
        TITLE_LABEL = os.path.basename(os.path.normpath(FLAT_MODEL_DIR))

    results_dir = evaluate_all(
        config_dict,
        flat_model_dir=FLAT_MODEL_DIR,
        hrl_model_dir=HRL_MODEL_DIR,
        sequential_model_dir=SEQUENTIAL_MODEL_DIR,
        num_seeds=NUM_SEEDS,
        max_steps=MAX_STEPS,
        mode=args.mode,
        flat_seed=FLAT_SEED,
        hrl_seed=HRL_SEED,
        sequential_seed=SEQUENTIAL_SEED,
    )

    if args.mode in ("robust", "both"):
        plot_robustness_kde(results_dir, NUM_SEEDS, TITLE_LABEL, FLAT_SEED, HRL_SEED, SEQUENTIAL_SEED)
        plot_robustness_degradation(results_dir, NUM_SEEDS, TITLE_LABEL, FLAT_SEED, HRL_SEED, SEQUENTIAL_SEED)

    if args.mode in ("metrics", "both"):
        plot_bay_uniformity(results_dir, NUM_SEEDS, TITLE_LABEL, FLAT_SEED, HRL_SEED, SEQUENTIAL_SEED)
        plot_partial_stacks(results_dir, NUM_SEEDS, TITLE_LABEL, FLAT_SEED, HRL_SEED, SEQUENTIAL_SEED)
