"""
training_plots.py
-----------------
Plot training curves for flat RL, hierarchical RL and Sequential HPPO seeds
fetched from wandb.

The x-axis is environment steps, read from each run's global_step (the SB3
num_timesteps that MaskedEvalCallback logs with every evaluation), so all
methods share the same exact axis. A run without global_step raises an error
instead of being plotted on an approximate axis.

Usage (script):
    python training_plots.py
"""

import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import List, Optional

ENTITY  = None  # wandb user or team; None = default entity of the logged-in API key
PROJECT = "stack-rl"

# Colors matching plots.py
FLAT_COLOR = "#59A14F"
HRL_COLOR  = "#F28E2B"
SEQUENTIAL_COLOR = "#E15759"


# ============================================================
# HELPERS
# ============================================================

def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average.
    alpha=1.0 → no smoothing (raw signal)
    alpha→0   → maximum smoothing
    (Same convention as wandb's smoothing slider inverted:
     wandb slider 0.9 ≈ alpha 0.1 here.)
    """
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _project_path(api: wandb.Api) -> str:
    """"<entity>/<project>" of the runs; ENTITY=None uses the API key's default entity."""
    return f"{ENTITY or api.default_entity}/{PROJECT}"


def _fetch(api: wandb.Api, run_map: dict, name: str):
    """
    Return (steps, rewards) numpy arrays for a single wandb run, where steps
    are environment steps (global_step) of each evaluation.
    """
    if name not in run_map:
        raise KeyError(
            f"Run '{name}' not found in {_project_path(api)}.\n"
            f"Available runs: {sorted(run_map.keys())}"
        )
    run  = api.run(f"{_project_path(api)}/{run_map[name]}")
    # scan_history only returns rows that contain every requested key
    hist = run.scan_history(keys=["global_step", "eval/mean_reward"])
    df   = pd.DataFrame(hist)
    if df.empty:
        raise ValueError(
            f"Run '{name}' has no eval/mean_reward logged with global_step. "
            "Train it with run.py (wandb enabled, sync_tensorboard) so every "
            "evaluation is logged at its environment step."
        )
    df   = df.dropna(subset=["eval/mean_reward"]).sort_values("global_step")
    return df["global_step"].to_numpy(dtype=float), df["eval/mean_reward"].to_numpy(dtype=float)


def _fetch_cached(
    api: wandb.Api,
    run_map: dict,
    name: str,
    cache_dir: str,
):
    """
    Like _fetch but saves the result to <cache_dir>/<run_name>.npz on first
    call and loads from disk on subsequent calls, avoiding repeated wandb queries.
    Caches not marked as global_step (e.g. written by an older version of this
    script) are re-fetched.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, name.replace("/", "_") + ".npz")
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            if "x_axis" in data and str(data["x_axis"]) == "global_step":
                print(f"  {name}  [from cache]")
                return data["steps"], data["rewards"]
        print(f"  {name}  [cache not in global_step, re-fetching from wandb …]")
    else:
        print(f"  {name}  [fetching from wandb …]")
    steps, rewards = _fetch(api, run_map, name)
    np.savez(cache_path, steps=steps, rewards=rewards, x_axis="global_step")
    print(f"    cached → {cache_path}")
    return steps, rewards


def _to_common_grid(
    steps_list: List[np.ndarray],
    rewards_list: List[np.ndarray],
    n_points: int = 500,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
):
    """
    Interpolate all seed curves onto a shared step grid.
    lo/hi can be supplied externally to enforce a global range across groups.
    Returns (grid, matrix) where matrix.shape == (n_seeds, n_points).
    """
    if lo is None:
        lo = max(s[0]  for s in steps_list)
    if hi is None:
        hi = min(s[-1] for s in steps_list)
    grid = np.linspace(lo, hi, n_points)

    rows = []
    for steps, rewards in zip(steps_list, rewards_list):
        f = interp1d(
            steps, rewards,
            kind="linear",
            bounds_error=False,
            fill_value=(rewards[0], rewards[-1]),
        )
        rows.append(f(grid))

    return grid, np.array(rows)  # (n_seeds, n_points)


# ============================================================
# MAIN PLOT FUNCTION
# ============================================================

def plot_training_curves(
    flat_run_names: List[str],
    hrl_run_names: List[str],
    sequential_run_names: List[str],
    env_name: str = "env",
    display_name: Optional[str] = None,
    ema_alpha: float = 0.1,
    n_points: int = 500,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Parameters
    ----------
    flat_run_names : list[str]
        wandb run names for the 3 flat RL seeds.
    hrl_run_names : list[str]
        wandb run names for the 3 hierarchical RL seeds.
    sequential_run_names : list[str]
        wandb run names for the 3 Sequential HPPO seeds (run.py --sequential_hppo
        --wandb_run_name).
    env_name : str
        Short key used for the save directory (e.g. "massive"). NOT used
        for the plot title or filename label.
    display_name : str or None
        Human-readable environment label shown in the plot title and used
        as the save filename slug (e.g. "Massive (10x8x5)"). When omitted,
        falls back to env_name.
    ema_alpha : float
        EMA weight for the current value (0 < alpha ≤ 1).
        Lower = smoother (default 0.1 ≈ heavy wandb smoothing).
    n_points : int
        Resolution of the interpolated step grid.
    title : str or None
        Plot title. Defaults to display_name (or env_name if display_name
        is not set).
    save_path : str or None
        Base file path (no extension). Defaults to
        "results/training_curves/<env_name>/training_curves".
        Pass an empty string to skip saving.
    """
    label = display_name if display_name is not None else env_name
    if title is None:
        title = label
    if save_path is None:
        save_path = os.path.join("results", "training_curves", env_name, "training_curves")

    # Cache directory sits next to the plot output
    cache_dir = os.path.join(os.path.dirname(save_path) or ".", "wandb_cache")

    print("Connecting to wandb …")
    api       = wandb.Api()
    runs_iter = list(api.runs(_project_path(api)))
    run_map   = {r.name: r.id for r in runs_iter}

    # A rerun with a reused name would make run_map silently keep only one of them
    requested = flat_run_names + hrl_run_names + sequential_run_names
    duplicates = sorted({n for n in requested if sum(r.name == n for r in runs_iter) > 1})
    if duplicates:
        raise ValueError(
            f"Run names matching more than one run in {_project_path(api)}: {duplicates}. "
            "Rename or delete the extra runs so each name is unique."
        )

    groups = [
        ("Flat RL",           flat_run_names, FLAT_COLOR),
        ("Hierarchical RL",   hrl_run_names,  HRL_COLOR),
        ("Sequential HPPO",   sequential_run_names, SEQUENTIAL_COLOR),
    ]

    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     18,
        "legend.fontsize":    10,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Fetch all runs first to compute a global step range ──
    # x is environment steps (global_step) for every run of every group
    all_steps_lists, all_rewards_lists = [], []
    for label, run_names, color in groups:
        print(f"\nFetching {label} runs …")
        steps_list, rewards_list = [], []
        for name in run_names:
            s, r = _fetch_cached(api, run_map, name, cache_dir)
            steps_list.append(s)
            rewards_list.append(r)
        all_steps_lists.append(steps_list)
        all_rewards_lists.append(rewards_list)

    # Global range: clip all curves to the shortest run across all groups
    global_lo = max(s[0]  for sl in all_steps_lists for s in sl)
    global_hi = min(s[-1] for sl in all_steps_lists for s in sl)

    print(f"\nGlobal step range: {global_lo:.0f} – {global_hi:.0f}")

    for (label, run_names, color), steps_list, rewards_list in zip(
        groups, all_steps_lists, all_rewards_lists
    ):
        grid, matrix = _to_common_grid(
            steps_list, rewards_list, n_points, lo=global_lo, hi=global_hi
        )

        # Raw seed lines — faint to show actual fluctuations
        for curve in matrix:
            ax.plot(
                grid, curve,
                color=color, alpha=0.35, linewidth=1.2,
            )

        # EMA line per seed — one bold line each, same color
        for i, curve in enumerate(matrix):
            smooth = _ema(curve, alpha=ema_alpha)
            ax.plot(
                grid, smooth,
                color=color, linewidth=2.0, alpha=0.85,
                label=label if i == 0 else "_nolegend_",
            )

    # Format x-axis ticks as "0.2M", "1M", "5M" etc.
    def _millions_fmt(x, _pos):
        m = x / 1_000_000
        if m == int(m):
            return f"{int(m)}M"
        # strip trailing zeros (e.g. 0.50 → "0.5M")
        return f"{m:.2g}M"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_millions_fmt))

    ax.set_xlabel("Training Steps (millions)")
    ax.set_ylabel("Eval Mean Reward")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    fig.tight_layout()

    if save_path:
        base = save_path.rstrip("/\\")
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        print(f"\nSaved: {base}.pdf / .png")

    plt.show()
    return fig, ax


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    FLAT_RUNS = [
        "flat-pointer-small-1M-s1",
        "flat-pointer-small-1M-s2",
        "flat-pointer-small-1M-s3"
    ]
    HRL_RUNS = [
        "hrl-pointer-small-1M-s1",
        "hrl-pointer-small-1M-s2",
        "hrl-pointer-small-1M-s3"
    ]
    SEQUENTIAL_RUNS = [
        "sequential-hppo-pointer-small-1M-s1",
        "sequential-hppo-pointer-small-1M-s2",
        "sequential-hppo-pointer-small-1M-s3"
    ]

    plot_training_curves(
        flat_run_names=FLAT_RUNS,
        hrl_run_names=HRL_RUNS,
        sequential_run_names=SEQUENTIAL_RUNS,
        env_name="small",
        display_name="Small (3x4x3)",
        ema_alpha=0.1,
    )
