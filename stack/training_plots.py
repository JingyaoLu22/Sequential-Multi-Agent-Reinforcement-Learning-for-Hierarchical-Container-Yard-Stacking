"""
training_plots.py
-----------------
Plot evaluation-reward training curves for flat RL, hierarchical RL
and Sequential HPPO seeds, fetched from wandb (plot_training_curves) or
read from local evaluations.csv files (plot_csv_training_curves).

Usage (from the repo root):
    python -m stack.training_plots                # the W&B example below
    python -m stack.training_plots --sequential_dirs runs/s1 runs/s2 runs/s3 \
        --display_name "Small (3x4x3)" --save_path runs/training_curves_3seeds
"""

import csv
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

ENTITY  = "aritrabancode-university-of-amsterdam"
PROJECT = "stack-rl"

# Colors matching plots.py
FLAT_COLOR       = "#59A14F"
HRL_COLOR        = "#F28E2B"
SEQUENTIAL_COLOR = "#E15759"

PLOT_STYLE = {
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     18,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}


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


def _fetch(api, run_map: dict, name: str):
    """Return (steps, rewards) numpy arrays for a single wandb run."""
    import pandas as pd

    if name not in run_map:
        raise KeyError(
            f"Run '{name}' not found in {ENTITY}/{PROJECT}.\n"
            f"Available runs: {sorted(run_map.keys())}"
        )
    run  = api.run(f"{ENTITY}/{PROJECT}/{run_map[name]}")
    hist = run.scan_history(keys=["_step", "eval/mean_reward"])
    df   = pd.DataFrame(hist).dropna(subset=["eval/mean_reward"])
    df   = df.sort_values("_step")
    return df["_step"].to_numpy(dtype=float), df["eval/mean_reward"].to_numpy(dtype=float)


def _fetch_cached(
    api,
    run_map: dict,
    name: str,
    cache_dir: str,
):
    """
    Like _fetch but saves the result to <cache_dir>/<run_name>.npz on first
    call and loads from disk on subsequent calls, avoiding repeated wandb queries.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, name.replace("/", "_") + ".npz")
    if os.path.exists(cache_path):
        print(f"  {name}  [from cache]")
        with np.load(cache_path) as data:
            return data["steps"], data["rewards"]
    print(f"  {name}  [fetching from wandb …]")
    steps, rewards = _fetch(api, run_map, name)
    np.savez(cache_path, steps=steps, rewards=rewards)
    print(f"    cached → {cache_path}")
    return steps, rewards


def load_csv_curve(source):
    """
    Return (steps, rewards) from an evaluations.csv, or from a run
    directory containing training_logs/evaluations.csv.
    """
    path = Path(source)
    if path.is_dir():
        path = path / "training_logs" / "evaluations.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} has no evaluation records yet.")
    steps = np.array([float(row["timesteps"]) for row in rows])
    rewards = np.array([float(row["mean_reward"]) for row in rows])
    order = np.argsort(steps, kind="stable")
    return steps[order], rewards[order]


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
    return grid, np.array([np.interp(grid, s, r) for s, r in zip(steps_list, rewards_list)])


def _plot_groups(
    groups,
    title: str,
    ema_alpha: float,
    n_points: int,
    save_path,
    show: bool,
):
    """
    groups: [(label, color, [(steps, rewards), ...]), ...]. All curves are
    clipped to the step range every seed of every group covers.
    """
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    curves = [curve for _label, _color, group in groups for curve in group]
    global_lo = max(s[0]  for s, _ in curves)
    global_hi = min(s[-1] for s, _ in curves)
    if global_lo > global_hi:
        raise ValueError("The seed histories have no common training-step interval.")

    with mpl.rc_context(PLOT_STYLE):
        if show:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            # No pyplot: safe to call repeatedly during headless training.
            fig = Figure(figsize=(9, 5))
            ax = fig.subplots()

        # A new run may have only its first evaluation: show it as a point.
        marker = "o" if global_lo == global_hi else None

        for label, color, group in groups:
            grid, matrix = _to_common_grid(
                [s for s, _ in group], [r for _, r in group], n_points, lo=global_lo, hi=global_hi
            )

            # Raw seed lines — faint to show actual fluctuations
            for curve in matrix:
                ax.plot(grid, curve, color=color, alpha=0.35, linewidth=1.2, marker=marker)

            # EMA line per seed — one bold line each, same color
            for i, curve in enumerate(matrix):
                ax.plot(
                    grid, _ema(curve, alpha=ema_alpha),
                    color=color, linewidth=2.0, alpha=0.85, marker=marker,
                    label=label if i == 0 else "_nolegend_",
                )

        # Format x-axis ticks as "0.2M", "1M", "5M" etc.
        def _millions_fmt(x, _pos):
            m = x / 1_000_000
            if m == int(m):
                return f"{int(m)}M"
            # strip trailing zeros (e.g. 0.50 → "0.5M")
            return f"{m:.2g}M"

        ax.xaxis.set_major_formatter(FuncFormatter(_millions_fmt))

        ax.set_xlabel("Training Steps (millions)")
        ax.set_ylabel("Eval Mean Reward")
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        fig.tight_layout()

        if save_path:
            base = str(save_path).rstrip("/\\")
            os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
            fig.savefig(base + ".pdf", bbox_inches="tight")
            fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
            print(f"\nSaved: {base}.pdf / .png")

        if show:
            plt.show()

    return fig, ax


# ============================================================
# MAIN PLOT FUNCTIONS
# ============================================================

def plot_training_curves(
    flat_run_names: List[str],
    hrl_run_names: List[str],
    env_name: str = "env",
    display_name: Optional[str] = None,
    ema_alpha: float = 0.1,
    n_points: int = 500,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    step_correction: bool = False,
):
    """
    Parameters
    ----------
    flat_run_names : list[str]
        wandb run names for the 3 flat RL seeds.
    hrl_run_names : list[str]
        wandb run names for the 3 hierarchical RL seeds.
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
    step_correction : bool
        When True, multiplies all step values on the x-axis by 197.
        Useful when wandb logged steps are in "update" units and you
        want to convert to environment steps.
    """
    import wandb

    label = display_name if display_name is not None else env_name
    if save_path is None:
        save_path = os.path.join("results", "training_curves", env_name, "training_curves")

    # Cache directory sits next to the plot output
    cache_dir = os.path.join(os.path.dirname(save_path) or ".", "wandb_cache")

    print("Connecting to wandb …")
    api       = wandb.Api()
    runs_iter = api.runs(f"{ENTITY}/{PROJECT}")
    run_map   = {r.name: r.id for r in runs_iter}

    STEP_MULTIPLIER = 197 if step_correction else 1

    groups = []
    for group_label, run_names, color in [
        ("Flat RL",           flat_run_names, FLAT_COLOR),
        ("Hierarchical RL",   hrl_run_names,  HRL_COLOR),
    ]:
        print(f"\nFetching {group_label} runs …")
        curves = []
        for name in run_names:
            steps, rewards = _fetch_cached(api, run_map, name, cache_dir)
            curves.append((steps * STEP_MULTIPLIER, rewards))
        groups.append((group_label, color, curves))

    return _plot_groups(groups, title or label, ema_alpha, n_points, save_path, show=True)


def plot_csv_training_curves(
    *,
    sequential_run_dirs: Sequence = (),
    hrl_run_dirs: Sequence = (),
    flat_run_dirs: Sequence = (),
    env_name: str = "env",
    display_name: Optional[str] = None,
    ema_alpha: float = 0.1,
    n_points: int = 500,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Same plot as plot_training_curves(), from local runs: each entry is a
    run directory (with training_logs/evaluations.csv) or a CSV path, one
    per seed. Each non-empty group is one color in the legend.
    """
    label = display_name if display_name is not None else env_name
    if save_path is None:
        save_path = os.path.join("results", "training_curves", env_name, "training_curves")

    groups = [
        (group_label, color, [load_csv_curve(source) for source in sources])
        for group_label, sources, color in [
            ("Flat RL",           flat_run_dirs,       FLAT_COLOR),
            ("Hierarchical RL",   hrl_run_dirs,        HRL_COLOR),
            ("Sequential HPPO",   sequential_run_dirs, SEQUENTIAL_COLOR),
        ]
        if sources
    ]
    if not groups:
        raise ValueError("No run directories given.")

    return _plot_groups(groups, title or label, ema_alpha, n_points, save_path, show)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot seed runs from their local evaluations.csv; "
                    "without run directories, plot the W&B example below."
    )
    parser.add_argument("--sequential_dirs", nargs="+", default=[],
                        help="Sequential HPPO run directories, one per seed.")
    parser.add_argument("--hrl_dirs", nargs="+", default=[],
                        help="Hierarchical RL run directories, one per seed.")
    parser.add_argument("--flat_dirs", nargs="+", default=[],
                        help="Flat RL run directories, one per seed.")
    parser.add_argument("--display_name")
    parser.add_argument("--save_path", help="Output base path without extension.")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--plot_points", type=int, default=500)
    args = parser.parse_args()

    if args.sequential_dirs or args.hrl_dirs or args.flat_dirs:
        plot_csv_training_curves(
            sequential_run_dirs=args.sequential_dirs,
            hrl_run_dirs=args.hrl_dirs,
            flat_run_dirs=args.flat_dirs,
            display_name=args.display_name,
            ema_alpha=args.ema_alpha,
            n_points=args.plot_points,
            save_path=args.save_path,
        )
    else:
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

        plot_training_curves(
            flat_run_names=FLAT_RUNS,
            hrl_run_names=HRL_RUNS,
            env_name="small",
            display_name="Small (3x4x3)",
            ema_alpha=0.1,
            step_correction=True
        )
