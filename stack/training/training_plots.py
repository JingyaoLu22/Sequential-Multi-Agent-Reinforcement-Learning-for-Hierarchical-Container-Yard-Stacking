from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

FLAT_COLOR = "#59A14F"
HRL_COLOR = "#F28E2B"
PLOT_STYLE = {
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 18,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
}


def _ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Current-sample weight alpha; alpha=1 gives the unsmoothed curve."""
    if not 0 < alpha <= 1:
        raise ValueError("ema_alpha must satisfy 0 < alpha <= 1.")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("EMA expects a one-dimensional curve.")
    out = np.empty_like(values, dtype=float)
    if not len(values):
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _clean_curve(steps, rewards):
    steps, rewards = np.asarray(steps, dtype=float), np.asarray(rewards, dtype=float)
    if steps.ndim != 1 or rewards.ndim != 1 or steps.shape != rewards.shape:
        raise ValueError("Steps and rewards must be equally sized one-dimensional arrays.")
    valid = np.isfinite(steps) & np.isfinite(rewards) & (steps >= 0)
    steps, rewards = steps[valid], rewards[valid]
    if not len(steps):
        raise ValueError("No finite evaluation records found; run training evaluation first.")
    order = np.argsort(steps, kind="stable")
    steps, rewards = steps[order], rewards[order]
    # If the same step was re-evaluated, the last record wins.
    keep = np.r_[steps[:-1] != steps[1:], True]
    return steps[keep], rewards[keep]


def load_curve(source: str | Path, step_multiplier: float = 1.0):
    
    if not np.isfinite(step_multiplier) or step_multiplier <= 0:
        raise ValueError("step_multiplier must be finite and positive.")
    path = Path(source).expanduser()
    if path.is_dir():
        candidates = [path / "training_logs" / "evaluations.csv",
                      path / "evaluations.csv", path / "evaluations.npz"]
        path = next((p for p in candidates if p.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(
            f"No evaluation history at {path}. Model weights alone cannot "
            "reconstruct earlier training evaluation rewards."
        )
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if {"steps", "rewards"} <= set(data.files):
                steps, rewards = data["steps"], data["rewards"]
            elif {"timesteps", "results"} <= set(data.files):
                steps, rewards = data["timesteps"], data["results"]
                if rewards.ndim == 2:
                    rewards = rewards.mean(axis=1)
            else:
                raise ValueError(f"{path}: expected steps/rewards or timesteps/results.")
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []
            step_key = next((k for k in ("timesteps", "total_environment_steps",
                                        "train/environment_steps", "steps") if k in columns), None)
            reward_key = next((k for k in ("eval/mean_reward", "mean_reward")
                               if k in columns), None)
            if step_key is None or reward_key is None:
                raise ValueError(
                    f"{path}: requires environment steps and eval/mean_reward or mean_reward. "
                    "rollout mean_episode_reward is a different metric; _step alone is ambiguous."
                )
            pairs = []
            for row in reader:
                try:
                    pairs.append((float(row[step_key]), float(row[reward_key])))
                except (TypeError, ValueError):
                    continue  # W&B exports can contain sparse metric rows.
        steps, rewards = np.array(pairs, dtype=float).reshape(-1, 2).T
    else:
        raise ValueError(f"Unsupported history file: {path}; use CSV or NPZ.")
    steps, rewards = _clean_curve(steps, rewards)
    return _clean_curve(steps * step_multiplier, rewards)


def _to_common_grid(steps_list, rewards_list, n_points=500, lo=None, hi=None):
    """Linear interpolation within the overlap; never invent a missing tail."""
    if len(steps_list) != len(rewards_list) or not len(steps_list):
        raise ValueError("Supply one reward curve for each nonempty step array.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2.")
    curves = [_clean_curve(s, r) for s, r in zip(steps_list, rewards_list)]
    overlap_lo = max(s[0] for s, _ in curves)
    overlap_hi = min(s[-1] for s, _ in curves)
    lo = overlap_lo if lo is None else float(lo)
    hi = overlap_hi if hi is None else float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError("The seed histories have no common training-step interval.")
    if lo < overlap_lo or hi > overlap_hi:
        raise ValueError("Requested grid extends beyond available seed histories.")
    # A new run may have only its first periodic evaluation: show a real point.
    grid = np.array([lo]) if lo == hi else np.linspace(lo, hi, n_points)
    matrix = np.array([np.interp(grid, s, r) for s, r in curves])
    return grid, matrix


def plot_training_curves(
    flat_run_dirs: Optional[Sequence[str | Path]] = None,
    hrl_run_dirs: Optional[Sequence[str | Path]] = None,
    env_name: str = "env",
    display_name: Optional[str] = None,
    ema_alpha: float = 0.1,
    n_points: int = 500,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    flat_step_multiplier: float = 1.0,
    hrl_step_multiplier: float = 1.0,
):
    _ema(np.array([], dtype=float), ema_alpha)  # validate before reading files
    groups = []
    for label, sources, color, multiplier in [
        ("Flat RL", flat_run_dirs, FLAT_COLOR, flat_step_multiplier),
        ("Hierarchical RL", hrl_run_dirs, HRL_COLOR, hrl_step_multiplier),
    ]:
        if isinstance(sources, (str, Path)):
            raise TypeError("Run paths must be a list, e.g. hrl_run_dirs=[path].")
        if sources is not None and len(sources):
            groups.append((label, [load_curve(p, multiplier) for p in sources], color))
    if not groups:
        raise ValueError("Supply at least one HRL or Flat RL evaluation history.")
    global_lo = max(s[0] for _, curves, _ in groups for s, _ in curves)
    global_hi = min(s[-1] for _, curves, _ in groups for s, _ in curves)
    # Validate before creating a figure, including disjoint histories.
    aligned = [(label, *_to_common_grid([s for s, _ in curves], [r for _, r in curves],
                                       n_points, global_lo, global_hi), color)
               for label, curves, color in groups]

    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    label = display_name or env_name
    if save_path is None:
        save_path = Path("results") / "training_curves" / env_name / "training_curves"
    with mpl.rc_context(PLOT_STYLE):
        if show:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = Figure(figsize=(9, 5))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
        for group_label, grid, matrix, color in aligned:
            for curve in matrix:
                ax.plot(grid, curve, color=color, alpha=0.35, linewidth=1.2,
                        marker="o" if len(grid) == 1 else None)
            for i, curve in enumerate(matrix):
                ax.plot(grid, _ema(curve, ema_alpha), color=color, linewidth=2.0,
                        alpha=0.85, marker="o" if len(grid) == 1 else None,
                        label=group_label if i == 0 else "_nolegend_")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.3g}M"))
        if global_lo == global_hi:
            ax.set_xlim(max(0, global_lo - 1), global_hi + max(1, global_hi * .05))
        ax.set_xlabel("Training Steps (millions)")
        ax.set_ylabel("Eval Mean Reward")
        ax.set_title(title or label, fontsize=18, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        fig.tight_layout()
        if save_path:
            base = Path(str(save_path).rstrip("/\\"))
            if base.suffix.lower() in (".png", ".pdf"):
                base = base.with_suffix("")
            base.parent.mkdir(parents=True, exist_ok=True)
            # Readers in Jupyter never see a half-written live image.
            for extension in ("png", "pdf"):
                target = Path(str(base) + "." + extension)
                temporary = target.with_name(target.name + ".tmp")
                try:
                    fig.savefig(temporary, format=extension, dpi=300, bbox_inches="tight")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        if show:
            plt.show()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrl_dirs", nargs="+", default=[])
    parser.add_argument("--flat_dirs", nargs="+", default=[])
    parser.add_argument("--env_name", default="env")
    parser.add_argument("--display_name")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--n_points", type=int, default=500)
    parser.add_argument("--save_path")
    parser.add_argument("--flat_step_multiplier", type=float, default=1.0)
    parser.add_argument("--hrl_step_multiplier", type=float, default=1.0)
    parser.add_argument("--show", action="store_true", help="Also open an interactive figure.")
    args = parser.parse_args()
    try:
        fig, _ = plot_training_curves(
            flat_run_dirs=args.flat_dirs, hrl_run_dirs=args.hrl_dirs,
            env_name=args.env_name, display_name=args.display_name,
            ema_alpha=args.ema_alpha, n_points=args.n_points,
            save_path=args.save_path, show=args.show,
            flat_step_multiplier=args.flat_step_multiplier,
            hrl_step_multiplier=args.hrl_step_multiplier,
        )
    except (ValueError, FileNotFoundError, TypeError) as error:
        parser.error(str(error))
    fig.clear()
    print("Training curves generated (PNG + PDF unless saving was disabled).")


if __name__ == "__main__":
    main()