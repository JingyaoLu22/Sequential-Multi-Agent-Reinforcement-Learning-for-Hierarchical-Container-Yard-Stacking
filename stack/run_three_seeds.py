"""Run independent Sequential Multi-Agent PPO seeds and merge their curves.

The default command launches seeds 1, 2 and 3 as three separate Python
processes. Runs are grouped by scale as <output_root>/<run_prefix>/s1, s2 and
s3. Each process therefore owns a different save directory, so checkpoints and
resume state can never be mistaken for another seed. Once all runs finish,
their evaluation CSV files are combined with the same per-seed interpolation
and EMA convention used by :mod:`stack.training.training_plots`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ENVIRONMENT_SIZES = (
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
)

YARD_SHAPES = {
    "small": (3, 3, 3),
    "small_with_margin": (3, 4, 3),
    "medium": (4, 4, 4),
    "medium_with_margin": (4, 5, 4),
    "large": (6, 6, 5),
    "large_with_margin": (6, 7, 5),
    "large_v2": (8, 5, 5),
    "large_v2_with_margin": (8, 7, 5),
    "large_v3": (10, 6, 5),
    "large_v3_with_margin": (10, 7, 5),
    "large_v4": (10, 8, 5),
    "large_v4_with_margin": (10, 9, 5),
}


def _default_prefix(size: str) -> str:
    return "sequential_hppo_" + size.removesuffix("_with_margin")


def _managed_option_in(extra_args: Sequence[str]) -> str | None:
    managed = {
        "--size",
        "--seed",
        "--device",
        "--timesteps",
        "--training_profile",
        "--profile",
        "--eval_freq",
        "--n_eval_episodes",
        "--eval_seed",
        "--eval_seed_mode",
        "--ema_alpha",
        "--plot_points",
        "--save_dir",
        "--wandb_project",
        "--wandb_run_name",
        "--no_wandb",
        "--fresh",
    }
    for token in extra_args:
        option = token.split("=", 1)[0]
        if option in managed:
            return option
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_and_tee(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        header = (
            "\n"
            + "=" * 78
            + "\n"
            + datetime.now(timezone.utc).isoformat()
            + "\n"
            + shlex.join(command)
            + "\n"
            + "=" * 78
            + "\n"
        )
        print(header, end="", flush=True)
        log_file.write(header)
        log_file.flush()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=ENVIRONMENT_SIZES,
                        default="small_with_margin")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                        help="Independent training seeds (default: 1 2 3).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto")
    parser.add_argument("--training_profile", "--profile", default="auto",
                        choices=["auto", "small", "medium", "large", "massive"])
    parser.add_argument("--timesteps", type=int,
                        help="Optional override; Small profile defaults to 1,000,000.")
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, default=100_000,
                        help="First episode seed at the first evaluation point.")
    parser.add_argument("--eval_seed_mode", choices=["rolling", "fixed"],
                        default="rolling")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--plot_points", type=int, default=500)
    parser.add_argument("--output_root", type=Path,
                        default=Path("./models_trained"))
    parser.add_argument("--run_prefix",
                        help=(
                            "Scale-directory name under output_root; derived "
                            "from --size by default."
                        ))
    parser.add_argument("--plot_path", type=Path,
                        help="Output base path without extension.")
    parser.add_argument("--display_name",
                        help="Title used for the final combined figure.")
    parser.add_argument("--fresh", action="store_true",
                        help="Start all three runs from new networks.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B; local CSV logging is always enabled.")
    parser.add_argument("--wandb_project", default="stack-sequential-hppo")
    parser.add_argument("--no_live_plots", action="store_true",
                        help="Skip per-run live plots; still create the final plot.")
    parser.add_argument("--plot_only", action="store_true",
                        help="Do not train; merge the existing seed histories.")
    parser.add_argument(
        "run_args",
        nargs=argparse.REMAINDER,
        help="Extra stack.run_sequential_hppo arguments after '--', e.g. -- --batch_size 32.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Training seeds must be nonnegative.")
    if args.eval_freq <= 0 or args.n_eval_episodes <= 0:
        raise ValueError("Evaluation frequency and episode count must be positive.")
    if args.timesteps is not None and args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if not 0 < args.ema_alpha <= 1:
        raise ValueError("--ema_alpha must satisfy 0 < alpha <= 1.")
    if args.plot_points < 2:
        raise ValueError("--plot_points must be at least 2.")

    extra_args = list(args.run_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    blocked = _managed_option_in(extra_args)
    if blocked is not None:
        raise ValueError(
            f"{blocked} is managed by run_three_seeds; pass it before '--'."
        )

    prefix = args.run_prefix or _default_prefix(args.size)
    output_root = args.output_root.expanduser()
    experiment_dir = output_root / prefix
    run_dirs = [experiment_dir / f"s{seed}" for seed in args.seeds]
    plot_path = (
        args.plot_path.expanduser()
        if args.plot_path is not None
        else experiment_dir / "training_curves_3seeds"
    )
    manifest_path = experiment_dir / "three_seeds_manifest.json"
    manifest = {
        "size": args.size,
        "seeds": args.seeds,
        "run_directories": [str(path) for path in run_dirs],
        "evaluation": {
            "frequency": args.eval_freq,
            "n_episodes": args.n_eval_episodes,
            "initial_base_seed": args.eval_seed,
            "seed_mode": args.eval_seed_mode,
            "deterministic": True,
            "evaluate_at_step_zero": False,
        },
        "plot": {
            "ema_alpha": args.ema_alpha,
            "n_points": args.plot_points,
            "save_path": str(plot_path),
        },
        "runs": [],
    }
    _write_json(manifest_path, manifest)

    if not args.plot_only:
        for seed, run_dir in zip(args.seeds, run_dirs):
            command = [
                sys.executable,
                "-u",
                "-m",
                "stack.run_sequential_hppo",
                "--size",
                args.size,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--training_profile",
                args.training_profile,
                "--eval_freq",
                str(args.eval_freq),
                "--n_eval_episodes",
                str(args.n_eval_episodes),
                "--eval_seed",
                str(args.eval_seed),
                "--eval_seed_mode",
                args.eval_seed_mode,
                "--ema_alpha",
                str(args.ema_alpha),
                "--plot_points",
                str(args.plot_points),
                "--save_model",
                "--save_dir",
                str(run_dir),
            ]
            if args.timesteps is not None:
                command.extend(["--timesteps", str(args.timesteps)])
            if args.fresh:
                command.append("--fresh")
            if args.no_live_plots:
                command.append("--no_training_plots")
            if args.wandb:
                command.extend(
                    [
                        "--wandb_project",
                        args.wandb_project,
                        "--wandb_run_name",
                        f"{prefix}-s{seed}",
                    ]
                )
            else:
                command.append("--no_wandb")
            command.extend(extra_args)

            run_record = {
                "seed": seed,
                "save_dir": str(run_dir),
                "command": command,
                "status": "running",
            }
            manifest["runs"].append(run_record)
            _write_json(manifest_path, manifest)
            try:
                _run_and_tee(command, run_dir / "training_logs" / "console.log")
            except BaseException:
                run_record["status"] = "failed_or_interrupted"
                _write_json(manifest_path, manifest)
                raise
            run_record["status"] = "completed"
            _write_json(manifest_path, manifest)

    from .training.training_plots import plot_training_curves

    yard = "x".join(str(value) for value in YARD_SHAPES[args.size])
    display_name = args.display_name or (
        f"{args.size.removesuffix('_with_margin').replace('_', ' ').title()} "
        f"({yard})"
    )
    fig, _ = plot_training_curves(
        hrl_run_dirs=run_dirs,
        env_name=args.size.removesuffix("_with_margin"),
        display_name=display_name,
        ema_alpha=args.ema_alpha,
        n_points=args.plot_points,
        save_path=plot_path,
        show=False,
    )
    fig.clear()
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(f"\nCombined three-seed curve: {plot_path}.png / .pdf")
    print(f"Experiment manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
