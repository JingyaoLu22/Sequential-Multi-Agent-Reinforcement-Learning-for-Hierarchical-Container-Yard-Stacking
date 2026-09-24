"""Durable local metrics and periodic evaluation for sequential PPO.

This module does not save or restore model/optimizer/environment checkpoints.
resume_step is only a history-alignment hook for a separately restored trainer.
"""

from __future__ import annotations

import csv
import json
import os
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _atomic_csv(path, columns, rows):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingMonitor:
    """Persist metrics before rendering, and preserve histories of earlier runs.

    A fresh run in an occupied log directory archives the earlier directory.
    A genuinely restored run can pass resume_step: records beyond that step
    are discarded from the active timeline, with the original CSV backed up.
    Supplying a step is not itself checkpoint restoration.
    """

    def __init__(self, log_dir, metadata, ema_alpha=0.1, n_points=500,
                 plot=True, resume_step=None):
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must satisfy 0 < alpha <= 1.")
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer >= 2.")
        if resume_step is not None and (int(resume_step) != resume_step or resume_step < 0):
            raise ValueError("resume_step must be a nonnegative integer.")
        self.log_dir = Path(log_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        known_files = ("evaluations.csv", "rollouts.csv", "run_metadata.json")
        if resume_step is None and any((self.log_dir / name).exists() for name in known_files):
            archive = self.log_dir.with_name(self.log_dir.name + "_previous_" + stamp)
            self.log_dir.rename(archive)
            print(f"Fresh training run: previous training logs preserved at {archive}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_path = self.log_dir / "evaluations.csv"
        self.rollout_path = self.log_dir / "rollouts.csv"
        self.ema_alpha, self.n_points, self.plot_enabled = ema_alpha, n_points, plot
        self.metadata = metadata
        metadata_path = self.log_dir / "run_metadata.json"
        if resume_step is not None and metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("environment", "evaluation", "seed"):
                if previous.get(key) != json.loads(json.dumps(metadata.get(key))):
                    raise ValueError(f"Cannot append a resumed history with different {key}.")
        self.evaluations = self._read(self.eval_path)
        if resume_step is not None:
            for path in (self.eval_path, self.rollout_path):
                rows = self._read(path)
                retained = [r for r in rows if float(r["timesteps"]) <= resume_step]
                if len(retained) != len(rows):
                    path.rename(path.with_name(path.stem + "_before_resume_" + stamp + ".csv"))
                    _atomic_csv(path, list(rows[0]), retained)
            self.evaluations = self._read(self.eval_path)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
        self._rollout_columns = None
        if self.rollout_path.exists():
            with self.rollout_path.open(newline="", encoding="utf-8") as file:
                self._rollout_columns = csv.DictReader(file).fieldnames

    @staticmethod
    def _read(path):
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    @property
    def last_eval_step(self):
        return int(float(self.evaluations[-1]["timesteps"])) if self.evaluations else None

    def record_rollout(self, stats):
        row = {"timesteps": int(stats["total_environment_steps"]), **stats}
        if self._rollout_columns is None:
            self._rollout_columns = list(row)
            _atomic_csv(self.rollout_path, self._rollout_columns, [])
        with self.rollout_path.open("a", newline="", encoding="utf-8") as file:
            # extrasaction="ignore": self._rollout_columns can be an
            # older, narrower column set read back from an existing
            # rollouts.csv on resume (see __init__ above). `row` may
            # contain keys that file's header predates - e.g. resuming
            # a run whose log directory was created before the
            # per-phase profiling columns existed. Silently dropping
            # those extra keys for this resumed file keeps resume
            # working (DictWriter's default is to raise ValueError on
            # any key not in fieldnames); it does not affect a fresh
            # run, whose columns are captured from `row` itself above.
            csv.DictWriter(
                file,
                fieldnames=self._rollout_columns,
                extrasaction="ignore",
            ).writerow(row)
            file.flush()
            os.fsync(file.fileno())

    def record_evaluation(
        self,
        timesteps,
        summary,
        *,
        evaluation_index=None,
        base_seed=None,
    ):
        timesteps = int(timesteps)
        if self.last_eval_step is not None and timesteps < self.last_eval_step:
            raise ValueError("Evaluation step moved backwards; align history to the restored checkpoint first.")
        row = {
            "timesteps": timesteps,
            "evaluation_index": (
                "" if evaluation_index is None else int(evaluation_index)
            ),
            "base_seed": "" if base_seed is None else int(base_seed),
            "last_seed": (
                ""
                if base_seed is None
                else int(base_seed) + int(summary.n_episodes) - 1
            ),
            "mean_reward": summary.mean_reward,
            "std_reward": summary.std_reward,
            "mean_episode_length": summary.mean_episode_length,
            "completion_rate": summary.completion_rate,
            "n_episodes": summary.n_episodes,
            "episode_rewards": json.dumps(summary.episode_rewards),
        }
        if self.last_eval_step == timesteps:
            self.evaluations[-1] = row
        else:
            self.evaluations.append(row)
        _atomic_csv(self.eval_path, list(row), self.evaluations)

    def refresh_plot(self):
        if not self.plot_enabled or not self.evaluations:
            return
        try:
            from .training_plots import plot_training_curves
            fig, _ = plot_training_curves(
                hrl_run_dirs=[self.eval_path],
                display_name=self.metadata["display_name"],
                ema_alpha=self.ema_alpha, n_points=self.n_points,
                save_path=self.log_dir / "training_curves", show=False,
            )
            fig.clear()
        except (ImportError, OSError, ValueError) as error:
            warnings.warn(f"Training plot could not be refreshed: {error}. "
                          f"Evaluation data is saved at {self.eval_path}.", stacklevel=2)


def evaluation_base_seed(initial_seed, evaluation_index, n_episodes, mode="rolling"):
    """Return the first episode seed for one periodic evaluation.

    ``rolling`` matches Aritra's important behaviour: every evaluation point
    consumes a new, non-overlapping block of episode seeds. ``fixed`` is kept
    only so older experiments can deliberately reproduce their old protocol.
    """
    initial_seed = int(initial_seed)
    evaluation_index = int(evaluation_index)
    n_episodes = int(n_episodes)
    if evaluation_index < 0:
        raise ValueError("evaluation_index must be nonnegative.")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    if mode == "rolling":
        base_seed = initial_seed + evaluation_index * n_episodes
    elif mode == "fixed":
        base_seed = initial_seed
    else:
        raise ValueError("mode must be 'rolling' or 'fixed'.")
    if base_seed < 0 or base_seed + n_episodes - 1 > np.iinfo(np.uint32).max:
        raise ValueError("Evaluation seed block must fit in the uint32 range.")
    return base_seed


def evaluate_for_training(environment_config, agent_b, agent_r, n_episodes, base_seed, eval_batch_size=8):
    """Batched evaluation on separate environments; restore modes and RNG.

    Evaluation transitions are never added to the training rollout or counter.
    The caller chooses ``base_seed``; episode i uses ``base_seed + i``, exactly
    as when this ran episodes one at a time - see evaluate_policy_batched()'s
    docstring for why batching eval_batch_size episodes together does not
    change per-episode seeds, action sequences, or results.
    """
    import torch
    from ..evaluation.evaluate import evaluate_policy_batched

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for agent in (agent_b, agent_r) for module in agent.modules()]
    try:
        agent_b.eval()
        agent_r.eval()
        _episodes, summary = evaluate_policy_batched(
            environment_config=environment_config,
            agent_b=agent_b,
            agent_r=agent_r,
            n_episodes=n_episodes,
            base_seed=base_seed,
            eval_batch_size=eval_batch_size,
            deterministic=True,
            verbose_steps=False,
            verbose=False,
        )
        return summary
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
