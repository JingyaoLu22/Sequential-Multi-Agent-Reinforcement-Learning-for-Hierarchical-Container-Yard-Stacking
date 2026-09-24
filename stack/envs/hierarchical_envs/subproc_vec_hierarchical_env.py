"""
True multiprocessing vector wrapper around HierarchicalEnv.

Drop-in replacement for VecHierarchicalEnv exposing the identical
public interface (reset(), get_global_state(), get_bay_action_mask(),
get_row_decision_input(), step(), close(), plus the *_space /
inner_env / num_envs attributes) so SequentialPPOTrainer and
evaluate.py can use either backend interchangeably (see
VecHierarchicalEnv's docstring, which explains why the in-process
for-loop was originally judged sufficient).

Each of the N HierarchicalEnv copies runs in its OWN OS process,
stepped through a request/response protocol over a
multiprocessing.Pipe. Commands are sent to every worker first and only
then received back from every worker, so the N environments actually
execute concurrently instead of one-at-a-time (see reset()/
get_row_decision_input()/step() below) - this is the same fan-out/
fan-in pattern SB3's SubprocVecEnv uses.

Per hierarchical decision, collect_rollout() calls, in order:
get_global_state(), get_bay_action_mask(), get_row_decision_input(),
step(). The first two are always read immediately after a reset() or
step() already produced them (see HierarchicalEnv's StateSnapshot -
they are cheap re-reads of the current state, not new computation), so
this class caches them locally instead of paying an IPC round trip for
values a worker already sent back. That leaves exactly two real IPC
round trips per hierarchical decision - get_row_decision_input() and
step() - each one fanned out across all N worker processes at once.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hierarchical_env import HierarchicalEnv


def _worker(
    remote: Connection,
    parent_remote: Connection,
    config: Optional[Dict],
    render_mode: Optional[str],
) -> None:
    """
    Entry point run inside each worker process.

    Owns exactly one HierarchicalEnv instance for the lifetime of the
    process and answers commands sent over `remote` until "close".
    """

    parent_remote.close()

    env = HierarchicalEnv(
        config=config,
        render_mode=render_mode,
    )

    try:
        while True:

            cmd, data = remote.recv()

            if cmd == "reset":
                observation, _info = env.reset(seed=data)
                bay_mask = env.get_bay_action_mask()
                remote.send((observation, bay_mask))

            elif cmd == "row_decision":
                row_observation, row_action_mask = (
                    env.get_row_decision_input(data)
                )
                remote.send((row_observation, row_action_mask))

            elif cmd == "step":
                bay_idx, row_idx = data
                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(bay_idx, row_idx)

                if terminated or truncated:
                    next_state, _reset_info = env.reset()

                bay_mask = env.get_bay_action_mask()

                remote.send(
                    (
                        next_state,
                        bay_mask,
                        reward,
                        terminated,
                        truncated,
                        info,
                    )
                )

            elif cmd == "get_spaces":
                remote.send(
                    (
                        env.bay_observation_space,
                        env.bay_action_space,
                        env.row_observation_space,
                        env.row_action_space,
                        env.global_observation_space,
                    )
                )

            elif cmd == "close":
                remote.close()
                break

            else:
                raise RuntimeError(
                    f"Unknown SubprocVecHierarchicalEnv worker "
                    f"command: {cmd!r}"
                )

    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        env.close()


class SubprocVecHierarchicalEnv:
    """
    Batched collection of independent HierarchicalEnv instances, each
    stepped in its own OS process.

    Parameters
    ----------
    config : dict | None
        StackEnv configuration, identical for every inner environment.

    num_envs : int
        Number of parallel environment copies / worker processes.

    base_seed : int | None
        If given, each inner environment is reset once at construction
        time with seed=base_seed + i, mirroring VecHierarchicalEnv.

    render_mode : str | None
        Passed directly to every HierarchicalEnv (inside its worker
        process).

    start_method : str | None
        multiprocessing start method. Defaults to "spawn" on every
        platform (rather than following the OS default, which is
        "fork" on Linux) because the parent process typically already
        holds a CUDA context (PyTorch models on GPU) by the time this
        class is constructed, and forking a process with an
        initialized CUDA context is unsafe/unsupported.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        num_envs: int = 1,
        base_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        start_method: Optional[str] = None,
    ) -> None:

        if num_envs <= 0:
            raise ValueError(
                "num_envs must be positive."
            )

        self.num_envs = int(num_envs)
        self.closed = False

        # ==============================================================
        # Reference env for spaces / inner_env only.
        #
        # Never reset() or step()'d - it exists purely so callers can
        # read env.bay_observation_space / env.inner_env.yard_shape
        # etc. right after construction, exactly as VecHierarchicalEnv
        # allows via self.envs[0], without an extra IPC round trip to
        # a worker.
        # ==============================================================

        reference_env = HierarchicalEnv(
            config=config,
            render_mode=None,
        )

        self.bay_observation_space = (
            reference_env.bay_observation_space
        )
        self.bay_action_space = (
            reference_env.bay_action_space
        )
        self.row_observation_space = (
            reference_env.row_observation_space
        )
        self.row_action_space = (
            reference_env.row_action_space
        )
        self.global_observation_space = (
            reference_env.global_observation_space
        )
        self.inner_env = reference_env.inner_env

        # ==============================================================
        # Spawn one worker process per environment.
        # ==============================================================

        ctx = mp.get_context(start_method or "spawn")

        pipes = [ctx.Pipe() for _ in range(self.num_envs)]
        self.remotes: List[Connection] = [p[0] for p in pipes]
        work_remotes: List[Connection] = [p[1] for p in pipes]

        self.processes = []

        for work_remote, remote in zip(work_remotes, self.remotes):

            process = ctx.Process(
                target=_worker,
                args=(work_remote, remote, config, render_mode),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

            # Only the worker's end should stay open in this process.
            work_remote.close()

        # ==============================================================
        # Cached current-state snapshot, mirrors HierarchicalEnv's own
        # "must reset() before reading" contract - see get_global_state
        # / get_bay_action_mask below.
        # ==============================================================

        self._cached_global_state: Optional[np.ndarray] = None
        self._cached_bay_mask: Optional[np.ndarray] = None

        if base_seed is not None:
            self.reset(
                seeds=[
                    base_seed + i
                    for i in range(self.num_envs)
                ]
            )

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
        seeds: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Reset every inner environment.

        See VecHierarchicalEnv.reset() for the full parameter contract
        - behavior is identical here.
        """

        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(
                "seeds must contain exactly one seed per environment. "
                f"Expected {self.num_envs}, received {len(seeds)}."
            )

        for i, remote in enumerate(self.remotes):
            seed = seeds[i] if seeds is not None else None
            remote.send(("reset", seed))

        results = [remote.recv() for remote in self.remotes]
        observations, bay_masks = zip(*results)

        global_state = np.stack(observations).astype(np.float32)
        bay_mask = np.stack(bay_masks).astype(bool)

        self._cached_global_state = global_state
        self._cached_bay_mask = bay_mask

        return global_state

    # ==================================================================
    # Global / Agent B observation
    # ==================================================================

    def get_global_state(self) -> np.ndarray:
        """
        Stacked centralized-critic global state, shape
        (num_envs, obs_dim).

        Served from the local cache populated by the last reset()/
        step() call - see this module's docstring.
        """

        if self._cached_global_state is None:
            raise RuntimeError(
                "SubprocVecHierarchicalEnv.reset() must be called "
                "before reading any observation/mask."
            )

        return self._cached_global_state

    # ==================================================================
    # Agent B action mask
    # ==================================================================

    def get_bay_action_mask(self) -> np.ndarray:
        """
        Stacked bay validity mask, shape (num_envs, n_bays).

        Served from the local cache - see get_global_state().
        """

        if self._cached_bay_mask is None:
            raise RuntimeError(
                "SubprocVecHierarchicalEnv.reset() must be called "
                "before reading any observation/mask."
            )

        return self._cached_bay_mask

    # ==================================================================
    # Agent R observation + mask
    # ==================================================================

    def get_row_decision_input(
        self,
        bay_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-env row observation/mask for the bay each env's Agent B
        chose. Fanned out to every worker process and awaited, so the
        N environments compute this concurrently.
        """

        if len(bay_actions) != self.num_envs:
            raise ValueError(
                "bay_actions must contain exactly one action per "
                f"environment. Expected {self.num_envs}, "
                f"received {len(bay_actions)}."
            )

        for remote, bay_action in zip(self.remotes, bay_actions):
            remote.send(("row_decision", int(bay_action)))

        results = [remote.recv() for remote in self.remotes]
        row_observations, row_action_masks = zip(*results)

        return (
            np.stack(row_observations).astype(np.float32),
            np.stack(row_action_masks).astype(bool),
        )

    # ==================================================================
    # Environment transition
    # ==================================================================

    def step(
        self,
        bay_actions: np.ndarray,
        row_actions: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Dict],
    ]:
        """
        Step every inner environment (in its own process) and
        auto-reset any that finished. Same fan-out/fan-in pattern as
        get_row_decision_input() - see VecHierarchicalEnv.step() for
        the full return-value contract, which this matches exactly.
        """

        if (
            len(bay_actions) != self.num_envs
            or len(row_actions) != self.num_envs
        ):
            raise ValueError(
                "bay_actions/row_actions must contain exactly one "
                f"action per environment ({self.num_envs})."
            )

        for remote, bay_action, row_action in zip(
            self.remotes, bay_actions, row_actions
        ):
            remote.send(
                ("step", (int(bay_action), int(row_action)))
            )

        results = [remote.recv() for remote in self.remotes]

        next_states = []
        bay_masks = []
        # float64 for the same reason as VecHierarchicalEnv.step():
        # env.step() returns a native Python float, and every existing
        # call site re-casts this array before use anyway.
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict] = []

        for i, (
            next_state,
            bay_mask,
            reward,
            env_terminated,
            env_truncated,
            info,
        ) in enumerate(results):

            next_states.append(next_state)
            bay_masks.append(bay_mask)
            rewards[i] = reward
            terminated[i] = env_terminated
            truncated[i] = env_truncated
            infos.append(info)

        next_global_state = np.stack(next_states).astype(np.float32)
        next_bay_mask = np.stack(bay_masks).astype(bool)

        self._cached_global_state = next_global_state
        self._cached_bay_mask = next_bay_mask

        return (
            next_global_state,
            rewards,
            terminated,
            truncated,
            infos,
        )

    # ==================================================================
    # Closing
    # ==================================================================

    def close(self) -> None:
        """
        Ask every worker process to exit and wait for them to do so.
        """

        if self.closed:
            return

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass

        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()

        self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
