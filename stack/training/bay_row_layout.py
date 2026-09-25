"""
Bay/row view of StackEnv's flat stack action space.

StackEnv exposes one Discrete(n_bays * n_rows) action and, for
observation_type="stack_features_v3", one feature token per stack. Both
are laid out bay-major / row-minor:

    action      = bay_idx * n_rows + row_idx
    observation = [bay 0 row 0, bay 0 row 1, ..., bay 1 row 0, ...]

so every input the two hierarchical agents need is a tensor view of the
original StackEnv observation and action mask - no environment wrapper:

    Agent B mask         action_mask.reshape(N, n_bays, n_rows).any(-1)
    Agent R mask         action_mask.reshape(N, n_bays, n_rows)[i, bay_i]
    Agent R observation  observation.reshape(N, n_bays, -1)[i, bay_i]
    StackEnv action      bay * n_rows + row

select_hierarchical_action() composes them into the one Bay -> Row
decision that training rollouts and evaluation both use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from ..envs.stack_gym import StackEnv
from ..models.pointer_actor import PointerActor


@dataclass(frozen=True)
class BayRowLayout:
    """
    Dimensions and spaces of the hierarchical split of one StackEnv.

    observation_space:
        StackEnv's global observation space, seen by Agent B and the
        centralized critic.

    row_observation_space:
        One bay's slice of it, seen by Agent R.
    """

    n_bays: int
    n_rows: int
    observation_space: spaces.Box
    row_observation_space: spaces.Box

    @classmethod
    def from_env(
        cls,
        env: Union[StackEnv, VecEnv],
    ) -> "BayRowLayout":
        """Read the layout from a StackEnv or a VecEnv of StackEnvs."""

        if isinstance(env, VecEnv):
            yard_shape = env.get_attr("yard_shape", indices=0)[0]
            observation_type = env.get_attr("observation_type", indices=0)[0]
        else:
            yard_shape = env.yard_shape
            observation_type = env.observation_type

        # Other observation types are not one token per stack, so
        # slicing them per bay would silently produce garbage.
        if observation_type != "stack_features_v3":
            raise ValueError(
                "Sequential HPPO requires "
                "observation_type='stack_features_v3', got "
                f"{observation_type!r}."
            )

        observation_space = env.observation_space

        if not (
            isinstance(observation_space, spaces.Box)
            and len(observation_space.shape) == 1
        ):
            raise TypeError(
                "Sequential HPPO requires a flat Box observation; set "
                "StackEnv's action_mask option to 'default'."
            )

        n_bays = int(yard_shape[0])
        n_rows = int(yard_shape[1])
        obs_dim = observation_space.shape[0]

        if obs_dim % (n_bays * n_rows):
            raise ValueError(
                f"Observation length {obs_dim} is not a whole number of "
                f"stack tokens for a {n_bays} x {n_rows} yard."
            )

        row_obs_dim = obs_dim // n_bays

        # stack_features_v3 bounds are identical for every token, so the
        # first bay's bounds apply to any bay.
        row_observation_space = spaces.Box(
            low=observation_space.low[:row_obs_dim],
            high=observation_space.high[:row_obs_dim],
            dtype=observation_space.dtype,
        )

        return cls(
            n_bays=n_bays,
            n_rows=n_rows,
            observation_space=observation_space,
            row_observation_space=row_observation_space,
        )

    def build_actors(
        self,
        **actor_kwargs,
    ) -> Tuple[PointerActor, PointerActor]:
        """Agent B's actor over bays and Agent R's actor over rows.

        Agent B scores each bay from its n_rows mean-pooled stack tokens;
        for Agent R every row of the selected bay is one token/action.
        """

        bay_actor = PointerActor(
            self.observation_space,
            n_stacks=self.n_bays * self.n_rows,
            tokens_per_action=self.n_rows,
            **actor_kwargs,
        )

        row_actor = PointerActor(
            self.row_observation_space,
            n_stacks=self.n_rows,
            **actor_kwargs,
        )

        return bay_actor, row_actor

    def bay_masks(
        self,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        """(N, n_bays * n_rows) -> (N, n_bays): a bay is valid iff any
        of its rows is."""

        return action_masks.reshape(-1, self.n_bays, self.n_rows).any(dim=-1)

    def row_inputs(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        bay_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Agent R's observation (N, row_obs_dim) and mask (N, n_rows)
        for the bay each environment's Agent B selected."""

        env_indices = torch.arange(
            bay_actions.shape[0],
            device=bay_actions.device,
        )

        row_observations = observations.reshape(
            -1, self.n_bays, self.row_observation_space.shape[0]
        )[env_indices, bay_actions]

        row_masks = action_masks.reshape(
            -1, self.n_bays, self.n_rows
        )[env_indices, bay_actions]

        return row_observations, row_masks

    def stack_actions(
        self,
        bay_actions: Union[torch.Tensor, np.ndarray],
        row_actions: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """StackEnv action for each (bay, row) pair, the inverse of
        StackEnv._action_to_bay_row()."""

        return bay_actions * self.n_rows + row_actions


@dataclass(frozen=True)
class HierarchicalAction:
    """One Bay -> Row decision for N environments.

    Every field except stack_actions is a (N, ...) tensor on the actors'
    device; stack_actions is the (N,) NumPy array StackEnv steps with.
    """

    bay_masks: torch.Tensor
    bay_actions: torch.Tensor
    bay_log_probs: torch.Tensor
    row_observations: torch.Tensor
    row_masks: torch.Tensor
    row_actions: torch.Tensor
    row_log_probs: torch.Tensor
    stack_actions: np.ndarray


@torch.no_grad()
def select_hierarchical_action(
    layout: BayRowLayout,
    bay_actor: PointerActor,
    row_actor: PointerActor,
    observations: torch.Tensor,
    action_masks: np.ndarray,
    deterministic: bool = False,
) -> HierarchicalAction:
    """
    global observation -> Bay mask -> Bay actor -> selected bay
    -> Row observation / Row mask -> Row actor -> StackEnv action

    ``observations`` (N, obs_dim) is StackEnv's observation on the
    actors' device; ``action_masks`` (N, n_bays * n_rows) is StackEnv's
    own action mask as returned by get_action_masks(). The environment
    does not step between the two decisions, and the only GPU -> CPU
    synchronization is the final copy of the StackEnv actions.
    """

    # Checked on the CPU copy, so it never waits for the GPU. Every bay
    # and row mask derived below then has a valid action too.
    if not action_masks.any(axis=-1).all():
        raise RuntimeError("An environment has no valid action at a decision point.")

    action_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=observations.device)

    bay_masks = layout.bay_masks(action_masks)
    bay_actions, bay_log_probs = bay_actor.act(observations, bay_masks, deterministic)

    row_observations, row_masks = layout.row_inputs(observations, action_masks, bay_actions)
    row_actions, row_log_probs = row_actor.act(row_observations, row_masks, deterministic)

    return HierarchicalAction(
        bay_masks=bay_masks,
        bay_actions=bay_actions,
        bay_log_probs=bay_log_probs,
        row_observations=row_observations,
        row_masks=row_masks,
        row_actions=row_actions,
        row_log_probs=row_log_probs,
        stack_actions=layout.stack_actions(bay_actions, row_actions).cpu().numpy(),
    )
