from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from stable_baselines3.common.type_aliases import TensorDict

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
from gymnasium import spaces
from typing import Any, Optional, Union, NamedTuple
import torch as th
import warnings

class MaskableReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor

class MaskableDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor


class MaskableReplayBuffer(ReplayBuffer):
    action_masks: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)        

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.action_masks[batch_inds, env_indices, :].reshape(-1, self.mask_dims),
        )
        return MaskableReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> MaskableReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
class MaskableDictReplayBuffer(DictReplayBuffer):
    action_masks: np.ndarray
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskableDictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return MaskableDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            action_masks=self.to_torch(self.action_masks[batch_inds, env_indices, :].reshape(-1, self.mask_dims)),
        )    

