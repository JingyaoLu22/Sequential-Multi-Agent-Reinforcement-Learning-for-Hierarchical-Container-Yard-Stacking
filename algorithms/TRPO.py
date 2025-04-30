import copy
import warnings
from functools import partial
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.ppo_mask.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

SelfMaskableTRPO = TypeVar("SelfMaskableTRPO", bound="MaskableTRPO")


class MaskableTRPO(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization (TRPO)

    Paper: https://arxiv.org/abs/1502.05477
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    and Stable Baselines (TRPO from https://github.com/hill-a/stable-baselines)

    Introduction to TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size for the value function
    :param gamma: Discount factor
    :param cg_max_steps: maximum number of steps in the Conjugate Gradient algorithm
        for computing the Hessian vector product
    :param cg_damping: damping in the Hessian vector product computation
    :param line_search_shrinking_factor: step-size reduction factor for the line-search
        (i.e., ``theta_new = theta + alpha^i * step``)
    :param line_search_max_iter: maximum number of iteration
        for the backtracking line-search
    :param n_critic_updates: number of critic updates per policy update
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param normalize_advantage: Whether to normalize or not the advantage
    :param target_kl: Target Kullback-Leibler divergence between updates.
        Should be small for stability. Values like 0.01, 0.05.
    :param sub_sampling_factor: Sub-sample the batch to make computation faster
        see p40-42 of John Schulman thesis http://joschu.net/docs/thesis.pdf
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`trpo_policies`
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: MaskableActorCriticPolicy
    rollout_buffer: MaskableRolloutBuffer

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,  # entropy bonus is not used by TRPO
            vf_coef=0.0,  # value function is optimized separately
            max_grad_norm=0.0,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            if normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        # Conjugate gradients parameters
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        # Backtracking line search parameters
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        self.target_kl = target_kl
        self.n_critic_updates = n_critic_updates
        self.sub_sampling_factor = sub_sampling_factor

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = MaskableDictRolloutBuffer
            else:
                self.rollout_buffer_class = MaskableRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        if not isinstance(self.policy, MaskableActorCriticPolicy):
            raise ValueError("Policy must subclass MaskableActorCriticPolicy")

        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) -> tuple[list[nn.Parameter], th.Tensor, th.Tensor, list[tuple[int, ...]]]:
        """
        Compute actor gradients for kl div and surrogate objectives.

        :param kl_div: The KL divergence objective
        :param policy_objective: The surrogate objective ("classic" policy gradient)
        :return: List of actor params, gradients and gradients shape.
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # The policy objective is also called surrogate objective
        policy_objective_gradients_list = []
        # Contains the gradients of the KL divergence
        grad_kl_list = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters
        grad_shape: list[tuple[int, ...]] = []
        # Contains the parameters which have non-zeros KL divergence gradients
        # The list is used during the line-search to apply the step to each parameters
        actor_params: list[nn.Parameter] = []

        for name, param in self.policy.named_parameters():
            # Skip parameters related to value function based on name
            # this work for built-in policies only (not custom ones)
            if "value" in name:
                continue

            # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            # If the gradient is not zero (not None), we store the parameter in the actor_params list
            # and add the gradient and its shape to grad_kl and grad_shape respectively
            if kl_param_grad is not None:
                # If the parameter impacts the KL divergence (i.e. the policy)
                # we compute the gradient of the policy objective w.r.t to the parameter
                # this avoids computing the gradient if it's not going to be used in the conjugate gradient step
                policy_objective_grad, *_ = th.autograd.grad(policy_objective, param, retain_graph=True, only_inputs=True)

                grad_shape.append(kl_param_grad.shape)
                grad_kl_list.append(kl_param_grad.reshape(-1))
                policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)

        # Gradients are concatenated before the conjugate gradient step
        policy_objective_gradients = th.cat(policy_objective_gradients_list)
        grad_kl = th.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    def predict(  # type: ignore[override]
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # old values, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    None,  # type: ignore[arg-type]  # returns, not used here
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()
            action_masks = rollout_data.action_masks

            with th.no_grad():
                # Note: is copy enough, no need for deepcopy?
                # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
                # directly to avoid PyTorch errors.
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

            distribution = self.policy.get_distribution(rollout_data.observations)
            
            if action_masks is not None:
                distribution.apply_masking(action_masks)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean()

            # KL divergence
            kl_div = kl_divergence(distribution, old_distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            # Maximal step length
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= th.matmul(
                search_direction, hessian_vector_product_fn(search_direction, retain_graph=False)
            )
            line_search_max_step_size = th.sqrt(line_search_max_step_size)  # type: ignore[assignment, arg-type]

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    # Applying the scaled step direction
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx : (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(rollout_data.observations)
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()

                    # New KL-divergence
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    # Constraint criteria:
                    # we need to improve the surrogate policy objective
                    # while being close enough (in term of kl div) to the old policy
                    if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(actor_params, original_actor_params):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def hessian_vector_product(
        self, params: list[nn.Parameter], grad_kl: th.Tensor, vector: th.Tensor, retain_graph: bool = True
    ) -> th.Tensor:
        """
        Computes the matrix-vector product with the Fisher information matrix.

        :param params: list of parameters used to compute the Hessian
        :param grad_kl: flattened gradient of the KL divergence between the old and new policy
        :param vector: vector to compute the dot product the hessian-vector dot product with
        :param retain_graph: if True, the graph will be kept after computing the Hessian
        :return: Hessian-vector dot product (with damping)
        """
        jacobian_vector_product = (grad_kl * vector).sum()
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def learn(
        self: SelfMaskableTRPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TRPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskableTRPO:
        iteration = 0
        
        total_timesteps, callback = self._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, use_masking=use_masking)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self
