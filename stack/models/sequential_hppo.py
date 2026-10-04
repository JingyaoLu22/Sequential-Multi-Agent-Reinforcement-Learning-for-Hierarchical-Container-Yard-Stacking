"""
Sequential HPPO algorithm for the Sequential Hierarchical Policy
(models/sequential_hierarchical_policy.py).

MaskablePPO with the sequential update of the README's "Sequential HPPO
Update" (HAPPO-style): train() updates the critic, then the Bay actor, then
the Row actor with the advantages M_B * A, where
M_B = pi_bay,new / pi_bay,old. Rollouts, GAE, logging, callbacks, saving
and loading are MaskablePPO's.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.utils import explained_variance

from models.transformer_bay_policy import MaskableBayTransformerPolicy


class SequentialHPPO(MaskablePPO):
    """
    MaskablePPO whose train() updates the critic, then the Bay actor, then
    the Row actor with the advantages M_B * A, where M_B = pi_bay,new / pi_bay,old
    (HAPPO). Rollouts, GAE, logging, callbacks, saving and loading are MaskablePPO's.
    """

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        policy = self.policy
        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        # The whole rollout, with each stack action split into its bay and row
        rollout = next(self.rollout_buffer.get())
        observations, advantages, returns = rollout.observations, rollout.advantages, rollout.returns
        actions = rollout.actions.long().flatten()
        action_masks = rollout.action_masks.bool()
        bay_actions = actions // policy.n_rows_per_bay
        row_actions = actions % policy.n_rows_per_bay
        bay_masks = policy.bay_masks(action_masks)
        row_observations, row_masks = policy.row_inputs(observations, action_masks, bay_actions)

        # log pi_old of each actor (the buffer holds their sum). The actors have not
        # changed since the rollout, which collect_rollouts ran in eval mode.
        policy.set_training_mode(False)
        with torch.no_grad():
            old_bay_log_prob = policy.bay_actor.get_distribution(observations, bay_masks).log_prob(bay_actions)
            old_row_log_prob = policy.row_actor.get_distribution(row_observations, row_masks).log_prob(row_actions)
        policy.set_training_mode(True)

        # 1. Critic: V(s) towards the GAE returns
        value_losses = []
        for _ in range(self.n_epochs):
            for idx in self._minibatch_indices(len(actions)):
                values = policy.critic.predict_values(observations[idx]).flatten()
                value_loss = F.mse_loss(returns[idx], values)
                value_losses.append(value_loss.item())
                self._optimizer_step(policy.critic, self.vf_coef * value_loss)

        # 2. Bay actor
        bay_logs = self._train_actor(
            policy.bay_actor, observations, bay_actions, bay_masks, old_bay_log_prob, advantages, clip_range
        )

        # 3. M_B from the updated Bay actor, which stays fixed during the Row update
        policy.bay_actor.set_training_mode(False)
        with torch.no_grad():
            new_bay_log_prob = policy.bay_actor.get_distribution(observations, bay_masks).log_prob(bay_actions)
        bay_correction = torch.exp(new_bay_log_prob - old_bay_log_prob)

        # 4. Row actor, with the advantages M_B * A
        row_logs = self._train_actor(
            policy.row_actor, row_observations, row_actions, row_masks, old_row_log_prob, advantages, clip_range,
            advantage_scale=bay_correction,
        )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs: MaskablePPO.train's train/ keys, with separate entropy,
        # policy-gradient, KL and clip-fraction entries for each actor
        for name, logs in (("bay", bay_logs), ("row", row_logs)):
            for key, value in logs.items():
                self.logger.record(f"train/{name}_{key}", value)
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/bay_correction", bay_correction.mean().item())

    def _train_actor(
        self,
        actor: MaskableBayTransformerPolicy,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        clip_range: float,
        advantage_scale: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        MaskablePPO.train's clipped-surrogate epochs for one actor (without the
        value loss); advantage_scale multiplies the normalized advantages.
        """
        entropy_losses, pg_losses, clip_fractions = [], [], []
        for _ in range(self.n_epochs):
            approx_kl_divs = []
            for idx in self._minibatch_indices(len(actions)):
                distribution = actor.get_distribution(observations[idx], action_masks[idx])
                log_prob = distribution.log_prob(actions[idx])

                # Normalize advantage
                minibatch_advantages = advantages[idx]
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(minibatch_advantages) > 1:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (
                        minibatch_advantages.std() + 1e-8
                    )
                if advantage_scale is not None:
                    minibatch_advantages = advantage_scale[idx] * minibatch_advantages

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - old_log_prob[idx])

                # clipped surrogate loss
                policy_loss_1 = minibatch_advantages * ratio
                policy_loss_2 = minibatch_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Entropy loss favor exploration
                entropy_loss = -torch.mean(distribution.entropy())
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob[idx]
                    approx_kl_divs.append(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy())

                self._optimizer_step(actor, policy_loss + self.ent_coef * entropy_loss)

        return {
            "entropy_loss": np.mean(entropy_losses),
            "policy_gradient_loss": np.mean(pg_losses),
            "approx_kl": np.mean(approx_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }

    def _minibatch_indices(self, n_samples: int):
        """One shuffled pass of minibatch indices, as RolloutBuffer.get."""
        indices = torch.as_tensor(np.random.permutation(n_samples), device=self.device)
        for start in range(0, n_samples, self.batch_size):
            yield indices[start : start + self.batch_size]

    def _optimizer_step(self, network: torch.nn.Module, loss: torch.Tensor) -> None:
        """MaskablePPO.train's optimization step; only ``network`` has gradients."""
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
