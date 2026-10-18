"""
Sequential HPPO algorithm for the Separate-Observation Sequential Hierarchical
Policy (models/separate_obs_sequential_hierarchical_policy.py).

MaskablePPO whose train() updates three networks one after another: the
critic, then the Bay actor, then the Row actor (HAPPO-style; Kuba et al.,
"Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
ICLR 2022, arXiv:2109.11251).

The Row actor is updated after the Bay actor has already changed, so its
advantages are reweighted by M_B = pi_bay,new(b|s) / pi_bay,old(b|s) at the
bay b taken in the rollout. Since pi(b, r|s) = pi_bay(b|s) * pi_row(r|s, b),
M_B times the Row actor's own ratio is the ratio of the joint policy, so the
Row surrogate estimates the improvement of the joint policy rather than of
the Row actor in isolation. M_B is computed once after the Bay update and
kept fixed (no gradient, not clipped) during the Row update. Unlike HAPPO,
which draws a random agent order, the order is fixed (Bay before Row),
because the Row actor is conditioned on the chosen bay.

With share_bay_critic_encoder=True the critic is the Bay actor's value head,
on the Bay actor's encoder. A separate critic stage would then change the
Bay actor, without clipping, before the Bay stage, so the critic and the Bay
actor are instead updated together in one loss, as in MaskablePPO.train:
clipped surrogate + ent_coef * entropy loss + vf_coef * value loss. M_B is
computed after this stage, so it is still the ratio of the Bay actor after
all changes to its encoder, and the Row actor keeps its own encoder, so the
Row update cannot change the Bay actor.

Rollouts, GAE, logging, callbacks, saving and loading are MaskablePPO's.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class SequentialHPPO(MaskablePPO):
    """
    MaskablePPO with the sequential critic -> Bay actor -> Row actor update in
    train() (Bay actor and critic together -> Row actor with the policy's
    share_bay_critic_encoder); see the module docstring for why the Row
    actor's advantages are multiplied by M_B.
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
        # Optional: clip range for the value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # The whole rollout, with each stack action split into its bay and row
        rollout = next(self.rollout_buffer.get())
        observations, advantages, returns = rollout.observations, rollout.advantages, rollout.returns
        old_values = rollout.old_values
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

        if policy.share_bay_critic_encoder:
            # 1-2. Bay actor and critic together: the critic is the Bay actor's value
            # head, so its loss joins the Bay actor's, weighted by vf_coef as in
            # MaskablePPO.train, and target_kl also stops the critic's epochs.
            bay_logs = self._train_actor(
                policy.bay_actor, observations, bay_actions, bay_masks, old_bay_log_prob, advantages, clip_range,
                value_targets=(returns, old_values), clip_range_vf=clip_range_vf,
            )
            mean_value_loss = bay_logs.pop("value_loss")
        else:
            # 1. Critic: V(s) towards the GAE returns. The critic has its own loss and
            # parameters, so vf_coef (which weighs the value loss against the policy
            # loss in MaskablePPO's joint loss) is not used. target_kl does not stop
            # the critic's epochs; it applies to each actor in _train_actor.
            value_losses = []
            for _ in range(self.n_epochs):
                for idx in self._minibatch_indices(len(actions)):
                    values = policy.critic.predict_values(observations[idx]).flatten()
                    value_loss = self._value_loss(values, returns[idx], old_values[idx], clip_range_vf)
                    value_losses.append(value_loss.item())
                    self._optimizer_step(policy.critic, value_loss)
            mean_value_loss = np.mean(value_losses)

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
        self.logger.record("train/value_loss", mean_value_loss)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/bay_correction", bay_correction.mean().item())

    def _train_actor(
        self,
        actor: MaskableActorCriticPolicy,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        clip_range: float,
        advantage_scale: torch.Tensor | None = None,
        value_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
        clip_range_vf: float | None = None,
    ) -> dict[str, float]:
        """
        MaskablePPO.train's clipped-surrogate epochs for one actor (without the
        value loss); advantage_scale multiplies the normalized advantages. With
        value_targets = (returns, old_values), the actor's own value head is
        trained in the same loss with vf_coef, as in MaskablePPO.train, and the
        logs include value_loss. With target_kl set, this actor's epochs stop
        early when its approximate KL exceeds 1.5 * target_kl, as in
        MaskablePPO.train.
        """
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for idx in self._minibatch_indices(len(actions)):
                if value_targets is None:
                    distribution = actor.get_distribution(observations[idx], action_masks[idx])
                    log_prob = distribution.log_prob(actions[idx])
                    entropy = distribution.entropy()
                else:
                    # One encoder pass for the actor and its value head
                    values, log_prob, entropy = actor.evaluate_actions(
                        observations[idx], actions[idx], action_masks=action_masks[idx]
                    )

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
                entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss
                if value_targets is not None:
                    returns, old_values = value_targets
                    value_loss = self._value_loss(values.flatten(), returns[idx], old_values[idx], clip_range_vf)
                    value_losses.append(value_loss.item())
                    loss = loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob[idx]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self._optimizer_step(actor, loss)

            if not continue_training:
                break

        logs = {
            "entropy_loss": np.mean(entropy_losses),
            "policy_gradient_loss": np.mean(pg_losses),
            "approx_kl": np.mean(approx_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }
        if value_targets is not None:
            logs["value_loss"] = np.mean(value_losses)
        return logs

    def _value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        clip_range_vf: float | None,
    ) -> torch.Tensor:
        """MaskablePPO.train's value loss, with the optional value clipping."""
        if clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
        return F.mse_loss(returns, values_pred)

    def _minibatch_indices(self, n_samples: int):
        """One shuffled pass of minibatch indices, as RolloutBuffer.get."""
        indices = torch.as_tensor(np.random.permutation(n_samples), device=self.device)
        for start in range(0, n_samples, self.batch_size):
            yield indices[start : start + self.batch_size]

    def _optimizer_step(self, network: torch.nn.Module, loss: torch.Tensor) -> None:
        """MaskablePPO.train's optimization step; only ``network`` has gradients."""
        # set_to_none=True is what keeps the other networks frozen: they keep
        # grad=None, so Adam skips them and their momentum can't move them.
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
