"""
Learning-rate / ent_coef linear decay and target_kl early stopping in
SequentialPPOTrainer, and that a checkpoint from before these settings
existed still resumes.
"""

import pytest
import torch

from stack.training.checkpoint import resume_training_state, save_training_state
from stack.training.rollout_buffer import JointRolloutBuffer

from .common import algorithm, environment, training_system

ENVIRONMENT = environment(2)


def test_learning_rate_and_ent_coef_decay_linearly_to_their_final_values() -> None:
    config = algorithm(buffer_size=20, batch_size=10, n_epochs=1, total_timesteps=100,
                       learning_rate=1e-3, final_learning_rate=1e-4, ent_coef=0.1, final_ent_coef=0.0)
    env, trainer = training_system(ENVIRONMENT, config)
    env.close()
    optimizers = (trainer.bay_optimizer, trainer.row_optimizer, trainer.critic_optimizer)

    for step, lr, ent in ((0, 1e-3, 0.1), (50, 5.5e-4, 0.05), (100, 1e-4, 0.0), (200, 1e-4, 0.0)):
        trainer.total_environment_steps = step
        trainer.update_schedules()
        assert trainer.current_learning_rate == pytest.approx(lr)
        assert trainer.current_ent_coef == pytest.approx(ent)
        assert all(group["lr"] == pytest.approx(lr) for opt in optimizers for group in opt.param_groups)


def test_without_final_values_the_schedules_are_constant() -> None:
    env, trainer = training_system(ENVIRONMENT, algorithm(buffer_size=20, batch_size=10, n_epochs=1))
    trainer.total_environment_steps = trainer.total_timesteps // 2
    stats = trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 20, 2, "cpu"))
    env.close()
    assert stats["learning_rate"] == trainer.learning_rate
    assert stats["ent_coef"] == trainer.ent_coef


@pytest.mark.parametrize("target_kl, stopped", [(None, False), (1e-12, True)])
def test_target_kl_stops_each_actor_update_before_the_offending_step(target_kl, stopped) -> None:
    config = algorithm(buffer_size=40, batch_size=10, n_epochs=3, learning_rate=1e-2, target_kl=target_kl)
    env, trainer = training_system(ENVIRONMENT, config)
    stats = trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 40, 2, "cpu"))
    env.close()

    for name in ("bay", "row"):
        assert stats[f"{name}_early_stopped"] == float(stopped)
        # 4 minibatches x 3 epochs. The first minibatch has KL 0 (dropout
        # off), so a tiny target_kl still allows exactly one step.
        assert stats[f"{name}_minibatch_updates"] == (1.0 if stopped else 12.0)


def test_checkpoint_without_the_new_settings_still_resumes(tmp_path) -> None:
    config = algorithm(buffer_size=20, batch_size=10, n_epochs=1)
    env, trainer = training_system(ENVIRONMENT, config)
    old_config = config.to_dict()
    for key in ("final_learning_rate", "final_ent_coef", "target_kl"):
        del old_config[key]

    class OldConfig:  # a HierarchicalConfig as saved by the previous code
        def to_dict(self):
            return old_config

    save_training_state(tmp_path, trainer, env, ENVIRONMENT, OldConfig(), best_eval_reward=0.0)
    resume_training_state(tmp_path, trainer, env, ENVIRONMENT, config)
    env.close()
