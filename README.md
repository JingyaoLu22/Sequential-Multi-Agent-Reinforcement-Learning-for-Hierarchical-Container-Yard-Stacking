# Hierarchical Reinforcement Learning with Pointer Networks for Scalable Container Yard Stacking

![Hierarchical RL Container Stacking](stack/assets/hrl_best.png)

This repo contains two separate, independent pipelines, split into their own
top-level directories:

- **`stack/`** — the container-**yard**-stacking problem (this thesis's focus):
  hierarchical RL with pointer networks. Documented below.
- **`stow/`** — the original vessel/ship stowage-planning problem (loading
  containers onto a ship). See [Stow](#stow) below.

## Installation

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`:

```
uv sync            # training runtime + dev tools (Jupyter, pytest)
uv sync --no-dev   # training runtime only
```

`pip install -e .` also works (versions then come from the `pyproject.toml` ranges, not the lock). Both make the `stack` package importable everywhere.

Docker builds the same locked runtime; choose what to run at `docker run`:

```
docker build -t multi-ppo .
docker run --rm --gpus all -v "$(pwd)/models_trained:/app/models_trained" multi-ppo \
    python -m stack.run_sequential_hppo --size small_with_margin --save_model --save_dir models_trained/small/s1
```

## Stack

### Pretrained Models

Download pretrained models here: https://drive.google.com/drive/folders/1pjUggGEmB8jCeRyCO5HIxaIHGiA4wnli?usp=drive_link

After downloading move the flat and hierarchical folders inside the already existing `stack/models` folder.

### Getting Started

Use `stack/explore.ipynb` to get started and run inference with trained models.

### Training

Use `stack/run.py` and its parameters to start training models. It is part of the `stack` package, so run it as a module (`python -m stack.run`) from the repo root. Example commands:

For training Hierarchical RL model on small environments with all constraints (40 feet and IMO) on :
```
uv run python -m stack.run --size small_with_margin --eval_freq 25000 --timesteps 100000 --joint_hierarchical --use_transformer --save_model --save_dir ./stack/models --save_filename ppo_joint_hier_small_imo40ft
```

For training Flat RL model on small environments with all constraints (40 feet and IMO) on :
```
uv run python -m stack.run --size small_with_margin --eval_freq 25000 --timesteps 100000 --use_transformer --save_model --save_dir ./stack/models --save_filename ppo_joint_flat_small_imo40ft
```

### Major Files Overview

| File | Description |
|---|---|
| `stack/run.py` | CLI entry point that parses args, builds the env config, and launches training. |
| `stack/train.py` | Core training utilities: builds envs/models and runs training with sb3 callbacks. |
| `stack/utils.py` | Shared helpers: env factories, action-masking, and eval/checkpoint callbacks. |
| `stack/configs/environments.py` | Environment sizes and `set_config()`: the one environment/reward definition every entry point imports. |
| `stack/envs/stack_gym.py` | Gymnasium environment simulating container yard stacking (vessel->yard). |
| `stack/agents/hierarchical_rule_based_agent.py` | Heuristic baseline agents for yard stacking. |
| `stack/models/transformer_policy.py` | Transformer encoder and Pointer Network decoder implementation.|
| `stack/models/joint_hierarchical_policy.py` | Code for Hierarchical RL policy |

### Sequential HPPO

Sequential multi-agent PPO with a hierarchical decision: every container placement is a bay decision followed by a row decision inside that bay, trained with HAPPO-style sequential updates and one centralized critic.

#### Method

- **Bay actor** (Agent B) sees the whole yard and picks a bay.
- **Row actor** (Agent R) sees only the chosen bay and picks a row. StackEnv then takes one step with action `bay * n_rows + row`.
- **Centralized critic** estimates `V(s)` from the whole yard; its GAE advantage `A` is shared by both actors.
- **Sequential PPO update**: critic → Bay actor with `A` → correction ratio `M_B = π_B,new / π_B,old` (detached) → Row actor with `M_B · A`.

#### Architecture

```text
StackEnv observation + action mask        (plain StackEnv in an SB3 DummyVecEnv / SubprocVecEnv)
        |
        |-- bay mask = mask.any over rows ------> Bay actor ---- bay
        |                                                          |
        |-- rows of the chosen bay (obs + mask) -> Row actor ---- row
        |                                                          |
        |                                    StackEnv.step(bay * n_rows + row)
        |
        '---------------------------------------> Centralized critic -> GAE -> sequential PPO update
```

#### Components

| Part | Where | What |
|---|---|---|
| Actor | `stack/models/pointer_actor.py` | One Transformer + pointer-network actor class for both bays and rows (built on `TransformerFeaturesExtractor` / `PointerDecoder`). |
| Critic | `stack/models/centralized_critic.py` | State-value critic over the whole yard. |
| Trainer | `stack/training/sequential_trainer.py` | Rollout collection, GAE, critic update, one PPO actor update used for both actors. |
| Decision | `stack/training/bay_row_layout.py` | Bay/row views of StackEnv's observation and mask, and the Bay → Row decision shared by training and evaluation. |
| Buffer | `stack/training/rollout_buffer.py` | On-device `(steps, envs)` rollout storage with vectorized GAE. |
| Checkpoints | `stack/training/checkpoint.py` | Model exports, `latest_training_state.pt`, resume. |
| Evaluation | `stack/evaluation/evaluate.py` | Batched deterministic evaluation of saved actors. |
| Config | `stack/configs/` | Environment sizes (`environments.py`) and per-size PPO training profiles (`hierarchical_config.py`). |

#### Training

```
python -m stack.run_sequential_hppo --size small_with_margin --seed 1 \
    --num_envs 16 --vec_backend subproc --save_model --save_dir ./models_trained/sequential_hppo_small/s1
```

The size selects the environment and its training profile (Small / Medium / Large / Massive); every hyperparameter can be overridden from the CLI. The environment, including `reward_norm` / `reward_clip`, comes from the same `set_config()` as `stack/run.py`, so both pipelines train on the same rewards for the same `--size`; change them only explicitly with `--[no-]reward_norm` / `--[no-]reward_clip`. The run directory receives `best_*` / `final_*` models and `training_logs/` (evaluation curve, `evaluations.csv`, `rollouts.csv`); W&B gets every statistic unless `--no_wandb`. `--profile` adds per-phase timings (it synchronizes CUDA, so it is off by default).

Plot several seeds together with `python -m stack.training_plots --hrl_dirs <run>/s1 <run>/s2 <run>/s3`, and add them to the Flat / HRL / rule-based comparison with `stack/plots.py`'s `evaluate(..., sequential_model_dir=<run>)`; run the comparison from the repo root with `python -m stack.plots`.

#### Evaluation

```
python -m stack.evaluation.evaluate --model_dir ./models_trained/sequential_hppo_small/s1 \
    --prefix best --episodes 100 --seed 1000 --output_dir ./evaluation_results/small_s1
```

Episode `i` uses seed `--seed + i`. Training-time evaluation and this script use the same reward scale (raw rewards by default, `--eval_use_training_rewards` for the training scale in both), so the numbers are directly comparable.

#### Resume

Re-run the same training command: if `--save_dir` contains `latest_training_state.pt`, training continues from it, including optimizer states, step counters, best evaluation reward and RNG states. `--timesteps`, the evaluation and checkpoint frequencies and `--num_envs` may change; a checkpoint with a different environment or model/training settings is refused, naming the mismatching keys. `--fresh` starts over and ignores the checkpoint.

All options: `python -m stack.run_sequential_hppo --help` and `python -m stack.evaluation.evaluate --help`.

## Stow

The vessel/ship stowage-planning problem: loading containers onto a ship, as opposed
to stacking them in a yard. This code predates the Stack work above and is kept
separate under `stow/`.

| File | Description |
|---|---|
| `stow/main.py` | Small enumeration-solver demo on `stow/envs/stowage_gym.py`. |
| `stow/train_spge.py` | Training entry point for the Stow RL baselines (uses `stow/hpo_spge.py` for algo/env registries and config loading). |
| `stow/hpo_spge.py` | Hyperparameter optimization script; also defines the algorithm/environment registries and YAML config loader used by `train_spge.py`. |
| `stow/ilp_solver.py` / `stow/ilp_solver_mc.py` | ILP-based stowage solver baselines. |
| `stow/algorithms/` | Custom SB3-compatible algorithm implementations (A2C, DQN, QRDQN, TRPO) used as Stow baselines. |
| `stow/envs/` | Stowage-planning Gymnasium/PettingZoo environments. |
| `stow/configs/` | YAML configs consumed by `train_spge.py` / `hpo_spge.py`. |

Example training command (run from the repo root):
```
uv run stow/train_spge.py --config spge.ppo.yaml
```
