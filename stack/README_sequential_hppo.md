# Sequential Multi-Agent Proximal Policy Optimization with Hierarchical Decision Decomposition for Scalable Container Yard Stacking

This repository implements a hierarchical multi-agent reinforcement learning framework for container-yard stacking.

Each placement decision is decomposed into two sequential stages:

1. **Bay selection** — `AgentB` selects a yard bay.
2. **Row selection** — `AgentR` selects a row/stack inside the selected bay.

The two actors use different observation spaces and are trained with sequential PPO updates. A centralized critic evaluates the global state and supports centralized training with decentralized decision-making.

The project is designed for experiments across multiple yard scales and supports parallel environment rollout (in-process or true multiprocessing), GPU inference, checkpointing, multi-seed training, plotting, and final evaluation.

---

## Method Overview

The decision process is:

```text
Global yard state
      |
      v
  AgentB
(Bay Policy)
      |
      | selected bay
      v
Selected-bay observation
      |
      v
  AgentR
(Row Policy)
      |
      | selected row
      v
Environment step
      |
      v
Centralized Critic
```

The training pipeline follows:

```text
Rollout collection
      |
      v
GAE computation
      |
      v
Centralized critic update
      |
      v
AgentB PPO update
      |
      v
Detached Bay sequence ratio
      |
      v
AgentR PPO update
```

The implementation separates the Bay and Row actors while keeping the critic centralized.

---

## Main Components

```text
stack/
├── agents/
│   ├── agent_b.py
│   └── agent_r.py
├── configs/
│   └── hierarchical_config.py
├── envs/
│   ├── stack_gym.py          (existing, unchanged)
│   └── hierarchical_envs/
│       ├── hierarchical_env.py
│       ├── bay_env.py
│       ├── row_env.py
│       ├── vec_hierarchical_env.py
│       └── subproc_vec_hierarchical_env.py
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── models/
│   ├── bay_policy.py
│   ├── row_policy.py
│   └── centralized_critic.py
├── training/
│   ├── advantage.py
│   ├── rollout_buffer.py
│   ├── sequential_trainer.py
│   ├── training_monitor.py
│   ├── training_plots.py
│   └── profiling.py
├── run_sequential_hppo.py
└── run_three_seeds.py
```

`vec_hierarchical_env.py` (in-process) and `subproc_vec_hierarchical_env.py` (true multiprocessing) each run several independent `HierarchicalEnv` copies together during rollout collection — see [Parallel Rollout](#parallel-rollout).

---

## Environment Scales

The main experiments use the `*_with_margin` configurations.

| Scale | CLI size | Vessel shape | Yard shape | Containers | Groups | Default training steps |
|---|---|---:|---:|---:|---:|---:|
| Small | `small_with_margin` | 3 × 3 × 3 | 3 × 4 × 3 | 27 | 3 | 1M |
| Medium | `medium_with_margin` | 4 × 4 × 4 | 4 × 5 × 4 | 64 | 4 | 2M |
| Large | `large_with_margin` | 6 × 6 × 5 | 6 × 7 × 5 | 180 | 6 | 10M |
| Massive | `large_v4_with_margin` | 10 × 8 × 5 | 10 × 9 × 5 | 400 | 10 | 28M |

Training profiles are defined in:

```text
stack/configs/hierarchical_config.py
```

Current profile defaults:

| Profile | Timesteps | Learning rate | Buffer | Batch | Epochs | Clip | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 1M | 1e-4 | 2048 | 64 | 3 | 0.20 | 0.15 |
| Medium | 2M | 1e-4 | 4096 | 128 | 3 | 0.20 | 0.15 |
| Large | 10M | 1e-4 | 8192 | 256 | 3 | 0.15 | 0.15 |
| Massive | 28M | 1e-4 | 8192 | 256 | 5 | 0.15 | 0.05 |

---

## Installation

### Local environment

Create and activate a Python environment:

```bash
conda create -n stowage-env python=3.12 -y
conda activate stowage-env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check that the project can be imported:

```bash
python -c "import stack; print('stack import OK')"
```

Check the available CLI arguments:

```bash
python -m stack.run_sequential_hppo --help
```

---

## Docker

Build the image from the repository root:

```bash
docker build -t multi-ppo:latest .
```

Test the image:

```bash
docker run --rm multi-ppo:latest
```

On a machine with an NVIDIA GPU:

```bash
docker run --rm \
  --gpus all \
  multi-ppo:latest \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Example training run through Docker:

```bash
docker run --rm \
  --gpus all \
  --shm-size=2g \
  -v "$(pwd)/models_trained:/app/models_trained" \
  multi-ppo:latest \
  python -u -m stack.run_sequential_hppo \
  --size small_with_margin \
  --seed 1 \
  --no_wandb \
  --save_model \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --save_dir ./models_trained/sequential_hppo_small/s1
```

---

# Experiments

Run all commands from the repository root.

## Small

### Seed 1

```bash
python -u -m stack.run_sequential_hppo \
  --size small_with_margin \
  --seed 1 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_small/s1
```

### Seed 2

```bash
python -u -m stack.run_sequential_hppo \
  --size small_with_margin \
  --seed 2 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_small/s2
```

### Seed 3

```bash
python -u -m stack.run_sequential_hppo \
  --size small_with_margin \
  --seed 3 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_small/s3
```

### Plot all three Small runs

```bash
python -m stack.run_three_seeds \
  --plot_only \
  --size small_with_margin \
  --seeds 1 2 3 \
  --output_root ./models_trained \
  --run_prefix sequential_hppo_small
```

---

## Medium

### Seed 1

```bash
python -u -m stack.run_sequential_hppo \
  --size medium_with_margin \
  --seed 1 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_medium/s1
```

### Seed 2

```bash
python -u -m stack.run_sequential_hppo \
  --size medium_with_margin \
  --seed 2 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_medium/s2
```

### Seed 3

```bash
python -u -m stack.run_sequential_hppo \
  --size medium_with_margin \
  --seed 3 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_medium/s3
```

### Plot all three Medium runs

```bash
python -m stack.run_three_seeds \
  --plot_only \
  --size medium_with_margin \
  --seeds 1 2 3 \
  --output_root ./models_trained \
  --run_prefix sequential_hppo_medium
```

---

## Large

### Seed 1

```bash
python -u -m stack.run_sequential_hppo \
  --size large_with_margin \
  --seed 1 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_large/s1
```

### Seed 2

```bash
python -u -m stack.run_sequential_hppo \
  --size large_with_margin \
  --seed 2 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_large/s2
```

### Seed 3

```bash
python -u -m stack.run_sequential_hppo \
  --size large_with_margin \
  --seed 3 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_large/s3
```

### Plot all three Large runs

```bash
python -m stack.run_three_seeds \
  --plot_only \
  --size large_with_margin \
  --seeds 1 2 3 \
  --output_root ./models_trained \
  --run_prefix sequential_hppo_large
```

---

## Massive

The Massive experiment uses:

```text
large_v4_with_margin
```

### Seed 1

```bash
python -u -m stack.run_sequential_hppo \
  --size large_v4_with_margin \
  --seed 1 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_massive/s1
```

### Seed 2

```bash
python -u -m stack.run_sequential_hppo \
  --size large_v4_with_margin \
  --seed 2 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_massive/s2
```

### Seed 3

```bash
python -u -m stack.run_sequential_hppo \
  --size large_v4_with_margin \
  --seed 3 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_massive/s3
```

### Plot all three Massive runs

```bash
python -m stack.run_three_seeds \
  --plot_only \
  --size large_v4_with_margin \
  --seeds 1 2 3 \
  --output_root ./models_trained \
  --run_prefix sequential_hppo_massive
```

---

# Evaluation

Example: evaluate the best checkpoint from Massive Seed 3 over 100 episodes.

```bash
python -m stack.evaluation.evaluate \
  --model_dir ./models_trained/sequential_hppo_massive/s3 \
  --prefix best \
  --episodes 100 \
  --seed 1000 \
  --raw_rewards \
  --device auto \
  --output_dir ./evaluation_results/sequential_hppo_massive/s3
```

The evaluation seed acts as the base seed for the evaluation episodes.

For example:

```text
Episode 0  -> seed 1000
Episode 1  -> seed 1001
Episode 2  -> seed 1002
...
Episode 99 -> seed 1099
```

`--raw_rewards` evaluates using the unnormalized and unclipped reward signal.

`--device auto` uses CUDA when available and otherwise falls back to CPU.

**Reward-normalization default mismatch.** Periodic evaluation *during* training (`stack.run_sequential_hppo`) defaults to raw/unnormalized rewards and only uses the training-time normalization if you pass `--eval_use_training_rewards`. This standalone `stack.evaluation.evaluate` script defaults the *other* way — it reuses whatever `reward_norm`/`reward_clip` the checkpoint was originally trained with (`True`/`True` for every size), unless you pass `--raw_rewards`. Running this script without `--raw_rewards` therefore reports numbers on a different scale than the `eval/mean_reward` curve logged during training for the same checkpoint — this can look like a regression when it's really just a default-flag mismatch between the two tools. Keep `--raw_rewards` on (as in the example above) to stay on the same scale as the training curve, unless you deliberately want the normalized/clipped scale.

---

## Checkpoints and Resume

When `--save_model` is enabled, checkpoints are stored under the selected `--save_dir`.

Typical saved files include:

```text
best_agent_b.pt
best_agent_r.pt
best_critic.pt
best_config.json
best_evaluation.json
latest_training_state.pt
training_logs/
```

The training pipeline supports resuming from the latest saved state. Resuming with a different `--num_envs` than the original run is fine — it's a rollout-collection runtime setting, not part of the compatibility check between a checkpoint and the current run.

To force a new run and ignore existing checkpoints:

```bash
python -u -m stack.run_sequential_hppo \
  --size small_with_margin \
  --seed 1 \
  --fresh \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_small/s1
```

---

## Parallel Rollout

Two flags control this: `--num_envs` (how many environment copies to run together) and `--vec_backend` (how they're run).

### `--num_envs`

Runs N copies of the environment together and batches their Agent B / Agent R / critic forward passes into one call instead of N separate calls. This matters most on GPU, where a single-environment forward pass wastes most of its time on overhead (data transfer, kernel launch) rather than actual computation — batching spreads that fixed overhead across N environments instead of paying it N times.

### `--vec_backend`

Controls *how* the N copies are executed:

| | `subproc` (used by every command in this README) | `sync` (the CLI's own built-in default) |
|---|---|---|
| How it runs | Each copy in its own OS process | All copies in a for-loop, one process |
| Real parallelism | Yes — copies execute concurrently | No — only the GPU forward pass is batched |
| Overhead | Some (inter-process communication) | None |
| Best for | The default choice | Very small `--num_envs`, limited CPU cores, or a sandbox that can't spawn processes |


For example:

```bash
python -u -m stack.run_sequential_hppo \
  --size large_v4_with_margin \
  --seed 1 \
  --device auto \
  --num_envs 16 \
  --vec_backend subproc \
  --no_wandb \
  --save_model \
  --save_dir ./models_trained/sequential_hppo_massive/s1
```


- `--num_envs` defaults to `1` — existing commands keep working unchanged unless you opt in.
- `--buffer_size` auto-rounds up to a multiple of `--num_envs`, so the last rollout of a run may collect a few extra transitions past `--timesteps`. Expected, nothing to configure.
- `--num_envs` only affects **training**. Evaluation (`stack.evaluation.evaluate`) has its own flag instead, `--eval_batch_size` (default 8) — same idea, different name.
- Each of the N copies gets its own seed (`--seed` + its index), so they run different trajectories instead of repeating the same one N times.
- Resuming with a different `--num_envs` than the original run is fine — it's a runtime setting, not part of the saved checkpoint.

The implementation is located in:

```text
stack/envs/hierarchical_envs/vec_hierarchical_env.py           # sync backend
stack/envs/hierarchical_envs/subproc_vec_hierarchical_env.py   # subproc backend
```

---

## Useful CLI Options

Some frequently used options are:

```text
--size
--seed
--device
--num_envs
--vec_backend
--timesteps
--training_profile
--buffer_size
--batch_size
--n_epochs
--learning_rate
--clip_range
--ent_coef
--eval_freq
--n_eval_episodes
--eval_batch_size
--eval_seed
--eval_use_training_rewards
--no_eval
--no_training_plots
--no_profiling
--save_model
--save_dir
--fresh
--no_wandb
```

For the complete interface:

```bash
python -m stack.run_sequential_hppo --help
```

---

## Output Structure

A typical multi-seed experiment produces:

```text
models_trained/
├── sequential_hppo_small/
│   ├── s1/
│   ├── s2/
│   └── s3/
├── sequential_hppo_medium/
│   ├── s1/
│   ├── s2/
│   └── s3/
├── sequential_hppo_large/
│   ├── s1/
│   ├── s2/
│   └── s3/
└── sequential_hppo_massive/
    ├── s1/
    ├── s2/
    └── s3/
```

Evaluation outputs are stored separately:

```text
evaluation_results/
└── ...
```

Each seed corresponds to an independent training run.
