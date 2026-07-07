# Hierarchical Reinforcement Learning with Pointer Networks for Scalable Container Yard Stacking

![Hierarchical RL Container Stacking](assets/hrl_best.png)

## Installation

Install dependencies using `pyproject.toml` and uv.


## Pretrained Models

Download pretrained models here: https://drive.google.com/drive/folders/1pjUggGEmB8jCeRyCO5HIxaIHGiA4wnli?usp=drive_link

 After downloading move the flat and hierarchical folders inside the already existing models folder

## Getting Started

Use `explore.ipynb` to get started and run inference with trained models.

## Training

Use `run.py` and its parameters to start training models. See the `bash/` folder for sample bash scripts used for training.

## File Overview

| File | Description |
|---|---|
| `run.py` | CLI entry point that parses args, builds the env config, and launches training via wandb. |
| `train.py` | Core training utilities: builds envs/models and runs MaskablePPO training with callbacks. |
| `utils.py` | Shared helpers: env factories, action-masking, and eval/checkpoint callbacks. |
| `envs/stack_gym.py` | Gymnasium environment simulating container yard stacking from a vessel. |
| `agents/hierarchical_rule_based_agent.py` | Rule-based baseline agent for the hierarchical (high/low level) action scheme. |
| `models/transformer_policy.py` | Transformer encoder + Pointer Network decoder policy for stack selection. |
| `models/joint_hierarchical_policy.py` | Joint autoregressive policy selecting bay then stack in a single action. |
