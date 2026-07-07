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

Use `run.py` and its parameters to start training models. Example commands:


For training Hierarchical RL model on small environments with all constraints (40 feet and IMO) on :  
```
uv run run.py --size small_with_margin --eval_freq 25000 --timesteps 100000 --joint_hierarchical --use_transformer --save_model --save_dir ./models --save_filename ppo_joint_hier_small_imo40ft
```

For training Flat RL model on small environments with all constraints (40 feet and IMO) on :  
```
uv run run.py --size small_with_margin --eval_freq 25000 --timesteps 100000 --use_transformer --save_model --save_dir ./models --save_filename ppo_joint_flat_small_imo40ft
```

## Major Files Overview

| File | Description |
|---|---|
| `run.py` | CLI entry point that parses args, builds the env config, and launches training. |
| `train.py` | Core training utilities: builds envs/models and runs training with sb3 callbacks. |
| `utils.py` | Shared helpers: env factories, action-masking, and eval/checkpoint callbacks. |
| `envs/stack_gym.py` | Gymnasium environment simulating container yard stacking (vessel->yard). |
| `agents/hierarchical_rule_based_agent.py` | Heuristic baseline agents for yard stacking. |
| `models/transformer_policy.py` | Transformer encoder and Pointer Network decoder implementation.|
| `models/joint_hierarchical_policy.py` | Code for Hierarchical RL policy |
