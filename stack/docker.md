# Docker (Stack)

Build the Stack training image on a Linux GPU server and train Sequential HPPO
in a container.

## Prerequisites (on the server)

- Docker 24+
- NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Driver new enough for CUDA 13**: `uv.lock` pins `torch==2.14.1`, which on Linux
  ships with CUDA 13. Run `nvidia-smi` and check the `CUDA Version` in the top
  right: it must be **13.0 or higher** (driver 580+).
- A W&B API key (optional; pass `--no_wandb` to train without it)

## Build

Build on the server, **from the repo root** (not from `stack/`), because the
build needs the root `pyproject.toml` and `uv.lock`:

```bash
git clone https://github.com/loadmaster-ai/stowage_env.git
cd stowage_env
docker build -f stack/Dockerfile -t stowage-stack .
```

Check that the container sees the GPU:

```bash
docker run --rm --gpus all stowage-stack \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

This should print `True` and the CUDA version.

After changing code, run the same `docker build` command again. The dependency
layer is cached, so only the code layer is rebuilt.

## Train Sequential HPPO

Run from the repo root:

```bash
mkdir -p outputs runs wandb
export WANDB_API_KEY=<your key>

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e WANDB_API_KEY \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/runs:/app/runs" \
  -v "$PWD/wandb:/app/wandb" \
  stowage-stack \
  python stack/run.py --size small_with_margin --eval_freq 25000 --timesteps 100000 \
    --sequential_hppo --save_model --save_dir /app/outputs \
    --save_filename ppo_sequential_hppo_small_imo40ft
```

This is the README's Sequential HPPO command with two changes:

- `python` instead of `uv run`.
- `--save_dir /app/outputs` instead of `./stack/models`. `stack/models` is not
  mounted, so models saved there are lost when the container exits.

Run `mkdir -p` first. If the folders don't exist, Docker creates them as root
and the container (running as you via `--user`) cannot write to them.

| Container path | Contents | Host folder |
|---|---|---|
| `/app/outputs` | Best model, checkpoints, final model (`<save_filename>/...`) | `./outputs` |
| `/app/runs` | TensorBoard logs (only written when W&B is on) | `./runs` |
| `/app/wandb` | Local W&B run files | `./wandb` |

### W&B options

- No W&B: drop `-e WANDB_API_KEY` and add `--no_wandb` to the `run.py` arguments.
- Server without internet: add `-e WANDB_MODE=offline`, then upload later from a
  machine with internet using `wandb sync wandb/offline-run-*`.

### Long runs

Run in the background so the job survives closing your SSH session: replace
`--rm` with `-d --name sequential-hppo`, then follow the output with
`docker logs -f sequential-hppo`. To pin a specific GPU, use
`--gpus '"device=1"'` instead of `--gpus all`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails with `"/uv.lock": not found` or `failed to read dockerfile` | Run `docker build` from the repo root with `-f stack/Dockerfile`, not from inside `stack/` |
| `Using device: cpu` | Add `--gpus all`; check that the NVIDIA Container Toolkit is installed and the driver is new enough |
| `PermissionError` on `outputs/`, `runs/` or `wandb/` | Create the folders with `mkdir -p` before `docker run` |
| W&B asks you to log in | `WANDB_API_KEY` is not exported in your shell, or add `--no_wandb` |
