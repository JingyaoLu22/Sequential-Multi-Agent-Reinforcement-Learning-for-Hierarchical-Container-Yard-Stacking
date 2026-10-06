# syntax=docker/dockerfile:1

# Stack training image for a Linux x86_64 server with an NVIDIA GPU.
# PyTorch from PyPI bundles its own CUDA runtime, so no nvidia/cuda base image
# is needed; the host only needs the NVIDIA driver and NVIDIA Container Toolkit.
#
# Build from the repo root, since the dependencies live in the root
# pyproject.toml and uv.lock:
#   docker build -f stack/Dockerfile -t stowage-stack .
# What gets sent to the build is set by stack/Dockerfile.dockerignore.
#
# Train Sequential HPPO, also from the repo root:
#   mkdir -p outputs runs wandb
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -e WANDB_API_KEY \
#     -v "$PWD/outputs:/app/outputs" -v "$PWD/runs:/app/runs" -v "$PWD/wandb:/app/wandb" \
#     stowage-stack \
#     python stack/run.py --size small_with_margin --eval_freq 25000 --timesteps 100000 \
#       --sequential_hppo --save_model --save_dir /app/outputs \
#       --save_filename ppo_sequential_hppo_small_imo40ft
# See stack/docker.md for prerequisites and W&B options.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies in their own layer so code edits don't trigger a reinstall.
# pyproject.toml has no [build-system], so there is no project package to
# install after copying the code.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Only the Stack code goes into the image; stow/ is left out
COPY stack/ stack/

# HOME=/tmp gives wandb a writable config/cache dir when run with --user
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

CMD ["python", "stack/run.py", "--help"]
