# Sequential HPPO / Stack training image.
#
#   docker build -t multi-ppo .
#   docker run --rm --gpus all -v "$(pwd)/models_trained:/app/models_trained" multi-ppo \
#       python -m stack.run_sequential_hppo --size small_with_margin --num_envs 16 \
#       --vec_backend subproc --save_model --save_dir models_trained/small/s1
#
# Every Python dependency comes from uv.lock (torch 2.14.0 built for CUDA 12.6),
# without the dev group (Jupyter, ipykernel, pytest).

FROM python:3.12.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# pygame (via gymnasium rendering) needs the SDL2 runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY stack ./stack
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# No default command: pass the trainer to run to `docker run` (see above).
