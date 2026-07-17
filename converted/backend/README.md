# NNModelling Training Backend

The backend receives a complete network and Hydra training configuration as
one JSON job document. It stores queue metadata in persistent Valkey and keeps
logs, resolved configurations, checkpoints and results in the backend job
directory.

## Local development

From `converted/`:

```bash
valkey-server backend/valkey.conf --dir /tmp/nnmodelling-valkey
PYTHONPATH=src NNM_VALKEY_URL=valkey://127.0.0.1:6379/0 \
  uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. The frontend Vite server
proxies `/api` to this address. Without `NNM_BACKEND_ARTIFACT_ROOT`, job
artifacts are stored in `converted/jobs/<job-id>/`, next to the installed
backend. Set that variable only when the artifacts must live on another
mounted filesystem, such as a Docker volume or shared Slurm filesystem.

The common local operations are also available through `just`:

```bash
just --list       # show all available recipes
just valkey       # start persistent Valkey in the foreground
just backend      # start FastAPI in the foreground
just health       # check the API health endpoint
just datasets     # list dataset classes discovered by the backend
just jobs         # list jobs stored in Valkey
just test         # run the remote-backend pytest suite
just docker-up    # build and start the Docker deployment
just docker-down  # stop Docker services, preserving volumes
```

Recipe details:

| Recipe | What it does |
|---|---|
| `just` / `just --list` | Lists the available backend operations. |
| `just valkey` | Creates `converted/valkey-data/` if needed and starts Valkey in the foreground with the repository persistence configuration. |
| `just backend` | Starts FastAPI/Uvicorn on `127.0.0.1:8000`, using the local Valkey instance and the default `converted/jobs/` artifact directory. |
| `just health` | Calls `GET /health` and fails if the API is unreachable. |
| `just datasets` | Calls `GET /datasets` and prints dataset classes available in the backend Python environment. |
| `just jobs` | Calls `GET /jobs` and prints the known training jobs and their states. |
| `just test` | Runs `src/tests/test_remote_backend.py` with `uv`; it uses `/tmp/nnmodelling-uv-cache` by default when `UV_CACHE_DIR` is not set. |
| `just docker-up` | Builds and starts the persistent Valkey/backend Docker Compose deployment in the foreground. |
| `just docker-down` | Stops the Docker Compose services without removing their volumes or stored artifacts. |

The host, API port, Valkey port, and uv cache can be overridden without
editing the file:

```bash
NNM_BACKEND_HOST=0.0.0.0 NNM_BACKEND_PORT=8001 just backend
NNM_VALKEY_PORT=6380 just valkey
UV_CACHE_DIR=/var/cache/uv just test
```

## Docker

```bash
cd converted/backend
docker compose up --build
```

The Valkey data volume and backend job-artifact volume are persistent across
container restarts. Valkey uses AOF plus periodic RDB snapshots.

## Slurm

Set `NNM_ENABLE_SLURM=1` and configure the Slurm unit with environment
variables such as:

```text
NNM_SLURM_PARTITION=gpu
NNM_SLURM_ACCOUNT=project-name
NNM_SLURM_SSH_HOST=cluster   # omit when sbatch runs locally
NNM_SLURM_PROJECT_DIR=/shared/NNModelling/converted
NNM_SLURM_CPU=32
NNM_SLURM_MEMORY_GB=128
NNM_SLURM_GPU=2
NNM_SLURM_GPU_TYPE=A100
```

The backend generates the batch script and invokes `sbatch` directly or over
SSH. Resource requests are converted to Slurm CPU, memory, GPU and node
options. The remote project and artifact directories must be visible to the
compute node when SSH submission is used.

## MCP

The MCP server remains a thin proxy. Set `NNM_BACKEND_URL` and use the remote
training MCP tools to submit, inspect, stream events for, or cancel the same
FastAPI jobs. Job state is never duplicated in MCP.
