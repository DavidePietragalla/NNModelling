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
