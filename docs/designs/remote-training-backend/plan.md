# Remote Training Backend and Training Configuration

**Status**: Design plan  
**Related issue**: [#14](https://github.com/LucaSforza/NNModelling/issues/14)  
**Date**: 2026-07-17

## 1. Objective

Add an optional FastAPI backend that receives a complete training job as one
JSON document, queues it in persistent Valkey, and executes it either:

- locally, on the machine where the backend is installed; or
- through Slurm, using a generated `sbatch` script.

The job JSON contains both the network and the complete Hydra training
configuration. The frontend talks directly to FastAPI for the interactive
training UI. The MCP server also exposes the same capabilities as an optional
agent-facing proxy, without duplicating job state.

The first implementation runs one job at a time, but the data model and
scheduler must already support multiple compute units and future concurrency.

## 2. Decisions

### 2.1 One self-contained job document

The frontend sends one document containing:

1. the compiled network (`NNTree`);
2. the Hydra training configuration;
3. the resource request;
4. the queue priority.

The existing diagram save format does not need to change in the first
iteration. The frontend assembles the job document when the user submits a
training job. The submitted document is stored unchanged as
`requested_config.json`.

Example:

```json
{
  "schema_version": 1,
  "network": {
    "format": "nntree",
    "value": {}
  },
  "training": {
    "seed": 42,
    "dataset": {
      "_target_": "dataset.mnist.MNISTDataset",
      "batch_size": 32,
      "num_workers": 4,
      "train_size": 0.8
    },
    "optimizer": {
      "_target_": "torch.optim.Adam",
      "lr": 0.001
    },
    "trainer": {
      "max_epochs": 20,
      "accelerator": "auto"
    },
    "wandb": {
      "project": "NeuralNetworks"
    },
    "early_stopping": {
      "patience": 3,
      "min_delta": 0.0
    },
    "overrides": [
      "trainer.max_epochs=10",
      "optimizer.lr=0.0001"
    ]
  },
  "resources": {
    "cpu": 8,
    "memory_gb": 32,
    "gpu": 1,
    "gpu_memory_gb": 16,
    "gpu_type": "A100",
    "node": null
  },
  "priority": 50
}
```

The dataset target is a Python import path represented as a string. The
canonical Hydra form is an object with `_target_`; a shorthand string may be
accepted by the API and normalized to this form.

`overrides` contains arbitrary Hydra override expressions. They are processed
through Hydra/OmegaConf APIs, not passed to a shell. The backend always runs
its own known Python entry points; the client cannot submit an arbitrary shell
command.

### 2.2 Resource requests

`resources` describes what a job needs. It is not a reference to a specific
machine:

```json
{
  "cpu": 8,
  "memory_gb": 32,
  "gpu": 1,
  "gpu_memory_gb": 16,
  "gpu_type": "A100",
  "node": null
}
```

It is used to:

- generate Slurm options such as CPU, memory, GPU and node constraints;
- reject impossible requests on a local CPU-only backend;
- select a compatible compute unit;
- preserve the future ability to run multiple jobs concurrently.

`node: null` means that any compatible node may be used. The first version
does not model individual nodes or individual GPUs as Valkey entities. A
future version can add that hierarchy without changing the job resource
request.

### 2.3 Priority and scheduling

There is one global queue.

Scheduling order is:

1. highest priority first;
2. FIFO for jobs with equal priority;
3. round robin between compatible compute units.

The initial maximum number of running jobs is one. There is no automatic
retry, `paused`, or `retrying` state. A job that fails is permanently marked
`failed` and retains all available diagnostic output.

The public job states are:

```text
queued → running → succeeded
                 ├→ failed
                 └→ cancelled
```

Submission priority is temporal: a user can submit a higher-priority job that
will be selected before older lower-priority jobs.

### 2.4 Heartbeat, but no lease

The system will implement heartbeat monitoring. It will not implement leases.

A worker or executor periodically reports that a running job is alive. The
heartbeat is diagnostic and is used to detect a stale or lost execution. It is
not a temporary ownership lock and it does not automatically release a
resource.

For local execution, the backend monitors the child process and records its
heartbeat. For Slurm execution, the backend stores the Slurm job ID, polls
Slurm, and the generated runner reports heartbeats when possible.

If the backend restarts:

- queued jobs are recovered from Valkey;
- Slurm jobs are reconciled through their Slurm job IDs;
- local jobs are checked through their stored process information;
- an execution that cannot be found is marked `failed` with a restart/lost
  process reason and its existing logs are preserved.

Cluster jobs are allowed to clean themselves up through Slurm. The backend
does not keep a resource lease after a job has ended.

## 3. Architecture

```text
┌──────────────────────┐       ┌──────────────────────┐
│ Svelte frontend       │       │ MCP server           │
│ Training mode         │       │ optional agent proxy │
└──────────┬───────────┘       └──────────┬───────────┘
           │ REST + SSE                   │ REST client
           └──────────────┬───────────────┘
                          ▼
┌──────────────────────┐
│ FastAPI backend       │
│                       │
│ Job API               │
│ Config service        │
│ Scheduler             │
│ Executor abstraction  │
└───────┬─────────┬────┘
        │         │
        │         ├── LocalExecutor → local Python process
        │         └── SlurmExecutor → sbatch / SSH + sbatch
        │
        ├── Valkey: queue, state, metadata, heartbeats
        └── Backend filesystem: configs, logs, checkpoints, results
```

Valkey is the persistent control plane. Large files are kept on the
filesystem where the backend is installed, with references and metadata in
Valkey.

## 4. Dataset discovery

The backend exposes the dataset classes available in its own Python
environment:

```text
GET /datasets
GET /datasets/{target}/schema
```

Discovery scans the trusted `dataset` package and identifies subclasses of
the project's `Dataset` base class. It returns:

- import target;
- class name and documentation;
- constructor parameters;
- parameter types and defaults when they can be inferred from the signature;
- optional backend-provided metadata.

Adding a new dataset class to the backend environment makes it available to
the frontend without hardcoding the class in Svelte. Dataset classes remain
backend-side code and are never uploaded by the client.

## 5. Hydra integration

The current converter exposes only a small number of high-level arguments,
while `main.py` consumes the complete composed Hydra configuration. The
backend therefore needs a library-oriented conversion path instead of
building the job around CLI flags.

Planned conversion flow:

```text
job JSON
  ├── network.value
  └── training
        ↓
validate and normalize
        ↓
generate Hydra config directory
        ↓
apply training.overrides
        ↓
save resolved_config.yaml
        ↓
run main.py with the generated config directory
```

The existing CLI remains supported for backwards compatibility. A new library
entry point will accept the job configuration directly, generate all Hydra
groups, and preserve arbitrary override expressions.

Each job directory contains at least:

```text
jobs/{job_id}/
├── requested_config.json
├── resolved_config.yaml
├── cfg/
├── stdout.log
├── stderr.log
├── checkpoints/
└── results/
```

The requested document is the reproducibility record of what the user sent;
the resolved YAML records what Hydra actually executed after defaults and
overrides.

## 6. FastAPI API

### 6.1 Job endpoints

```text
GET    /health
POST   /jobs
GET    /jobs
GET    /jobs/{job_id}
GET    /jobs/{job_id}/logs
GET    /jobs/{job_id}/events
DELETE /jobs/{job_id}
```

`POST /jobs` validates the complete job document, stores it, assigns a job
ID, and puts the job in the global queue. It does not execute client-provided
commands.

`GET /jobs/{job_id}/events` is an SSE endpoint for status changes, log lines,
metrics and the W&B URL when it becomes available. REST remains responsible
for submission, inspection and cancellation.

The MCP server may call the same API through a small set of thin tools, for
example:

```text
submit_training_job
get_training_job
get_training_job_logs
cancel_training_job
list_training_datasets
```

MCP reuses the backend job IDs, status model and error model. It does not own a
second queue, scheduler, job database, or copy of the training configuration.
The frontend can use FastAPI directly while an agent uses MCP for the same
jobs.

### 6.2 Error preservation

The backend captures:

- converter stdout and stderr;
- training stdout and stderr;
- executor submission errors;
- Slurm output and exit status;
- backend-side exception details.

The API returns a short error summary, while the complete output remains
available through the job logs and filesystem artifacts.

### 6.3 W&B

The training configuration can enable W&B. Once the run URL is known, it is
stored in the job metadata and emitted through the SSE stream. The frontend
shows an `Open in W&B` action that opens the URL in another browser tab.

## 7. Valkey data model

Valkey will run persistently with a Docker volume and append-only persistence.
The backend's job directory also needs a persistent volume because logs and
checkpoints are not stored in Valkey.

Initial keys:

```text
job:{id}                    → hash: status, priority, timestamps, executor data
job:{id}:config             → string: original job JSON
job:{id}:heartbeat          → string/hash: last heartbeat timestamp and details
job:{id}:events             → stream/list of recent event references
queue:jobs                  → sorted set or equivalent priority/FIFO queue
compute_units               → ordered set of unit IDs
compute_unit:{id}           → JSON/hash: executor and capacity configuration
scheduler:round_robin       → integer cursor
```

The queue claim and state transition must be atomic. A Valkey Lua script or a
transactional command sequence will ensure that the scheduler cannot claim
the same queued job twice when concurrency is later increased.

There are deliberately no `lease:{resource}:{job}` keys. Heartbeat data is
not a lease and is never used as a resource lock.

## 8. Compute units and executors

The first data model treats a compute unit as a logical execution profile,
not as every physical node or GPU.

Example local unit:

```json
{
  "id": "local",
  "kind": "local",
  "capacity": {
    "cpu": 16,
    "memory_gb": 64,
    "gpu": 0
  }
}
```

Example Slurm unit:

```json
{
  "id": "slurm-main",
  "kind": "slurm",
  "partition": "gpu",
  "account": "project-name",
  "ssh_host": "cluster",
  "capacity": {
    "gpu": 2
  }
}
```

For Slurm, the unit represents a configured cluster/partition profile. Slurm
chooses the physical node. Individual node/GPU entities are explicitly
deferred.

### 8.1 Common executor interface

```python
class Executor:
    def submit(self, job): ...
    def status(self, job): ...
    def cancel(self, job): ...
    def collect_logs(self, job): ...
```

`LocalExecutor` runs the known Python training entry point directly.

`SlurmExecutor`:

1. creates the job workspace;
2. generates the batch script from validated job data;
3. calls local `sbatch` when the backend is on the cluster;
4. calls `ssh <host> sbatch` when remote submission is configured;
5. sends the script through stdin rather than interpolating it into a shell
   command;
6. stores the Slurm Job ID;
7. polls `squeue`/`sacct`;
8. uses `scancel` for API cancellation.

The backend never accepts an arbitrary command or arbitrary batch script from
the frontend.

## 9. Frontend training mode

The frontend receives a separate training mode, independent of the node
sidebar.

Planned component responsibilities:

```text
TrainingSidebar.svelte
├── dataset selector and Hydra parameter form
├── optimizer configuration
├── trainer configuration
├── early-stopping and W&B configuration
├── arbitrary Hydra overrides editor
├── resource request form
├── priority input
└── submit button

TrainingJobsPanel.svelte
├── queued/running/completed jobs
├── live status and logs
├── error details
├── cancel action
└── Open in W&B action
```

Dataset forms are generated from `/datasets` metadata. Standard training
sections can initially use explicit frontend schemas and later be exposed
through the same backend schema mechanism.

On submission the frontend combines the current compiled NNTree with the
training configuration and sends one JSON job document to FastAPI.

## 10. Persistence and recovery

The local deployment will include:

- FastAPI container or process;
- Valkey container with a persistent volume;
- backend job-artifact directory with a persistent volume.

On backend startup, recovery performs:

1. reload Valkey metadata;
2. requeue jobs that were `queued`;
3. reconcile running local processes;
4. reconcile Slurm jobs using stored Job IDs;
5. mark unverifiable executions as `failed`;
6. preserve their output and failure reason.

This provides recovery after a backend restart without pretending that Valkey
itself owns the training process.

## 11. Implementation phases

### Phase 0 — Contracts and fixtures

- define the versioned job JSON schema;
- define public job states and event types;
- create representative CPU, local-GPU and Slurm fixtures;
- document resource units and priority semantics.

### Phase 1 — Hydra job conversion

- refactor conversion into a reusable Python API;
- accept the complete training configuration;
- support arbitrary Hydra overrides;
- generate `requested_config.json` and `resolved_config.yaml`;
- preserve the existing CLI and tests.

### Phase 2 — FastAPI core

- create backend package and Pydantic models;
- implement dataset discovery;
- implement job directories and artifact paths;
- implement job submission, inspection, logs and cancellation;
- add SSE events.

### Phase 3 — Valkey queue and persistence

- add persistent Valkey configuration;
- implement job metadata and queue keys;
- implement priority/FIFO ordering;
- implement round-robin compute-unit selection;
- implement atomic claim/state transitions;
- implement heartbeat recording;
- implement startup recovery.

### Phase 4 — Local executor

- execute one local job at a time;
- capture stdout/stderr;
- detect exit status;
- expose heartbeats;
- validate CPU-only resource requests;
- add CPU MNIST integration test.

### Phase 5 — Slurm executor

- add Slurm compute-unit configuration;
- generate safe batch scripts;
- support local `sbatch` and SSH submission;
- poll Slurm state and collect output;
- support cancellation via `scancel`;
- test with a mocked Slurm command layer.

### Phase 6 — Frontend training mode

- add the separate training sidebar;
- load dataset classes and parameters dynamically;
- add Hydra configuration sections;
- add arbitrary override editor;
- add resource and priority forms;
- submit complete job JSON;
- display live jobs, logs, errors and W&B links.

### Phase 7 — MCP integration

- add MCP tools that call the FastAPI job API;
- expose dataset discovery and job submission through MCP;
- expose status, logs, cancellation and W&B URL retrieval;
- reuse the backend's job IDs and error model;
- ensure MCP never keeps an independent copy of job state;
- add MCP unit tests with a mocked FastAPI client;
- rebuild and test `mcp-server` after the TypeScript changes.

### Phase 8 — Packaging and documentation

- add Dockerfile and persistent Valkey configuration;
- document local CPU installation;
- document cluster installation;
- document SSH and non-SSH Slurm modes;
- add end-to-end API tests;
- document direct frontend and MCP-mediated workflows;
- add deployment examples.

## 12. Acceptance criteria

The first complete version is accepted when:

- the frontend can submit a network and complete Hydra configuration as one
  JSON document;
- a dataset class installed only in the backend appears in `/datasets`;
- arbitrary Hydra overrides are applied without accepting shell commands;
- a local CPU backend can queue and run an MNIST job;
- Valkey preserves the queue and job metadata after restart;
- stdout, stderr and failure details are retained;
- priority and FIFO ordering work;
- compute units are selected round robin when multiple compatible units are
  configured;
- the same job API works with `LocalExecutor` and `SlurmExecutor`;
- Slurm submission works both through local `sbatch` and SSH;
- heartbeat information is visible for running jobs;
- the frontend can follow a job and open its W&B run in another tab;
- MCP can submit, inspect, retrieve logs for, and cancel the same jobs through
  FastAPI;
- existing conversion and Python tests remain green.
