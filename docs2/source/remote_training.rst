Remote Training Backend
=======================

NNModelling includes an optional remote-training backend. The browser sends a
complete training request to FastAPI, while Valkey stores job state and the
backend filesystem stores configurations, logs, checkpoints, and results.

This page documents the features implemented in the initial version. The final
section lists only planned future features.

Architecture
------------

.. code-block:: text

   Browser
      │ REST + SSE
      ▼
   FastAPI backend ─── Valkey (persistent queue and job state)
      │
      ├── LocalExecutor ── Python/Lightning process
      └── SlurmExecutor ── generated sbatch script
      │
      └── converted/jobs/{job_id}/ (configs, logs, checkpoints)

The frontend uses ``TrainingSidebar.svelte`` as a separate mode from the node
property sidebar. The backend owns scheduling and state; the optional MCP
training tools are only an HTTP proxy and do not maintain another queue.

Job JSON
--------

``POST /jobs`` receives one JSON document. It contains the compiled network and
the complete Hydra configuration:

.. code-block:: json

   {
     "schema_version": 1,
     "network": {"format": "nntree", "value": {}},
     "training": {
       "seed": 42,
       "dataset": {
         "_target_": "dataset.mnist.MNISTDataset",
         "batch_size": 32,
         "num_workers": 4,
         "train_size": 0.8
       },
       "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
       "trainer": {"max_epochs": 20, "accelerator": "auto"},
       "wandb": {"project": "NeuralNetworks", "mode": "online"},
       "early_stopping": {"patience": 3, "min_delta": 0.0},
       "overrides": ["trainer.max_epochs=10"]
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

The dataset is a Python import path. The backend normalizes the shorthand
string to Hydra's ``_target_`` form. Hydra overrides are interpreted through
Hydra/OmegaConf; clients cannot send shell commands.

Resources and priority
~~~~~~~~~~~~~~~~~~~~~~

The resource request describes CPU, memory, GPU count, GPU memory, GPU type,
and an optional node selector. ``priority`` is an integer: larger values are
selected first, and equal priorities are FIFO.

The initial scheduler runs one job at a time. Compatible executor profiles are
selected round robin. A job has one of these states:

.. code-block:: text

   queued → running → succeeded
                    ├→ failed
                    └→ cancelled

There are no retry or lease states. Heartbeats record executor activity and
timestamps for running jobs. Errors and stdout/stderr remain available after a
failure.

API
---

.. list-table::
   :header-rows: 1

   * - Method
     - Endpoint
     - Purpose
   * - GET
     - ``/health``
     - Backend health check
   * - GET
     - ``/datasets``
     - Dataset classes installed in the backend environment
   * - GET
     - ``/compute-units``
     - Configured local and Slurm executor profiles
   * - POST
     - ``/jobs``
     - Validate, persist, and enqueue a job
   * - GET
     - ``/jobs`` and ``/jobs/{id}``
     - List or inspect job state
   * - GET
     - ``/jobs/{id}/logs``
     - Retrieve captured stdout and stderr
   * - GET
     - ``/jobs/{id}/events``
     - Stream lifecycle events with SSE
   * - DELETE
     - ``/jobs/{id}``
     - Cancel a queued or running job

Dataset discovery
~~~~~~~~~~~~~~~~~

``GET /datasets`` inspects the trusted ``dataset`` Python package installed
with the backend. It returns each class's target string, documentation, and
constructor parameters. Installing a dataset class on a training machine is
enough to make it available to the frontend.

Valkey persistence
------------------

Valkey is the persistent control plane. The implementation uses:

.. code-block:: text

   job:{id}                  JSON job metadata
   job:{id}:events           lifecycle event stream
   queue:priorities          priority index
   queue:priority:{value}    FIFO bucket for one priority

Queue claiming is atomic through a Lua operation. The Valkey deployment uses
AOF and periodic RDB snapshots. Job files are kept outside Valkey on the
backend filesystem.

Artifacts and installation
--------------------------

When ``NNM_BACKEND_ARTIFACT_ROOT`` is not set, artifacts are stored next to the
backend installation:

.. code-block:: text

   converted/jobs/{job_id}/
   ├── requested_config.json
   ├── resolved_config.yaml
   ├── cfg/
   ├── stdout.log
   ├── stderr.log
   └── checkpoints and results

Set ``NNM_BACKEND_ARTIFACT_ROOT`` for a persistent Docker volume or a shared
Slurm filesystem. The directory is ignored by Git.

Local setup
-----------

From ``converted/``, start a persistent Valkey instance and the API:

.. code-block:: bash

   mkdir -p valkey-data
   valkey-server backend/valkey.conf --dir valkey-data

In another terminal:

.. code-block:: bash

   PYTHONPATH=src NNM_VALKEY_URL=valkey://127.0.0.1:6379/0 \
     uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000

The frontend proxies ``/api`` to the backend. Docker Compose provides the same
services with persistent Valkey and job-artifact volumes.

Executors
---------

``LocalExecutor`` launches the repository's fixed ``src/main.py`` entrypoint
and captures its output. ``SlurmExecutor`` generates a batch script from the
validated request and submits it either with local ``sbatch`` or through
``ssh <host> sbatch``. The backend never accepts an arbitrary client command or
batch script.

Model loading
-------------

The editor's ``Carica`` button loads source Svelte Flow JSON files containing
``nodes`` and ``edges``. Parsing failures are visible and do not replace the
current diagram. Files in ``examples/nntrees/`` are already-converted NNTree
artifacts; the editable source files are in ``examples/diagrams/``.

Source-model validation can select an explicit path:

.. code-block:: bash

   NNM_MODEL_PATH=examples/diagrams/transformer_classifier.json \
     pnpm --dir front-end test:integration:model

The test rejects an ``examples/nntrees/`` path, runs the frontend type engine,
and compiles the source diagram before the Python pipeline is used.

Testing
-------

Relevant checks are:

.. code-block:: bash

   pnpm --dir front-end test
   pnpm --dir front-end check
   pnpm --dir front-end build
   cd converted
   PYTHONPATH=src uv run pytest src/tests/test_remote_backend.py -q

The direct Chrome workflow also verifies model loading, visible type
diagnostics, dataset selection, job submission, and a completed local MNIST
training job.

Future features
---------------

The following features are planned for later versions:

* concurrent execution of multiple compatible jobs;
* a physical Slurm resource model in which a two-GPU node can expose two
  individually schedulable GPU units with memory capabilities;
* dataset-aware input dtype metadata for stricter validation of models such as
  Transformer networks using ``Embedding``.

