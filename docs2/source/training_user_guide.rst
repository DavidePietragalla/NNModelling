Connect the frontend to a training backend
==========================================

The Training sidebar can send the diagram to an NNModelling backend running
on another machine. Connections use administrator-approved browser pairing:
there are no usernames or passwords, and an unapproved browser cannot inspect
datasets, compute units, jobs, logs, or events.

.. warning::

   This pairing system is designed for a trusted LAN. Do not expose the
   backend directly to the public Internet. Plain HTTP does not encrypt the
   bearer token; use HTTPS whenever the LAN is not fully trusted.

Before connecting
-----------------

Ask the backend administrator for its URL, for example
``http://192.168.1.20:8000``. The administrator must allow the exact Origin
from which your NNModelling frontend is served. An Origin includes scheme,
host, and port, such as ``http://192.168.1.30:5174``.

Only ``GET /health`` and the short pairing exchange are available before
approval. The administrator never needs to send you a password or token.

Request a connection
--------------------

1. Open NNModelling and select **Training** in the editor toolbar.
2. Enter the absolute HTTP or HTTPS backend URL.
3. Optionally enter a recognizable device name, such as
   ``Laptop laboratorio``.
4. Select **Richiedi connessione**.
5. Compare the six-digit code shown in the sidebar with the code seen by the
   administrator. Do not ask the administrator to approve if the codes differ.
6. Wait on the same screen. The frontend checks the request periodically and
   unlocks the training controls after approval.

The request itself expires after a short interval, normally ten minutes. A
rejected or expired request can be replaced by selecting the backend again and
creating a new one.

What the browser stores
-----------------------

The browser stores the backend URL, the opaque connection token, and the
connection identifier in ``localStorage``. Refreshing the page or restarting
the browser therefore preserves a non-expired connection.

Treat the browser profile as a credential: another process that can read its
site storage can act as this connection. The token is sent in the
``Authorization`` header and is never placed in a URL. Job events use an
authenticated fetch-based SSE stream for the same reason.

Session lifetime and renewal
----------------------------

The usual lifetime is 24 hours. The backend administrator can select a
different duration for one approval. The sidebar shows the current expiry
time.

When a connection expires, choose **Richiedi rinnovo**. The administrator must
approve the new request. A successful renewal keeps the same connection
identity, so jobs created before expiry remain visible.

A revoked connection cannot renew itself. It must pair as a new connection,
and the new identity does not inherit jobs owned by the revoked identity.

Submit and inspect training jobs
--------------------------------

After approval, the sidebar loads dataset classes and compute capabilities
from the backend. Configure the dataset, optimizer, trainer, W&B settings,
requested CPU/RAM/GPU resources, and priority, then select **Invia training**.

Priority is a non-negative integer. Larger values are scheduled first; equal
priorities retain FIFO order. The initial scheduler runs one compatible job at
a time and can use either the local machine or a configured Slurm executor.

Job states are:

.. code-block:: text

   queued -> running -> succeeded
                    |-> failed
                    `-> cancelled

Selecting **Invia training** opens a terminal-like tab for the new job. It
follows stdout and stderr incrementally while the job is queued or running,
and remains available afterwards from **Apri terminale**. Both queued and
running jobs can be cancelled.

With W&B mode set to ``online``, a second tab opens in a waiting state. It is
redirected to the live W&B run as soon as the remote training process reports
its public run URL. If the browser blocks a popup, the job row still exposes
**Apri W&B** once that URL is available.

Install and use an exported model
---------------------------------

Before submitting a job, optionally set **Nome pacchetto** in the training
sidebar. Enter only the suffix, for example ``mnist_classifier``: NNModelling
creates the distribution and Python module ``nnm_mnist_classifier``. The name
must begin with a letter and may then contain only letters, digits, and
underscores. When a job completes successfully, select **Scarica wheel** in
the job row and install the downloaded file in a Python 3.12+ environment:

.. code-block:: bash

   pip install ./nnm_mnist_classifier-0.1.0-py3-none-any.whl

The wheel contains the resolved graph, safe ``safetensors`` weights and its
declared input adapter. It does not need the training backend, W&B, Lightning
or the training dataset. Every model offers a universal tensor API:

.. code-block:: python

   import torch
   from nnm_mnist_classifier import load_model

   model = load_model(device="cpu")
   logits = model.predict_tensor(batch_tensor)

``predict_tensor`` expects a batch tensor already prepared for the model. A
model trained with MNIST also offers ``model.predict(path_or_bytes_or_image)``;
it applies the saved grayscale, resize and normalization configuration before
inference. A model trained with ``EnronSpamDataset`` also accepts one raw email
string through ``model.predict(email_text)``. Its exported text adapter uses the
saved Hugging Face tokenizer name and maximum length. The first inference may
download that tokenizer unless it is already available in the local Hugging
Face cache; install the wheel's ``transformers`` dependency and provision the
tokenizer cache in offline environments.

Dataset class metadata
-----------------------

The training backend discovers the dataset classes installed in its trusted
environment. A dataset can declare its fixed classification cardinality through
the ``Dataset.num_classes(config)`` class method, without constructing or
downloading the dataset. The training sidebar displays this value and includes
it in the job request. The backend independently resolves the same metadata,
so requests from older clients are still correct. For example, MNIST declares
10 classes and Enron Spam declares 2. A supplied ``training.num_classes`` that
conflicts with a dataset's declared value is rejected before conversion.

Only the browser connection that created a job can download its wheel. The
wheel contains model parameters: store and distribute it as a model artifact,
not as a public URL.

Job privacy
-----------

Each job is assigned to the connection that submitted it. A browser can list,
inspect, stream, read logs for, and cancel only its own jobs. Attempting to
address another connection's job returns ``404`` without confirming that the
job exists.

The backend administrator can manage every job from the backend machine. Jobs
created by older NNModelling versions have no owner and are visible only to the
administrator.

Disconnect or revoke
--------------------

The sidebar offers two different actions:

``Dimentica su questo browser``
   Deletes the local URL and token. The backend connection remains valid until
   expiry or administrator revocation. Use this only when you intentionally
   want to remove local state without affecting another tab sharing it.

``Disconnetti e revoca``
   Revokes the connection immediately on the backend and then clears local
   state. Other tabs using the same browser storage lose access as well.

Common connection errors
------------------------

Backend unreachable or CORS Origin not authorized
   Confirm the URL and network route. If ``GET /health`` works outside the
   browser, ask the administrator to add the frontend's exact Origin to
   ``NNM_ALLOWED_ORIGINS`` and restart the backend.

Connection pending
   The administrator has not yet approved the displayed request. Compare the
   verification code before approval.

Session expired
   Request a renewal. Existing owned jobs return after approval.

Session revoked
   Pair again as a new connection or contact the administrator. Revocation is
   immediate and cannot be reversed from the old token.

No jobs are shown
   Job lists are intentionally scoped to the current connection. Jobs from a
   previous revoked or forgotten identity are not transferred automatically.

Security limitations
--------------------

Pairing prevents anonymous access and isolates job ownership, but an approved
browser is still trusted to submit training configuration. The current Hydra
configuration surface remains broad; server-side allowlists for all Hydra
targets and overrides are a separate hardening task.

The MCP server's remote-training HTTP client does not yet implement this
pairing token. Browser diagram tools remain usable, but MCP remote-training
tools must not be pointed at the protected backend in this version.
