# NNModelling — Python Codegen Target

This directory converts NNTree JSON diagrams (exported from the visual editor) into runnable PyTorch/Lightning models. It is the backend half of the NNModelling DSL pipeline.

## Pipeline

```
Diagram → NNTree JSON → convert.py → Hydra YAML configs → main.py → training
```

## Setup

Requires Python 3.12. Install dependencies with `uv`:

```bash
uv sync
```

## Usage

### Step 1: Generate Configuration

```bash
python src/convert.py [json_path] [output_dir] [options]
```

**Positional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `json_path` | `../converted_minst.json` | Path to NNTree JSON exported from the visual editor |
| `output_dir` | `cfg` | Output directory for generated Hydra configs |

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-classes N` | `None` | Required when the loss node has `taskType: "classification"` |
| `--dataset D` | `dataset.mnist.MNISTDataset` | Dataset class path (e.g. `dataset.autoencoder_mnist.AutoencoderMNIST`) |
| `--early-stop-patience N` | `3` | Early stopping patience |
| `--early-stop-min-delta F` | `0.0` | Early stopping minimum delta |
| `--max-epochs N` | `20` | Maximum training epochs |

**Output structure** (created under `output_dir/`):

```
cfg/
├── base.yaml                  # Root config composing all sub-configs
├── net/custom_sequence.yaml   # Network architecture (from NNTree JSON)
├── optimizer/adam.yaml        # Optimizer config (Adam, lr=0.001)
├── trainer/default.yaml       # Trainer config (max_epochs, accelerator)
├── dataset/dataset.yaml       # Dataset class, batch_size, train/val split
├── wandb/wandb.yaml           # W&B project settings
└── early_stopping/default.yaml # EarlyStopping parameters
```

### Step 2: Train

```bash
python src/main.py --config-path <dir> --config-name <name>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config-path` | `../cfg` | Path to the Hydra config directory |
| `--config-name` | `base` | Config file name (without `.yaml`) |

The training script:

- Instantiates the network dynamically from the generated config via Hydra's `instantiate()`
- Auto-detects **classification** vs **regression** based on the loss node's `taskType` (Accuracy or MSE metric)
- Logs metrics to **Weights & Biases** (project: `NeuralNetworks`)
- Saves the trained model to `weights.pt`
- Applies **early stopping** based on validation metric

## End-to-End Example

Train the autoencoder from the included example JSON:

```bash
# Generate configs
python src/convert.py auto_encoder.json cfg \
  --dataset dataset.autoencoder_mnist.AutoencoderMNIST \
  --max-epochs 50

# Train
python src/main.py --config-path cfg --config-name base
```

## Architecture

### `net/base.py` — Dynamic DAG Network

The `Net` class (`LightningModule`) builds a `ModuleDict` dynamically from the config. It uses a **BFS topological sort** to execute the computation graph at runtime:

1. Starts from the root (Input) node
2. Tracks in-degree to know when all parents of a join node are ready
3. Users explicitly place Flatten nodes for dimension adjustment
4. Supports sequential chains, join merges (Addition, Einsum), and subflow containers

### `ops/` — Join Operations

| Module | Description |
|--------|-------------|
| `ops.Addition` | Element-wise sum of multiple branch outputs |
| `ops.Einsum` | Tensor contraction via `torch.einsum` |

### `dataset/` — Dataset Classes

| Class | Description |
|-------|-------------|
| `dataset.mnist.MNISTDataset` | Standard MNIST classification (28×28 images → digit labels) |
| `dataset.autoencoder_mnist.AutoencoderMNIST` | MNIST autoencoder (image → same image as target) |

## Project Structure

```
converted/
├── src/
│   ├── convert.py                # NNTree JSON → Hydra YAML configs
│   ├── main.py                   # Training entry point (Hydra + Lightning)
│   ├── net/base.py               # Dynamic DAG LightningModule
│   ├── ops/addition.py           # Element-wise join
│   ├── ops/einsum.py             # Einsum join
│   ├── dataset/ds.py             # Abstract dataset base
│   ├── dataset/mnist.py          # MNIST classification dataset
│   └── dataset/autoencoder_mnist.py  # MNIST autoencoder dataset
├── auto_encoder.json             # Example NNTree JSON (autoencoder)
├── pyproject.toml                # Dependencies (hydra, lightning, torch, wandb)
└── README.md
```
