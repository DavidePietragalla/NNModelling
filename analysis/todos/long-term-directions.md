# NNModelling — Long-Term Directions

## Inferred Vision

NNModelling is best understood as a **visual programming language for neural network architecture design**. Its philosophy is:

1. **Diagrams are source code** — the visual graph is the ground truth. The generated Python is a projection, not the canonical representation.
2. **Explicit over implicit** — Flatten is a node, Fork is a node, joins must be placed manually. Nothing happens behind the user's back.
3. **Composability as a first-class principle** — subflows can contain subflows can contain subflows. Behavioral stereotypes (Repeat, HorizontalRepeat) can wrap arbitrary subgraphs.
4. **Production-readiness by default** — the output is not a toy. It's a full Lightning+Hydra+wandb training pipeline with configurable datasets, early stopping, and metrics.
5. **Minimalist architecture** — the frontend has one state class, zero state management libraries, pure Svelte 5 reactivity. The backend has ~150-line Net class, ~90-line Subflow engine. Everything is small, focused, and understandable by one person.

The long-term trajectory is clear: evolve from a **model design tool** into a **full visual ML IDE** — covering the entire lifecycle from architecture sketching through training, debugging, and deployment — while preserving the core philosophy of visual-first, explicit, composable design.

---

## Direction 1: Shape-Aware Graph Engine

### Vision

Transform the connection system from pure topology validation ("is this handle free?") into a **type system for tensors**. The editor should understand tensor shapes through the graph and warn when dimensions misalign — the way TypeScript warns about type mismatches.

### Why It's a Natural Evolution

The current system validates **only handle occupancy**, not semantic correctness. A user can connect a Conv2d (output: `[B, C, H, W]`) directly to a Linear (expects `[B, features]`) without any warning. This is the single largest source of silent errors in diagram construction.

The architecture already has the foundation: `stereotype.ts` knows every module's `pythonClassName` and params. `convert.py` already parses shapes from params. The conversion pipeline already does topological traversal. Adding forward shape propagation to this traversal is a natural extension — it's the same algorithm, different payload.

### What Architectural Changes It Would Require

1. **Shape inference rules per stereotype** — each stereotype JSON would need output-shape derivation rules (e.g., `Linear(in_features, out_features)` → output `[B, out_features]`). This could be declarative in the JSON or a small inference engine.
2. **Forward shape propagation** — extend `nnTree.processNode()` or create a separate `ShapeInference` pass that walks the DAG and annotates edges with tensor shapes.
3. **Connection validation** — `checkValidConnection()` would compare the source node's output shape against the target's expected input shape.
4. **Visual shape annotations** — edge labels showing tensor dimensions (e.g., `[64, 350]` on the edge from Linear to Tanh). Node handles could color-code shape compatibility (green = match, yellow = reshape needed, red = mismatch).
5. **Runtime shape feedback** — after training, actual tensor shapes could be fed back to the editor to correct inference rules.

### Potential Benefits

- **Eliminates silent dimension errors** — the most common bug in hand-designed architectures
- **Educational value** — users see how shapes flow through the network, understanding why `in_features` must match
- **Enables automatic reshaping suggestions** — the editor could offer to insert a Flatten or Linear projection when shapes don't match
- **Foundation for architecture search** — a shape-aware graph enables automated exploration that respects dimensional constraints

### Potential Risks

- **Complexity** — shape inference rules for every PyTorch module (especially dynamic ones like AdaptiveAvgPool2d or operations with multiple interpretations) is substantial work
- **False positives** — shape inference that doesn't account for batch dimension flexibility or dynamic shapes could produce spurious warnings
- **Maintenance burden** — every new stereotype requires shape rules, doubling the cost of adding modules
- **Shape inference can never be perfect** — some shapes are only known at runtime (e.g., `Flatten` output depends on actual input spatial dims, which depend on prior pooling). The system must gracefully handle "unknown" shapes

---

## Direction 2: Collaborative Visual ML Platform

### Vision

Evolve from a single-user desktop tool into a **collaborative visual ML platform** where teams design architectures together in real-time, share subflow components as reusable assets, and iterate on experiments with shared visibility.

### Why It's a Natural Evolution

The project is already structured as a web application (Svelte 5 + Vite). The diagram format is pure JSON. Subflows are already designed as self-contained, nestable components. These are the precise primitives needed for:

- **Real-time collaboration** — Operational Transformation or CRDT on the nodes/edges JSON
- **Component marketplace** — subflows are already exportable as JSON; a gallery of community-contributed components (ResNet blocks, transformer encoders, UNet blocks) is a natural next step
- **Experiment sharing** — a diagram + trained weights + metrics = a reproducible experiment that can be shared, forked, and compared

The transition from single-user to collaborative is less architectural and more operational (needs a server, auth, storage) — but the data model is ready.

### What Architectural Changes It Would Require

1. **Backend server** — FastAPI or similar to handle user auth, diagram storage, collaboration sessions, and experiment tracking. The TODO.md already plans a FastAPI training server.
2. **Database** — PostgreSQL for user accounts, diagrams, experiments, and component registry.
3. **Real-time sync** — WebSocket-based CRDT (e.g., Yjs) or simple operational transform for concurrent diagram editing.
4. **Component registry** — each subflow can be published with metadata (name, description, input/output shape contracts, tags). Versioning for components so diagrams don't break on updates.
5. **Access control** — private diagrams, team-shared components, public gallery.
6. **Compute integration** — connect to cloud GPU services (the Brev CLI skill is already available) for remote training from the web UI.

### Potential Benefits

- **Community growth** — a component marketplace creates network effects
- **Reproducibility** — shared diagrams + configs = truly reproducible ML experiments
- **Education** — students can fork published architectures, modify them, and see results
- **Enterprise adoption** — teams can share architecture standards as subflow libraries

### Potential Risks

- **Massive scope increase** — requires backend engineering, auth, storage, deployment, monitoring
- **User acquisition challenge** — collaborative tools need a critical mass of users to be useful
- **Complexity kills simplicity** — the project's core strength is its minimalism. Adding accounts, permissions, and collaboration infrastructure could overwhelm the simple visual editor experience
- **Data privacy** — users' model architectures may be proprietary; self-hosting becomes necessary

---

## Direction 3: Architecture Search & Optimization

### Vision

The visual graph IS the search space. Given a dataset and a task, the system should be able to **automatically explore the graph space** — adding/removing layers, adjusting dimensions, inserting skip connections — to find architectures that maximize accuracy while respecting compute budgets.

### Why It's a Natural Evolution

NNModelling's core data structure is a DAG with typed nodes and typed connections. This is exactly the representation used in neural architecture search (NAS) literature. The project already has:

- A discrete search space (node types, connections, params)
- A deterministic compiler to executable models (convert.py + base.py)
- A training pipeline with metrics (main.py)
- A mechanism for repeated structures (Repeat, HorizontalRepeat)

The missing piece is the search algorithm — mutating the graph, training, evaluating, and selecting — which can be built entirely on the existing infrastructure.

### What Architectural Changes It Would Require

1. **Graph mutation operators** — programmatic API for adding/removing nodes, rewiring connections, changing params. This currently lives in `Diagram.svelte.ts` as UI methods; would need a headless version.
2. **Search strategies** — evolutionary algorithms (mutate + crossover), reinforcement learning (controller RNN), or gradient-based (DARTS-like differentiable architecture search).
3. **Evaluation pipeline** — train → evaluate → score → mutate → repeat. Needs resource management (GPU scheduling, early stopping of poor candidates).
4. **Search space constraints** — define valid/invalid graph patterns (e.g., max depth, allowed connections, required input/output shapes) to keep search tractable.
5. **Results visualization** — show the search trajectory in the editor, let users browse candidate architectures, compare Pareto frontiers (accuracy vs. params vs. FLOPs).

### Potential Benefits

- **Automated architecture design** — users specify dataset + task, system proposes architectures
- **Democratizes NAS** — currently NAS is accessible only to researchers with large compute clusters; a visual tool makes it tangible
- **Human-in-the-loop** — the visual editor lets users guide search, inspect candidates, and inject domain knowledge
- **Novel architecture discovery** — search may find non-obvious graph structures that outperform hand-designed ones

### Potential Risks

- **Compute cost** — NAS is notoriously expensive. Even with early stopping and weight sharing, training hundreds of candidate architectures requires significant GPU resources.
- **Search space explosion** — the full DAG space is enormous. Without careful constraints, search may never converge.
- **Overfitting to validation** — automated search can overfit to the validation metric, producing architectures that don't generalize.
- **Interpretability** — automatically discovered architectures may be hard to understand or justify.

---

## Direction 4: Multi-Backend Code Generation

### Vision

The diagram is backend-agnostic. Today it compiles to PyTorch/Lightning. Tomorrow it should compile to **JAX/Flax, TensorFlow/Keras, ONNX, and even hardware platforms** (Core ML for Apple Silicon, TensorRT for NVIDIA inference, TFLite for mobile).

### Why It's a Natural Evolution

The conversion pipeline is already a clean separation: `Diagram → NNTree → JSON → convert.py → YAML configs`. The NNTree JSON is a backend-agnostic intermediate representation. The Python side maps each `pythonClassName` to a Hydra `_target_`. Adding a new backend means adding:

1. A new stereotype-to-backend mapping (e.g., `"nn.Linear"` → `"flax.linen.Dense"`)
2. A new `ops/` implementation for custom operations
3. A new training pipeline (replacing Lightning with Flax's training loop, or Keras' `fit()`)

The front-end diagram editor doesn't change at all. The architecture supports this cleanly.

### What Architectural Changes It Would Require

1. **Backend abstraction layer** — currently `pythonClassName` maps directly to `torch.nn.*`. Would need a backend registry where `"Linear"` maps to different `_target_` strings per backend.
2. **Per-backend stereotype extensions** — some stereotypes exist only in one backend (e.g., `nn.MultiheadAttention` is PyTorch-specific; JAX/Flax may not have an equivalent).
3. **Per-backend training pipelines** — `main.py` is tightly coupled to Lightning. Would need `main_jax.py`, `main_keras.py`, or a configurable training loop.
4. **ONNX/TensorRT export** — rather than training, these are inference-only backends. Would need an export path: `diagram → PyTorch model → ONNX → TensorRT`.
5. **Performance comparison** — train/benchmark the same architecture across backends and show comparative results (throughput, memory, accuracy).

### Potential Benefits

- **Unlocks deployment** — export to ONNX/TensorRT/CoreML for production inference
- **Ecosystem flexibility** — users can design in the visual editor, then train in their preferred framework
- **Benchmarking tool** — compare framework implementations of the same architecture
- **Future-proof** — the diagram format outlives any single framework

### Potential Risks

- **Lowest-common-denominator** — not all PyTorch modules have equivalents in JAX or Keras. The shared stereotype set might shrink to what's universally available.
- **Maintenance multiplier** — each new backend doubles the test surface, ops implementations, and training pipeline code.
- **PyTorch dominance** — in practice, most users will stay with PyTorch. JAX/TF backends may see low usage.
- **Subtle behavioral differences** — different frameworks have different defaults (weight init, epsilon values, padding modes). The visual editor would need to abstract these or expose them.

---

## Direction 5: Training Loop Abstraction & Multi-Task Architectures

### Vision

Move beyond simple classification/regression training loops. Support **GAN training** (generator + discriminator alternating), **contrastive learning** (Siamese networks with triplet loss), **reinforcement learning** (policy + value networks), and **multi-task learning** (shared backbone, task-specific heads) — all specified visually.

### Why It's a Natural Evolution

The current system detects `taskType` from the loss node: classification or regression. This is a one-model, one-loss, one-metric model. But many modern architectures don't fit this pattern:

- **GANs**: two models (generator + discriminator) with adversarial loss
- **Autoencoders**: already supported (image→image with MSELoss), but VAE requires KL divergence + reconstruction loss combined
- **Siamese networks**: shared weights, contrastive loss
- **Multi-task**: single backbone, multiple heads, combined loss function
- **Knowledge distillation**: teacher model + student model + distillation loss

The subflow concept already supports encapsulation. A GAN could be: `Input → Generator(Subflow) → Discriminator(Subflow) → AdversarialLoss`. The architecture is ready for multiple models inside one diagram.

### What Architectural Changes It Would Require

1. **Multi-output diagrams** — currently one root, one loss node. Need multiple model outputs and multiple loss functions.
2. **Training loop config** — replace the single `trainer.fit()` with configurable training steps. Define alternating updates, gradient accumulation, loss weighting.
3. **Weight sharing** — Siamese networks need two branches with shared weights. This could be a new stereotype or a `shared_weights` flag on subflow instances.
4. **Loss composition** — combine multiple losses with configurable weights. A new "LossComposer" join type that applies `α * loss_a + β * loss_b`.
5. **Custom training steps** — allow users to define per-step logic visually (e.g., "step 1: train discriminator on real+fake; step 2: train generator via discriminator feedback").

### Potential Benefits

- **Expands the addressable problem space** — current system handles classification, regression, autoencoders; this would add GANs, contrastive learning, multi-task, distillation
- **Visual debugging of complex training** — see the alternating generator/discriminator losses, understand why GANs are unstable
- **Makes cutting-edge techniques accessible** — users can visually construct architectures from papers

### Potential Risks

- **Training loop complexity** — Lightning's `training_step` is already an abstraction; adding multiple alternating steps pushes the abstraction to its limits
- **Over-engineering** — many users just need classification. Complex training loops may clutter the UX.
- **Debugging difficulty** — visually debugging a GAN training loop (mode collapse, gradient issues) is harder than debugging a simple classifier

---

## Direction 6: Visual Training Monitor & Interactive Debugging

### Vision

Close the loop between design and training. The visual editor should show **live training metrics** overlaid on the architecture diagram — which layers are learning, where gradients are vanishing, which skip connections carry the most signal. The diagram becomes a **live dashboard** for model behavior.

### Why It's a Natural Evolution

The visual editor already shows the architecture. The training pipeline already captures metrics (wandb, validation accuracy, loss). The missing link is bringing training feedback back into the editor:

- **Per-node analytics** — show gradient norms, activation statistics, weight distributions per layer
- **Attention visualization** — for transformer diagrams, show attention head patterns
- **Loss landscape** — show which architectural changes affect training dynamics
- **Bottleneck detection** — highlight layers where activations saturate or gradients vanish

The Subflow concept is perfect for this: you could drill into a subflow to see its internal metrics, or collapse it for a summary.

### What Architectural Changes It Would Require

1. **Training hooks** — PyTorch hooks on each module to capture activations, gradients, and weights during training
2. **Streaming metrics** — WebSocket or polling from training process to editor showing per-layer statistics
3. **Visual overlays** — node colors/sizes could encode gradient norm, activation sparsity, weight magnitude
4. **Time-series replay** — replay training progress, watching the network learn
5. **Comparison mode** — side-by-side comparison of two training runs on the same architecture

### Potential Benefits

- **Demystifies training** — users see WHY their model isn't learning (vanishing gradients in deep networks, dead ReLUs, overfitting patterns)
- **Faster debugging** — instead of staring at loss curves, users see which layers are the problem
- **Educational** — teaching tool for understanding training dynamics

### Potential Risks

- **Performance overhead** — capturing per-layer statistics slows training significantly
- **Information overload** — too many metrics on screen; users need smart defaults and progressive disclosure
- **Coupling editor to training runtime** — the editor currently works offline; live training monitoring requires a running process connection

---

## Direction 7: Deployment & Serving Pipeline

### Vision

One-click from diagram to deployed API. The visual editor should generate not just training code, but a **production serving endpoint** — FastAPI server, Docker container, or cloud function — with the trained weights baked in.

### Why It's a Natural Evolution

The inference script (`infer.py`) already loads a trained model and runs predictions. The missing pieces are:

- **API wrapping** — FastAPI endpoint with `/predict` accepting JSON/images/text
- **Containerization** — Dockerfile with the model + server
- **Cloud deployment** — scripts for AWS SageMaker, GCP Vertex AI, HuggingFace Spaces
- **Model optimization** — quantization, pruning, ONNX export for faster inference

The project already has the Brev CLI skill for GPU cloud management — connecting the visual editor to Brev for one-click cloud deployment is a natural integration.

### What Architectural Changes It Would Require

1. **Serving code generator** — parallel to `convert.py`, a `serve.py` that generates a FastAPI server wrapping the trained model
2. **Dockerfile generator** — auto-generate Dockerfile from the model's dependencies
3. **Optimization pipeline** — auto-apply quantization (int8), ONNX export, TensorRT optimization
4. **Deployment targets** — Brev CLI, AWS SageMaker, HuggingFace Spaces, GCP Cloud Run
5. **API schema generation** — determine input/output shapes from the diagram and generate OpenAPI/Swagger docs

### Potential Benefits

- **End-to-end ML workflow** — design → train → deploy, all from one tool
- **Production-ready by default** — the generated serving code follows best practices (batching, health checks, logging)
- **Lowers the deployment barrier** — many ML practitioners struggle with the ops side

### Potential Risks

- **Scope creep** — deployment is a separate discipline from architecture design
- **One-size-fits-all serving** — different use cases need different serving patterns (batch inference, streaming, edge deployment)
- **Security** — auto-generated servers need careful security review before production use

---

## Synthesis: Most Natural Evolutionary Path

The seven directions above are not independent. They form a coherent evolution:

```
Shape-Aware Engine → enables → Architecture Search
                              → Multi-Backend Generation
                              → Training Loop Abstraction

Visual Training Monitor ← depends on → Training Loop Abstraction

Deployment Pipeline ← depends on → Multi-Backend Generation
                                 → Shape-Aware Engine (for ONNX export)

Collaborative Platform ← depends on → Component Registry (subflow sharing)
                                    → Deployment Pipeline (shared compute)
```

The **Shape-Aware Graph Engine** (Direction 1) is the foundational direction. Every other long-term direction either depends on it or is significantly enhanced by it. It's the "type system" for the visual language — just as TypeScript supercharges JavaScript, a shape system would supercharge NNModelling.

The **Collaborative Platform** (Direction 2) is the direction that creates network effects and community, but it requires significant infrastructure investment and changes the project's nature from a tool to a platform.

The **Architecture Search** (Direction 3) and **Training Monitor** (Direction 6) are the directions that most directly serve the power user — the ML researcher who already knows architectures but wants tools to explore and understand them more deeply.

**Deployment** (Direction 7) is the "last mile" — it completes the workflow but is the least architecturally innovative direction.

A pragmatic long-term roadmap might be: Shape Engine (foundation) → Training Monitor (user feedback) → Architecture Search (automation) → Collaborative Platform (scale).

---
User analysis: I prefer direction 1 -> direction 4 -> direction 5. Direction 2 is to me changed, but the base idea is nice. The other directions are interesting, but for the moment I don't want to implement them.
