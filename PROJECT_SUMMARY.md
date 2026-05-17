# Project Summary: My-Flow-App

## Overview
A **Domain Specific Language (DSL) for creating Neural Networks** using a visual, diagrammatic editor. This tool allows users to design, configure, and convert neural network architectures to Python/PyTorch code.

## Technology Stack
- **Frontend Framework**: Svelte 5 with Svelte Flow
- **Build Tool**: Vite 8
- **Visualization**: @xyflow/svelte (Svelte Flow) for drag-and-drop node editing
- **Language**: TypeScript

## Core Components

### 1. Visual Editor (front-end/src/)
- **App.svelte**: Entry point using SvelteFlowProvider
- **FlowCanvas.svelte**: Main canvas with node editor, toolbar (save/load/convert), and sidebar
- **Diagram.svelte.ts**: Reactive state management for nodes and edges
- **stereotype.ts**: Stereotype system for reusable node templates

### 2. Stereotypes (Stereotypes/)
JSON-based templates defining neural network modules:
- **Modules/**: 23+ node types (Input, Linear, Conv2d, ReLU, etc.)
- **Joins/**: Special nodes for merging flows (Addition, Einsum)

Each stereotype defines:
- Python class mapping (e.g., `nn.Linear`)
- Parameters with defaults and positions
- Visual appearance (color, dimensions)
- Category (Input, Linear, Conv2d, Loss, Join)

### 3. Conversion System (front-end/src/conversion/)
- **nnTree.ts**: Converts visual diagram to a tree structure representing the neural network
- Handles sequential layers, forks, and loss nodes
- Outputs JSON representation for Python code generation

## Key Features

### Node Types
- **Module nodes**: Standard neural network layers (single input, multiple outputs)
- **Input nodes**: Network entry points (no input, single output)
- **Join nodes**: Merge multiple inputs into one (explicit join operation)
- **Subflow nodes**: Nested submodels with hierarchical positioning
- **Loss nodes**: Output nodes for training (BCE, CrossEntropy, MSE, etc.)

### DSL Concepts
1. **Stereotypes**: Reusable templates for modules with pre-configured parameters
2. **Expression Language**: Custom expressions for modules (via stereotype `expr` field)
3. **Join Expressions**: Operations like `x + y` for merge points
4. **Hierarchical Submodels**: Nested models with parent-child positioning

### Workflow
1. Drag module nodes from sidebar onto canvas
2. Connect nodes to define data flow
3. Configure parameters via sidebar
4. Save/Load diagrams as JSON
5. Convert to Python code (via NNTree → JSON → PyTorch)

## Architecture

```
Diagram Class (state)
  ├── nodes[]: Array of SvelteFlow nodes
  ├── edges[]: Array of connections
  └── stereotypes[]: Loaded from Stereotypes/ directory

Node Types
  ├── custom: Standard modules
  ├── join: Merge nodes
  └── subflow: Nested models

Conversion Pipeline
  Diagram → NNTree → JSON → Python/PyTorch
```

## Current State
- ✅ Visual node editing with Svelte Flow
- ✅ Node creation, deletion, and connection
- ✅ Parameter configuration sidebar
- ✅ Save/Load diagram persistence (JSON)
- ✅ Stereotype system for module templates
- ✅ Submodel support (nested graphs)
- ⚠️ Conversion to Python: Basic NNTree structure implemented, needs full code generation

## Development Commands
```bash
cd front-end
npm run dev      # Start dev server
npm run build    # Build for production
npm run preview  # Preview built app
```
