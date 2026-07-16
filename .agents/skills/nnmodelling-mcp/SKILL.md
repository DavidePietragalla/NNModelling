---
name: nnmodelling-mcp
description: Operate NNModelling through its browser-backed MCP server. Use when Codex must start or diagnose the frontend, Chromium DevTools, WebSocket bridge, or MCP stdio server; select a browser tab; inspect or edit a neural-network diagram; query tensor types; capture screenshots or hover tooltips; convert a diagram; train or run inference; or diagnose leaked ports and disconnected browser sessions.
---

# NNModelling MCP

Treat the browser's `DiagramCore` as the only diagram state. The MCP server is
a thin stdio-to-WebSocket proxy and cannot manipulate a graph until a frontend
tab connects to `ws://localhost:9339`.

## Start the stack

Run commands from the repository root. Use
`.agents/skills/nnmodelling-mcp/scripts/nnm-stack.sh status` first and reuse
healthy processes.

1. Start the frontend in a persistent terminal:

   ```bash
   .agents/skills/nnmodelling-mcp/scripts/nnm-stack.sh frontend
   ```

2. Start Chromium in a separate persistent terminal. This uses an isolated
   profile and exposes Chrome DevTools on port 9223:

   ```bash
   .agents/skills/nnmodelling-mcp/scripts/nnm-stack.sh browser
   ```

3. Start the MCP stdio server through the client's configured MCP transport.
   For direct debugging in a persistent terminal:

   ```bash
   .agents/skills/nnmodelling-mcp/scripts/nnm-stack.sh mcp
   ```

4. Verify ports and endpoints:

   ```bash
   .agents/skills/nnmodelling-mcp/scripts/nnm-stack.sh status
   ```

Defaults are frontend `http://127.0.0.1:5174`, browser DevTools `9223`, and
browser WebSocket bridge `9339`. Override the first two with
`NNM_FRONTEND_URL` and `NNM_CDP_PORT`.

If Chromium is not installed as Flatpak, set `NNM_BROWSER_COMMAND` to a working
executable. Do not use this machine's `~/.local/bin/google-chrome` wrapper
without checking it: it may depend on an unavailable `cobalt` binary.
Launching the Flatpak GUI normally requires elevated execution approval; ask
for it through the command tool instead of falling back to an unrelated browser.

## Connect over raw stdio when MCP tools are not exposed

Keep `pnpm --dir mcp-server start` in a PTY and send one JSON object per line.
Initialize before tool calls:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"codex","version":"1.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
```

Call a tool with:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_graph","arguments":{}}}
```

Wait for `Browser tab connected`. With multiple tabs, call
`list_browser_tabs` and `select_browser_tab`; never guess which tab is active.

## Manipulate and inspect diagrams

Use MCP tools instead of editing browser state or diagram JSON directly:

- Inspect: `get_graph`, `get_node`, `get_type_info`, `get_edges`.
- Mutate: `create_node`, `connect_nodes`, `disconnect_nodes`,
  `update_parameters`, `move_nodes`, `delete_nodes`.
- View: `fit_view`, `center_view`, `clear_selection`.
- Validate before conversion: call `get_type_info` with `refresh: true` and
  require no hard errors.

Preserve handle order for non-commutative joins. Use `in-0`, `in-1`, etc., and
never rely on traversal order.

After structural edits, arrange nodes vertically or horizontally with
`move_nodes`, clear selection, and call `fit_view`. This leaves the user's tab
readable instead of overlapping newly-created nodes.

## Inspect types and screenshots

`get_type_info` returns JSON-safe input/output shapes, dtype, errors, warnings,
and suggestions. A Loss is conceptually a layer with rank-1 output `[B]`, even
though the current Python backend treats it as a terminal objective.

Use `capture_screenshot` with:

- `pageUrl` when several DevTools pages exist;
- `reloadPage: true` to discard broken HMR state, knowing that reload resets
  the in-memory diagram;
- `hoverNodeId` to display a node's output tensor tooltip;
- an explicit `/tmp/*.png` `outputPath` for later inspection.

When reloading, wait for the browser tab to reconnect, then recreate or import
the diagram before capturing an annotated screenshot.

## Convert and train

1. Build a type-correct graph with an explicit `Flatten` before a Linear layer
   when an image dataset supplies `[B,C,H,W]` tensors.
2. For a small MNIST classifier, prefer:

   ```text
   Input → Flatten → Linear(784,64) → ReLU → Linear(64,10)
         → CrossEntropyLoss
   ```

   Do not add Softmax before CrossEntropyLoss; it expects logits.
3. Call `execute_conversion` with a fresh output directory, MNIST dataset,
   `numClasses: 10`, and the intended epoch count.
4. Inspect the generated config or conversion result. Require
   `taskType: classification` and confirm the expected layers.
5. Call `execute_training` with `device: cpu` and a small `maxEpochs` value.
   The server translates these to Hydra overrides and writes each run to an
   isolated `/tmp/nnmodelling-training-*` directory.
6. Verify `success: true` and that `checkpointPath` exists. Preserve the path
   if inference will follow.

Avoid transformer and autoencoder training unless the user explicitly requests
it. For smoke testing, train only two or three small networks at most.

## Diagnose failures

- `ECONNREFUSED` on 9339: MCP server is absent or still starting.
- MCP reports no selected tab: frontend is absent, reloading, or connected to
  another server instance.
- Screenshot cannot find a page: start Chromium with `--remote-debugging-port`
  and pass the exact `pageUrl`.
- Hydra rejects `--max-epochs` or `--device`: the server build is stale; rebuild
  and restart it. Current code must emit `trainer.*` overrides.
- Linear matrix mismatch on MNIST: add `Flatten`; static Input metadata alone
  does not reshape runtime tensors.
- Port leak: run the status script and inspect the owning PID. Stop only the
  stale process, with user approval when ownership is uncertain. Never start a
  second MCP server on 9339.

After changing MCP TypeScript, run `pnpm --dir mcp-server test`, then restart
the stdio server so the live process uses the new `dist/` build.
