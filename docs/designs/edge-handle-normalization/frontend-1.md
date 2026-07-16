# Edge Handle Normalization on Import

## Objective

Fix backwards compatibility: old diagrams (saved before Phase 14) have edges
without `sourceHandle` and `targetHandle` fields. Since Phase 14 added explicit
`id="in"` / `id="out"` to Handles in `CustomNode.svelte`, SvelteFlow cannot
match these edges to handles, making them **invisible on the canvas**.

The fix adds default handle IDs (`"out"` for source, `"in"` for target) to edges
during `importFromJson` when those fields are missing.

## Context — Why This Happened

**Before Phase 14:**
- `CustomNode.svelte` Handles had no `id` attribute → SvelteFlow defaulted handles to `id=null`
- Edges created by drag-and-drop also had `null` handles → everything matched
- Export saved edges without `sourceHandle`/`targetHandle`

**Phase 14** added `id="in"` / `id="out"` to Handles and defaulted `addEdge` to
use `"out"`/`"in"`. New diagrams work fine. Old diagrams break because edges
have `undefined` handles while Handles have explicit IDs.

## Files to Modify

### `front-end/src/core/DiagramCore.ts`

**Method:** `importFromJson` (line 662)

**Change:** After `this.edges = parsedData.edges`, iterate over edges and fill
in missing handle fields with safe defaults.

**Logic:**

```ts
// Normalize edge handles for backward compatibility
// Old diagrams (pre-Phase 14) lack sourceHandle/targetHandle;
// SvelteFlow needs them to match Handle id="in"/"out"
this.edges = this.edges.map((edge: any) => {
  if (!edge.sourceHandle) edge.sourceHandle = "out";
  if (!edge.targetHandle) edge.targetHandle = "in";
  return edge;
});
```

**Rationale for defaults:**
- `sourceHandle = "out"` — universal for all node types (both `custom` and `join` nodes use `id="out"` on source Handle)
- `targetHandle = "in"` — correct for `custom` nodes. For `join` nodes, old files ALREADY have explicit `targetHandle` (e.g. `"in-0"`, `"in-1"`) because joins were added after Phase 14ish. The `if (!edge.targetHandle)` guard ensures join edges are NOT overwritten.

## Test Plan

1. Load `examples/diagrams/mninst.json` via the UI — verify edges render correctly
2. Load `examples/diagrams/mnist_skips.json` — verify join edges keep their `targetHandle` (`"in-0"`, `"in-1"`)
3. Load `modello_ai_nuovo.json` — verify no regression (edges already have handles)
4. Run existing unit tests to confirm no breakage: `cd front-end && npm run test`
5. Run integration smoke tests: `cd front-end && npm run test:integration:smoke`

## No Other Files Affected

This is a pure `importFromJson` change. No changes to `addEdge`, `exportToJson`,
or any Svelte components required.
