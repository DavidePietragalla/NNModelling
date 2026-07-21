---
name: chrome-direct
description: Directly inspect and operate a live Chrome/Chromium tab through the Chrome DevTools Protocol (CDP), including DOM inspection, clicks, form input, navigation, console evaluation, network verification, and screenshots. Use when the user asks to use Chrome directly, reproduce or verify a browser UI flow, diagnose a live frontend page, or explicitly says not to use MCP/browser proxy tools.
---

# Direct Chrome interaction

Use Chrome DevTools Protocol directly. Do not start or call an MCP server for
this workflow. The bundled `scripts/cdp.mjs` helper uses the browser's HTTP
debug endpoint to discover tabs and a WebSocket CDP connection to operate the
selected page.

## Preconditions

1. Ensure Chrome/Chromium is running with remote debugging enabled, normally on
   `127.0.0.1:9223`.
2. Confirm the endpoint before trying to interact:

   ```bash
   curl --silent --show-error --max-time 3 http://127.0.0.1:9223/json/list
   ```

3. If there is no page target, start the browser using the repository's normal
   browser launcher or ask the user before changing their browser session.
4. Select a `type: "page"` target. Never guess a stale WebSocket URL; refresh
   the target list after navigation or a tab reload.

## Core workflow

1. List targets and identify the tab by URL/title.
2. Inspect the current page with `eval` before mutating it. Check the visible
   text, controls, selected values, and relevant error containers.
3. Perform the smallest user-equivalent action: click a visible button, set a
   form control, select an option, or navigate to a URL.
4. Wait briefly for the UI to settle, then inspect the DOM again. Verify the
   result through visible state and, when useful, browser console/network data.
5. Capture a screenshot after the successful state or when diagnosing a visual
   problem. Store screenshots under `/tmp` unless the user requests a repo
   artifact.

For model-loading flows, the successful state is not the file-picker click:
after selecting the JSON, verify that the file input disappeared, the canvas
contains the expected model nodes, and no load-error alert is visible. Then
always capture a screenshot of the loaded canvas and inspect it with
`view_image`. Include the screenshot path in the handoff. A type-check warning
visible on a loaded model must be reported separately from a file-loading
failure.

Use the helper from the skill directory:

```bash
SKILL=.agents/skills/chrome-direct
node "$SKILL/scripts/cdp.mjs" list --port 9223
node "$SKILL/scripts/cdp.mjs" eval --port 9223 --tab 0 \
  --expression 'JSON.stringify({url: location.href, title: document.title})'
node "$SKILL/scripts/cdp.mjs" screenshot --port 9223 --tab 0 \
  --output /tmp/chrome-page.png
```

The helper prints JSON for `list` and `eval`; `screenshot` prints the output
path and byte count. Use `view_image` on a local screenshot when visual
inspection is needed.

## User-equivalent DOM actions

Prefer the helper commands over ad-hoc WebSocket code:

```bash
node "$SKILL/scripts/cdp.mjs" click --port 9223 --tab 0 --text 'Training'
node "$SKILL/scripts/cdp.mjs" input --port 9223 --tab 0 \
  --selector 'input[name="priority"]' --value '99'
node "$SKILL/scripts/cdp.mjs" select --port 9223 --tab 0 \
  --selector 'select' --value 'dataset.mnist.MNISTDataset'
node "$SKILL/scripts/cdp.mjs" reload --port 9223 --tab 0
```

When a control has no stable selector, inspect its DOM and use a narrowly
scoped text selector or a direct `eval` expression. For Svelte/React inputs,
the helper uses the native element value setter and dispatches both `input` and
`change`, so framework bindings observe the change.

Do not call application internals, mutate framework stores, or inject a script
that bypasses the UI when the user asked to test the user flow. CDP DOM events
and page navigation are appropriate; direct API calls are useful only for
diagnosing a network/backend issue and must be clearly separated from the UI
test.

## Diagnosing a failed page

For an error such as `502 Bad Gateway`:

1. Inspect the visible error text and current URL.
2. Use CDP `Network.enable`/`Runtime.evaluate` only to identify the failing
   request and response; do not hide the error by changing the page state.
3. Check the frontend proxy target and backend listener from the terminal.
4. Fix or start the intended backend, reload the page through CDP, and repeat
   the same UI action.
5. Verify that the error element disappears and that the expected controls or
   result appear.

For a page that appears stale, call `reload`, wait for `document.readyState ===
"complete"`, then inspect again. After a frontend hot-module reload, rediscover
the target WebSocket URL.

## Screenshots and handoff

Capture screenshots only after the state is meaningful: an error state for a
bug report, or a completed interaction for a success report. Report the exact
URL, tab selection, action performed, visible result, and screenshot path. Do
not claim that a UI action worked based only on a successful CDP command; check
the resulting DOM or network-visible state.

For a loaded neural-network diagram, the minimum screenshot check is:

```bash
node "$SKILL/scripts/cdp.mjs" screenshot --port 9223 --tab 0 \
  --output /tmp/chrome-model-loaded.png
```

Then call `view_image` on `/tmp/chrome-model-loaded.png` before reporting the
result. Keep the screenshot in `/tmp` unless it is explicitly requested as a
repository artifact.

## Bundled resource

`scripts/cdp.mjs` is the deterministic CDP client. Read it when extending the
helper or when a command fails; otherwise execute it directly.
