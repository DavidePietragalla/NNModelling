#!/usr/bin/env node

/**
 * Small dependency-free Chrome DevTools Protocol client.
 *
 * Node 22+ provides a global WebSocket implementation. The script deliberately
 * uses only the CDP HTTP endpoint and page WebSocket; it does not use MCP.
 */

import { writeFile } from "node:fs/promises";

function usage() {
  console.error(`Usage:
  cdp.mjs list [--port 9223]
  cdp.mjs eval [--port 9223] [--tab 0|URL] --expression <javascript>
  cdp.mjs click [--port 9223] [--tab 0|URL] --text <visible text>
  cdp.mjs input [--port 9223] [--tab 0|URL] --selector <css> --value <text>
  cdp.mjs select [--port 9223] [--tab 0|URL] --selector <css> --value <option value>
  cdp.mjs upload [--port 9223] [--tab 0|URL] --selector <css> --file <path>
  cdp.mjs reload [--port 9223] [--tab 0|URL]
  cdp.mjs screenshot [--port 9223] [--tab 0|URL] --output <path>`);
  process.exit(2);
}

function args(argv) {
  const [command, ...rest] = argv;
  if (!command) usage();
  const options = { command, port: 9223 };
  for (let index = 0; index < rest.length; index += 1) {
    const value = rest[index];
    if (!value.startsWith("--")) continue;
    const key = value.slice(2);
    if (key === "port") options.port = Number(rest[++index]);
    else options[key] = rest[++index];
  }
  return options;
}

async function targets(port) {
  const response = await fetch(`http://127.0.0.1:${port}/json/list`);
  if (!response.ok) throw new Error(`Chrome target discovery failed: ${response.status}`);
  return (await response.json()).filter((target) => target.type === "page");
}

function targetFor(items, value) {
  if (value === undefined) return items[0];
  const index = Number(value);
  if (Number.isInteger(index) && String(index) === value) return items[index];
  return items.find((item) => item.url === value || item.id === value);
}

function requireTarget(items, value) {
  const target = targetFor(items, value);
  if (!target) throw new Error(`Chrome page target not found: ${value ?? "first page"}`);
  if (!target.webSocketDebuggerUrl) throw new Error("Target has no CDP WebSocket URL");
  return target;
}

function connect(target) {
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  let nextId = 1;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    const resolve = pending.get(message.id);
    if (!resolve) return;
    pending.delete(message.id);
    if (message.error) resolve.reject(new Error(message.error.message));
    else resolve.resolve(message.result);
  });
  const open = new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  async function call(method, params = {}) {
    await open;
    const id = nextId++;
    const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
    socket.send(JSON.stringify({ id, method, params }));
    return result;
  }
  async function close() {
    await open.catch(() => {});
    socket.close();
  }
  return { call, close };
}

async function evaluate(cdp, expression, awaitPromise = true) {
  const result = await cdp.call("Runtime.evaluate", {
    expression,
    awaitPromise,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  }
  return result.result?.value;
}

function required(options, name) {
  if (!options[name]) {
    console.error(`Missing --${name}`);
    usage();
  }
  return options[name];
}

async function main() {
  const options = args(process.argv.slice(2));
  const items = await targets(options.port);
  if (options.command === "list") {
    console.log(JSON.stringify(items, null, 2));
    return;
  }
  const target = requireTarget(items, options.tab);
  const cdp = connect(target);
  try {
    if (options.command === "eval") {
      console.log(JSON.stringify(await evaluate(cdp, required(options, "expression"))));
    } else if (options.command === "click") {
      const text = JSON.stringify(required(options, "text"));
      const expression = `(() => { const wanted = ${text}; const node = [...document.querySelectorAll('button,a,[role="button"]')].find((element) => element.textContent?.trim().includes(wanted)); if (!node) throw new Error('Visible control not found: ' + wanted); node.click(); return { clicked: node.textContent?.trim() }; })()`;
      console.log(JSON.stringify(await evaluate(cdp, expression)));
    } else if (options.command === "input") {
      const selector = JSON.stringify(required(options, "selector"));
      const value = JSON.stringify(required(options, "value"));
      const expression = `(() => { const node = document.querySelector(${selector}); if (!node) throw new Error('Input not found: ' + ${selector}); const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set || Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(node, ${value}); node.dispatchEvent(new Event('input', { bubbles: true })); node.dispatchEvent(new Event('change', { bubbles: true })); return { selector: ${selector}, value: node.value }; })()`;
      console.log(JSON.stringify(await evaluate(cdp, expression)));
    } else if (options.command === "select") {
      const selector = JSON.stringify(required(options, "selector"));
      const value = JSON.stringify(required(options, "value"));
      const expression = `(() => { const node = document.querySelector(${selector}); if (!node) throw new Error('Select not found: ' + ${selector}); node.value = ${value}; node.dispatchEvent(new Event('input', { bubbles: true })); node.dispatchEvent(new Event('change', { bubbles: true })); return { selector: ${selector}, value: node.value }; })()`;
      console.log(JSON.stringify(await evaluate(cdp, expression)));
    } else if (options.command === "upload") {
      const selector = required(options, "selector");
      const file = required(options, "file");
      await cdp.call("DOM.enable");
      const documentNode = await cdp.call("DOM.getDocument", { depth: -1 });
      const node = await cdp.call("DOM.querySelector", {
        nodeId: documentNode.root.nodeId,
        selector,
      });
      if (!node.nodeId) throw new Error(`File input not found: ${selector}`);
      await cdp.call("DOM.setFileInputFiles", { nodeId: node.nodeId, files: [file] });
      console.log(JSON.stringify({ selector, file }));
    } else if (options.command === "reload") {
      await cdp.call("Page.enable");
      await cdp.call("Page.reload", { ignoreCache: false });
      console.log(JSON.stringify({ reloaded: target.url }));
    } else if (options.command === "screenshot") {
      await cdp.call("Page.enable");
      const result = await cdp.call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
      const output = required(options, "output");
      await writeFile(output, Buffer.from(result.data, "base64"));
      console.log(JSON.stringify({ output, bytes: Buffer.byteLength(result.data, "base64") }));
    } else usage();
  } finally {
    await cdp.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
