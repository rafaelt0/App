#!/usr/bin/env node
// Dependency-free Chrome DevTools Protocol driver for verify-b3lab.
// Normally invoked through `b3lab-verify browser <command> ...`, which sets
// CDP_PORT, B3LAB_URL, and DOWNLOAD_DIR from the run's state file.

const usage = `usage: cdp.mjs <command> [args]
  open <path|url>                         navigate and wait for document load
  click <name> [--role R] [--contains] [--nth N] [--download]
                                          real mouse click on a visible, enabled element
                                          roles: button (default), link, option, checkbox, summary, any
  fill <label> <value>                    focus a labelled input, select all, type value
  set <label> <value>                     set a range/number input via native setter + input event
  select <label> <option>...              choose option(s) by visible text or value in a <select>
  press <key>                             Enter | Escape | Tab | ArrowDown | ArrowUp
  wait-text <text> [--gone] [--timeout S] poll visible page text (default 120 s)
  wait-idle [--timeout S]                 wait until no loading spinner is visible for 750 ms
  audit                                   print JSON layout/a11y metrics for the current viewport
  text [selector]                         print visible text (default: main)
  snapshot <path>                         write accessibility tree as indented text
  screenshot <path> [--full]              write PNG of viewport (or full page)
  viewport <width> <height>               resize the browser window
  eval <expression>                       READ-ONLY inspection; never use to drive the app`;

const [command, ...rest] = process.argv.slice(2);
if (!command || command === "-h" || command === "--help") { console.log(usage); process.exit(command ? 0 : 2); }
const port = process.env.CDP_PORT;
if (!port) fail("CDP_PORT is not set. Run through `b3lab-verify browser ...`.");

function fail(message) { console.error(`cdp: ${message}`); process.exit(1); }
function flag(name) { const i = rest.indexOf(name); if (i < 0) return false; rest.splice(i, 1); return true; }
function option(name, fallback) { const i = rest.indexOf(name); if (i < 0) return fallback; const [, value] = rest.splice(i, 2); return value; }
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const version = await fetch(`http://127.0.0.1:${port}/json/version`).then(r => r.json()).catch(() => fail(`no browser on CDP port ${port}`));
const socket = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let nextId = 0;
const pending = new Map();
const listeners = new Set();
socket.onmessage = event => {
  const message = JSON.parse(event.data);
  if (message.id !== undefined) {
    const entry = pending.get(message.id); pending.delete(message.id);
    message.error ? entry.reject(new Error(`${entry.method}: ${message.error.message}`)) : entry.resolve(message.result);
  } else for (const listener of listeners) listener(message);
};
function send(method, params = {}, sessionId) {
  const id = ++nextId;
  socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject, method }));
}
const { targetInfos } = await send("Target.getTargets");
let target = targetInfos.find(info => info.type === "page");
if (!target) target = { targetId: (await send("Target.createTarget", { url: "about:blank" })).targetId };
const { sessionId } = await send("Target.attachToTarget", { targetId: target.targetId, flatten: true });
const page = (method, params) => send(method, params, sessionId);

async function evaluate(expression) {
  const result = await page("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
  return result.result.value;
}

// Page-side lookup by role and accessible-ish name (aria-label, label text, or visible text).
const helpers = String.raw`
const norm = s => (s ?? "").replace(/\s+/g, " ").trim();
const visible = el => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
// Name excludes nested controls and the control's aria-describedby hints (description, not name).
const labelText = (label, el) => { const copy = label.cloneNode(true); const described = (el?.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean);
  copy.querySelectorAll("input,select,textarea,output,ul,button").forEach(n => n.remove()); described.forEach(id => copy.querySelector("[id=\"" + id + "\"]")?.remove()); return norm(copy.textContent); };
const nameOf = el => {
  if (el.getAttribute("aria-label")) return norm(el.getAttribute("aria-label"));
  if (el.matches("input,select,textarea")) { const label = el.labels?.[0]; return label ? labelText(label, el) : ""; }
  return norm(el.innerText);
};
const selectors = { button: "button,[role=button]", link: "a[href]", option: "[role=option] button,[role=option]", checkbox: "input[type=checkbox]", summary: "summary", any: "button,[role=button],a[href],[role=option],summary,input,select,textarea" };
function find(role, name, contains, nth) {
  const wanted = norm(name);
  const all = [...document.querySelectorAll(selectors[role] ?? role)].filter(visible);
  const hits = all.filter(el => contains ? nameOf(el).includes(wanted) : nameOf(el) === wanted);
  if (!hits.length) throw new Error("no visible " + role + " named " + JSON.stringify(wanted) + (contains ? " (contains)" : "") + "; candidates: " + JSON.stringify(all.map(nameOf).filter(Boolean).slice(0, 40)));
  if (nth >= hits.length) throw new Error(hits.length + " match(es) for " + JSON.stringify(wanted) + ", --nth " + nth + " out of range");
  return hits[nth];
}
function control(name) {
  const wanted = norm(name);
  const controls = [...document.querySelectorAll("input,select,textarea")].filter(visible);
  const hits = controls.filter(el => nameOf(el) === wanted);
  if (!hits.length) throw new Error("no visible control labelled " + JSON.stringify(wanted) + "; labels: " + JSON.stringify(controls.map(nameOf).filter(Boolean)));
  return hits[0];
}
function point(el) {
  el.scrollIntoView({ block: "center", inline: "center" });
  const box = el.getBoundingClientRect();
  const x = box.left + box.width / 2, y = box.top + box.height / 2;
  const hit = document.elementFromPoint(x, y);
  if (!hit || !(el === hit || el.contains(hit) || hit.contains(el) || (el.labels && [...el.labels].some(l => l.contains(hit))))) throw new Error("element is covered by " + (hit?.outerHTML.slice(0, 120) ?? "nothing"));
  return { x, y, disabled: !!el.disabled || el.getAttribute("aria-disabled") === "true" };
}`;

async function mouseClick(x, y) {
  await page("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await page("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await page("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
}
const keys = {
  Enter: { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r" },
  Escape: { key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 },
  Tab: { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 },
  ArrowDown: { key: "ArrowDown", code: "ArrowDown", windowsVirtualKeyCode: 40 },
  ArrowUp: { key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38 },
};
async function press(name) {
  const key = keys[name] ?? fail(`unknown key ${name}`);
  await page("Input.dispatchKeyEvent", { type: key.text ? "keyDown" : "rawKeyDown", ...key });
  await page("Input.dispatchKeyEvent", { type: "keyUp", ...key, text: undefined });
}
async function waitLoaded(timeout = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await evaluate("document.readyState").catch(() => "") === "complete") return;
    await sleep(200);
  }
  fail("page did not finish loading");
}

try {
  switch (command) {
    case "open": {
      const [where] = rest;
      if (!where) fail("open needs a path or URL");
      const url = /^https?:/.test(where) ? where : new URL(where, process.env.B3LAB_URL ?? fail("B3LAB_URL is not set")).href;
      const result = await page("Page.navigate", { url });
      if (result.errorText) fail(`navigation failed: ${result.errorText}`);
      await sleep(300);
      await waitLoaded();
      console.log(`opened ${url}`);
      break;
    }
    case "click": {
      const role = option("--role", "button"), nth = Number(option("--nth", "0")), contains = flag("--contains"), download = flag("--download");
      const [name] = rest;
      if (!name) fail("click needs a name");
      const target = await evaluate(`(() => { ${helpers}; return point(find(${JSON.stringify(role)}, ${JSON.stringify(name)}, ${contains}, ${nth})); })()`);
      if (target.disabled) fail(`${role} ${JSON.stringify(name)} is disabled`);
      let finished;
      if (download) {
        const dir = process.env.DOWNLOAD_DIR ?? fail("DOWNLOAD_DIR is not set");
        await send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: dir, eventsEnabled: true });
        let filename = "";
        finished = new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error("download did not complete within 60 s")), 60000);
          listeners.add(message => {
            if (message.method === "Browser.downloadWillBegin") filename = message.params.suggestedFilename;
            if (message.method === "Browser.downloadProgress" && message.params.state === "completed") { clearTimeout(timer); resolve(`${dir}/${filename}`); }
            if (message.method === "Browser.downloadProgress" && message.params.state === "canceled") { clearTimeout(timer); reject(new Error("download canceled")); }
          });
        });
      }
      await mouseClick(target.x, target.y);
      console.log(`clicked ${role} ${JSON.stringify(name)}`);
      if (finished) console.log(`downloaded ${await finished}`);
      break;
    }
    case "fill": {
      const [label, value] = rest;
      if (label === undefined || value === undefined) fail("fill needs <label> <value>");
      const target = await evaluate(`(() => { ${helpers}; const el = control(${JSON.stringify(label)}); const p = point(el); el.focus(); return p; })()`);
      if (target.disabled) fail(`control ${JSON.stringify(label)} is disabled`);
      await page("Input.dispatchKeyEvent", { type: "rawKeyDown", key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2, commands: ["selectAll"] });
      await page("Input.dispatchKeyEvent", { type: "keyUp", key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2 });
      // Replace the selection in one edit: controlled inputs such as Number(value) re-render "0" if cleared first.
      if (value) await page("Input.insertText", { text: value });
      else {
        await page("Input.dispatchKeyEvent", { type: "rawKeyDown", key: "Backspace", code: "Backspace", windowsVirtualKeyCode: 8 });
        await page("Input.dispatchKeyEvent", { type: "keyUp", key: "Backspace", code: "Backspace", windowsVirtualKeyCode: 8 });
      }
      const now = await evaluate("document.activeElement?.value ?? ''");
      console.log(`filled ${JSON.stringify(label)} -> ${JSON.stringify(now)}`);
      break;
    }
    case "set": {
      const [label, value] = rest;
      if (label === undefined || value === undefined) fail("set needs <label> <value>");
      const now = await evaluate(`(() => { ${helpers}; const el = control(${JSON.stringify(label)}); if (el.disabled) throw new Error("control is disabled"); point(el);
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, ${JSON.stringify(value)});
        el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); return el.value; })()`);
      console.log(`set ${JSON.stringify(label)} -> ${JSON.stringify(now)}`);
      break;
    }
    case "select": {
      const [label, ...choices] = rest;
      if (!label || !choices.length) fail("select needs <label> <option>...");
      const chosen = await evaluate(`(() => { ${helpers}; const el = control(${JSON.stringify(label)}); if (el.tagName !== "SELECT") throw new Error("control is not a <select>"); if (el.disabled) throw new Error("select is disabled"); point(el);
        const wanted = ${JSON.stringify(choices)}.map(norm);
        const options = [...el.options];
        for (const want of wanted) if (!options.some(o => norm(o.text) === want || o.value === want)) throw new Error("no option " + JSON.stringify(want) + "; options: " + JSON.stringify(options.map(o => norm(o.text))));
        if (!el.multiple && wanted.length > 1) throw new Error("select is single-choice");
        const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set;
        if (el.multiple) options.forEach(o => { o.selected = wanted.includes(norm(o.text)) || wanted.includes(o.value); });
        else setter.call(el, options.find(o => norm(o.text) === wanted[0] || o.value === wanted[0]).value);
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return [...el.selectedOptions].map(o => norm(o.text)); })()`);
      console.log(`selected ${JSON.stringify(label)} -> ${JSON.stringify(chosen)}`);
      break;
    }
    case "wait-idle": {
      const timeout = Number(option("--timeout", "300")) * 1000;
      const start = Date.now();
      let quietSince = 0;
      while (Date.now() - start < timeout) {
        const busy = await evaluate(`[...document.querySelectorAll(".loading-spinner")].some(n => n.getClientRects().length)`).catch(() => true);
        if (busy) quietSince = 0;
        else if (!quietSince) quietSince = Date.now();
        else if (Date.now() - quietSince >= 750) { console.log(`idle after ${((Date.now() - start) / 1000).toFixed(1)} s`); process.exit(0); }
        await sleep(150);
      }
      fail(`still loading after ${timeout / 1000} s`);
      break;
    }
    case "audit": {
      console.log(JSON.stringify(await evaluate(`(() => { ${helpers};
        const shown = [...document.querySelectorAll("button,a[href],input:not([type=hidden]),select,textarea,summary")].filter(visible);
        const small = shown.filter(el => { const r = el.getBoundingClientRect(); const inline = el.matches("a") && getComputedStyle(el).display === "inline"; return !inline && !el.matches("input[type=checkbox],input[type=radio]") && (r.height < 44 || r.width < 24); });
        const unlabeled = shown.filter(el => el.matches("input,select,textarea") && !nameOf(el) && !el.getAttribute("aria-labelledby"));
        const overflowing = [...document.querySelectorAll("main *")].filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.right > innerWidth + 1 && !el.closest(".table-scroll,.table-wrap,.plotly-panel"); });
        return {
          url: location.pathname, viewport: innerWidth + "x" + innerHeight, docHeight: document.documentElement.scrollHeight,
          pageOverflowX: Math.max(0, document.documentElement.scrollWidth - innerWidth),
          overflowingElements: overflowing.slice(0, 8).map(el => el.tagName.toLowerCase() + (el.className && typeof el.className === "string" ? "." + el.className.split(" ")[0] : "")),
          smallTargets: small.length, smallTargetNames: [...new Set(small.map(el => (el.tagName.toLowerCase() + ":" + (nameOf(el) || el.className)).slice(0, 60)))].slice(0, 12),
          unlabeledControls: unlabeled.map(el => el.outerHTML.slice(0, 80)),
          h1: [...document.querySelectorAll("h1")].map(n => norm(n.innerText)), h2: [...document.querySelectorAll("main h2")].filter(visible).map(n => norm(n.innerText)),
          alerts: [...document.querySelectorAll("[role=alert]")].map(n => norm(n.innerText).slice(0, 100)),
        }; })()`)));
      break;
    }
    case "press": {
      const [key] = rest;
      await press(key ?? fail("press needs a key"));
      console.log(`pressed ${key}`);
      break;
    }
    case "wait-text": {
      const gone = flag("--gone"), timeout = Number(option("--timeout", "120")) * 1000;
      const [text] = rest;
      if (!text) fail("wait-text needs text");
      const start = Date.now();
      while (Date.now() - start < timeout) {
        const present = await evaluate(`document.body?.innerText.replace(/\\s+/g, " ").includes(${JSON.stringify(text.replace(/\s+/g, " "))}) ?? false`).catch(() => false);
        if (present !== gone) { console.log(`${gone ? "gone" : "found"} ${JSON.stringify(text)} after ${((Date.now() - start) / 1000).toFixed(1)} s`); process.exit(0); }
        await sleep(250);
      }
      const alerts = await evaluate(`[...document.querySelectorAll('[role=alert]')].map(n => n.innerText).join(" | ")`).catch(() => "");
      fail(`timed out after ${timeout / 1000} s waiting for ${JSON.stringify(text)} to ${gone ? "disappear" : "appear"}${alerts ? `; alerts: ${alerts}` : ""}`);
      break;
    }
    case "text": {
      const selector = rest[0] ?? "main";
      console.log(await evaluate(`document.querySelector(${JSON.stringify(selector)})?.innerText ?? ""`));
      break;
    }
    case "snapshot": {
      const [path] = rest;
      if (!path) fail("snapshot needs a path");
      await page("Accessibility.enable");
      const { nodes } = await page("Accessibility.getFullAXTree");
      const byId = new Map(nodes.map(node => [node.nodeId, node]));
      const lines = [`url: ${await evaluate("location.href")}`];
      const skip = new Set(["none", "generic", "InlineTextBox", "LineBreak"]);
      const walk = (node, depth) => {
        const role = node.role?.value ?? "";
        const name = node.name?.value ?? "";
        const shown = !node.ignored && !(skip.has(role) && !name) && !(role === "StaticText" && !name.trim());
        if (shown) {
          const props = (node.properties ?? []).filter(p => ["disabled", "checked", "expanded", "selected", "pressed", "focused", "level"].includes(p.name) && p.value.value !== false).map(p => `${p.name}=${p.value.value}`);
          const value = node.value?.value !== undefined && node.value.value !== "" ? ` value=${JSON.stringify(node.value.value)}` : "";
          lines.push(`${"  ".repeat(depth)}${role} ${JSON.stringify(name)}${value}${props.length ? ` [${props.join(" ")}]` : ""}`);
        }
        for (const child of node.childIds ?? []) { const next = byId.get(child); if (next) walk(next, shown ? depth + 1 : depth); }
      };
      walk(nodes[0], 0);
      const { writeFileSync, mkdirSync } = await import("node:fs");
      const { dirname } = await import("node:path");
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, lines.join("\n") + "\n");
      console.log(`snapshot ${path} (${lines.length} lines)`);
      break;
    }
    case "screenshot": {
      const full = flag("--full");
      const [path] = rest;
      if (!path) fail("screenshot needs a path");
      const { data } = await page("Page.captureScreenshot", { format: "png", captureBeyondViewport: full });
      const { writeFileSync, mkdirSync } = await import("node:fs");
      const { dirname } = await import("node:path");
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, Buffer.from(data, "base64"));
      console.log(`screenshot ${path}`);
      break;
    }
    case "viewport": {
      const [width, height] = rest.map(Number);
      if (!width || !height) fail("viewport needs <width> <height>");
      const { windowId } = await send("Browser.getWindowForTarget", { targetId: target.targetId });
      await send("Browser.setWindowBounds", { windowId, bounds: { width, height, windowState: "normal" } });
      console.log(`viewport ${await evaluate("innerWidth + 'x' + innerHeight")}`);
      break;
    }
    case "eval": {
      console.log(JSON.stringify(await evaluate(rest.join(" ")), null, 2));
      break;
    }
    default:
      fail(`unknown command ${command}\n${usage}`);
  }
} catch (error) {
  fail(error.message);
}
process.exit(0);
