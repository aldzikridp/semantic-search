// Gortex extension for the Pi coding agent (earendil-works/pi).
//
// Pi has no MCP support by design — the extension API's registerTool
// covers it. This extension does two things:
//
//   1. Exposes Gortex's graph tools as native Pi tools through a
//      persistent MCP stdio bridge: one `gortex mcp` child per session,
//      spoken to over standard JSON-RPC 2.0. The eager tools/list
//      surface is registered up front; the deferred catalogue is
//      reached via the server's own `tools_search` tool, whose
//      promotions are re-synced into Pi's registry on
//      notifications/tools/list_changed.
//
//   2. Re-creates the "prefer graph tools over raw file reads"
//      enforcement by forwarding lifecycle events to
//      `gortex hook --agent=pi` and applying its decision — the Go side
//      owns the postures, identical to every other agent.
//
// This file is written verbatim by the `pi` agent adapter, except for
// four templated sentinels replaced with JSON literals at install time:
//
//   "/nix/store/4s7z445yjfp0nsshxzsaavb4y3lwsyzm-gortex-0.63.8/bin/gortex"          -> string: resolved `gortex` binary path
//   ["/nix/store/4s7z445yjfp0nsshxzsaavb4y3lwsyzm-gortex-0.63.8/bin/gortex","hook","--agent=pi"]    -> array: full hook argv
//   true      -> boolean: wire read-discipline enforcement
//                              (false when installed with --no-hooks)
//   "core" -> string: default eager tool preset;
//                              GORTEX_TOOLS overrides it at runtime
//
// NOTE: Pi is pre-1.0. The event names and built-in tool names mapped in
// normalizeToolCall() below are pinned to Pi's documented API
// (packages/coding-agent/docs/extensions.md); if an upgrade renames one,
// adjust the small maps here — the Go wire contract stays unchanged.

// Zero runtime dependencies, by design: a single-file auto-discovered
// extension has no node_modules, so any non-built-in import would throw
// at load and take the whole extension down. Tool `parameters` are the
// MCP server's JSON-Schema objects, passed verbatim.
import { execFileSync, spawn } from "node:child_process";

const GORTEX_BIN: string = "~/.bin/gortex";
const HOOK_ARGV: string[] = ["~/.bin/gortex","hook","--agent=pi"];
const ENFORCE: boolean = true;

// Which eager preset to expose as native Pi tools. The daemon itself
// defaults to the `core` preset in `defer` mode, so for `core` (and
// `full`) the bridge passes nothing and the daemon's own surface applies.
// A non-default preset (edit|nav|readonly) is forwarded to the `gortex
// mcp` proxy via GORTEX_TOOLS, always with GORTEX_TOOLS_MODE=defer — in
// hide mode the proxy would -32601 calls to tools promoted later by
// tools_search. Anything unrecognised (a typo, a stale baked value) is
// NOT forwarded: server-side it would parse as a one-tool allow-list and
// silently filter every promoted tool out of the session — fail open to
// the daemon's default surface instead. GORTEX_TOOLS in the environment
// overrides the value baked at install time.
const TOOLS_PRESET: string = ((process.env && process.env.GORTEX_TOOLS) || "core").trim();

// Identity reported as MCP clientInfo, plus the gortex/wire capability
// declared in the initialize request: the daemon reads it to learn which
// compact wire formats this client decodes, so list-shaped tools default
// to GCX1 for this session without a server-side client-name allowlist
// entry.
const CLIENT_NAME = "pi";
const CLIENT_VERSION = "1.0.0";
const WIRE_FORMATS: string[] = ["gcx"];

// Names the Gortex tools are registered under in Pi — usually the bare
// daemon name, except for a few aliased to dodge Pi's built-ins (see
// piAliasName). Tracks the post-alias name.
const gortexToolNames = new Set<string>();

// ---------------------------------------------------------------------------
// Daemon lifecycle
// ---------------------------------------------------------------------------

// ensureDaemon brings the shared Gortex daemon up before the MCP child
// dials it. Idempotent, fire-and-forget (we don't block on the launcher's
// ≤60s readiness poll), and never throws — a missing binary or spawn
// failure must not take the extension down. The bridge's initialize retry
// absorbs the warm-up window.
function ensureDaemon(): void {
  try {
    const child = spawn(GORTEX_BIN, ["daemon", "start", "--detach"], {
      stdio: "ignore",
      detached: true,
    });
    child.on("error", () => { }); // binary missing / spawn failure — swallow
    // No teardown counterpart, by design: the daemon is shared, long-lived
    // infrastructure that outlives the session.
    child.unref();
  } catch {
  }
}

// ---------------------------------------------------------------------------
// MCP stdio client (newline-delimited JSON-RPC 2.0 over a `gortex mcp` child)
// ---------------------------------------------------------------------------

const INIT_TIMEOUT_MS = 60_000;
const RPC_TIMEOUT_MS = 30_000;
// tools/call cap: generous enough for long analyzers (sast sweeps,
// reviews, enrich passes), finite so a wedged daemon behind a
// still-alive child can't hang the agent turn forever.
const CALL_TIMEOUT_MS = 600_000;
// A single JSON-RPC frame past this cap kills the child: the stream
// can't be resynced mid-frame, and the old shell-out path errored at
// execFileSync's 64 MiB maxBuffer for the same reason.
const MAX_BUFFER_BYTES = 64 * 1024 * 1024;

interface PendingRequest {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
  timer?: ReturnType<typeof setTimeout>;
}

class MCPStdioClient {
  private child: any = null;
  private buffer = "";
  private searchStart = 0;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private exited = true;
  private respawnPromise: Promise<void> | null = null;

  // Fired on notifications/tools/list_changed (server-side promotion).
  onToolsListChanged: (() => void) | null = null;

  private childEnv(): Record<string, string | undefined> {
    const env: Record<string, string | undefined> = { ...process.env };
    const preset = TOOLS_PRESET.toLowerCase();
    // The daemon's own default surface already IS core/defer; only a
    // recognised non-default preset needs the proxy-side narrowing.
    // Unknown values are dropped (fail open) — forwarded verbatim they
    // would parse as a one-tool allow-list and leave the session with
    // no callable graph tools at all.
    const FORWARDED_PRESETS = new Set(["edit", "nav", "readonly"]);
    if (!env.GORTEX_TOOLS && FORWARDED_PRESETS.has(preset)) {
      env.GORTEX_TOOLS = TOOLS_PRESET;
    }
    // Never let a preset hide-block tools promoted later by tools_search.
    if (env.GORTEX_TOOLS && !env.GORTEX_TOOLS_MODE) env.GORTEX_TOOLS_MODE = "defer";
    return env;
  }

  private spawnChild(): void {
    const child = spawn(GORTEX_BIN, ["mcp"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: this.childEnv(),
    });
    this.child = child;
    this.buffer = "";
    this.searchStart = 0;
    this.exited = false;
    child.stdout.on("data", (chunk: Buffer) => this.onData(chunk));
    // Consume stderr so the child never blocks on a full pipe; routing it
    // to the terminal would corrupt Pi's TUI.
    child.stderr.on("data", () => { });
    child.stdin.on("error", () => { }); // EPIPE race: child may exit before stdin.write() finishes
    child.on("error", () => this.markExited(new Error("gortex mcp spawn failed")));
    child.on("exit", () => this.markExited(new Error("gortex mcp exited")));
  }

  private markExited(err: Error): void {
    if (this.exited && this.pending.size === 0) return;
    this.exited = true;
    this.child = null;
    for (const [, p] of this.pending) {
      if (p.timer) clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
  }

  private onData(chunk: Buffer): void {
    this.buffer += chunk.toString("utf8");
    let nl: number;
    // Resume the newline scan where the previous chunk left off — a
    // large frame split across many chunks must not re-scan the whole
    // growing buffer from index 0 on every data event.
    while ((nl = this.buffer.indexOf("\n", this.searchStart)) >= 0) {
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      this.searchStart = 0;
      if (!line) continue;
      let msg: any;
      try {
        msg = JSON.parse(line);
      } catch {
        continue; // non-JSON noise — ignore
      }
      this.dispatch(msg);
    }
    this.searchStart = this.buffer.length;
    if (this.buffer.length > MAX_BUFFER_BYTES) {
      // One frame past the cap is fatal for this child (no way to
      // resync mid-frame): pending requests reject, the next call
      // respawns a fresh child.
      this.buffer = "";
      this.searchStart = 0;
      const child = this.child;
      this.markExited(new Error(`gortex mcp frame exceeded ${MAX_BUFFER_BYTES} bytes`));
      try {
        child?.kill();
      } catch {
      }
    }
  }

  private dispatch(msg: any): void {
    if (msg && typeof msg.id === "number" && this.pending.has(msg.id)) {
      const p = this.pending.get(msg.id)!;
      this.pending.delete(msg.id);
      if (p.timer) clearTimeout(p.timer);
      if (msg.error) {
        p.reject(new Error(msg.error.message || `JSON-RPC error ${msg.error.code}`));
      } else {
        p.resolve(msg.result);
      }
      return;
    }
    if (msg && msg.method === "notifications/tools/list_changed") {
      try {
        this.onToolsListChanged?.();
      } catch {
        // best effort — never break the read loop.
      }
    }
  }

  private send(obj: Record<string, unknown>): void {
    if (!this.child || this.exited) throw new Error("gortex mcp is not running");
    this.child.stdin.write(JSON.stringify(obj) + "\n");
  }

  // A single shared respawn: concurrent callers of request() against a
  // dead child all await the same spawn+initialize, so a burst of tool
  // calls can never fork multiple children.
  private respawn(): Promise<void> {
    if (!this.respawnPromise) {
      this.respawnPromise = (async () => {
        this.spawnChild();
        await this.initialize();
      })().finally(() => {
        this.respawnPromise = null;
      });
    }
    return this.respawnPromise;
  }

  async request(method: string, params: unknown, timeoutMs: number = RPC_TIMEOUT_MS): Promise<any> {
    if (this.exited) await this.respawn();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const entry: PendingRequest = { resolve, reject };
      if (Number.isFinite(timeoutMs)) {
        entry.timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`${method} timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }
      this.pending.set(id, entry);
      try {
        this.send({ jsonrpc: "2.0", id, method, params });
      } catch (err: any) {
        this.pending.delete(id);
        if (entry.timer) clearTimeout(entry.timer);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  private notify(method: string, params: unknown): void {
    try {
      this.send({ jsonrpc: "2.0", method, params });
    } catch {
      // notifications are best-effort.
    }
  }

  private async initialize(): Promise<void> {
    await this.request(
      "initialize",
      {
        protocolVersion: "2025-06-18",
        // gortex/wire self-declares this client's compact-format
        // decoders; the daemon prefers it over its name allowlist.
        capabilities: { experimental: { "gortex/wire": WIRE_FORMATS } },
        clientInfo: { name: CLIENT_NAME, version: CLIENT_VERSION },
      },
      INIT_TIMEOUT_MS,
    );
    this.notify("notifications/initialized", {});
  }

  // start spawns the child and runs the MCP handshake, retrying once
  // after a 1s backoff — the daemon may still be warming up right after
  // ensureDaemon() kicked it off.
  async start(): Promise<void> {
    this.spawnChild();
    try {
      await this.initialize();
    } catch {
      await new Promise((r) => setTimeout(r, 1_000));
      if (this.exited) this.spawnChild();
      await this.initialize();
    }
  }

  async listTools(): Promise<any[]> {
    const res = await this.request("tools/list", {});
    return Array.isArray(res?.tools) ? res.tools : [];
  }

  // Tool calls get a generous cap rather than none: long-running
  // analyzers (sast sweeps, reviews, enrich passes) must be able to
  // finish, but a wedged daemon behind a still-alive child must not
  // hang the agent turn forever.
  async callTool(name: string, args: Record<string, unknown>): Promise<any> {
    return this.request("tools/call", { name, arguments: args ?? {} }, CALL_TIMEOUT_MS);
  }

  stop(): void {
    const child = this.child;
    this.markExited(new Error("gortex mcp bridge stopped"));
    try {
      child?.kill();
    } catch {
    }
  }
}

// The session's live bridge; claimed synchronously at session_start (so
// an overlapping restart can stop it) and nulled when the handshake
// fails or the session is superseded.
let client: MCPStdioClient | null = null;

// Why the bridge is down this session (empty when it's up). Surfaced
// once through the orientation injection so the user learns that
// /reload retries the handshake.
let bridgeError = "";

// textFromResult joins the text parts of an MCP tools/call result.
function textFromResult(result: any): string {
  const parts = Array.isArray(result?.content) ? result.content : [];
  return parts
    .filter((c: any) => c && c.type === "text" && typeof c.text === "string")
    .map((c: any) => c.text)
    .join("\n");
}

interface PiDecision {
  block?: boolean;
  reason?: string;
  additional_context?: string;
  orientation?: string;
}

// callHook sends a normalized event envelope to `gortex hook --agent=pi`
// and parses the PiDecision it writes back. Fail-open: any error returns
// an empty decision so the extension never blocks Pi's flow on a hook
// hiccup (daemon down, parse error, timeout).
function callHook(envelope: Record<string, unknown>): PiDecision {
  try {
    const out = execFileSync(HOOK_ARGV[0], HOOK_ARGV.slice(1), {
      input: JSON.stringify(envelope),
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
      timeout: 5_000,
    }).trim();
    if (!out) return {};
    return JSON.parse(out) as PiDecision;
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Tool-call normalization (Pi vocabulary -> canonical Claude-Code vocabulary)
// ---------------------------------------------------------------------------
//
// The Go enrich() classifier switches on Claude-Code tool names
// ("Read"/"Grep"/"Glob"/"Bash"/"Edit"/"Write") and reads canonical input
// keys ("file_path"/"pattern"/"command"). Pi uses its own lowercase names
// and input shapes, so we translate here — keeping all Pi-specific
// knowledge in this Pi-specific file.
const toolNameMap: Record<string, string> = {
  read: "Read",
  grep: "Grep",
  find: "Glob",
  ls: "Glob",
  bash: "Bash",
  edit: "Edit",
  write: "Write",
};

function firstString(obj: Record<string, unknown>, keys: string[]): string | undefined {
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === "string" && v !== "") return v;
  }
  return undefined;
}

function normalizeToolCall(
  piName: string,
  piInput: Record<string, unknown>,
): { tool_name: string; tool_input: Record<string, unknown> } {
  const canonical = toolNameMap[piName.toLowerCase()] ?? piName;
  const out: Record<string, unknown> = { ...piInput };

  const path = firstString(piInput, ["file_path", "path", "file", "filepath", "absolute_path"]);
  if (path !== undefined) out.file_path = path;

  let pattern = firstString(piInput, ["pattern", "glob", "query", "regex", "name"]);
  if (pattern === undefined && canonical === "Glob") pattern = path;
  if (pattern !== undefined) out.pattern = pattern;

  const command = firstString(piInput, ["command", "cmd", "script"]);
  if (command !== undefined) out.command = command;

  return { tool_name: canonical, tool_input: out };
}

// ---------------------------------------------------------------------------
// Dynamic tool registration
// ---------------------------------------------------------------------------

interface ToolDescriptor {
  name: string;
  description?: string;
  inputSchema?: unknown;
}

// Pi's reserved built-in names (mirrors toolNameMap's keys). Pi lets an
// extension silently replace a built-in by reusing its name, but our
// generic bridge doesn't match the built-in's result shape — breaking
// Pi's own rendering and any extension hooked to the built-in (e.g. a
// lint-on-edit plugin). Only "edit"/"read" collide today; checking all
// seven guards against future facade additions too.
const PI_RESERVED_TOOL_NAMES = new Set(Object.keys(toolNameMap));
const PI_ALIAS_PREFIX = "gortex_";

// piAliasName is unchanged unless name collides with a built-in above, in
// which case it's registered under a `gortex_`-prefixed alias instead.
function piAliasName(name: string): string {
  return PI_RESERVED_TOOL_NAMES.has(name) ? PI_ALIAS_PREFIX + name : name;
}

// piAliasNote tells the model which tools got aliased this session, so it
// can translate bare names in Gortex's own guidance. Pi-local by design —
// not a fact any other agent needs.
function piAliasNote(): string {
  const aliased = Array.from(gortexToolNames).filter((n) => n.startsWith(PI_ALIAS_PREFIX));
  if (aliased.length === 0) return "";
  const pairs = aliased
    .map((n) => `\`${n.slice(PI_ALIAS_PREFIX.length)}\` -> \`${n}\``)
    .join(", ");
  return (
    `[Gortex] This session renamed these Gortex tools to avoid colliding with Pi's own built-ins ` +
    `of the same name: ${pairs}. Wherever Gortex's guidance or tool descriptions mention the bare name, call the renamed one instead.`
  );
}

// safeRegister registers under the alias name (piAliasName), tolerating a
// redundant registration — a reload can re-fire session_start without
// resetting the registry. Returns the name actually registered.
function safeRegister(pi: any, def: any): string {
  def.name = piAliasName(def.name);
  def.label = def.name;
  try {
    pi.registerTool(def);
  } catch {
    // already live this session — the existing registration stands.
  }
  return def.name;
}

// Lazily resolved TUI Text component, used only to collapse a gortex
// tool's result to nothing until the user expands it (ctrl+o). Resolved
// via require (not a static import) so a Pi packaging that can't
// provide @earendil-works/pi-tui to a dependency-free single-file
// extension degrades to Pi's default renderer instead of throwing at
// load and taking tool registration down with it — the same hazard the
// file's zero-runtime-dependency design already avoids for typebox.
let TuiText: (new (text: string, x: number, y: number) => any) | undefined;
try {
  TuiText = require("@earendil-works/pi-tui").Text;
} catch {
  TuiText = undefined;
}

// registerOneTool registers a single Gortex tool as a native Pi tool
// under its bare daemon name, whose execute() forwards to the MCP
// bridge's tools/call. The MCP input schema is passed verbatim as Pi's
// parameters (Pi validates plain JSON Schema), so the model sees real
// parameter docs. Idempotent — a name already registered is skipped.
// Adds the name to gortexToolNames so the read-discipline postures treat
// a call to it as a graph query, not a raw file read. Used both for the
// eager tools/list surface and for tools promoted later by a
// tools_search call (picked up via the list_changed re-sync).
//
// renderResult keeps the collapsed view to just the tool name (Pi's
// fallback renderCall already shows that) — no preview, no summary.
// ctrl+o (expand) is the only way to see the actual output.
function registerOneTool(pi: any, desc: ToolDescriptor): void {
  const name = desc.name;
  if (!name) return;
  if (gortexToolNames.has(piAliasName(name))) return;
  const parameters =
    desc.inputSchema && typeof desc.inputSchema === "object"
      ? desc.inputSchema
      : { type: "object", properties: {} };
  const def: any = {
    name,
    label: name,
    description: (desc.description || name).trim(),
    parameters,
    async execute(_id: string, params: Record<string, unknown>) {
      if (!client) throw new Error(`gortex ${name}: MCP bridge is not connected`);
      let result: any;
      try {
        result = await client.callTool(name, params ?? {});
      } catch (err: any) {
        throw new Error(`gortex ${name} failed: ${err?.message || String(err)}`);
      }
      if (name === "tools_search") {
        // The daemon just promoted the matches and fired list_changed;
        // register them NOW (bypassing the debounce) so every tool the
        // reply cites is already callable when the model reads it.
        await syncTools(pi);
      }
      const text = textFromResult(result) ||
        JSON.stringify(result?.structuredContent ?? result ?? {});
      if (result?.isError) throw new Error(text || `gortex ${name} failed`);
      return { content: [{ type: "text", text }], details: {} };
    },
  };
  if (TuiText) {
    def.renderResult = (result: any, opts: { expanded?: boolean }) => {
      if (!opts?.expanded) return new TuiText!("", 0, 0);
      const text = textFromResult(result) ||
        JSON.stringify(result?.structuredContent ?? result ?? {});
      return new TuiText!(text, 0, 0);
    };
  }
  gortexToolNames.add(safeRegister(pi, def));
}

// registerGortexTools fetches the daemon's eager surface (tools/list —
// the configured preset, full schemas included) and registers each entry,
// including the server's `tools_search` discovery tool.
async function registerGortexTools(pi: any): Promise<void> {
  if (!client) return;
  for (const desc of await client.listTools()) {
    if (!desc) continue;
    registerOneTool(pi, desc);
  }
}

// syncTools re-fetches tools/list after a notifications/tools/list_changed
// and registers anything new. This is how tools promoted by a
// tools_search call become callable Pi tools.
async function syncTools(pi: any): Promise<void> {
  try {
    await registerGortexTools(pi);
  } catch {
    // transient — the next promotion or session retries.
  }
}

// The daemon fires one list_changed per promoted tool, so a tools_search
// sweep arrives as a notification burst; debounce to a single trailing
// re-fetch instead of one full tools/list round-trip per notification.
// (The tools_search execute() path awaits syncTools directly, bypassing
// the debounce, so the reply never waits on this timer.)
const SYNC_DEBOUNCE_MS = 200;
let syncTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleSyncTools(pi: any): void {
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(() => {
    syncTimer = null;
    void syncTools(pi);
  }, SYNC_DEBOUNCE_MS);
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: any) {
  let orientationInjected = false;
  // Orientation awaiting injection into the next LLM call. The `context`
  // hook appends it as a tail user message rather than mutating
  // systemPrompt: a systemPrompt change sits at messages[0] and invalidates
  // prefix prompt caching. Computed once per session.
  let pendingOrientation = "";

  // Pi resets the session's tool registry on every session_start, so
  // (re)register here. Clear the name guard first — it persists across
  // sessions and would otherwise suppress re-registration. The previous
  // session's bridge child (if any) is stopped before a fresh handshake.
  pi.on("session_start", async () => {
    orientationInjected = false;
    pendingOrientation = "";
    bridgeError = "";
    ensureDaemon();
    gortexToolNames.clear();
    if (syncTimer) {
      clearTimeout(syncTimer);
      syncTimer = null;
    }
    if (client) {
      client.stop();
      client = null;
    }
    const c = new MCPStdioClient();
    // Claim the slot before the first await: an overlapping
    // session_start (rapid /new or /reload) then sees and stops THIS
    // bridge instead of leaking its child mid-handshake.
    client = c;
    try {
      await c.start();
      if (client !== c) return; // superseded while handshaking
      c.onToolsListChanged = () => {
        scheduleSyncTools(pi);
      };
      await registerGortexTools(pi);
    } catch (err: any) {
      // daemon unreachable / binary missing / handshake or tools/list
      // failure — no graph tools this session; /reload (or the next
      // session_start) retries.
      c.stop();
      if (client === c) {
        client = null;
        bridgeError = err?.message || String(err);
      }
    }
  });

  // Fires before the agent loop's first LLM call. It can't mutate messages
  // itself, so it just parks the orientation for the `context` hook.
  pi.on("before_agent_start", () => {
    if (orientationInjected) return;
    const decision = callHook({ event: "session_start", cwd: pi?.cwd ?? process.cwd() });
    const parts: string[] = [];
    if (bridgeError) {
      parts.push(
        `[Gortex] graph tools are unavailable this session (${bridgeError}). ` +
        `Tell the user to run /reload to retry the connection.`,
      );
      bridgeError = "";
    }
    if (decision.orientation) parts.push(decision.orientation);
    const aliasNote = piAliasNote();
    if (aliasNote) parts.push(aliasNote);
    if (parts.length > 0) {
      pendingOrientation = parts.join("\n\n");
      orientationInjected = true;
    }
    return;
  });

  // Fires before each LLM call with a mutable message array. Append the
  // parked orientation once, then clear it so it isn't repeated each call.
  pi.on("context", (event: any) => {
    if (!pendingOrientation) return;
    const text = pendingOrientation;
    pendingOrientation = "";
    try {
      if (event && Array.isArray(event.messages)) {
        event.messages.push({ role: "user", content: text });
        return { messages: event.messages };
      }
    } catch {
      // best effort — never break context assembly.
    }
    return;
  });

  if (!ENFORCE) return;

  // Enforcement: every non-Gortex tool call is checked against the Go hook.
  pi.on("tool_call", async (event: any, ctx: any) => {
    const piName: string = event?.toolName ?? "";
    const piInput: Record<string, unknown> = event?.input ?? {};
    const isGortexTool = gortexToolNames.has(piName);

    const norm = normalizeToolCall(piName, piInput);
    const decision = callHook({
      event: "tool_call",
      tool_name: norm.tool_name,
      tool_input: norm.tool_input,
      cwd: ctx?.cwd ?? pi?.cwd ?? process.cwd(),
      session_id: ctx?.sessionManager?.sessionId ?? "",
      is_gortex_tool: isGortexTool,
    });

    if (decision.block) {
      return { block: true, reason: decision.reason ?? "[Gortex] blocked — prefer graph tools." };
    }
    if (decision.additional_context) {
      // Soft guidance: surface it without blocking the call.
      try {
        pi.sendMessage(
          { customType: "gortex", content: decision.additional_context, display: true },
          { deliverAs: "steer" },
        );
      } catch {
        // sendMessage shape can vary across Pi versions; never fatal.
      }
    }
    return;
  });
}
