// The shell's plumbing: a command registry the ribbon reads and documents write, the active
// document context, and portals into the docked panes (Properties, Problems).
//
// A document registers its commands with useCommands(); only the active document's layer is
// used, on top of the app-wide layer. The ribbon renders every command from ribbonSpec and
// looks up its binding here: no binding = disabled (with the spec's "when" hint).
import { createContext, ReactNode, useContext, useLayoutEffect, useRef, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";

export interface Binding {
  run?: () => void;                         // buttons
  checked?: boolean;                        // toggles and checkboxes
  disabled?: boolean | string;              // string = why (shown in the tooltip)
  value?: string | number;                  // fields
  options?: (string | [string, string])[];  // dropdown fields: value or [value, label]
  set?: (v: string) => void;                // fields and checkboxes ("true"/"false")
  label?: string;                           // dynamic label (status-like buttons)
  hidden?: boolean;
}
export type Bindings = Record<string, Binding | undefined>;

class Store {
  private layers = new Map<string, Bindings>();
  private activeKey = "";
  setActive(key: string) { if (key !== this.activeKey) { this.activeKey = key; this.bump(); } }
  private listeners = new Set<() => void>();
  private v = 0;
  private pending = false;
  subscribe = (f: () => void) => { this.listeners.add(f); return () => { this.listeners.delete(f); }; };
  version = () => this.v;
  set(owner: string, b: Bindings) { this.layers.set(owner, b); this.bump(); }
  remove(owner: string) { if (this.layers.delete(owner)) this.bump(); }
  get(id: string): Binding | undefined {
    const pre = `doc:${this.activeKey}:`;
    for (const [k, v] of this.layers) if (k.startsWith(pre) && v[id]) return v[id];
    return this.layers.get("app")?.[id];
  }
  private bump() {
    this.v++;
    if (this.pending) return;
    this.pending = true;
    queueMicrotask(() => { this.pending = false; this.listeners.forEach((f) => f()); });
  }
}
export const commands = new Store();
/** Re-render when any binding changes (ribbon, title bar, status bar). */
export function useRegistryVersion() { return useSyncExternalStore(commands.subscribe, commands.version); }
export function run(id: string) { const b = commands.get(id); if (b?.run && !b.disabled) { b.run(); return true; } return false; }

// ── active document ─────────────────────────────────────────────────────────
export interface DocInfo { key: string; active: boolean }
export const DocCtx = createContext<DocInfo>({ key: "", active: true });
export const useDoc = () => useContext(DocCtx);

/** Register ribbon bindings while this document is active. Call every render; `slot` lets
 *  several components of one document each contribute their own commands. */
export function useCommands(b: Bindings, slot = "main") {
  const { active, key } = useDoc();
  const ref = useRef(b); ref.current = b;
  const name = `doc:${key}:${slot}`;
  useLayoutEffect(() => { if (active) commands.set(name, ref.current); });
  useLayoutEffect(() => () => commands.remove(name), [name]);
}
export function useAppCommands(b: Bindings) {
  const ref = useRef(b); ref.current = b;
  useLayoutEffect(() => { commands.set("app", ref.current); });
}

// ── pane portals ────────────────────────────────────────────────────────────
export interface PaneTargets { properties: HTMLElement | null; problems: HTMLElement | null; setProblemCount: (key: string, n: number) => void }
export const PaneCtx = createContext<PaneTargets>({ properties: null, problems: null, setProblemCount: () => {} });
/** Render children into the Properties pane while this document is active. */
export function ToProperties({ children }: { children: ReactNode }) {
  const { active } = useDoc(); const { properties } = useContext(PaneCtx);
  return active && properties ? createPortal(children, properties) : null;
}
export interface Problem { severity: "error" | "warning" | "info" | "ok"; where: string; message: string }
export function ToProblems({ items }: { items: Problem[] }) {
  const { active, key } = useDoc(); const { problems, setProblemCount } = useContext(PaneCtx);
  const n = items.filter((p) => p.severity === "error" || p.severity === "warning").length;
  useLayoutEffect(() => { if (active) setProblemCount(key, n); }, [active, key, n, setProblemCount]);
  if (!active || !problems) return null;
  return createPortal(
    <table className="grid"><thead><tr><th style={{ width: 22 }} /><th style={{ width: 80 }}>severity</th><th style={{ width: 260 }}>where</th><th>message</th></tr></thead>
      <tbody>{items.map((p, i) => (
        <tr key={i}><td><SevIcon s={p.severity} /></td><td>{p.severity}</td><td className="mono">{p.where}</td><td>{p.message}</td></tr>
      ))}{items.length === 0 && <tr><td /><td colSpan={3} className="muted">No problems.</td></tr>}</tbody></table>, problems);
}
function SevIcon({ s }: { s: Problem["severity"] }) {
  const c = { error: "#C42B1C", warning: "#C27C00", info: "#2F6FDD", ok: "#2E8B3D" }[s];
  return <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 5, background: c }} />;
}

// ── app-wide UI state shared with documents (filters, follow, density) ──────
export interface UiState {
  jobKind: string; jobScene: string; jobStatus: string; hideFinished: boolean; follow: boolean; wrap: boolean;
  focus: number | null;   // job selected in the Output panel / Monitor (Cancel, Re-run, Save log act on it)
}
export const UiCtx = createContext<{ ui: UiState; setUi: (p: Partial<UiState>) => void }>({
  ui: { jobKind: "all", jobScene: "all", jobStatus: "all", hideFinished: false, follow: true, wrap: true, focus: null }, setUi: () => {},
});
export const useUi = () => useContext(UiCtx);
