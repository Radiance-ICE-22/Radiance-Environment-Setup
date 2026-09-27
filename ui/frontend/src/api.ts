// Thin client for the Galley backend (ui/backend/galley/app.py).

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";

export interface Job {
  id: number;
  kind: string;
  label: string;
  argv: string[];
  params: Record<string, unknown>;
  scene: string | null;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  returncode: number | null;
}

export interface StepState { step: string; done: boolean; when: string | null; fingerprint: string | null }
export interface Model { run: string; config: string; checkpoint: string | null; checkpoint_mb: number | null }
export interface ArchivedModel { run: string; complete: boolean; checkpoint_mb: number | null }
export interface SceneStatus { scene: string; steps: StepState[]; results: Record<string, any>; models: Model[] }
export interface SceneSummary { scene: string; has_workspace: boolean; models: number; loadable: boolean; has_state: boolean }
export interface Machine {
  gpu: { name: string; vram_mib: number; live: { used_mib: number; total_mib: number; util_pct: number; temp_c: number } | null };
  disk: { free_gb: number; total_gb: number } | null;
  paths: Record<string, string | null>;
  defaults: Record<string, unknown>;
}
export interface ConfigItem { name: string; modified: number; kind?: string }
export type Family = "captures" | "courses" | "pilots" | "frames" | "methods" | "nnio";
export const FAMILIES: Family[] = ["captures", "courses", "pilots", "frames", "methods", "nnio"];

export const STEPS = ["preflight", "probe", "transcode", "aruco", "config", "patch", "sfm", "train",
  "verify", "bounds", "course", "simulate", "validate", "record"] as const;
export type Step = (typeof STEPS)[number];

export interface FigsRun {
  scene: string;
  video?: string;
  marker_id?: number;
  marker_length?: number;
  num_images?: number;
  num_marked?: number;
  course?: string;
  frame?: string;
  pilot?: string;
  method?: string;
  margin?: number;
  allow_outside?: boolean;
  from_step?: Step;
  only?: Step;
  stop_after?: Step;
  redo?: Step[];
  train_iters?: number;
  downscale?: number;
  cache_images?: "cpu" | "gpu";
  train_vis?: "viewer" | "tensorboard" | "viewer+tensorboard";
  train_args?: string[];
  archive_old?: boolean;
}

const TOKEN_KEY = "galley.token";
export function getToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function setToken(t: string | null) {
  try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  const tok = getToken();
  if (tok) headers["Authorization"] = `Bearer ${tok}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail
        : Array.isArray(j.detail) ? j.detail.map((d: any) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ")
        : JSON.stringify(j);
    } catch { /* not json */ }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => req<{ ok: boolean; current_job: number | null; env_script: boolean; pipeline: boolean }>("GET", "/health"),
  machine: () => req<Machine>("GET", "/machine"),
  scenes: () => req<SceneSummary[]>("GET", "/scenes"),
  scene: (s: string) => req<SceneStatus>("GET", `/scenes/${encodeURIComponent(s)}`),
  runs: (scene?: string) => req<any[]>("GET", `/runs${scene ? `?scene=${encodeURIComponent(scene)}` : ""}`),
  videos: () => req<{ name: string; mb: number }[]>("GET", "/videos"),
  configs: (f: Family) => req<ConfigItem[]>("GET", `/configs/${f}`),
  config: (f: Family, n: string) => req<any>("GET", `/configs/${f}/${encodeURIComponent(n)}`),
  saveConfig: (f: Family, n: string, data: unknown, overwrite = true) =>
    req<{ path: string; mirrored: string | null }>("PUT", `/configs/${f}/${encodeURIComponent(n)}?overwrite=${overwrite}`, data),
  validateConfig: (f: Family, data: unknown) => req<{ ok: boolean }>("POST", `/configs/${f}/validate`, data),
  jobs: (scene?: string) => req<Job[]>("GET", `/jobs${scene ? `?scene=${encodeURIComponent(scene)}` : ""}`),
  job: (id: number) => req<Job>("GET", `/jobs/${id}`),
  submitFigs: (r: FigsRun) => req<{ id: number }>("POST", "/jobs/figs", r),
  submitSelftest: (seconds: number) => req<{ id: number }>("POST", "/jobs/selftest", { seconds }),
  cancel: (id: number) => req<Job>("POST", `/jobs/${id}/cancel`),
  models: (s: string) => req<{ active: Model[]; archived: ArchivedModel[] }>("GET", `/scenes/${encodeURIComponent(s)}/models`),
  archiveModel: (s: string, run: string) => req<{ cleared_steps: string[] }>("POST", `/scenes/${encodeURIComponent(s)}/models/${encodeURIComponent(run)}/archive`),
  promoteModel: (s: string, run: string) => req<{ archived: string[]; cleared_steps: string[] }>("POST", `/scenes/${encodeURIComponent(s)}/models/${encodeURIComponent(run)}/promote`),
  metrics: (s: string, run?: string) => req<{ run: string | null; series: Record<string, [number, number][]> }>(
    "GET", `/scenes/${encodeURIComponent(s)}/metrics${run ? `?run=${encodeURIComponent(run)}` : ""}`),
};

export function flightUrl(scene: string) {
  const tok = getToken();
  return `/api/scenes/${encodeURIComponent(scene)}/flight${tok ? `?token=${encodeURIComponent(tok)}` : ""}`;
}

export type StreamMsg =
  | { type: "line"; seq: number; line: string }
  | { type: "progress"; line: string }
  | { type: "status"; status: JobStatus; returncode?: number | null };

export function streamJob(id: number, onMsg: (m: StreamMsg) => void, onClose: () => void): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const tok = getToken();
  const ws = new WebSocket(`${proto}://${location.host}/api/jobs/${id}/stream${tok ? `?token=${encodeURIComponent(tok)}` : ""}`);
  ws.onmessage = (e) => onMsg(JSON.parse(e.data));
  ws.onclose = onClose;
  return () => ws.close();
}

export const active = (s: JobStatus) => s === "queued" || s === "running";

export function ago(ts: number | null): string {
  if (!ts) return "—";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} h ago`;
  return new Date(ts * 1000).toLocaleString();
}

export function duration(j: Job): string {
  if (!j.started) return "—";
  const s = (j.finished ?? Date.now() / 1000) - j.started;
  return s < 60 ? `${s.toFixed(0)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${(s / 3600).toFixed(1)} h`;
}
