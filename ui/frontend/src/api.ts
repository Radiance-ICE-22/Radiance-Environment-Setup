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

// ── video staging (ui/backend/galley/videos.py) ─────────────────────────────
export interface VideoFile { name: string; mb: number; bytes: number; modified: number }
export interface VideoProbe {
  codec: string | null; profile: string | null; pix_fmt: string; width: number | null; height: number | null; rotation: number;
  fps: number | null; avg_fps: number | null; vfr: boolean; duration: number | null; frames: number | null;
  bit_depth: number; hdr: boolean; color_transfer: string | null; audio: boolean; device: string | null; created: string | null; bytes: number | null;
}
export interface PartialUpload { id: string; name: string; size: number; offset: number; modified: number; created: number; updated: number; overwrite: boolean }
export interface UploadsInfo { dir: string; free_bytes: number | null; max_chunk: number; uploads: PartialUpload[] }
/** Allowed by the backend: letters, digits, _ - . ; a video extension. */
export const VIDEO_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.(mov|mp4|m4v|mkv|avi|webm|mts)$/i;
export const VIDEO_EXT = [".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".mts"];

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
        : j.detail && typeof j.detail.message === "string" ? j.detail.message
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
  videos: () => req<VideoFile[]>("GET", "/videos"),
  probeVideo: (name: string) => req<VideoProbe>("GET", `/videos/${encodeURIComponent(name)}/probe`),
  deleteVideo: (name: string) => req<{ deleted: string }>("DELETE", `/videos/${encodeURIComponent(name)}`),
  uploads: () => req<UploadsInfo>("GET", "/uploads"),
  abortUpload: (id: string) => req<{ aborted: string }>("DELETE", `/uploads/${id}`),
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
  log: (id: number, after = -1, limit = 20000) => req<{ seq: number; ts: number; line: string }[]>("GET", `/jobs/${id}/log?after=${after}&limit=${limit}`),
  models: (s: string) => req<{ active: Model[]; archived: ArchivedModel[] }>("GET", `/scenes/${encodeURIComponent(s)}/models`),
  archiveModel: (s: string, run: string) => req<{ cleared_steps: string[] }>("POST", `/scenes/${encodeURIComponent(s)}/models/${encodeURIComponent(run)}/archive`),
  promoteModel: (s: string, run: string) => req<{ archived: string[]; cleared_steps: string[] }>("POST", `/scenes/${encodeURIComponent(s)}/models/${encodeURIComponent(run)}/promote`),
  metrics: (s: string, run?: string) => req<{ run: string | null; series: Record<string, [number, number][]> }>(
    "GET", `/scenes/${encodeURIComponent(s)}/metrics${run ? `?run=${encodeURIComponent(run)}` : ""}`),
};

// ── course editor (Phase 3) ──────────────────────────────────────────────────
type V3 = [number, number, number];
export interface Geometry {
  scene: string;
  camera_path: V3[];
  camera_box: { lo: V3; hi: V3 };
  waypoint_box: { lo: V3; hi: V3; margin: number };
  bounds_splat: { lo: V3; hi: V3 };
  points: number[] | null;          // flat x,y,z in the course frame
  colors: number[] | null;          // flat r,g,b 0..255
  points_box?: { lo: V3; hi: V3 };  // 1st..99th percentile
  n_points: number;
  n_points_sent: number;
  warning?: string;
}
export interface Preview {
  mode: "fixed" | "expert";
  pilot: string; frame: string; hz: number; kT: number | null; use_l2_time: boolean; solve_s: number;
  keyframes: { name: string; t_file: number; t_solved: number; pos: V3; yaw: number }[];
  duration_file: number; duration_solved: number;
  t: number[]; pos: V3[]; vel: V3[]; acc: V3[]; yaw: number[]; speed: number[]; acc_norm: number[];
  quat?: [number, number, number, number][];   // FiGS attitude, body FRD, [x, y, z, w] (absent from older tools)
  stats: { v_max: number; v_mean: number; a_max: number; length_m: number; nonfinite_inputs: number };
  inputs: { names: string[]; lower: number[]; upper: number[]; u: number[][]; max_use: (number | null)[];
    violations: Record<string, [number, number][]> };
  clearance: { threshold: number; k: number; d: number[]; min: number; at_t: number; at_pos: V3;
    body_radius?: number; min_centre?: number;   // d/min are gaps to the drone's sphere when body_radius > 0
    below: [number, number][]; n_points: number; nearest_min: number; nearest_at_t: number; note: string } | null;
  inside: { keyframes: { name: string; inside: boolean }[]; outside_intervals: [number, number][]; outside_frac: number } | null;
}
export interface PreviewRequest {
  course: unknown; scene?: string; pilot?: string; frame?: string; mode?: "fixed" | "expert"; clearance?: number; clearance_k?: number;
  body_radius?: number;
}
export const courseApi = {
  geometry: (scene: string, margin = 0.5) =>
    req<Geometry>("GET", `/scenes/${encodeURIComponent(scene)}/geometry?margin=${margin}`),
  preview: (r: PreviewRequest) => req<Preview>("POST", "/courses/preview", r),
  lint: (name: string) => req<{ int_cells: string[] }>("GET", `/courses/${encodeURIComponent(name)}/lint`),
  /** Export (first time) and describe the scene's browser splat. */
  splat: (scene: string, build = true) => req<SplatMeta>("POST", `/scenes/${encodeURIComponent(scene)}/splat?build=${build}`),
};
export interface SplatMeta {
  file: string | null; cached: boolean; run: string; n_total?: number; n_written?: number; bytes?: number;
  min_opacity?: number; step?: number; seconds?: number; checkpoint?: string; checkpoint_mb?: number;
  box_course?: { lo: V3; hi: V3 };
}
export function splatUrl(scene: string, file: string) {
  const tok = getToken();
  return `/api/scenes/${encodeURIComponent(scene)}/splat/${encodeURIComponent(file)}${tok ? `?token=${encodeURIComponent(tok)}` : ""}`;
}

// ── semantic features (docs/SEMANTICS.md) ──────────────────────────────────
export const SEM_STEPS = ["preflight", "cameras", "teachers", "lift", "export", "fmgs", "bake"] as const;
export type SemStep = (typeof SEM_STEPS)[number];
export type SemBackend = "lift" | "fmgs";
/** semantic_pipeline.py's steps per backend (the first three are shared). */
export const SEM_BACKEND_STEPS: Record<SemBackend, SemStep[]> = {
  lift: ["preflight", "cameras", "teachers", "lift", "export"], fmgs: ["preflight", "cameras", "teachers", "fmgs", "bake"],
};
export interface SemanticRun {
  scene: string; backend?: SemBackend; teachers?: ("clip" | "dino")[]; feat_width?: number; dino_width?: number; batch?: number;
  from_step?: SemStep; only?: SemStep; stop_after?: SemStep; redo?: SemStep[];
  fmgs_steps?: number; fmgs_width?: number; fmgs_variant?: "auto" | "faithful" | "blite"; fmgs_table?: number;
}
export interface SemTable {
  run: string; backend: SemBackend; rows: number | null; key: string | null; stale: boolean; active_run: boolean;
  teacher_tag: string | null; order_sha: string | null; created: string | null; mb: number; seen_rows: number | null;
  lift: { seconds: number | null; passes: number | null; peak_vram_mib: number | null; render_width: number | null; views: number | null };
  fmgs: { steps: number | null; variant: string | null; fallback: { level: number; name: string } | null; loss_first: number | null;
    loss_last: number | null; peak_vram_mib_device: number | null; it_per_s: number | null } | null;
}
export interface SemStatus {
  scene: string; run: string | null; key: string | null; steps: { step: SemStep; done: boolean; when: string | null }[];
  config: Record<string, any>; results: Record<string, any>; tables: SemTable[]; ready: SemBackend[];
  queries: { total: number; annotated: number }; busy: number | null | boolean; script: boolean;
}
export interface Candidate {
  rank: number; score: number; n: number; large: boolean; centroid: V3; box: { lo: V3; hi: V3 };
  approach: V3; cameras: number; gap: number | null; gap_ok: boolean | null; centroid_splat: V3;
}
export interface QueryResult {
  text: string; negatives: string[]; threshold: number; tau: number; peak: number; rel_alpha: number;
  rel_pct: Record<string, number>; n_selected: number; rel_max: number; rel_p99: number; voxel: number;
  candidates: Candidate[]; margin: number | null; ambiguous: boolean; frame: string; ms: Record<string, number>;
}
export interface TableInfo { key: string | null; n: number; backend: string | null; teacher_tag: string | null; order_sha: string | null; run: string | null }
export interface QueryReply { result: QueryResult; table: TableInfo; stale: boolean; worker_ms: number | null; relevancy_id: string | null }
export interface QueryReq {
  text: string; backend?: SemBackend; negatives?: string[]; threshold?: number; rel_alpha?: number; standoff?: number;
  margin?: number; top?: number; relevancy?: boolean;
}
export interface LabelsReply { scores: [string, number][]; seen: boolean; position_splat: V3; table: TableInfo; stale: boolean }
export interface Annotation { text: string; position: V3 | null; set?: string; [k: string]: unknown }
export interface Annotations { version?: number; frame?: string; queries: Annotation[]; updated?: string; [k: string]: unknown }
export interface WorkerStatus { running: boolean; pid: number | null; started: number | null; last_used: number | null; idle_s: number; restarts: number; script: string; log: string }

/** GET raw bytes (relevancy, PCA) with the token header; returns the bytes and the response headers. */
async function bytes(path: string): Promise<{ data: Uint8Array; headers: Headers }> {
  const tok = getToken();
  const r = await fetch(`/api${path}`, { headers: tok ? { Authorization: `Bearer ${tok}` } : {} });
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : j.detail?.message ?? JSON.stringify(j); } catch { /* not json */ }
    throw new ApiError(r.status, msg);
  }
  return { data: new Uint8Array(await r.arrayBuffer()), headers: r.headers };
}
const sc = (s: string) => `/scenes/${encodeURIComponent(s)}/semantics`;
export const semApi = {
  status: (scene: string) => req<SemStatus>("GET", sc(scene)),
  submit: (r: SemanticRun) => req<{ id: number }>("POST", "/jobs/semantics", r),
  query: (scene: string, q: QueryReq) => req<QueryReply>("POST", `${sc(scene)}/query`, q),
  relevancy: async (scene: string, rid: string) => (await bytes(`${sc(scene)}/relevancy/${encodeURIComponent(rid)}`)).data,
  pca: async (scene: string, backend: SemBackend) => {
    const r = await bytes(`${sc(scene)}/${backend}/pca`);
    return { data: r.data, stale: r.headers.get("X-Table-Stale") === "1", key: r.headers.get("X-Table-Key") };
  },
  labels: (scene: string, index: number, labels: string[], backend: SemBackend = "lift") =>
    req<LabelsReply>("POST", `${sc(scene)}/labels`, { index, labels, backend }),
  queries: (scene: string) => req<Annotations>("GET", `${sc(scene)}/queries`),
  saveQueries: (scene: string, a: Annotations) => req<Annotations>("PUT", `${sc(scene)}/queries`, a),
  worker: () => req<WorkerStatus>("GET", "/semantics/worker"),
  stopWorker: () => req<{ stopped: boolean }>("POST", "/semantics/worker/stop"),
};

// ── SV-Net cohorts (Phase 4) ─────────────────────────────────────────────────
export const SV_STEPS = ["preflight", "rollout", "observe", "train_hist", "train_comm", "deploy"] as const;
export type SvStep = (typeof SV_STEPS)[number];
export interface SvnetRun {
  cohort: string; scene?: string; courses?: string[]; method?: string; roster?: string[];
  expert?: string; frame?: string; nro_ds?: number; use_compress?: boolean; subsample?: number;
  hist_epochs?: number; comm_epochs?: number; lr?: number; batch_size?: number; lim_sv?: number;
  comm_eval?: string; deploy_course?: string; deploy_method?: string;
  fresh?: ("histNet" | "commNet")[]; from_step?: SvStep; only?: SvStep; stop_after?: SvStep; redo?: SvStep[];
}
export interface CohortSummary {
  cohort: string; scene: string | null; courses: string[] | null; method: string | null; roster: string[] | null;
  done: string[]; students: Record<string, number>; managed: boolean;
}
export interface LossLog {
  log: string; epochs: number; train_s: number; n_train: number; n_test: number;
  loss_train: [number, number][]; loss_test: [number, number][]; eval_tte_upstream: [number, number][]; n_logs: number;
}
export interface DeployRow {
  role: "expert" | "student"; upstream_tte_mean: number; upstream_tte_best: number; upstream_pp: number;
  hz_mean: number; hz_worst: number; video: string | null;
  tte: { mean_m: number; max_m: number; "within_0.3m": number; final_mean_m: number; rollouts: number };
}
export interface CohortStatus {
  cohort: string; exists: boolean; data_dir: string;
  config: Partial<Record<keyof SvnetRun, any>>;
  steps: { step: SvStep; done: boolean; when: string | null }[];
  results: {
    estimate?: { rollouts: number; samples: number; rollout_gb: number; free_gb: number };
    rollout?: { courses: Record<string, { files: number; rollouts: number; samples: number; gb: number }>; wallclock: string; peak_vram_mib: number };
    observe?: { pilots: Record<string, Record<string, number>>; wallclock: string; peak_vram_mib: number };
    train_histNet?: { epochs: number; wallclock: string; peak_vram_mib: number; pilots: Record<string, LossLog> };
    train_commNet?: { epochs: number; wallclock: string; peak_vram_mib: number; deployment: [string, string, string] | null; pilots: Record<string, LossLog> };
    deploy?: { course: string; method: string; scene: string; pilots: Record<string, DeployRow>; wallclock: string; peak_vram_mib: number; finished: string };
    [k: string]: any;
  };
  live: Record<string, Partial<Record<"histNet" | "commNet", [number, number][]>>>;
  disk_gb: Record<string, number>;
}
export const svApi = {
  cohorts: () => req<CohortSummary[]>("GET", "/cohorts"),
  cohort: (c: string) => req<CohortStatus>("GET", `/cohorts/${encodeURIComponent(c)}`),
  submit: (r: SvnetRun) => req<{ id: number }>("POST", "/jobs/svnet", r),
  jobs: (c: string) => req<Job[]>("GET", `/jobs?cohort=${encodeURIComponent(c)}`),
};
export function cohortVideoUrl(cohort: string, file: string) {
  const tok = getToken();
  return `/api/cohorts/${encodeURIComponent(cohort)}/video/${encodeURIComponent(file)}${tok ? `?token=${encodeURIComponent(tok)}` : ""}`;
}

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

/** Queue a finished job again with the parameters it was submitted with. */
export function rerun(j: Job): Promise<{ id: number }> {
  const p = Object.fromEntries(Object.entries(j.params ?? {}).filter(([, v]) => v !== null && v !== undefined));
  if (j.kind === "figs") return api.submitFigs(p as unknown as FigsRun);
  if (j.kind === "svnet") return svApi.submit(p as unknown as SvnetRun);
  if (j.kind === "selftest") return api.submitSelftest(Number(p.seconds ?? 20));
  if (j.kind === "semantics") return semApi.submit(p as unknown as SemanticRun);
  return Promise.reject(new Error(`cannot re-run a ${j.kind} job`));
}
