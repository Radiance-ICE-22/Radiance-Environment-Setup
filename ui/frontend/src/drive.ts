// Google Drive imports (backend: ui/backend/galley/drive.py). The browser signs in with Google
// Identity Services and opens Google's file picker; the HOST downloads the picked files, so a
// slow home uplink or a relayed Tailscale link never carries the video. Imports keep running on
// the host if this tab closes; their state is polled from /api/drive/imports.
//
// Scope is drive.file: Galley can read only the files picked in the picker, nothing else in Drive.
// The access token (about an hour) goes to the host with each import and is kept in its memory only.
import { useSyncExternalStore } from "react";
import { ApiError, getToken as galleyToken } from "./api";
import { announce, safeName } from "./uploads";

export type DriveState = "queued" | "downloading" | "verifying" | "done" | "error" | "cancelled" | "interrupted";
export interface DriveImport {
  id: string; file_id: string; name: string; drive_name: string | null; size: number; received: number;
  state: DriveState; error: string | null; rate: number; started: number | null; finished: number | null; created: number;
}
export interface DriveCfg { enabled: boolean; client_id: string; api_key: string; app_id: string; scope: string }
export const driveActive = (i: DriveImport) => i.state === "queued" || i.state === "downloading" || i.state === "verifying";

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  const tok = galleyToken(); if (tok) headers["Authorization"] = `Bearer ${tok}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = j.detail;
    throw new ApiError(r.status, typeof d === "string" ? d : d?.message ?? (Array.isArray(d) ? d.map((x: any) => `${(x.loc ?? []).slice(-1)}: ${x.msg}`).join("; ") : r.statusText));
  }
  return j as T;
}
export const driveApi = {
  config: () => call<DriveCfg>("GET", "/drive/config"),
  setConfig: (c: { client_id: string; api_key: string; app_id: string }) => call<DriveCfg>("PUT", "/drive/config", c),
  imports: () => call<DriveImport[]>("GET", "/drive/imports"),
  start: (b: { file_id: string; name: string; token: string; overwrite?: boolean }) => call<DriveImport & { state: string }>("POST", "/drive/imports", b),
  resume: (id: string, token: string) => call<DriveImport>("POST", `/drive/imports/${id}/resume`, { token }),
  cancel: (id: string) => call<DriveImport>("POST", `/drive/imports/${id}/cancel`),
  dismiss: (id: string) => call<{ dismissed: string }>("DELETE", `/drive/imports/${id}`),
};

// ── polled store ─────────────────────────────────────────────────────────────
let snap: { list: DriveImport[]; cfg: DriveCfg | null; doneCount: number; setupOpen: boolean } = { list: [], cfg: null, doneCount: 0, setupOpen: false };
const subs = new Set<() => void>();
let timer: ReturnType<typeof setTimeout> | null = null;
const set = (p: Partial<typeof snap>) => { snap = { ...snap, ...p }; subs.forEach((f) => f()); };
export async function refreshDrive() {
  try {
    const list = await driveApi.imports();
    const done = list.filter((i) => i.state === "done").length;
    const prevDone = snap.list.filter((i) => i.state === "done").length;
    set({ list, doneCount: snap.doneCount + Math.max(0, done - prevDone) });
  } catch { /* backend without drive.py, or offline: keep the last list */ }
}
async function loop() {
  await refreshDrive();
  timer = setTimeout(loop, snap.list.some(driveActive) ? 1500 : 10000);
}
export async function refreshDriveConfig() {
  try { set({ cfg: await driveApi.config() }); } catch { set({ cfg: null }); }
}
export function useDrive() {
  return useSyncExternalStore((f) => {
    subs.add(f);
    if (!timer) { loop(); refreshDriveConfig(); }
    return () => { subs.delete(f); };
  }, () => snap);
}
const kick = () => { if (timer) clearTimeout(timer); loop(); };

// ── Google scripts, sign-in, picker ───────────────────────────────────────────
const G = () => (window as any).google;
const scripts = new Map<string, Promise<void>>();
function loadScript(src: string) {
  if (!scripts.has(src)) {
    scripts.set(src, new Promise<void>((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.async = true; s.onload = () => ok();
      s.onerror = () => { scripts.delete(src); fail(new Error(`could not load ${src} (is this computer online?)`)); };
      document.head.appendChild(s);
    }));
  }
  return scripts.get(src)!;
}
let pickerReady: Promise<void> | null = null;
/** Load Google's scripts ahead of the click, so the sign-in popup opens inside the user gesture. */
export function preloadGoogle() {
  if (!pickerReady) {
    pickerReady = Promise.all([loadScript("https://accounts.google.com/gsi/client"), loadScript("https://apis.google.com/js/api.js")])
      .then(() => new Promise<void>((ok) => (window as any).gapi.load("picker", { callback: () => ok() })))
      .catch((e) => { pickerReady = null; throw e; });
  }
  return pickerReady;
}

let cached: { token: string; exp: number } | null = null;
function signIn(cfg: DriveCfg): Promise<string> {
  if (cached && Date.now() < cached.exp - 120_000) return Promise.resolve(cached.token);
  return new Promise((ok, fail) => {
    const tc = G().accounts.oauth2.initTokenClient({
      client_id: cfg.client_id, scope: cfg.scope,
      callback: (r: any) => {
        if (r.error) return fail(new Error(`Google sign-in: ${r.error_description || r.error}`));
        cached = { token: r.access_token, exp: Date.now() + Number(r.expires_in || 3600) * 1000 };
        ok(r.access_token);
      },
      error_callback: (e: any) => fail(new Error(e?.type === "popup_closed" ? "sign-in window closed" : e?.type === "popup_failed_to_open"
        ? "the browser blocked Google's sign-in popup; allow popups for this page" : `Google sign-in failed (${e?.type ?? e})`)),
    });
    tc.requestAccessToken({ prompt: "" });
  });
}

export interface Picked { id: string; name: string; mimeType: string; sizeBytes?: number }
function openPicker(cfg: DriveCfg, token: string): Promise<Picked[]> {
  return new Promise((ok) => {
    const P = G().picker;
    const mine = new P.DocsView(P.ViewId.DOCS_VIDEOS).setIncludeFolders(true).setMode(P.DocsViewMode.LIST);
    const shared = new P.DocsView(P.ViewId.DOCS_VIDEOS).setEnableDrives(true).setIncludeFolders(true).setMode(P.DocsViewMode.LIST);
    new P.PickerBuilder()
      .setTitle("Choose phone videos for the host's video_captures/")
      .addView(mine).addView(shared)
      .enableFeature(P.Feature.MULTISELECT_ENABLED).enableFeature(P.Feature.SUPPORT_DRIVES)
      .setOAuthToken(token).setDeveloperKey(cfg.api_key).setAppId(cfg.app_id).setOrigin(location.origin)
      .setCallback((d: any) => {
        const a = d[P.Response.ACTION];
        if (a === P.Action.PICKED) ok((d[P.Response.DOCUMENTS] ?? []).map((x: any) => ({ id: x.id, name: x.name, mimeType: x.mimeType, sizeBytes: x.sizeBytes })));
        else if (a === P.Action.CANCEL) ok([]);
      })
      .build().setVisible(true);
  });
}

/** Sign in, pick videos in Google's picker, and have the host download them. Returns the names started. */
export async function pickFromDrive(cfg: DriveCfg, overwrite = false): Promise<{ started: string[]; skipped: string[] }> {
  await preloadGoogle();
  const token = await signIn(cfg);
  const docs = await openPicker(cfg, token);
  const started: string[] = [], skipped: string[] = [];
  for (const doc of docs) {
    const name = safeName(doc.name);
    try {
      const r = await driveApi.start({ file_id: doc.id, name, token, overwrite });
      if ((r.state as string) === "exists") skipped.push(`${name} (already on the host)`);
      started.push(name);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) cached = null;
      skipped.push(`${doc.name}: ${e instanceof Error ? e.message : e}`);
    }
  }
  if (started.length) announce(started[0]);
  kick();
  return { started, skipped };
}

export async function resumeDrive(cfg: DriveCfg, id: string) {
  await preloadGoogle();
  cached = null;                                    // the old token is why it stopped, or is gone after a restart
  const token = await signIn(cfg);
  await driveApi.resume(id, token);
  kick();
}
export async function cancelDrive(id: string) { await driveApi.cancel(id); kick(); }
export async function dismissDrive(id: string) { await driveApi.dismiss(id); kick(); }

export const openDriveSetup = (open = true) => set({ setupOpen: open });

/** The "Google Drive" command: set-up dialog until credentials exist, then sign-in and the picker. */
export async function startDriveFlow(): Promise<void> {
  if (!snap.cfg) await refreshDriveConfig();
  const cfg = snap.cfg;
  if (!cfg?.enabled) { openDriveSetup(); return; }
  try {
    const { skipped } = await pickFromDrive(cfg);
    if (skipped.length) alert(`Google Drive:\n${skipped.join("\n")}`);
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e));
  }
}
