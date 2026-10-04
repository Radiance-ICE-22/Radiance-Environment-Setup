// Resumable video uploads into the host's video_captures/ (backend: ui/backend/galley/videos.py).
// App-wide: an upload keeps going while you switch documents. One file is sent at a time, in
// 8 MiB chunks, each checked with SHA-256 where the browser allows it (secure contexts:
// localhost / https). A dropped connection retries with back-off; a closed tab or a restarted
// server resumes from the bytes already on the host when the same file is chosen again.
import { useSyncExternalStore } from "react";
import { getToken, VIDEO_EXT, VIDEO_RE } from "./api";

export const CHUNK = 8 * 2 ** 20;
const MAX_FAILS = 8;
const CHUNK_TIMEOUT_MS = 10 * 60 * 1000;

export type UpState = "waiting" | "starting" | "uploading" | "paused" | "finishing" | "done" | "error" | "cancelled";
export interface Upload {
  key: number; file: File; name: string; size: number;
  sent: number;            // bytes the host has confirmed (plus the current chunk's progress while uploading)
  resumedFrom: number;     // bytes already on the host when this attempt started
  state: UpState; id: string | null; error: string | null; exists: boolean; overwrite: boolean;
  rate: number;            // bytes/s, smoothed
  note: string | null; started: number; finished: number | null; retries: number;
}

let list: Upload[] = [];
let seq = 0;
let version = 0;
let lastAdded: { name: string; at: number } | null = null;
let doneCount = 0;
const subs = new Set<() => void>();
const xhrs = new Map<number, XMLHttpRequest>();
const emit = () => { version++; list = [...list]; subs.forEach((f) => f()); };
let throttled = 0;
const emitSoon = () => { const now = Date.now(); if (now - throttled > 200) { throttled = now; emit(); } };

const isActive = (u: Upload) => u.state === "starting" || u.state === "uploading" || u.state === "finishing";
export const pending = (u: Upload) => isActive(u) || u.state === "waiting";

function snapshot() { return { list, version, lastAdded, doneCount }; }
let snap = snapshot();
subs.add(() => { snap = snapshot(); });
export function useUploads() {
  return useSyncExternalStore((f) => { subs.add(f); return () => { subs.delete(f); }; }, () => snap);
}
export const uploadsNow = () => list;

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", (e) => {
    if (list.some(pending)) { e.preventDefault(); e.returnValue = ""; }   // the browser asks before leaving
  });
}

/** Backend-safe file name: spaces and other characters become '_'. */
export function safeName(file: string): string {
  const dot = file.lastIndexOf(".");
  const ext = dot > 0 ? file.slice(dot) : "";
  let stem = (dot > 0 ? file.slice(0, dot) : file).replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "");
  if (!stem) stem = "video";
  return `${stem.slice(0, 90)}${ext}`;
}
/** Scene-name suggestion from a video file name. */
export function sceneFromFile(name: string): string {
  return name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 64);
}
export const isVideoFile = (f: File) => VIDEO_EXT.some((e) => f.name.toLowerCase().endsWith(e)) || f.type.startsWith("video/");

export function addFiles(files: FileList | File[]): string[] {
  const names: string[] = [];
  for (const file of Array.from(files)) {
    const name = safeName(file.name);
    const base: Upload = { key: ++seq, file, name, size: file.size, sent: 0, resumedFrom: 0, state: "waiting", id: null, error: null,
      exists: false, overwrite: false, rate: 0, note: name !== file.name ? `renamed from ${file.name}` : null, started: Date.now(), finished: null, retries: 0 };
    if (!VIDEO_RE.test(name)) {
      list.push({ ...base, state: "error", error: `${file.name}: not a video file Galley accepts (${VIDEO_EXT.join(" ")})` });
      continue;
    }
    const dup = list.find((u) => u.name === name && (pending(u) || u.state === "paused"));
    if (dup) { names.push(name); continue; }           // already in the list
    list.push(base);
    names.push(name);
    lastAdded = { name, at: Date.now() };
  }
  emit();
  pump();
  return names;
}

/** Make `name` the video New capture selects (used by Google Drive imports too). */
export function announce(name: string) { lastAdded = { name, at: Date.now() }; emit(); }

/** Open the browser's file chooser (must run inside a click). */
export function pickFiles(onPicked?: (names: string[]) => void) {
  const inp = document.createElement("input");
  inp.type = "file";
  inp.accept = [...VIDEO_EXT, "video/*"].join(",");
  inp.multiple = true;
  inp.style.display = "none";
  document.body.appendChild(inp);             // some browsers drop change events of detached inputs
  inp.onchange = () => { const f = inp.files ? Array.from(inp.files) : []; inp.remove(); if (f.length) { const names = addFiles(f); onPicked?.(names); } };
  inp.addEventListener("cancel", () => inp.remove());
  inp.click();
}

function set(u: Upload, p: Partial<Upload>) {
  const i = list.findIndex((x) => x.key === u.key);
  if (i >= 0) { list[i] = Object.assign(list[i], p); }
}
const get = (key: number) => list.find((x) => x.key === key);

export function pause(key: number) {
  const u = get(key); if (!u || !(pending(u))) return;
  set(u, { state: "paused" }); xhrs.get(key)?.abort(); emit(); pump();
}
export function resume(key: number, overwrite?: boolean) {
  const u = get(key); if (!u || !["paused", "error"].includes(u.state)) return;
  set(u, { state: "waiting", error: null, exists: false, overwrite: overwrite ?? u.overwrite, retries: 0 }); emit(); pump();
}
export async function cancel(key: number) {
  const u = get(key); if (!u) return;
  const id = u.id;
  set(u, { state: "cancelled" }); xhrs.get(key)?.abort(); emit();
  if (id) await call("DELETE", `/uploads/${id}`).catch(() => undefined);   // drop the partial on the host
  pump();
}
export function dismiss(key: number) {
  list = list.filter((x) => x.key !== key || pending(x)); emit();
}

let running = false;
function pump() {
  if (running) return;
  const next = list.find((u) => u.state === "waiting");
  if (!next) return;
  running = true;
  run(next.key).finally(() => { running = false; pump(); });
}

class HttpErr extends Error {
  constructor(public status: number, message: string, public detail: any) { super(message); }
}
async function call(method: string, path: string, body?: unknown) {
  const headers: Record<string, string> = {};
  const tok = getToken(); if (tok) headers["Authorization"] = `Bearer ${tok}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new HttpErr(r.status, detailMsg(j.detail) ?? r.statusText, j.detail);
  return j;
}
const detailMsg = (d: any): string | undefined => typeof d === "string" ? d : d && typeof d.message === "string" ? d.message
  : Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : undefined;

async function sha256(blob: Blob): Promise<string | null> {
  if (!globalThis.crypto?.subtle) return null;        // plain http to a non-localhost host: no WebCrypto
  const h = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(h), (b) => b.toString(16).padStart(2, "0")).join("");
}

function putChunk(u: Upload, blob: Blob, offset: number, sha: string | null): Promise<any> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    xhrs.set(u.key, x);
    x.open("PUT", `/api/uploads/${u.id}?offset=${offset}`);
    const tok = getToken(); if (tok) x.setRequestHeader("Authorization", `Bearer ${tok}`);
    x.setRequestHeader("Content-Type", "application/octet-stream");
    if (sha) x.setRequestHeader("X-Chunk-SHA256", sha);
    x.timeout = CHUNK_TIMEOUT_MS;
    let lastT = performance.now(), lastB = 0;
    x.upload.onprogress = (e) => {
      const now = performance.now(), dt = (now - lastT) / 1000;
      if (dt > 0.25) {
        const r = (e.loaded - lastB) / dt;
        set(u, { rate: u.rate ? u.rate * 0.7 + r * 0.3 : r }); lastT = now; lastB = e.loaded;
      }
      set(u, { sent: offset + e.loaded }); emitSoon();
    };
    x.onload = () => {
      xhrs.delete(u.key);
      let j: any = {}; try { j = JSON.parse(x.responseText); } catch { /* empty */ }
      if (x.status >= 200 && x.status < 300) resolve(j);
      else reject(new HttpErr(x.status, detailMsg(j.detail) ?? x.statusText, j.detail));
    };
    x.onerror = () => { xhrs.delete(u.key); reject(new HttpErr(0, "connection lost", null)); };
    x.ontimeout = () => { xhrs.delete(u.key); reject(new HttpErr(0, "chunk timed out", null)); };
    x.onabort = () => { xhrs.delete(u.key); reject(new HttpErr(-1, "aborted", null)); };
    x.send(blob);
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function run(key: number) {
  const u = get(key); if (!u || u.state !== "waiting") return;
  const still = () => get(key)?.state === "uploading";
  set(u, { state: "starting", error: null, rate: 0 }); emit();
  let fails = 0;
  try {
    let start: any;
    for (;;) {
      try {
        start = await call("POST", "/uploads", { name: u.name, size: u.size, modified: Math.round(u.file.lastModified || 0), overwrite: u.overwrite });
        break;
      } catch (e) {
        if (e instanceof HttpErr && e.status !== 0 && e.status < 500) throw e;
        if (++fails >= MAX_FAILS) throw e;
        set(u, { retries: fails, note: `host not reachable, retrying (${fails}/${MAX_FAILS})` }); emit();
        await sleep(Math.min(30000, 1000 * 2 ** fails));
        if (get(key)?.state !== "starting") return;
      }
    }
    if (get(key)?.state !== "starting") {                 // paused or cancelled while asking
      if (get(key)?.state === "cancelled" && start.id) await call("DELETE", `/uploads/${start.id}`).catch(() => undefined);
      return;
    }
    if (start.status === "exists") {
      set(u, { state: "done", sent: u.size, resumedFrom: u.size, note: "already on the host (same name and size)", finished: Date.now() });
      doneCount++; emit(); return;
    }
    set(u, { id: start.id, sent: start.offset, resumedFrom: start.offset, state: "uploading", started: Date.now(),
      note: start.offset ? `resumed at ${(start.offset / 2 ** 20).toFixed(0)} MB` : u.note });
    emit();
    let offset: number = start.offset;
    fails = 0;
    while (offset < u.size) {
      if (!still()) return;
      const blob = u.file.slice(offset, Math.min(offset + CHUNK, u.size));
      try {
        const sha = await sha256(blob);
        if (!still()) return;
        const r = await putChunk(u, blob, offset, sha);
        offset = r.offset; fails = 0;
        set(u, { sent: offset, retries: 0 }); emitSoon();
      } catch (e) {
        if (!(e instanceof HttpErr)) throw e;
        if (e.status === -1) return;                                   // paused or cancelled
        if (e.status === 409 && typeof e.detail?.offset === "number") { offset = e.detail.offset; set(u, { sent: offset }); continue; }
        if (e.status === 400 && /checksum/.test(e.message) && fails < 3) { fails++; continue; }
        if (e.status !== 0 && e.status < 500) throw e;
        if (++fails >= MAX_FAILS) throw new Error(`${e.message}; ${MAX_FAILS} attempts failed. Resume continues from ${(offset / 2 ** 20).toFixed(0)} MB.`);
        set(u, { retries: fails, rate: 0, note: `${e.message}, retrying (${fails}/${MAX_FAILS})` }); emit();
        await sleep(Math.min(30000, 1000 * 2 ** fails));
        // after a reconnect the host may hold more or fewer bytes than we think: ask
        try {
          const info = await call("POST", "/uploads", { name: u.name, size: u.size, modified: Math.round(u.file.lastModified || 0), overwrite: u.overwrite });
          if (info.id) offset = info.offset;
        } catch { /* next PUT reports it */ }
      }
    }
    if (!still()) return;
    set(u, { state: "finishing", sent: u.size }); emit();
    await call("POST", `/uploads/${u.id}/complete`);
    set(u, { state: "done", finished: Date.now(), note: null }); doneCount++; emit();
  } catch (e) {
    const st = get(key)?.state;
    if (st === "paused" || st === "cancelled") return;
    set(u, { state: "error", rate: 0, error: e instanceof Error ? e.message : String(e),
      exists: e instanceof HttpErr && !!e.detail?.exists });
    emit();
  }
}

export function fmtBytes(n: number) {
  return n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(0)} MB` : `${(n / 1024).toFixed(0)} KB`;
}
export function fmtDur(s: number) {
  if (!isFinite(s) || s < 0) return "—";
  s = Math.round(s);
  return s >= 3600 ? `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`
    : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
export function eta(u: Upload) { return u.rate > 0 ? (u.size - u.sent) / u.rate : NaN; }
