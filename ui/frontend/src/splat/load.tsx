// Getting the scene's .splat into the page, once, for every document that shows it.
//
// useSplat(scene, on): asks the backend to export the browser splat from the active checkpoint
// (cached on the host until the model changes), downloads it with progress, and parses it
// (splat/format.ts). Parsed files are kept per URL — the course editor and the splat editor
// share one copy — for the last two scenes.
//
// <SplatLayer>: the renderer (SplatMesh) behind an error boundary, so a GPU or shader failure
// shows as a message instead of taking the whole 3D view down.
import { Component, ReactNode, useCallback, useEffect, useState } from "react";
import { ApiError, courseApi, getToken, SplatMeta, splatUrl } from "../api";
import { parseSplat, SplatData } from "./format";
import { SplatMesh } from "./SplatMesh";

export type SplatState = "idle" | "exporting" | "downloading" | "parsing" | "ready" | "error";
export interface SplatLoad { state: SplatState; meta: SplatMeta | null; data: SplatData | null; progress: number; err: string | null; url: string | null }
const IDLE: SplatLoad = { state: "idle", meta: null, data: null, progress: 0, err: null, url: null };

const cache = new Map<string, Promise<SplatData>>();
function remember(url: string, p: Promise<SplatData>) {
  cache.set(url, p);
  p.catch(() => cache.delete(url));
  while (cache.size > 2) cache.delete(cache.keys().next().value!);
}

async function download(url: string, onProgress: (f: number) => void): Promise<ArrayBuffer> {
  const tok = getToken();
  const r = await fetch(url, { headers: tok ? { Authorization: `Bearer ${tok}` } : {} });
  if (!r.ok) throw new ApiError(r.status, `${r.status} ${r.statusText}: the splat file`);
  const total = Number(r.headers.get("Content-Length")) || 0;
  if (!r.body || !total) return r.arrayBuffer();
  const out = new Uint8Array(total), rd = r.body.getReader();
  let got = 0;
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    if (got + value.length > total) throw new Error("the splat download is longer than its Content-Length");
    out.set(value, got); got += value.length; onProgress(got / total);
  }
  if (got !== total) throw new Error(`the splat download stopped at ${got} of ${total} bytes`);
  return out.buffer;
}

export function useSplat(scene: string | undefined, on: boolean): SplatLoad & { retry: () => void } {
  const [st, setSt] = useState<SplatLoad>(IDLE);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!scene || !on) { setSt(IDLE); return; }
    let alive = true;
    const set = (p: Partial<SplatLoad>) => alive && setSt((s) => ({ ...s, ...p }));
    (async () => {
      try {
        set({ ...IDLE, state: "exporting" });
        const meta = await courseApi.splat(scene);
        if (!meta.file) throw new Error("the backend exported no splat file");
        const url = splatUrl(scene, meta.file);
        set({ meta, url });
        let p = cache.get(url);
        if (!p) {
          set({ state: "downloading", progress: 0 });
          p = download(url, (f) => set({ progress: f })).then((buf) => { set({ state: "parsing" }); return parseSplat(buf); });
          remember(url, p);
        }
        const data = await p;
        set({ state: "ready", data, progress: 1 });
      } catch (e) { set({ state: "error", err: e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e) }); }
    })();
    return () => { alive = false; };
  }, [scene, on, nonce]);
  const retry = useCallback(() => setNonce((n) => n + 1), []);
  return { ...st, retry };
}

export function SplatLayer({ data, colors, onReady, onError, onRecolor }: {
  data: SplatData; colors?: Uint8Array | null; onReady?: () => void; onError?: (m: string) => void; onRecolor?: (t: { cpu: number; frame: number }) => void;
}) {
  return (
    <SplatBoundary key={data.n + ":" + data.buf.byteLength} onError={(m) => onError?.(m)}>
      <SplatMesh data={data} colors={colors} onReady={onReady} onRecolor={onRecolor} />
    </SplatBoundary>
  );
}
class SplatBoundary extends Component<{ children: ReactNode; onError: (m: string) => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(e: unknown) { this.props.onError(e instanceof Error ? e.message : String(e)); }
  render() { return this.state.failed ? null : this.props.children; }
}

export const mb = (b?: number | null) => (b ? `${(b / 2 ** 20).toFixed(1)} MB` : "—");
