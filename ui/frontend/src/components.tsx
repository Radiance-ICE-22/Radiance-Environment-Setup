import { useEffect, useState } from "react";
import { api, ApiError, FigsRun, JobStatus } from "./api";

export function Badge({ s }: { s: JobStatus | string }) {
  return <span className={`badge s-${s}`}>{s}</span>;
}

export function usePoll<T>(fn: () => Promise<T>, ms: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    let alive = true;
    const tick = () => fn().then((d) => { if (alive) { setData(d); setErr(null); } })
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)));
    tick();
    const t = ms > 0 ? setInterval(tick, ms) : undefined;
    return () => { alive = false; if (t) clearInterval(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n]);
  return { data, err, reload: () => setN((x) => x + 1) };
}

/** Number input that maps "" to undefined, so unset flags fall back to the script's defaults. */
export function NumField(p: { label: string; value: number | undefined; onChange: (v: number | undefined) => void;
  step?: number; min?: number; max?: number; placeholder?: string; hint?: string }) {
  return (
    <label className="f">
      {p.label}
      <input type="number" step={p.step ?? 1} min={p.min} max={p.max} placeholder={p.placeholder ?? "default"}
        value={p.value ?? ""} onChange={(e) => p.onChange(e.target.value === "" ? undefined : Number(e.target.value))} />
      {p.hint && <span className="hint">{p.hint}</span>}
    </label>
  );
}

export function Select<T extends string>(p: { label: string; value: T | undefined; options: readonly T[];
  onChange: (v: T | undefined) => void; allowDefault?: boolean; hint?: string }) {
  return (
    <label className="f">
      {p.label}
      <select value={p.value ?? ""} onChange={(e) => p.onChange((e.target.value || undefined) as T | undefined)}>
        {p.allowDefault !== false ? <option value="">default</option>
          : p.value === undefined && <option value="" disabled>choose…</option>}
        {p.options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
      {p.hint && <span className="hint">{p.hint}</span>}
    </label>
  );
}

/** Training options for the `train` step (figs_pipeline.py --train-*). */
export function TrainOptions({ r, set, vramMib }: { r: FigsRun; set: (p: Partial<FigsRun>) => void; vramMib?: number }) {
  const [extra, setExtra] = useState((r.train_args ?? []).join("\n"));
  const small = vramMib !== undefined && vramMib > 0 && vramMib < 6000;
  return (
    <fieldset>
      <legend>Splat training</legend>
      {small && <p className="note small">This machine has {vramMib} MiB of VRAM. The reference run peaked at 5203 MiB at
        960×540 (8 GB card), so use a CPU image cache and a larger downscale or fewer Gaussians.</p>}
      <div className="fields">
        <NumField label="Iterations" value={r.train_iters} min={100} max={200000} placeholder="30000"
          onChange={(v) => set({ train_iters: v })} />
        <NumField label="Image downscale" value={r.downscale} min={1} max={16} placeholder="auto (≤1600 px)"
          onChange={(v) => set({ downscale: v })} hint="1080p: auto = 2 (960×540)" />
        <Select label="Image cache" value={r.cache_images} options={["cpu", "gpu"] as const}
          onChange={(v) => set({ cache_images: v })} hint="cpu keeps images out of VRAM" />
        <Select label="Logging" value={r.train_vis} options={["tensorboard", "viewer", "viewer+tensorboard"] as const}
          onChange={(v) => set({ train_vis: v })} />
      </div>
      <label className="f" style={{ marginTop: 10 }}>
        Extra ns-train options, one per line
        <textarea rows={2} value={extra} placeholder="--pipeline.model.stop-split-at 10000"
          onChange={(e) => { setExtra(e.target.value); set({ train_args: e.target.value.split("\n").map((s) => s.trim()).filter(Boolean) }); }} />
        <span className="hint">Allowed: --pipeline.*, --optimizers.* options with one value.</span>
      </label>
      <label className="check small" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={!!r.archive_old} onChange={(e) => set({ archive_old: e.target.checked })} />
        Archive the scene's existing model first (FiGS loads exactly one per scene)
      </label>
    </fieldset>
  );
}

export function useSubmit() {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const submit = async (r: FigsRun) => {
    setBusy(true); setErr(null);
    try {
      const clean = Object.fromEntries(Object.entries(r).filter(([, v]) =>
        v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0) && v !== false)) as FigsRun;
      const { id } = await api.submitFigs(clean);
      location.hash = `#/jobs/${id}`;
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally { setBusy(false); }
  };
  return { busy, err, submit };
}
