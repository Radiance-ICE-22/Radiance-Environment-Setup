import { useEffect, useState } from "react";
import { api, ago, ApiError, ArchivedModel, FigsRun, flightUrl, Model, Step, STEPS } from "../api";
import { BarChart, LineChart } from "../charts";
import { Badge, Select, TrainOptions, useMachineTrainDefaults, usePoll, useSubmit } from "../components";

export default function ScenePage({ scene }: { scene: string }) {
  const st = usePoll(() => api.scene(scene), 5000, [scene]);
  const jobs = usePoll(() => api.jobs(scene), 5000, [scene]);
  const courses = usePoll(() => api.configs("courses"), 0);
  const machine = usePoll(api.machine, 0);
  const [videoKey, setVideoKey] = useState(0);
  const s = st.data;
  const res = s?.results ?? {};

  return (
    <>
      <div className="row"><h1>{scene}</h1><span className="spacer" /><a href="#/">← Overview</a></div>
      {st.err && <p className="err">{st.err}</p>}
      {s && (
        <div className="panel">
          <h2>Pipeline steps</h2>
          <div className="steps">
            {s.steps.map((x) => (
              <div key={x.step} className={`step ${x.done ? "done" : ""}`} title={x.fingerprint ?? ""}>
                <b>{x.step}</b>
                <span className={x.done ? "ok" : "muted"}>{x.done ? (x.when ?? "done").replace("T", " ") : "not done"}</span>
              </div>
            ))}
          </div>
          <p className="muted small">From <span className="mono">.figs_pipeline_state/{scene}/</span>. A step re-runs when a flag it depends on changes.</p>
        </div>
      )}

      <div className="grid2">
        <div className="panel">
          <h2>Trained model</h2>
          <Models scene={scene} onChange={st.reload} />
          {res.sfm && (
            <>
              <h3>Reconstruction</h3>
              <table><tbody>
                <tr><td>Registered</td><td className={res.sfm.pct < 90 ? "bad" : "ok"}>{res.sfm.registered} / {res.sfm.images} ({res.sfm.pct}%)</td></tr>
                <tr><td>Sparse points</td><td>{res.sfm.sparse_points?.toLocaleString()}</td></tr>
              </tbody></table>
            </>
          )}
          {res.train && (
            <>
              <h3>Last training</h3>
              <table><tbody>
                <tr><td>Wall clock</td><td>{res.train.wallclock}</td></tr>
                <tr><td>Peak VRAM</td><td>{res.train.peak_vram_mib} MiB</td></tr>
                <tr><td>Settings</td><td className="mono small">iters {res.train.iters ?? "default"} · downscale {res.train.downscale ?? "auto"} · cache {res.train.cache_images ?? "default"}</td></tr>
              </tbody></table>
            </>
          )}
          {res.bounds_splat && (
            <>
              <h3>Camera bounds (splat frame, z-up, m)</h3>
              <p className="mono small">{["x", "y", "z"].map((a) => `${a} ${res.bounds_splat[a][0]} … ${res.bounds_splat[a][1]}`).join("   ")}</p>
              <p className="muted small">Course frame is (x, −y, −z): altitude is negative z.</p>
            </>
          )}
        </div>
        <div className="panel">
          <h2>Last flight</h2>
          <video key={videoKey} controls src={flightUrl(scene)} onError={(e) => ((e.target as HTMLVideoElement).style.display = "none")} />
          {res.sim && (
            <table><tbody>
              <tr><td>Course</td><td>{res.course?.name}</td></tr>
              <tr><td>Frames</td><td>{res.sim.frames} at {res.sim.hz} Hz ({res.sim.duration_s} s)</td></tr>
              <tr><td>Tracking error</td><td className={res.sim.track_err_max_m > 0.5 ? "bad" : ""}>mean {res.sim.track_err_mean_m} m · max {res.sim.track_err_max_m} m</td></tr>
              <tr><td>Render check</td><td className={res.sim.dark_frames > 0 || res.sim.pixel_std < 5 ? "bad" : ""}>pixel std {res.sim.pixel_std} · dark frames {res.sim.dark_frames}</td></tr>
            </tbody></table>
          )}
          <button onClick={() => setVideoKey((k) => k + 1)}>Reload video</button>
        </div>
      </div>

      <div className="grid2">
        <TrainingCurve scene={scene} runKey={s?.models.map((m) => m.run).join(",") ?? ""} />
        {res.aruco?.histogram && <ArucoPanel a={res.aruco} />}
      </div>

      <div className="grid2">
        <FlyCourse scene={scene} courses={courses.data?.map((c) => c.name) ?? []} />
        <Retrain scene={scene} vram={machine.data?.gpu.vram_mib} />
      </div>
      <Advanced scene={scene} />

      <div className="panel">
        <h2>Jobs for this scene</h2>
        <table>
          <thead><tr><th>#</th><th>Job</th><th>Status</th><th>Created</th></tr></thead>
          <tbody>
            {jobs.data?.map((j) => (
              <tr key={j.id} className="click" onClick={() => (location.hash = `#/jobs/${j.id}`)}>
                <td>{j.id}</td><td>{j.label}</td><td><Badge s={j.status} /></td><td className="muted">{ago(j.created)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function FlyCourse({ scene, courses }: { scene: string; courses: string[] }) {
  const [r, setR] = useState<FigsRun>({ scene, from_step: "bounds" });
  const { busy, err, submit } = useSubmit();
  return (
    <div className="panel">
      <h2>Fly a course</h2>
      <p className="muted small">Runs bounds → course check → simulate → validate → record with the Viper MPC expert.</p>
      <div className="fields">
        <Select label="Course" value={r.course} options={courses} allowDefault={false} onChange={(v) => setR({ ...r, course: v })} />
        <label className="f">Frame<input value={r.frame ?? ""} placeholder="carl" onChange={(e) => setR({ ...r, frame: e.target.value || undefined })} /></label>
        <label className="f">Expert<input value={r.pilot ?? ""} placeholder="Viper" onChange={(e) => setR({ ...r, pilot: e.target.value || undefined })} /></label>
        <label className="f">Method<input value={r.method ?? ""} placeholder="eval_single" onChange={(e) => setR({ ...r, method: e.target.value || undefined })} /></label>
      </div>
      <label className="check small" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={!!r.allow_outside} onChange={(e) => setR({ ...r, allow_outside: e.target.checked })} />
        Fly even if waypoints leave the captured volume
      </label>
      {err && <p className="err">{err}</p>}
      <button className="primary" disabled={busy || !r.course} onClick={() => submit({ ...r, redo: ["simulate", "validate"] })}>Queue flight</button>
    </div>
  );
}

function Retrain({ scene, vram }: { scene: string; vram?: number }) {
  const [r, setR] = useState<FigsRun>({ scene, from_step: "train", stop_after: "verify", archive_old: true });
  const machine = usePoll(api.machine, 0);
  useMachineTrainDefaults(machine.data?.defaults, setR);
  const { busy, err, submit } = useSubmit();
  return (
    <div className="panel">
      <h2>Retrain the splat</h2>
      <p className="muted small">Reuses this scene's SfM (the <span className="mono">sfm</span> step) and runs only <span className="mono">train</span> and <span className="mono">verify</span>.</p>
      <TrainOptions r={r} set={(p) => setR({ ...r, ...p })} vramMib={vram} />
      {err && <p className="err">{err}</p>}
      <button className="primary" disabled={busy} onClick={() => submit({ ...r, redo: ["train", "verify"] })}>Queue training</button>
    </div>
  );
}

function Advanced({ scene }: { scene: string }) {
  const [open, setOpen] = useState(false);
  const [r, setR] = useState<FigsRun>({ scene });
  const { busy, err, submit } = useSubmit();
  if (!open) return <p><button onClick={() => setOpen(true)}>Run arbitrary steps…</button></p>;
  return (
    <div className="panel">
      <h2>Run steps</h2>
      <div className="fields">
        <Select label="Only" value={r.only} options={STEPS} onChange={(v) => setR({ ...r, only: v, from_step: undefined, stop_after: undefined })} />
        <Select label="From" value={r.from_step} options={STEPS} onChange={(v) => setR({ ...r, from_step: v, only: undefined })} />
        <Select label="Stop after" value={r.stop_after} options={STEPS} onChange={(v) => setR({ ...r, stop_after: v, only: undefined })} />
      </div>
      <h3>Force redo</h3>
      <div className="row small">
        {STEPS.map((x) => (
          <label key={x} className="row" style={{ gap: 4 }}>
            <input type="checkbox" checked={r.redo?.includes(x) ?? false}
              onChange={(e) => setR({ ...r, redo: e.target.checked ? [...(r.redo ?? []), x] : (r.redo ?? []).filter((y: Step) => y !== x) })} />
            <span className="mono">{x}</span>
          </label>
        ))}
      </div>
      {err && <p className="err">{err}</p>}
      <div className="row" style={{ marginTop: 10 }}>
        <button className="primary" disabled={busy} onClick={() => submit(r)}>Queue</button>
        <button onClick={() => setOpen(false)}>Close</button>
      </div>
    </div>
  );
}

function Models({ scene, onChange }: { scene: string; onChange: () => void }) {
  const [m, setM] = useState<{ active: Model[]; archived: ArchivedModel[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const load = () => api.models(scene).then(setM).catch((e) => setMsg(String(e.message ?? e)));
  useEffect(() => { load(); }, [scene]);
  const act = async (fn: () => Promise<{ cleared_steps: string[] }>, what: string) => {
    setBusy(true); setMsg(null);
    try {
      const r = await fn();
      setMsg(`${what}.${r.cleared_steps.length ? ` Cleared ${r.cleared_steps.join(", ")}: fly again to refresh the flight.` : ""}`);
      await load(); onChange();
    } catch (e) { setMsg(e instanceof ApiError ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  if (!m) return null;
  return (
    <>
      {m.active.length === 0 && <p className="bad">No active model: FiGS cannot load this scene.</p>}
      {m.active.length > 1 && <p className="bad">{m.active.length} active models: FiGS refuses to guess. Archive all but one.</p>}
      <table><tbody>
        {m.active.map((x) => (
          <tr key={x.run}>
            <td><span className="badge ok">active</span></td>
            <td className="mono small">{x.run}</td><td className="small">{x.checkpoint_mb ?? "?"} MB</td>
            <td><button disabled={busy} title="Move to gsplats/workspace/_archive/" onClick={() => {
              if (confirm(`Archive ${x.run}? The scene will have no active model until you promote one.`))
                act(() => api.archiveModel(scene, x.run), `Archived ${x.run}`);
            }}>Archive</button></td>
          </tr>
        ))}
        {m.archived.map((x) => (
          <tr key={x.run}>
            <td><span className="badge muted">{x.complete ? "archived" : "incomplete"}</span></td>
            <td className="mono small">{x.run}</td><td className="small">{x.checkpoint_mb ?? "—"}{x.checkpoint_mb ? " MB" : ""}</td>
            <td><button disabled={busy || !x.complete} title={x.complete ? "Make this the active model" : "Training did not finish"}
              onClick={() => act(() => api.promoteModel(scene, x.run), `Promoted ${x.run}`)}>Promote</button></td>
          </tr>
        ))}
      </tbody></table>
      {msg && <p className="small">{msg}</p>}
    </>
  );
}

function TrainingCurve({ scene, runKey }: { scene: string; runKey: string }) {
  const [data, setData] = useState<{ run: string | null; series: Record<string, [number, number][]> } | null>(null);
  const [tag, setTag] = useState<string | null>(null);
  const [log, setLog] = useState(true);
  useEffect(() => { api.metrics(scene).then(setData).catch(() => setData(null)); }, [scene, runKey]);
  const tags = Object.keys(data?.series ?? {}).sort();
  const pick = tag && tags.includes(tag) ? tag
    : tags.find((t) => /^train[ _]loss$/i.test(t)) ?? tags.find((t) => /loss/i.test(t)) ?? tags[0];
  return (
    <div className="panel">
      <div className="row">
        <h2 style={{ margin: 0 }}>Training curve</h2><span className="spacer" />
        {tags.length > 0 && (
          <>
            <select value={pick} onChange={(e) => setTag(e.target.value)}>{tags.map((t) => <option key={t}>{t}</option>)}</select>
            <label className="check small"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log scale</label>
          </>
        )}
      </div>
      {!data || tags.length === 0
        ? <p className="muted small">No TensorBoard events for the active model. Train with logging set to “tensorboard” to record curves.</p>
        : <>
            <LineChart points={data.series[pick!]} xLabel="step" yLabel={pick!} log={log} />
            <p className="muted small mono">{data.run}</p>
          </>}
    </div>
  );
}

function ArucoPanel({ a }: { a: any }) {
  const hist: { t0: number; hits: number; median_px: number | null }[] = a.histogram;
  return (
    <div className="panel">
      <h2>ArUco detections of marker {a.marker_id} per 10 s</h2>
      <BarChart bars={hist.map((h) => ({ label: `${h.t0}s`, value: h.hits }))} xLabel="from" yLabel="frames with the marker"
        detail={(i) => (hist[i].median_px ? `median size ${hist[i].median_px} px` : "not seen")} />
      <p className="small">
        ~{a.est_marked_frames} marked frames · spread over {a.windows_10s} windows
        <span className={a.windows_10s < 3 ? " bad" : " ok"}>{a.windows_10s < 3 ? " (short baseline)" : ""}</span>
        {" · "}median size <span className={a.median_px < 40 ? "bad" : ""}>{a.median_px} px</span>
      </p>
    </div>
  );
}
