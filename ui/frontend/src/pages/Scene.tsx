import { useState } from "react";
import { api, ago, FigsRun, flightUrl, Step, STEPS } from "../api";
import { Badge, Select, TrainOptions, usePoll, useSubmit } from "../components";

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
          {s?.models.length === 0 && <p className="bad">No trained model: FiGS cannot load this scene.</p>}
          {s && s.models.length > 1 && <p className="bad">{s.models.length} models: FiGS refuses to guess. Retrain with “archive existing model”, or archive one by hand.</p>}
          {s?.models.map((m) => (
            <div key={m.run} className="small"><span className="mono">{m.run}</span> · {m.checkpoint_mb ?? "?"} MB checkpoint
              <div className="muted mono">{m.config}</div></div>
          ))}
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
  const [r, setR] = useState<FigsRun>({ scene, from_step: "train", stop_after: "verify", archive_old: true, train_vis: "tensorboard" });
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
