// Scene & Splat document: pipeline steps, reconstruction, training, models, last flight.
// The Capture & Splat tab drives it: Reconstruct (SfM, Train + options), Models, Steps.
import { useEffect, useState } from "react";
import { active, api, ApiError, ArchivedModel, FigsRun, flightUrl, Model, Step, STEPS } from "../api";
import { BarChart, LineChart } from "../charts";
import { Select, TrainOptions, useMachineTrainDefaults, usePoll, useSubmit } from "../components";
import { Problem, ToProblems, ToProperties, useCommands, useUi } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { Pill, Prop, PropSection, QueueTable, StepStrip, Tile } from "../shell/Panes";

const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
type Sel = { kind: "active"; m: Model } | { kind: "archived"; m: ArchivedModel } | null;

export default function ScenePage({ scene }: { scene: string }) {
  const d = useAppData();
  const { ui, setUi } = useUi();
  const st = usePoll(() => api.scene(scene), 5000, [scene]);
  const models = usePoll(() => api.models(scene), 10000, [scene]);
  const s = st.data;
  const res = s?.results ?? {};
  const jobs = d.jobs.filter((j) => j.scene === scene);
  const busyJob = jobs.find((j) => active(j.status));
  const { busy, err, submit } = useSubmit();
  const [videoKey, setVideoKey] = useState(0);
  const [sel, setSel] = useState<Sel>(null);
  const [msg, setMsg] = useState<string | null>(null);

  // retrain options (Reconstruct group) and the step range (Steps group)
  const [tr, setTr] = useState<FigsRun>({ scene, archive_old: true });
  useMachineTrainDefaults(d.machine?.defaults, setTr);
  const setT = (p: Partial<FigsRun>) => setTr((x) => ({ ...x, ...p }));
  const [range, setRange] = useState<{ from?: Step; stop?: Step; redo: boolean }>({ redo: false });
  const runRange = () => {
    const from = range.from, stop = range.stop;
    const redo = range.redo ? STEPS.filter((_, i) => i >= (from ? STEPS.indexOf(from) : 0) && i <= (stop ? STEPS.indexOf(stop) : STEPS.length - 1)) : undefined;
    submit({ scene, from_step: from, stop_after: stop, redo: redo as Step[] | undefined });
  };

  // models
  const act = async (fn: () => Promise<{ cleared_steps: string[] }>, what: string) => {
    setMsg(null);
    try {
      const r = await fn();
      setMsg(`${what}.${r.cleared_steps.length ? ` Cleared ${r.cleared_steps.join(", ")}: fly again to refresh the flight.` : ""}`);
      models.reload(); st.reload(); d.reload("scenes"); setSel(null);
    } catch (e) { setMsg(e instanceof ApiError ? e.message : String(e)); }
  };
  const m = models.data;
  const activeRun = m?.active[0];
  const promotable = sel?.kind === "archived" ? (sel.m.complete ? sel.m : null) : m?.archived.find((x) => x.complete) ?? null;
  const archiveTarget = sel?.kind === "active" ? sel.m : activeRun;
  const doArchive = () => archiveTarget && confirm(`Archive ${archiveTarget.run}? The scene will have no active model until you promote one.`)
    && act(() => api.archiveModel(scene, archiveTarget.run), `Archived ${archiveTarget.run}`);
  const doPromote = () => promotable && act(() => api.promoteModel(scene, promotable.run), `Promoted ${promotable.run}`);
  const lock = busyJob ? `Job #${busyJob.id} for this scene is ${busyJob.status}.` : false;
  const retrain = () => submit({ ...tr, scene, from_step: "train", stop_after: "verify", redo: ["train", "verify"] });

  useCommands({
    "run": { run: runRange, disabled: busy && "Submitting…" },
    "steps.run": { run: runRange, disabled: busy && "Submitting…" },
    "steps.from": { value: range.from ?? "", options: [["", "first not done"], ...STEPS], set: (v) => setRange((r) => ({ ...r, from: (v || undefined) as Step })) },
    "steps.stop": { value: range.stop ?? "", options: [["", "last"], ...STEPS], set: (v) => setRange((r) => ({ ...r, stop: (v || undefined) as Step })) },
    "steps.redo": { checked: range.redo, set: (v) => setRange((r) => ({ ...r, redo: v === "true" })) },
    "splat.sfm": { run: () => { if (confirm(`Re-run structure-from-motion for ${scene}? It takes tens of minutes, and the model must be retrained after it.`)) submit({ scene, only: "sfm", redo: ["sfm"] }); }, disabled: lock },
    "splat.train": { run: retrain, disabled: lock || (busy && "Submitting…") },
    "splat.downscale": { value: tr.downscale ?? "", set: (v) => setT({ downscale: num(v) }) },
    "splat.iters": { value: tr.train_iters ?? "", set: (v) => setT({ train_iters: num(v) }) },
    "splat.cache": { value: tr.cache_images ?? "", options: [["", "default"], "cpu", "gpu"], set: (v) => setT({ cache_images: (v || undefined) as FigsRun["cache_images"] }) },
    "splat.verify": { run: () => submit({ scene, only: "verify", redo: ["verify"] }), disabled: lock },
    "splat.archive": archiveTarget ? { run: doArchive, disabled: lock } : { disabled: "No active model." },
    "splat.promote": promotable ? { run: doPromote, disabled: lock } : { disabled: "No complete archived run." },
    "splat.curve": { run: () => document.getElementById(`curve-${scene}`)?.scrollIntoView({ behavior: "smooth", block: "center" }) },
    "ctx.model": { checked: !!sel },
  });

  // problems
  const probs: Problem[] = [
    ...(st.err ? [{ severity: "error" as const, where: scene, message: st.err }] : []),
    ...(m && m.active.length === 0 ? [{ severity: "error" as const, where: "models", message: "No active model: FiGS cannot load this scene. Train, or promote an archived run." }] : []),
    ...(m && m.active.length > 1 ? [{ severity: "error" as const, where: "models", message: `${m.active.length} active models: FiGS refuses to guess. Archive all but one.` }] : []),
    ...(res.sfm && res.sfm.pct < 90 ? [{ severity: "error" as const, where: "sfm", message: `Only ${res.sfm.pct}% of images registered (verify fails under 90%).` }] : []),
    ...(res.aruco && res.aruco.windows_10s < 3 ? [{ severity: "warning" as const, where: "aruco", message: `Marker seen in only ${res.aruco.windows_10s} ten-second windows: short baseline for the scale.` }] : []),
    ...(res.aruco && res.aruco.median_px < 40 ? [{ severity: "warning" as const, where: "aruco", message: `Median marker size ${res.aruco.median_px} px: film closer.` }] : []),
    ...(res.sim && res.sim.track_err_max_m > 0.5 ? [{ severity: "warning" as const, where: "flight", message: `Tracking error up to ${res.sim.track_err_max_m} m on ${res.course?.name}.` }] : []),
    ...(res.sim && res.sim.dark_frames > 0 ? [{ severity: "warning" as const, where: "flight", message: `${res.sim.dark_frames} dark frames: the drone left the splat.` }] : []),
    ...jobs.filter((j) => j.status === "failed").slice(0, 5).map((j) => ({ severity: "info" as const, where: `job #${j.id}`, message: `${j.label} failed.` })),
  ];
  const fromIdx = range.from ? STEPS.indexOf(range.from) : -1, stopIdx = range.stop ? STEPS.indexOf(range.stop) : -1;

  return (
    <>
      <Tile title="Pipeline steps" icon="steps" meta={<>.figs_pipeline_state/{scene}/ · click a step to run from it (Steps group){busyJob && <> · <Pill s={busyJob.status} label={`#${busyJob.id}`} /></>}</>}>
        {s ? <StepStrip steps={s.steps.map((x, i) => ({ name: x.step, state: x.done ? "done" : "",
          sub: x.done ? (x.when ?? "done").replace("T", " ").slice(5, 16) : "not done", tip: x.fingerprint ? `Done. Fingerprint ${x.fingerprint}. A step re-runs when a flag it depends on changes.` : x.done ? "Done." : "Not done yet.",
          range: (fromIdx >= 0 || stopIdx >= 0) && i >= Math.max(0, fromIdx) && (stopIdx < 0 || i <= stopIdx) }))}
          onPick={(name) => setRange((r) => ({ ...r, from: name as Step }))} /> : <p className="muted">{st.err ?? "Loading…"}</p>}
      </Tile>

      <div className="tiles cols-3">
        <Tile title="Reconstruction" icon="sfm" meta="hloc + COLMAP">
          {res.sfm ? (
            <table className="kv"><tbody>
              <tr><td>Registered</td><td className={res.sfm.pct < 90 ? "bad" : "ok"}>{res.sfm.registered} / {res.sfm.images} ({res.sfm.pct}%)</td></tr>
              <tr><td>Sparse points</td><td>{res.sfm.sparse_points?.toLocaleString()}</td></tr>
              {res.bounds_splat && ["x", "y", "z"].map((a) => <tr key={a}><td>bounds {a} (splat, m)</td><td className="mono">{res.bounds_splat[a][0]} … {res.bounds_splat[a][1]}</td></tr>)}
              {res.train && <><tr><td>Training</td><td>{res.train.wallclock} · peak {res.train.peak_vram_mib} MiB</td></tr>
                <tr><td>Settings</td><td className="mono small">iters {res.train.iters ?? "default"} · downscale {res.train.downscale ?? "auto"} · cache {res.train.cache_images ?? "default"}</td></tr></>}
            </tbody></table>) : <p className="empty">No SfM result yet.</p>}
          {res.bounds_splat && <p className="muted small">Course frame is (x, −y, −z): altitude is negative z.</p>}
        </Tile>
        <TrainingCurve scene={scene} runKey={s?.models.map((x) => x.run).join(",") ?? ""} />
        <Tile title="Models" icon="splat" meta={m ? `${m.active.length} active · ${m.archived.length} archived` : undefined} className="flush">
          {m && (
            <table>
              <thead><tr><th>state</th><th>run</th><th className="num">MB</th></tr></thead>
              <tbody>
                {m.active.map((x) => (
                  <tr key={x.run} className={`click ${sel?.m.run === x.run ? "sel" : ""}`} onClick={() => setSel(sel?.m.run === x.run ? null : { kind: "active", m: x })}>
                    <td><Pill s="succeeded" label="active" /></td><td className="mono small">{x.run}</td><td className="num">{x.checkpoint_mb ?? "?"}</td></tr>))}
                {m.archived.map((x) => (
                  <tr key={x.run} className={`click ${sel?.m.run === x.run ? "sel" : ""}`} onClick={() => setSel(sel?.m.run === x.run ? null : { kind: "archived", m: x })}>
                    <td><Pill s={x.complete ? "queued" : "warning"} label={x.complete ? "archived" : "incomplete"} /></td><td className="mono small">{x.run}</td><td className="num">{x.checkpoint_mb ?? "—"}</td></tr>))}
                {m.active.length + m.archived.length === 0 && <tr><td colSpan={3} className="muted">No trained model.</td></tr>}
              </tbody>
            </table>)}
          {msg && <p className="small" style={{ padding: "4px 8px" }}>{msg}</p>}
          <p className="muted small" style={{ padding: "4px 8px" }}>Select a run for Model Tools (Archive, Promote). FiGS loads exactly one active model.</p>
        </Tile>
      </div>

      <div className="tiles cols-3">
        <Tile title="Last flight" icon="video" meta={res.course?.name} actions={<button className="small" onClick={() => setVideoKey((k) => k + 1)}>Reload</button>}>
          <video key={videoKey} controls src={flightUrl(scene)} onError={(e) => ((e.target as HTMLVideoElement).style.display = "none")} />
          {res.sim ? (
            <table className="kv" style={{ marginTop: 4 }}><tbody>
              <tr><td>Frames</td><td>{res.sim.frames} at {res.sim.hz} Hz ({res.sim.duration_s} s)</td></tr>
              <tr><td>Tracking error</td><td className={res.sim.track_err_max_m > 0.5 ? "bad" : ""}>mean {res.sim.track_err_mean_m} m · max {res.sim.track_err_max_m} m</td></tr>
              <tr><td>Render check</td><td className={res.sim.dark_frames > 0 || res.sim.pixel_std < 5 ? "bad" : ""}>pixel std {res.sim.pixel_std} · dark frames {res.sim.dark_frames}</td></tr>
            </tbody></table>) : <p className="empty">No flight yet: open the course editor to fly one.</p>}
        </Tile>
        {res.aruco?.histogram ? <ArucoPanel a={res.aruco} /> : <Tile title="ArUco detections" icon="marker"><p className="empty">No ArUco result.</p></Tile>}
        <FlyCourse scene={scene} courses={d.courses.map((c) => c.name)} />
      </div>

      <div className="tiles cols-2">
        <Tile title="Retrain the splat" icon="retrain" meta="reuses this scene's SfM; runs train and verify">
          <TrainOptions r={tr} set={setT} vramMib={d.machine?.gpu.vram_mib} />
          {err && <p className="err">{err}</p>}
          <button className="primary" disabled={busy || !!lock} onClick={retrain}>Queue training</button>
        </Tile>
        <Tile title="Jobs for this scene" icon="job" className="flush">
          <QueueTable jobs={jobs.slice(0, 15)} focus={ui.focus} setFocus={(id) => setUi({ focus: id })} compact />
        </Tile>
      </div>

      <ToProperties>
        {sel ? (
          <>
            <div className="props-title"><Icon name="splat" size={16} />{sel.m.run}</div>
            <PropSection title="Model">
              <Prop k="State" tone={sel.kind === "active" ? "ok" : undefined}>{sel.kind === "active" ? "active (FiGS loads it)" : sel.m.complete ? "archived, complete" : "archived, incomplete"}</Prop>
              <Prop k="Checkpoint">{sel.m.checkpoint_mb ?? "—"} MB</Prop>
              {sel.kind === "active" && <><Prop k="Config" mono>{sel.m.config}</Prop><Prop k="Checkpoint file" mono>{sel.m.checkpoint ?? "—"}</Prop></>}
              {sel.kind === "archived" && <Prop k="Folder" mono>gsplats/workspace/_archive/{scene}/</Prop>}
            </PropSection>
            <div className="row" style={{ padding: 8 }}>
              {sel.kind === "active" ? <button className="push" disabled={!!lock} onClick={doArchive}>Archive</button>
                : <button className="push" disabled={!!lock || !sel.m.complete} onClick={doPromote}>Promote</button>}
            </div>
          </>
        ) : (
          <>
            <div className="props-title"><Icon name="scene" size={16} />{scene}</div>
            <PropSection title="Scene">
              <Prop k="Steps done">{s ? `${s.steps.filter((x) => x.done).length} / ${s.steps.length}` : "—"}</Prop>
              <Prop k="Active model" tone={m?.active.length === 1 ? "ok" : "bad"}>{activeRun?.run ?? "none"}</Prop>
              <Prop k="Last course">{res.course?.name ?? "—"}</Prop>
              <Prop k="Jobs">{jobs.length}{busyJob ? ` (#${busyJob.id} ${busyJob.status})` : ""}</Prop>
            </PropSection>
            {res.sfm && <PropSection title="SfM"><Prop k="Registered">{res.sfm.pct}%</Prop><Prop k="Points">{res.sfm.sparse_points?.toLocaleString()}</Prop></PropSection>}
            {res.sim && <PropSection title="Last flight">{Object.entries(res.sim).map(([k, v]) => <Prop key={k} k={k} mono>{typeof v === "object" ? JSON.stringify(v) : String(v)}</Prop>)}</PropSection>}
            <PropSection title="Step range (Steps group)"><Prop k="From">{range.from ?? "first not done"}</Prop><Prop k="Stop after">{range.stop ?? "last"}</Prop><Prop k="Redo">{range.redo ? "yes" : "no"}</Prop></PropSection>
          </>
        )}
      </ToProperties>
      <ToProblems items={probs} />
    </>
  );
}

function FlyCourse({ scene, courses }: { scene: string; courses: string[] }) {
  const [r, setR] = useState<FigsRun>({ scene, from_step: "bounds" });
  const { busy, err, submit } = useSubmit();
  return (
    <Tile title="Fly a course" icon="fly" meta="bounds → course → simulate → validate → record">
      <div className="fields">
        <Select label="Course" value={r.course} options={courses} allowDefault={false} onChange={(v) => setR({ ...r, course: v })} />
        <label className="f">Frame<input value={r.frame ?? ""} placeholder="carl" onChange={(e) => setR({ ...r, frame: e.target.value || undefined })} /></label>
        <label className="f">Expert<input value={r.pilot ?? ""} placeholder="Viper" onChange={(e) => setR({ ...r, pilot: e.target.value || undefined })} /></label>
        <label className="f">Method<input value={r.method ?? ""} placeholder="eval_single" onChange={(e) => setR({ ...r, method: e.target.value || undefined })} /></label>
      </div>
      <label className="check small" style={{ marginTop: 6 }}>
        <input type="checkbox" checked={!!r.allow_outside} onChange={(e) => setR({ ...r, allow_outside: e.target.checked })} />Fly even if waypoints leave the captured volume
      </label>
      {err && <p className="err">{err}</p>}
      <div className="row" style={{ marginTop: 6 }}>
        <button className="primary" disabled={busy || !r.course} onClick={() => submit({ ...r, redo: ["simulate", "validate"] })}>Queue flight</button>
        <a href={`#/course/${scene}${r.course ? `/${r.course}` : ""}`}>Open in the course editor</a>
      </div>
    </Tile>
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
    <Tile title="Training (splatfacto)" icon="chart" id={`curve-${scene}`} meta={data?.run ?? undefined}
      actions={tags.length > 0 && <span className="row" style={{ marginLeft: 6, flexWrap: "nowrap" }}>
        <select value={pick} onChange={(e) => setTag(e.target.value)} style={{ maxWidth: 150 }}>{tags.map((t) => <option key={t}>{t}</option>)}</select>
        <label className="check small"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} />log</label></span>}>
      {!data || tags.length === 0
        ? <p className="empty">No TensorBoard events for the active model. Train with logging set to “tensorboard” to record curves.</p>
        : <LineChart points={data.series[pick!]} xLabel="step" yLabel={pick!} log={log} />}
    </Tile>
  );
}

function ArucoPanel({ a }: { a: any }) {
  const hist: { t0: number; hits: number; median_px: number | null }[] = a.histogram;
  return (
    <Tile title={`ArUco marker ${a.marker_id} per 10 s`} icon="marker" meta={`~${a.est_marked_frames} marked frames · ${a.windows_10s} windows · median ${a.median_px} px`}>
      <BarChart bars={hist.map((h) => ({ label: `${h.t0}s`, value: h.hits }))} xLabel="from" yLabel="frames with the marker"
        detail={(i) => (hist[i].median_px ? `median size ${hist[i].median_px} px` : "not seen")} />
      {a.windows_10s < 3 && <p className="bad small">Short baseline: the marker is seen in fewer than 3 windows.</p>}
    </Tile>
  );
}
