// Course editor (Phase 3): build a SousVide course over a captured scene, preview the
// expert's minimum-snap trajectory, check clearance and the capture volume, save, fly.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { active, api, ApiError, courseApi, flightUrl, Geometry, Job, Preview } from "../api";
import { TimeChart } from "../charts";
import { Badge, usePoll } from "../components";
import Scene3D, { Tool, ViewOpts } from "../course/Scene3D";
import {
  AXES, blankLoop, Cell, emptyAxes, Course, CourseFile, displayPos, fromFile, insertAfter, inside, MAX_ORDERS, ORDERS,
  pos0, problems, round, toFile, Vec3,
} from "../course/model";

const UPSTREAM = ["circuit", "traverse", "infinity", "button_prod"];
const go = (scene?: string, name?: string) => {
  const h = `#/course${scene ? `/${encodeURIComponent(scene)}` : ""}${scene && name ? `/${encodeURIComponent(name)}` : ""}`;
  if (location.hash !== h) location.hash = h;
};

export default function CoursePage({ scene, name }: { scene?: string; name?: string }) {
  const scenes = usePoll(api.scenes, 0);
  const courses = usePoll(() => api.configs("courses"), 0);
  const pilots = usePoll(() => api.configs("pilots"), 0);
  const frames = usePoll(() => api.configs("frames"), 0);

  // ── scene geometry ─────────────────────────────────────────────────────────
  const [geo, setGeo] = useState<Geometry | null>(null);
  const [geoErr, setGeoErr] = useState<string | null>(null);
  useEffect(() => {
    setGeo(null); setGeoErr(null);
    if (!scene) return;
    let alive = true;
    courseApi.geometry(scene).then((g) => alive && setGeo(g)).catch((e) => alive && setGeoErr(String(e.message ?? e)));
    return () => { alive = false; };
  }, [scene]);

  // ── course state, undo ─────────────────────────────────────────────────────
  const [course, setCourse] = useState<Course | null>(null);
  const [saved, setSaved] = useState<string>("");            // JSON of the last loaded/saved file
  const [saveName, setSaveName] = useState(name ?? "");
  const [intCells, setIntCells] = useState<string[]>([]);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const hist = useRef<Course[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [goalSel, setGoalSel] = useState(false);
  const [tool, setTool] = useState<Tool>("move");
  const [opts, setOpts] = useState<ViewOpts>({ points: true, colorBy: "rgb", pointSize: 0.025, cameraPath: true, boxes: true });

  useEffect(() => {
    setMsg(null); setSel(null); hist.current = [];
    if (!name) { setCourse(null); setSaved(""); setIntCells([]); return; }
    setSaveName(name);
    api.config("courses", name).then((d: CourseFile) => {
      const c = fromFile(d);
      setCourse(c); setSaved(JSON.stringify(toFile(c)));
    }).catch((e) => setMsg({ ok: false, text: String(e.message ?? e) }));
    courseApi.lint(name).then((l) => setIntCells(l.int_cells)).catch(() => setIntCells([]));
  }, [name]);

  const edit = useCallback((fn: (c: Course) => Course, push = true) => {
    setCourse((c) => c && fn(c));
    if (push && course) { hist.current.push(course); if (hist.current.length > 200) hist.current.shift(); }
  }, [course]);
  const undo = () => { const prev = hist.current.pop(); if (prev) setCourse(prev); };

  const file = useMemo(() => (course ? toFile(course) : null), [course]);
  const fileJson = useMemo(() => (file ? JSON.stringify(file) : ""), [file]);
  const dirty = !!course && fileJson !== saved;
  const probs = useMemo(() => (course ? problems(course) : []), [course]);

  // ── preview (auto, fixed times) and expert solve ───────────────────────────
  const [pilot, setPilot] = useState("Viper");
  const [frame, setFrame] = useState("carl");
  const [method, setMethod] = useState("eval_single");
  const [clearance, setClearance] = useState(0.3);
  const [clearanceK, setClearanceK] = useState(5);
  const [auto, setAuto] = useState(true);
  const [pv, setPv] = useState<Preview | null>(null);
  const [pvFor, setPvFor] = useState<string>("");            // fileJson the preview belongs to
  const [pvBusy, setPvBusy] = useState<null | "fixed" | "expert">(null);
  const [pvErr, setPvErr] = useState<string | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const seq = useRef(0);

  const runPreview = useCallback(async (mode: "fixed" | "expert") => {
    if (!file || probs.length) return;
    const my = ++seq.current, snapshot = fileJson;
    setPvBusy(mode); setPvErr(null);
    try {
      const r = await courseApi.preview({ course: file, scene, pilot, frame, mode, clearance, clearance_k: clearanceK });
      if (my === seq.current) { setPv(r); setPvFor(snapshot); }
    } catch (e) {
      if (my !== seq.current) return;
      if (e instanceof ApiError && e.status === 409) { setPvErr("waiting for the running solve to finish…"); setTimeout(() => my === seq.current && runPreview(mode), 2000); return; }
      setPvErr(e instanceof ApiError ? e.message : String(e));
    } finally { if (my === seq.current) setPvBusy(null); }
  }, [file, fileJson, probs.length, scene, pilot, frame, clearance, clearanceK]);

  useEffect(() => {
    if (!auto || !file || probs.length || pvBusy === "expert") return;
    const t = setTimeout(() => runPreview("fixed"), 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileJson, auto, scene, pilot, frame, clearance, clearanceK]);
  const stale = !!pv && pvFor !== fileJson;

  // ── keyframe editing helpers ───────────────────────────────────────────────
  const setCell = (i: number, row: number, col: number, v: Cell, push = true) => edit((c) => {
    const kfs = c.kfs.map((k, j) => {
      if (j !== i) return k;
      const fo = k.fo.map((r) => [...r]);
      while (fo[row].length <= col) fo[row].push(null);
      fo[row][col] = v;
      return { ...k, fo };
    });
    return { ...c, kfs };
  }, push);
  const setKf = (i: number, patch: Partial<{ name: string; t: number }>) =>
    edit((c) => ({ ...c, kfs: c.kfs.map((k, j) => (j === i ? { ...k, ...patch } : k)) }));
  const moveTo = (i: number, v: Vec3) => edit((c) => ({
    ...c, kfs: c.kfs.map((k, j) => j !== i ? k : { ...k, fo: k.fo.map((r, ax) => (ax < 3 && r[0] !== null ? [round(v[ax]), ...r.slice(1)] : r)) }),
  }), false);
  const remove = (i: number) => { if (course && course.kfs.length > 2) { edit((c) => ({ ...c, kfs: c.kfs.filter((_, j) => j !== i) })); setSel(null); } };
  const shift = (i: number, d: -1 | 1) => edit((c) => {
    const j = i + d;
    if (j < 0 || j >= c.kfs.length) return c;
    const kfs = [...c.kfs];
    kfs[i] = { ...c.kfs[j], t: c.kfs[i].t };             // swap order, keep times increasing
    kfs[j] = { ...c.kfs[i], t: c.kfs[j].t };
    return { ...c, kfs };
  });
  const applySolvedTimes = () => pv && edit((c) => ({
    ...c, kfs: c.kfs.map((k) => { const s = pv.keyframes.find((x) => x.name === k.name); return s ? { ...k, t: round(s.t_solved, 3) } : k; }),
  }));

  // keyboard: ctrl+z undo, Delete removes, Esc deselects, M/R/A tools
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
      else if ((e.key === "Delete" || e.key === "Backspace") && sel !== null) remove(sel);
      else if (e.key === "Escape") { setSel(null); setGoalSel(false); }
      else if (e.key === "m") setTool("move");
      else if (e.key === "r") setTool("yaw");
      else if (e.key === "a") setTool("add");
    };
    addEventListener("keydown", h);
    return () => removeEventListener("keydown", h);
  });

  // ── save ───────────────────────────────────────────────────────────────────
  const save = async (): Promise<boolean> => {
    if (!file || probs.length) { setMsg({ ok: false, text: "Fix the problems listed under the table first." }); return false; }
    const n = saveName.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(n)) { setMsg({ ok: false, text: "Course name: letters, digits, _ and -." }); return false; }
    const exists = courses.data?.some((c) => c.name === n);
    if (exists && n !== name && !confirm(`Overwrite the existing course “${n}”?`)) return false;
    if (UPSTREAM.includes(n) && !confirm(`“${n}” is an upstream SousVide course: a re-clone restores the original. Save anyway?`)) return false;
    try {
      const r = await api.saveConfig("courses", n, file, true);
      setSaved(fileJson); setIntCells([]);
      setMsg({ ok: true, text: `Saved ${r.path}${r.mirrored ? ` (copied to the overlay)` : ""}.` });
      courses.reload();
      if (n !== name) go(scene, n);
      return true;
    } catch (e) { setMsg({ ok: false, text: e instanceof ApiError ? e.message : String(e) }); return false; }
  };

  // ── fly the expert through the existing pipeline job ───────────────────────
  const [allowOutside, setAllowOutside] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [flyErr, setFlyErr] = useState<string | null>(null);
  const sceneSt = usePoll(() => (scene ? api.scene(scene) : Promise.resolve(null)), job && active(job.status) ? 5000 : 0, [scene, job?.status]);
  useEffect(() => {
    if (!job || !active(job.status)) return;
    const t = setInterval(() => api.job(job.id).then(setJob).catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [job?.id, job?.status]);
  const fly = async () => {
    setFlyErr(null);
    if (!scene) return;
    if ((dirty || !name || saveName !== name) && !(await save())) return;
    try {
      const n = saveName.trim();
      const { id } = await api.submitFigs({
        scene, course: n, pilot, frame, method, from_step: "course", stop_after: "record",
        redo: ["course", "simulate", "validate"], ...(allowOutside ? { allow_outside: true } : {}),
      });
      setJob(await api.job(id));
    } catch (e) { setFlyErr(e instanceof ApiError ? e.message : String(e)); }
  };

  const experts = (pilots.data ?? []).filter((p) => p.kind === "expert").map((p) => p.name);
  const loadable = (scenes.data ?? []).filter((s) => s.has_workspace);
  const box = geo?.camera_box ?? null;
  const selKf = sel !== null && course ? course.kfs[sel] : null;
  const res = sceneSt.data?.results;

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <>
      <div className="row">
        <h1 style={{ margin: 0 }}>Course editor</h1>
        <span className="muted small">course frame (x, −y, −z): z points down, altitude is −z</span>
        <span className="spacer" />
        {scene && <a href={`#/scene/${scene}`}>Scene {scene} →</a>}
      </div>

      <div className="panel row" style={{ marginTop: 12 }}>
        <label className="f">Scene
          <select value={scene ?? ""} onChange={(e) => go(e.target.value || undefined, name)}>
            <option value="">choose…</option>
            {loadable.map((s) => <option key={s.scene} value={s.scene}>{s.scene}{s.loadable ? "" : " (no single model)"}</option>)}
          </select>
        </label>
        <label className="f">Course
          <select value={name ?? ""} onChange={(e) => {
            if (dirty && !confirm("Discard unsaved changes?")) return;
            go(scene, e.target.value || undefined);
          }}>
            <option value="">choose…</option>
            {courses.data?.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
        </label>
        <button disabled={!geo} title="Four-corner loop inside the recommended waypoint box"
          onClick={() => { if (dirty && !confirm("Discard unsaved changes?")) return;
            const c = blankLoop(geo?.waypoint_box ?? null, geo?.camera_box ?? null); hist.current = []; setCourse(c); setSaved(""); setSaveName(`${scene}_loop`);
            setIntCells([]); if (name) go(scene); }}>New loop</button>
        <span className="spacer" />
        {course && (
          <>
            <label className="f">Save as<input value={saveName} onChange={(e) => setSaveName(e.target.value.trim())} style={{ width: 170 }} /></label>
            <button onClick={undo} disabled={!hist.current.length} title="Ctrl+Z">Undo</button>
            <button className="primary" disabled={!dirty && saveName === name} onClick={save}>Save</button>
            {dirty && <span className="warn small">unsaved</span>}
          </>
        )}
      </div>
      {msg && <p className={msg.ok ? "ok" : "err"}>{msg.text}</p>}
      {geoErr && <p className="err">{geoErr}</p>}
      {geo?.warning && <p className="note small">{geo.warning}</p>}
      {intCells.length > 0 && (
        <p className="note small">This file has {intCells.length} integer cell(s) ({intCells.slice(0, 4).join(", ")}{intCells.length > 4 ? ", …" : ""}).
          FiGS reads an integer as the previous cell's value, so it would not fly as written, and the <span className="mono">course</span> step now refuses it.
          Saving from this editor writes floats and fixes it.</p>
      )}
      {!scene && <p className="muted">Pick a scene: the editor needs its SfM output (point cloud and camera path).</p>}

      {scene && (
        <div className="course-grid">
          <div>
            <div className="viewport">
              {course || geo ? (
                <Scene3D geo={geo} course={course ?? { Nco: 6, kfs: [], forces: null, goal: null, extra: {}, wpExtra: {} }}
                  sel={sel} onSelect={setSel} preview={stale ? null : pv} cursor={cursor} tool={tool} opts={opts}
                  goalSelected={goalSel} onGoalSelect={setGoalSel}
                  onDragStart={() => course && hist.current.push(course)}
                  onMove={moveTo}
                  onYaw={(i, y) => setCell(i, 3, 0, round(y), false)}
                  onAdd={(v) => { if (!course) return; const i = sel ?? course.kfs.length - 2;
                    edit((c) => insertAfter(c, i, v)); setSel(Math.min(i, course.kfs.length - 2) + 1); setTool("move"); }}
                  onGoalMove={(v) => edit((c) => ({ ...c, goal: c.goal && { ...c.goal, position: v.map((x) => round(x)) as Vec3 } }), false)} />
              ) : <p className="muted" style={{ padding: 16 }}>Loading scene…</p>}
              <div className="overlay">
                <div className="seg">
                  {([["move", "Move", "M"], ["yaw", "Yaw", "R"], ["add", "Add", "A"]] as const).map(([k, l, key]) => (
                    <button key={k} className={tool === k ? "on" : ""} title={`${l} (${key})`} onClick={() => setTool(k)}>{l}</button>
                  ))}
                </div>
                <div className="seg">
                  <button className={opts.points ? "on" : ""} onClick={() => setOpts({ ...opts, points: !opts.points })}>Points</button>
                  <button className={opts.colorBy === "altitude" ? "on" : ""} title="Colour points by altitude"
                    onClick={() => setOpts({ ...opts, colorBy: opts.colorBy === "rgb" ? "altitude" : "rgb" })}>Altitude</button>
                  <button className={opts.cameraPath ? "on" : ""} onClick={() => setOpts({ ...opts, cameraPath: !opts.cameraPath })}>Camera path</button>
                  <button className={opts.boxes ? "on" : ""} onClick={() => setOpts({ ...opts, boxes: !opts.boxes })}>Boxes</button>
                </div>
                <input type="range" min={0.005} max={0.08} step={0.005} value={opts.pointSize} title="Point size"
                  onChange={(e) => setOpts({ ...opts, pointSize: Number(e.target.value) })} style={{ width: 90 }} />
              </div>
              <div className="legend">
                {tool === "add" && <div><b>Add:</b> click the tinted plane to insert a keyframe after the selected one, at its altitude.</div>}
                <div><span className="swatch" style={{ background: "#8a8f98" }} />camera box (where the camera went, not free space)</div>
                <div><span className="swatch" style={{ background: "#2f9e6e" }} />waypoint box (inset {geo?.waypoint_box.margin ?? 0.5} m)
                  {emptyAxes(geo?.waypoint_box ?? null).length > 0 && <span className="warn"> · empty in {emptyAxes(geo!.waypoint_box).join(", ")}: the camera spanned less than twice the margin</span>}</div>
                <div><span className="swatch" style={{ background: "linear-gradient(90deg,#3b82f6,#f59e0b)" }} />path, slow → fast
                  {pv && !stale ? ` (0–${pv.stats.v_max} m/s)` : ""} · <span className="swatch" style={{ background: "#e5484d" }} />too close / outside</div>
                {geo && <div>{geo.n_points_sent.toLocaleString()} of {geo.n_points.toLocaleString()} sparse points</div>}
              </div>
            </div>
            {course && <PreviewPanel pv={pv} stale={stale} busy={pvBusy} err={pvErr} cursor={cursor} setCursor={setCursor}
              auto={auto} setAuto={setAuto} run={runPreview} applyTimes={applySolvedTimes} canRun={!probs.length}
              clearance={clearance} setClearance={setClearance} clearanceK={clearanceK} setClearanceK={setClearanceK} />}
          </div>

          <div>
            {course ? (
              <div className="panel">
                <div className="row"><h2 style={{ margin: 0 }}>Keyframes</h2><span className="spacer" />
                  <button disabled={sel === null} onClick={() => { if (sel === null) return; edit((c) => insertAfter(c, sel)); setSel(Math.min(sel, course.kfs.length - 2) + 1); }}>Insert after</button>
                </div>
                <p className="muted small">Position cells; empty = free (the solver chooses). Click a row or a sphere to select; drag the gizmo to move.
                  The first and last keyframes must fix x, y, z and yaw.</p>
                <table className="kf">
                  <thead><tr><th>name</th><th>t (s)</th><th>x</th><th>y</th><th>z</th><th>yaw</th><th /></tr></thead>
                  <tbody>
                    {course.kfs.map((k, i) => {
                      const out = box ? !inside(pos0(k), box) : false;
                      return (
                        <tr key={i} className={`${sel === i ? "sel" : ""} ${out ? "outside" : ""}`} onClick={() => { setSel(i); setGoalSel(false); }}
                          title={out ? "outside the captured volume" : undefined}>
                          <td><TextIn value={k.name} onCommit={(v) => setKf(i, { name: v })} className="num" /></td>
                          <td><NumIn value={k.t} onCommit={(v) => v !== null && setKf(i, { t: v })} /></td>
                          {[0, 1, 2, 3].map((r) => (
                            <td key={r}><NumIn value={k.fo[r][0] ?? null} nullable placeholder={r < 3 && pvFor === fileJson && pv
                              ? `≈${displayPos(course, i, pv.keyframes.find((x) => x.name === k.name)?.pos)[r].toFixed(2)}` : "free"}
                              onCommit={(v) => setCell(i, r, 0, v)} /></td>
                          ))}
                          <td style={{ whiteSpace: "nowrap" }}>
                            <button className="small" style={{ padding: "2px 6px" }} disabled={i === 0} onClick={(e) => { e.stopPropagation(); shift(i, -1); }} title="Move up">↑</button>
                            <button className="small" style={{ padding: "2px 6px" }} disabled={course.kfs.length <= 2} onClick={(e) => { e.stopPropagation(); remove(i); }} title="Delete (Del)">✕</button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {probs.length > 0 && <ul className="err small">{probs.map((p, i) => <li key={i}>{p.msg}</li>)}</ul>}
                {selKf && sel !== null && (
                  <>
                    <h3>{selKf.name}: derivative constraints</h3>
                    <div className="matrix">
                      <span />{ORDERS.map((o) => <span key={o} className="muted">{o}</span>)}
                      {AXES.map((ax, r) => (
                        <FragmentRow key={ax} label={ax} row={selKf.fo[r]} onCommit={(c, v) => setCell(sel, r, c, v)} />
                      ))}
                    </div>
                    <p className="muted small">Units: m, m/s, m/s², … and rad for yaw. Empty = free. Endpoints normally pin velocity to 0;
                      FiGS pads missing columns as free. Yaw is unwrapped: keep consecutive values within π of each other.</p>
                  </>
                )}
                <div className="row small" style={{ marginTop: 8 }}>
                  <label className="f">Nco<input type="number" min={1} max={12} value={course.Nco} style={{ width: 70 }}
                    onChange={(e) => edit((c) => ({ ...c, Nco: Math.max(1, Math.min(12, Number(e.target.value) || 6)) }))} /></label>
                  <span className="muted">forces: {course.forces ? "custom (edit in Configs)" : "none"}</span>
                </div>
              </div>
            ) : scene && <div className="panel muted">Load a course or start a new loop.</div>}

            {course && (
              <div className="panel">
                <h2>Semantic goal <span className="muted small">(thesis hook)</span></h2>
                {course.goal ? (
                  <>
                    <div className="fields">
                      <label className="f">Label<input value={course.goal.label} placeholder="e.g. the red chair"
                        onChange={(e) => edit((c) => ({ ...c, goal: c.goal && { ...c.goal, label: e.target.value } }))} /></label>
                      {[0, 1, 2].map((a) => (
                        <label key={a} className="f">{"xyz"[a]}<NumIn value={course.goal!.position[a]} onCommit={(v) => v !== null &&
                          edit((c) => ({ ...c, goal: c.goal && { ...c.goal, position: c.goal.position.map((x, j) => (j === a ? v : x)) as Vec3 } }))} /></label>
                      ))}
                    </div>
                    <div className="row" style={{ marginTop: 8 }}>
                      <button onClick={() => setGoalSel(true)}>Select to drag</button>
                      <button onClick={() => { edit((c) => ({ ...c, goal: null })); setGoalSel(false); }}>Remove</button>
                    </div>
                  </>
                ) : (
                  <button onClick={() => {
                    const b = geo?.waypoint_box;
                    const p: Vec3 = sel !== null ? displayPos(course, sel) : b ? [0, 1, 2].map((a) => round((b.lo[a] + b.hi[a]) / 2)) as Vec3 : [0, 0, -1];
                    edit((c) => ({ ...c, goal: { label: "", position: p } })); setGoalSel(true);
                  }}>Place goal marker</button>
                )}
                <p className="muted small">Saved as <span className="mono">semantic_goal</span> in the course file. FiGS and SousVide read only
                  <span className="mono"> waypoints</span> and <span className="mono">forces</span>, so it does not change the flight; it marks where the
                  natural-language goal extension plugs in.</p>
              </div>
            )}

            {course && (
              <div className="panel">
                <h2>Fly the expert</h2>
                <p className="muted small">Saves, then runs <span className="mono">figs_pipeline.py --from course --stop-after record</span>: the volume check,
                  the MPC flight through the splat, video checks and a run record. One GPU job at a time.</p>
                <div className="fields">
                  <label className="f">Expert<select value={pilot} onChange={(e) => setPilot(e.target.value)}>
                    {(experts.length ? experts : ["Viper"]).map((x) => <option key={x}>{x}</option>)}</select></label>
                  <label className="f">Frame<select value={frame} onChange={(e) => setFrame(e.target.value)}>
                    {(frames.data?.map((f) => f.name) ?? ["carl"]).map((x) => <option key={x}>{x}</option>)}</select></label>
                  <label className="f">Method<input value={method} onChange={(e) => setMethod(e.target.value)} /></label>
                </div>
                <label className="check small" style={{ marginTop: 8 }}>
                  <input type="checkbox" checked={allowOutside} onChange={(e) => setAllowOutside(e.target.checked)} />
                  Fly even if keyframes leave the captured volume
                </label>
                {flyErr && <p className="err">{flyErr}</p>}
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="primary" disabled={!!probs.length || (!!job && active(job.status))} onClick={fly}>
                    {dirty || saveName !== name ? "Save and fly" : "Fly"}</button>
                  {job && <><Badge s={job.status} /><a href={`#/jobs/${job.id}`}>job #{job.id} log</a></>}
                </div>
                {job && !active(job.status) && res?.sim && res?.course?.name === name && (
                  <>
                    <table style={{ marginTop: 10 }}><tbody>
                      <tr><td>Tracking error</td><td className={res.sim.track_err_max_m > 0.5 ? "bad" : ""}>mean {res.sim.track_err_mean_m} m · max {res.sim.track_err_max_m} m</td></tr>
                      <tr><td>Render check</td><td className={res.sim.dark_frames > 0 || res.sim.pixel_std < 5 ? "bad" : ""}>pixel std {res.sim.pixel_std} · dark frames {res.sim.dark_frames} / {res.sim.frames}</td></tr>
                      <tr><td>Sim VRAM</td><td>{res.sim.peak_vram_mib} MiB · {res.sim.wallclock}</td></tr>
                    </tbody></table>
                    <video key={job.id} controls src={flightUrl(scene!)} style={{ marginTop: 8 }} />
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function PreviewPanel(p: {
  pv: Preview | null; stale: boolean; busy: null | "fixed" | "expert"; err: string | null;
  cursor: number | null; setCursor: (t: number | null) => void; auto: boolean; setAuto: (b: boolean) => void;
  run: (m: "fixed" | "expert") => void; applyTimes: () => void; canRun: boolean;
  clearance: number; setClearance: (v: number) => void; clearanceK: number; setClearanceK: (v: number) => void;
}) {
  const { pv } = p;
  const u = pv?.inputs;
  const thrust = u ? u.u[0].map((v) => (u.lower[0] < 0 ? v / u.lower[0] : v)) : [];
  const rateLim = u ? Math.max(...[1, 2, 3].map((i) => Math.max(Math.abs(u.lower[i]), Math.abs(u.upper[i])))) : 5;
  const rate = u ? u.u[1].map((_, k) => Math.max(Math.abs(u.u[1][k]), Math.abs(u.u[2][k]), Math.abs(u.u[3][k]))) : [];
  const alt = pv ? pv.pos.map((q) => -q[2]) : [];
  const moved = pv ? pv.keyframes.filter((k) => Math.abs(k.t_solved - k.t_file) > 0.005).length : 0;
  const viol = u ? Object.entries(u.violations) : [];
  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="row">
        <h2 style={{ margin: 0 }}>Trajectory preview</h2>
        {pv && <span className="muted small">{pv.mode === "expert" ? `re-timed like ${pv.pilot} (kT ${pv.kT})` : "minimum snap at the file's times"} · {pv.hz} Hz · solved in {pv.solve_s} s</span>}
        <span className="spacer" />
        <label className="check small"><input type="checkbox" checked={p.auto} onChange={(e) => p.setAuto(e.target.checked)} /> live</label>
        <button disabled={!p.canRun || !!p.busy} onClick={() => p.run("fixed")}>{p.busy === "fixed" ? "Solving…" : "Preview"}</button>
        <button disabled={!p.canRun || !!p.busy} onClick={() => p.run("expert")}
          title="MinTimeSnap with the expert's kT re-optimises segment times, as the flight does. Slow: about a minute or two.">
          {p.busy === "expert" ? "Re-timing… (1–2 min)" : "Re-time like expert"}</button>
      </div>
      {p.err && <p className="err small">{p.err}</p>}
      {p.stale && <p className="muted small">Edited since this preview{p.auto ? "; updating…" : "."}</p>}
      {!pv && !p.busy && <p className="muted small">No preview yet.</p>}
      {pv && (
        <>
          <div className="kpis">
            <Kpi v={`${pv.duration_solved} s`} l={pv.mode === "expert" ? `duration (file ${pv.duration_file} s)` : "duration"} />
            <Kpi v={`${pv.stats.length_m} m`} l="path length" />
            <Kpi v={`${pv.stats.v_max} m/s`} l={`max speed (mean ${pv.stats.v_mean})`} />
            <Kpi v={`${pv.stats.a_max} m/s²`} l="max acceleration" />
            <Kpi v={pct(u?.max_use[0])} l="thrust vs limit" bad={(u?.max_use[0] ?? 0) > 1} />
            <Kpi v={pct(u ? Math.max(...u.max_use.slice(1).map((x) => x ?? 0)) : null)} l="body rate vs limit" bad={viol.length > 0} />
            {pv.clearance && <Kpi v={`${pv.clearance.min} m`} l={`min clearance at ${pv.clearance.at_t} s`} bad={pv.clearance.min < pv.clearance.threshold} />}
            {pv.inside && <Kpi v={pct(pv.inside.outside_frac)} l="of path outside capture" bad={pv.inside.outside_frac > 0} />}
          </div>
          {viol.length > 0 && <p className="err small">Inputs beyond {pv.pilot}'s bounds: {viol.map(([k, iv]) => `${k} at ${iv.map(([a, b]) => `${a}–${b} s`).join(", ")}`).join("; ")}.
            The MPC will saturate there: move the keyframes around it apart, or fly with an expert copy that has a smaller kT.</p>}
          {pv.stats.nonfinite_inputs > 0 && <p className="err small">{pv.stats.nonfinite_inputs} samples have undefined inputs (free fall or a singular yaw).</p>}
          {pv.mode === "expert" && moved > 0 && (
            <p className="note small">The expert re-timed {moved} keyframe(s): the file's t values are only the solver's starting guess.
              <button style={{ marginLeft: 8 }} onClick={p.applyTimes}>Write solved times into the keyframes</button></p>
          )}
          <div className="grid2">
            <TimeChart t={pv.t} y={pv.speed} label="Speed" unit="m/s" cursor={p.cursor} onCursor={p.setCursor} />
            <TimeChart t={pv.t} y={pv.acc_norm} label="Acceleration" unit="m/s²" cursor={p.cursor} onCursor={p.setCursor} />
            <TimeChart t={pv.t} y={thrust} label="Thrust (fraction of limit)" unit="" refs={[{ y: 1, label: "limit" }]}
              bad={u?.violations.thrust ?? []} cursor={p.cursor} onCursor={p.setCursor} />
            <TimeChart t={pv.t} y={rate} label="Largest body rate |ω|" unit="rad/s" refs={[{ y: rateLim, label: "limit" }]}
              bad={[...(u?.violations.wx ?? []), ...(u?.violations.wy ?? []), ...(u?.violations.wz ?? [])]} cursor={p.cursor} onCursor={p.setCursor} />
            {pv.clearance && (
              <TimeChart t={pv.t} y={pv.clearance.d} label={pv.clearance.k > 1 ? `Clearance (${pv.clearance.k}th-nearest sparse point)` : "Clearance to nearest sparse point"} unit="m"
                refs={[{ y: pv.clearance.threshold, label: `${pv.clearance.threshold} m` }]} bad={pv.clearance.below}
                cursor={p.cursor} onCursor={p.setCursor} />
            )}
            <TimeChart t={pv.t} y={alt} label="Altitude (−z)" unit="m" zeroBased={false}
              bad={pv.inside?.outside_intervals ?? []} cursor={p.cursor} onCursor={p.setCursor} />
          </div>
          <div className="row small">
            <label className="f">Clearance threshold (m)<input type="number" step={0.05} min={0} max={2} value={p.clearance}
              onChange={(e) => p.setClearance(Math.max(0, Number(e.target.value) || 0))} /></label>
            <label className="f">Ignore outliers: k-th point<input type="number" step={1} min={1} max={50} value={p.clearanceK}
              onChange={(e) => p.setClearanceK(Math.min(50, Math.max(1, Math.round(Number(e.target.value) || 1))))} /></label>
            {pv.clearance && <span className="muted" style={{ maxWidth: 420 }}>{pv.clearance.note}.
              {pv.clearance.k > 1 && ` Nearest single point: ${pv.clearance.nearest_min} m at ${pv.clearance.nearest_at_t} s.`}</span>}
          </div>
        </>
      )}
    </div>
  );
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
function Kpi({ v, l, bad }: { v: string; l: string; bad?: boolean }) {
  return <div className={`kpi ${bad ? "bad" : ""}`}><b className={bad ? "bad" : ""}>{v}</b><span>{l}</span></div>;
}

function FragmentRow({ label, row, onCommit }: { label: string; row: Cell[]; onCommit: (col: number, v: Cell) => void }) {
  return (
    <>
      <span className="mono">{label}</span>
      {Array.from({ length: MAX_ORDERS }, (_, c) => (
        <NumIn key={c} value={c < row.length ? row[c] : null} nullable placeholder={c < row.length ? "free" : "—"}
          onCommit={(v) => onCommit(c, v)} />
      ))}
    </>
  );
}

/** Number input that commits on blur/Enter. Empty → null when nullable. */
function NumIn({ value, onCommit, nullable = false, placeholder }:
  { value: number | null; onCommit: (v: number | null) => void; nullable?: boolean; placeholder?: string }) {
  const [text, setText] = useState(value === null ? "" : String(value));
  const [focus, setFocus] = useState(false);
  useEffect(() => { if (!focus) setText(value === null ? "" : String(value)); }, [value, focus]);
  const commit = () => {
    const s = text.trim();
    if (s === "") { if (nullable) onCommit(null); else setText(value === null ? "" : String(value)); return; }
    const v = Number(s);
    if (Number.isFinite(v)) { if (v !== value) onCommit(v); } else setText(value === null ? "" : String(value));
  };
  return (
    <input className={value === null ? "free" : ""} value={text} placeholder={placeholder} inputMode="decimal"
      onFocus={() => setFocus(true)} onBlur={() => { setFocus(false); commit(); }}
      onChange={(e) => setText(e.target.value)}
      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
  );
}

function TextIn({ value, onCommit, className }: { value: string; onCommit: (v: string) => void; className?: string }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return (
    <input className={className} value={text} onChange={(e) => setText(e.target.value)}
      onBlur={() => { const s = text.trim(); if (s && s !== value) onCommit(s); else setText(value); }}
      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
  );
}
