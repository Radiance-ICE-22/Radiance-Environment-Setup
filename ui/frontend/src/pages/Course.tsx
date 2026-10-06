// Course workspace (Phase 3): build a SousVide course over a captured scene, preview the
// expert's minimum-snap trajectory, check clearance and the capture volume, save, fly.
// The Course tab (File, Edit, Preview, Checks, Show, Fly) and Keyframe Tools drive it; the
// selected keyframe's details and derivative matrix are in Properties, problems in Output.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { active, api, ApiError, courseApi, flightUrl, Geometry, Job, Preview } from "../api";
import { TimeChart } from "../charts";
import { usePoll } from "../components";
import { Problem as PaneProblem, ToProblems, ToProperties, useCommands, useDoc } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { Pill, Prop, PropSection, Splitter, Tile } from "../shell/Panes";
import Scene3D, { Tool, ViewOpts } from "../course/Scene3D";
import { useSplat } from "../splat/load";
import { useDrone } from "../course/Drone";
import {
  AXES, blankLoop, Cell, emptyAxes, Course, CourseFile, displayPos, fastSegments, fromFile, insertAfter, inside, MAX_ORDERS, ORDERS,
  pos0, problems, round, SemanticGoal, toFile, Vec3, withGoalAt,
} from "../course/model";

const UPSTREAM = ["circuit", "traverse", "infinity", "button_prod"];
const go = (scene?: string, name?: string) => {
  const h = `#/course${scene ? `/${encodeURIComponent(scene)}` : ""}${scene && name ? `/${encodeURIComponent(name)}` : ""}`;
  if (location.hash !== h) location.hash = h;
};

export default function CoursePage({ scene, name }: { scene?: string; name?: string }) {
  const d = useAppData();
  const courses = { data: d.courses, reload: () => d.reload("courses") };
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
  const [opts, setOpts] = useState<ViewOpts>({ points: true, colorBy: "rgb", pointSize: 0.025, cameraPath: true, boxes: true, drone: true });
  const drone = useDrone();
  const { active: docIsActive } = useDoc();
  const [navSpeed, setNavSpeed] = useState(() => { try { return Number(localStorage.getItem("galley.navSpeed")) || 1; } catch { return 1; } });

  // ── Gaussian splat (toggle): exported from the active checkpoint on first use, then cached ──
  // (splat/load.tsx: one download shared with the splat editor)
  const [splatOn, setSplatOn] = useState(false);
  useEffect(() => { setSplatOn(false); }, [scene]);
  const sp = useSplat(scene, splatOn);
  const [splatDrawn, setSplatDrawn] = useState(false);
  const [splatGpuErr, setSplatGpuErr] = useState<string | null>(null);
  useEffect(() => { setSplatDrawn(false); setSplatGpuErr(null); }, [sp.data]);
  const splat = {
    on: splatOn, meta: sp.meta, busy: sp.state === "exporting",
    state: !splatOn ? null : sp.state === "error" || splatGpuErr ? "error" : sp.state === "ready" && splatDrawn ? "ready" : "loading",
    err: splatGpuErr ? `Splat did not draw: ${splatGpuErr}` : sp.err ? `Splat did not load: ${sp.err}` : null,
  };
  const toggleSplat = () => {
    if (splatOn) { setSplatOn(false); return; }
    if (!scene) return;
    setSplatOn(true); setOpts((o) => ({ ...o, points: false }));
  };

  // the splat editor's Send to course saved this course: load it again (unless edited here)
  const [reloadN, setReloadN] = useState(0);
  const dirtyRef = useRef(false);
  useEffect(() => {
    const f = (e: Event) => { const n = (e as CustomEvent).detail?.name; if (n && n === name && !dirtyRef.current) setReloadN((k) => k + 1); };
    addEventListener("galley:course-saved", f); return () => removeEventListener("galley:course-saved", f);
  }, [name]);
  useEffect(() => {
    setMsg(null); setSel(null); hist.current = [];
    if (!name) { setCourse(null); setSaved(""); setIntCells([]); return; }
    setSaveName(name);
    api.config("courses", name).then((d: CourseFile) => {
      const c = fromFile(d);
      setCourse(c); setSaved(JSON.stringify(toFile(c)));
    }).catch((e) => setMsg({ ok: false, text: String(e.message ?? e) }));
    courseApi.lint(name).then((l) => setIntCells(l.int_cells)).catch(() => setIntCells([]));
  }, [name, reloadN]);

  const edit = useCallback((fn: (c: Course) => Course, push = true) => {
    setCourse((c) => c && fn(c));
    if (push && course) { hist.current.push(course); if (hist.current.length > 200) hist.current.shift(); }
  }, [course]);
  const undo = () => { const prev = hist.current.pop(); if (prev) setCourse(prev); };

  const file = useMemo(() => (course ? toFile(course) : null), [course]);
  const fileJson = useMemo(() => (file ? JSON.stringify(file) : ""), [file]);
  const dirty = !!course && fileJson !== saved;
  dirtyRef.current = dirty;
  const probs = useMemo(() => (course ? problems(course) : []), [course]);

  // ── preview (auto, fixed times) and expert solve ───────────────────────────
  const [pilot, setPilot] = useState("Viper");
  const [frame, setFrame] = useState("carl");
  const [method, setMethod] = useState("eval_single");
  // With the drone's size subtracted the threshold is a gap, so it is smaller than the old
  // centre-distance default (0.3 m) — 0.15 m gap ≈ 0.34 m centre distance for a 0.19 m drone.
  const [clearance, setClearance] = useState(0.15);
  const [useBody, setUseBody] = useState(true);
  const [clearanceK, setClearanceK] = useState(5);
  const [auto, setAuto] = useState(true);
  const [pv, setPv] = useState<Preview | null>(null);
  const [pvFor, setPvFor] = useState<string>("");            // fileJson the preview belongs to
  const [pvBusy, setPvBusy] = useState<null | "fixed" | "expert">(null);
  const [pvErr, setPvErr] = useState<string | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const seq = useRef(0);
  const bodyR = useBody && drone ? drone.meta.radius : 0;

  const runPreview = useCallback(async (mode: "fixed" | "expert") => {
    if (!file || probs.length) return;
    const my = ++seq.current, snapshot = fileJson;
    setPvBusy(mode); setPvErr(null);
    try {
      const r = await courseApi.preview({ course: file, scene, pilot, frame, mode, clearance, clearance_k: clearanceK, body_radius: bodyR });
      if (my === seq.current) { setPv(r); setPvFor(snapshot); }
    } catch (e) {
      if (my !== seq.current) return;
      if (e instanceof ApiError && e.status === 409) { setPvErr("waiting for the running solve to finish…"); setTimeout(() => my === seq.current && runPreview(mode), 2000); return; }
      setPvErr(e instanceof ApiError ? e.message : String(e));
    } finally { if (my === seq.current) setPvBusy(null); }
  }, [file, fileJson, probs.length, scene, pilot, frame, clearance, clearanceK, bodyR]);

  useEffect(() => {
    if (!auto || !file || probs.length || pvBusy === "expert") return;
    const t = setTimeout(() => runPreview("fixed"), 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileJson, auto, scene, pilot, frame, clearance, clearanceK, bodyR]);
  const stale = !!pv && pvFor !== fileJson;

  // ── play the preview: moves the cursor (charts and the drone in the 3D view) in real time ──
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;
  useEffect(() => {
    if (!playing) return;
    if (!pv || stale || pv.t.length < 2) { setPlaying(false); return; }
    const t0 = pv.t[0], t1 = pv.t[pv.t.length - 1];
    const c = cursorRef.current;
    const from = c !== null && c < t1 - 1e-3 ? c : t0;
    let raf = 0, last = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = from + ((now - start) / 1000) * speed;
      if (t >= t1) { setCursor(t1); setPlaying(false); return; }
      if (now - last > 30) { setCursor(t); last = now; }     // ~30 fps is plenty for the charts
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, pv, stale, speed]);

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

  // ── tiles: sizes (dragged splitters, remembered) and one tile maximized ───
  const [lay, setLayRaw] = useState<CourseLayout>(() => { try { return { ...LAY0, ...JSON.parse(localStorage.getItem(LAY_KEY) ?? "{}") }; } catch { return LAY0; } });
  const setLay = (p: Partial<CourseLayout>) => setLayRaw((l) => { const n = { ...l, ...p }; try { localStorage.setItem(LAY_KEY, JSON.stringify(n)); } catch { /* */ } return n; });
  const [max, setMax] = useState<null | "view" | "kf" | "charts">(null);
  const toggleMax = (t: "view" | "kf" | "charts") => setMax((m) => (m === t ? null : t));
  const wrap = useRef<HTMLDivElement>(null);

  // keyboard (document-local): Delete removes, Esc deselects, Ins inserts, M/R/A tools.
  // Ctrl+Z, Ctrl+S, F5 and F6 are global and reach this document through its bindings.
  const docActive = useRef(false);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (!docActive.current) return;
      const el = e.target as HTMLElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Escape" && max) { e.preventDefault(); setMax(null); return; }   // before the window's Esc (full screen)
      if (e.key === "Delete" && sel !== null) remove(sel);
      else if (e.key === "Insert" && sel !== null) insertSel();
      else if (e.key === "Escape") { setSel(null); setGoalSel(false); }
      else if (e.key === "m") setTool("move");
      else if (e.key === "r") setTool("yaw");
      else if (e.key === "a") setTool("add");
    };
    addEventListener("keydown", h, true);
    return () => removeEventListener("keydown", h, true);
  });

  // ── save ───────────────────────────────────────────────────────────────────
  const save = async (as?: string): Promise<boolean> => {
    if (!file || probs.length) { setMsg({ ok: false, text: "Fix the problems listed in Output ▸ Problems first." }); return false; }
    const n = (as ?? saveName).trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(n)) { setMsg({ ok: false, text: "Course name: letters, digits, _ and -." }); return false; }
    const exists = courses.data?.some((c) => c.name === n);
    if (exists && n !== name && !confirm(`Overwrite the existing course “${n}”?`)) return false;
    if (UPSTREAM.includes(n) && !confirm(`“${n}” is an upstream SousVide course: a re-clone restores the original. Save anyway?`)) return false;
    try {
      const r = await api.saveConfig("courses", n, file, true);
      setSaved(fileJson); setIntCells([]); setSaveName(n);
      setMsg({ ok: true, text: `Saved ${r.path}${r.mirrored ? ` (copied to the overlay)` : ""}.` });
      courses.reload();
      if (n !== name) go(scene, n);
      return true;
    } catch (e) { setMsg({ ok: false, text: e instanceof ApiError ? e.message : String(e) }); return false; }
  };
  const saveAs = () => { const n = prompt("Save the course as:", saveName || `${scene}_course`)?.trim(); if (n) save(n); };
  const newLoop = () => {
    if (dirty && !confirm("Discard unsaved changes?")) return;
    const c = blankLoop(geo?.waypoint_box ?? null, geo?.camera_box ?? null); hist.current = []; setCourse(c); setSaved(""); setSaveName(`${scene}_loop`);
    setIntCells([]); setSel(null); if (name) go(scene);
  };

  // ── fly the expert through the existing pipeline job ───────────────────────
  const [allowOutside, setAllowOutside] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [flyErr, setFlyErr] = useState<string | null>(null);
  const sceneSt = usePoll(() => (scene ? api.scene(scene) : Promise.resolve(null)), job && active(job.status) ? 5000 : 0, [scene, job?.status]);
  useEffect(() => { const j = job && d.jobs.find((x) => x.id === job.id); if (j && j.status !== job!.status) setJob(j); }, [d.jobs, job]);
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
      setJob(await api.job(id)); d.reload("jobs");
    } catch (e) { setFlyErr(e instanceof ApiError ? e.message : String(e)); }
  };

  const experts = (pilots.data ?? []).filter((p) => p.kind === "expert").map((p) => p.name);
  const box = geo?.camera_box ?? null;
  const selKf = sel !== null && course ? course.kfs[sel] : null;
  const res = sceneSt.data?.results;
  const solvedKf = (k: { name: string } | null) => (k && pv && !stale ? pv.keyframes.find((x) => x.name === k.name) ?? null : null);
  const insertSel = () => { if (sel === null || !course) return; edit((c) => insertAfter(c, sel)); setSel(Math.min(sel, course.kfs.length - 2) + 1); };
  const placeGoal = () => {
    if (!course) return;
    if (course.goal) { setGoalSel(true); setSel(null); return; }
    const b = geo?.waypoint_box;
    const p: Vec3 = sel !== null ? displayPos(course, sel) : b ? [0, 1, 2].map((a) => round((b.lo[a] + b.hi[a]) / 2)) as Vec3 : [0, 0, -1];
    edit((c) => ({ ...c, goal: { label: "", position: p } })); setGoalSel(true); setSel(null);
  };
  const editSel = (fn: (fo: Cell[][]) => Cell[][]) => sel !== null && edit((c) => ({ ...c, kfs: c.kfs.map((k, j) => (j === sel ? { ...k, fo: fn(k.fo.map((r) => [...r])) } : k)) }));
  const [showMatrix, setShowMatrix] = useState(true);
  const [cursorAt, setCursorAt] = useState<number | null>(null);
  const ci = pv && !stale && cursor !== null ? nearest(pv.t, cursor) : null;
  useEffect(() => setCursorAt(ci), [ci]);

  // ── ribbon ─────────────────────────────────────────────────────────────────
  const loadable = d.scenes.filter((s) => s.has_workspace);
  const noCourse = !course && "Load a course (Course box) or start a New loop.";
  const noPv = !course ? noCourse : probs.length ? "Fix the course problems first." : false;
  const yes = (v: string) => v === "true";
  useCommands({
    "course.scene": { value: scene ?? "", options: loadable.map((s) => [s.scene, s.loadable ? s.scene : `${s.scene} (no single model)`] as [string, string]), set: (v) => go(v || undefined, name) },
    "course.name": { value: name ?? "", options: (courses.data ?? []).map((c) => c.name), set: (v) => { if (dirty && !confirm("Discard unsaved changes?")) return; go(scene, v || undefined); }, disabled: !scene && "Pick a scene first." },
    "course.newloop": { run: newLoop, disabled: !geo && (scene ? "Loading the scene…" : "Pick a scene first.") },
    "file.save": { run: () => save(), disabled: noCourse || (!dirty && saveName === name && "Saved.") },
    "course.saveas": { run: saveAs, disabled: noCourse },
    "edit.undo": { run: undo, disabled: !hist.current.length && "Nothing to undo." },
    "course.lint": name ? { run: () => courseApi.lint(name).then((l) => {
      setIntCells(l.int_cells);
      const fast = l.fast_segments ?? [];
      const parts = [l.int_cells.length ? `${l.int_cells.length} integer cell(s): ${l.int_cells.slice(0, 6).join(", ")}. Saving writes floats.` : "No integer cells.",
        fast.length ? `${fast.length} leg(s) with far too little time in the saved file: ${fast.slice(0, 3).join("; ")}.` : "Keyframe times are plausible."];
      setMsg({ ok: !l.int_cells.length && !fast.length, text: parts.join(" ") });
    }) } : { disabled: "Save the course first." },
    "tool.move": { checked: tool === "move", run: () => setTool("move"), disabled: noCourse },
    "tool.yaw": { checked: tool === "yaw", run: () => setTool("yaw"), disabled: noCourse },
    "tool.add": { checked: tool === "add", run: () => setTool("add"), disabled: noCourse },
    "tool.goal": { checked: goalSel, run: placeGoal, disabled: noCourse },
    "prev.run": { run: () => runPreview("fixed"), disabled: noPv || (pvBusy ? "Solving…" : false) },
    "prev.retime": { run: () => runPreview("expert"), disabled: noPv || (pvBusy ? "Solving…" : false), label: pvBusy === "expert" ? "Re-timing…" : undefined },
    "prev.live": { checked: auto, set: (v) => setAuto(yes(v)) },
    "prev.play": { checked: playing, run: () => setPlaying(!playing), disabled: (!pv || stale) && "Preview first.", label: playing ? "Pause" : undefined },
    "prev.speed": { value: String(speed), options: [["0.25", "0.25×"], ["0.5", "0.5×"], ["1", "1×"], ["2", "2×"]], set: (v) => setSpeed(Number(v)) },
    "chk.gap": { value: clearance, set: (v) => setClearance(Math.max(0, Number(v) || 0)) },
    "chk.k": { value: clearanceK, set: (v) => setClearanceK(Math.min(50, Math.max(1, Math.round(Number(v) || 1)))) },
    "chk.body": drone ? { checked: useBody, set: (v) => setUseBody(yes(v)) } : { disabled: "No drone model (public/models/drone.json)." },
    "view.splat": { checked: splat.on, run: toggleSplat,
      disabled: !scene ? "Pick a scene first." : d.scenes.find((x) => x.scene === scene)?.loadable === false ? "This scene needs exactly one trained model (Capture & Splat ▸ Models)." : splat.busy ? "Exporting the splat from the checkpoint…" : false,
      label: splat.busy ? "Exporting…" : splat.on && splat.state === "loading" ? "Loading…" : undefined },
    "view.navspeed": { value: String(navSpeed), options: [["0.25", "0.25×"], ["0.5", "0.5×"], ["1", "1×"], ["2", "2×"], ["4", "4×"]],
      set: (v) => { setNavSpeed(Number(v)); try { localStorage.setItem("galley.navSpeed", v); } catch { /* */ } } },
    "view.points": { checked: opts.points, run: () => setOpts({ ...opts, points: !opts.points }) },
    "view.alt": { checked: opts.colorBy === "altitude", run: () => setOpts({ ...opts, colorBy: opts.colorBy === "rgb" ? "altitude" : "rgb" }) },
    "view.cam": { checked: opts.cameraPath, run: () => setOpts({ ...opts, cameraPath: !opts.cameraPath }) },
    "view.boxes": { checked: opts.boxes, run: () => setOpts({ ...opts, boxes: !opts.boxes }) },
    "view.drone": drone ? { checked: opts.drone, run: () => setOpts({ ...opts, drone: !opts.drone }) } : { disabled: "No drone model (public/models/drone.json)." },
    "view.psize": { value: opts.pointSize, set: (v) => setOpts({ ...opts, pointSize: Math.min(0.2, Math.max(0.002, Number(v) || 0.025)) }) },
    "fly.expert": { value: pilot, options: experts.length ? experts : ["Viper"], set: setPilot },
    "fly.frame": { value: frame, options: frames.data?.map((f) => f.name) ?? ["carl"], set: setFrame },
    "fly.method": { value: method, set: (v) => setMethod(v || "eval_single") },
    "fly.outside": { checked: allowOutside, set: (v) => setAllowOutside(yes(v)) },
    "fly.go": { run: fly, disabled: !scene ? "Pick a scene." : noPv || (!!job && active(job.status) && `Job #${job.id} is ${job.status}.`), label: dirty || saveName !== name ? "Save + fly" : undefined },
    "run": { run: fly, disabled: !scene ? "Pick a scene." : noPv || (!!job && active(job.status) && `Job #${job.id} is ${job.status}.`) },
    "tile.view": { checked: max === "view", run: () => toggleMax("view"), disabled: !scene && "Pick a scene first." },
    "tile.kf": { checked: max === "kf", run: () => toggleMax("kf"), disabled: noCourse },
    "tile.charts": { checked: max === "charts", run: () => toggleMax("charts"), disabled: (!pv && "Preview first (F6).") || noCourse },
    "tile.reset": { run: () => { setMax(null); setLay(LAY0); } },
    "ctx.keyframe": { checked: sel !== null && !!course },
    "kf.insert": { run: insertSel },
    "kf.delete": { run: () => sel !== null && remove(sel), disabled: (course?.kfs.length ?? 0) <= 2 && "A course keeps at least 2 keyframes." },
    "kf.up": { run: () => { if (sel) { shift(sel, -1); setSel(sel - 1); } }, disabled: sel === 0 && "Already first." },
    "kf.down": { run: () => { if (sel !== null && course && sel < course.kfs.length - 1) { shift(sel, 1); setSel(sel + 1); } }, disabled: !!course && sel === course.kfs.length - 1 && "Already last." },
    "kf.matrix": { checked: showMatrix, run: () => setShowMatrix(!showMatrix) },
    "kf.stop": { run: () => editSel((fo) => fo.map((r) => { while (r.length < 2) r.push(null); r[1] = 0; return r; })) },
    "kf.free": { run: () => editSel((fo) => fo.map((r) => r.map((v, c) => (c === 0 ? v : null)))) },
    "kf.fix": { run: () => sel !== null && course && editSel((fo) => { const p = displayPos(course, sel, solvedKf(selKf)?.pos as Vec3 | undefined); return fo.map((r, ax) => (ax < 3 && r[0] === null ? [round(p[ax]), ...r.slice(1)] : r)); }),
      disabled: !!selKf && [0, 1, 2].every((a) => selKf.fo[a][0] !== null) && "x, y and z are fixed already." },
    "kf.t": { value: selKf?.t ?? "", set: (v) => sel !== null && Number.isFinite(Number(v)) && v !== "" && setKf(sel, { t: Number(v) }) },
    "kf.solved": { run: applySolvedTimes, disabled: !(pv && pv.mode === "expert") && "Re-time first (Preview ▸ Re-time)." },
  });

  // ── problems ───────────────────────────────────────────────────────────────
  const paneProbs: PaneProblem[] = [
    ...(msg && !msg.ok ? [{ severity: "error" as const, where: "course", message: msg.text }] : []),
    ...(geoErr ? [{ severity: "error" as const, where: `scene ${scene}`, message: geoErr }] : []),
    ...(geo?.warning ? [{ severity: "warning" as const, where: `scene ${scene}`, message: geo.warning }] : []),
    ...probs.map((p) => ({ severity: "error" as const, where: p.kf !== undefined && course ? `keyframe ${course.kfs[p.kf]?.name}` : "course", message: p.msg })),
    ...(intCells.length ? [{ severity: "warning" as const, where: `${name}.json`, message: `${intCells.length} integer cell(s) (${intCells.slice(0, 4).join(", ")}${intCells.length > 4 ? ", …" : ""}): FiGS reads an integer as the previous cell's value. Saving from this editor writes floats.` }] : []),
    ...(course && box ? course.kfs.filter((k) => !inside(pos0(k), box)).map((k) => ({ severity: "warning" as const, where: `keyframe ${k.name}`, message: "Outside the captured volume: the splat renders mush there." })) : []),
    ...(course ? fastSegments(course).map((f) => ({ severity: "warning" as const, where: `timing ${f.split(":")[0]}`, message: `${f.slice(f.indexOf(":") + 2)}: far too little time for the distance. The expert's re-time cannot recover from such a starting guess (it stops at it), so this would fly as previewed. Give the leg about 1 s per metre in the t column.` })) : []),
    ...(pv && !stale && pv.timing?.retime_stalled ? [{ severity: "error" as const, where: "re-time", message: "The expert's time optimisation returned the file's times unchanged while the path breaks the input limits: it could not move from this starting guess. Space the keyframes' t values out (about 1 s per metre), then Re-time again." }] : []),
    ...(pv && !stale && pv.clearance && pv.clearance.min < pv.clearance.threshold ? [{ severity: "warning" as const, where: `t = ${pv.clearance.at_t} s`, message: `${(pv.clearance.body_radius ?? 0) > 0 ? "Gap" : "Clearance"} ${pv.clearance.min} m, under ${pv.clearance.threshold} m (${pv.clearance.below.map(([a, b]) => `${a}–${b} s`).join(", ")}).` }] : []),
    ...(pv && !stale ? Object.entries(pv.inputs.violations).map(([k, iv]) => ({ severity: "warning" as const, where: `input ${k}`, message: `Beyond ${pv.pilot}'s bounds at ${iv.map(([a, b]) => `${a}–${b} s`).join(", ")}: the MPC will saturate. Move the keyframes around it apart, or use an expert copy with a smaller kT.` })) : []),
    ...(pv && !stale && pv.inside && pv.inside.outside_frac > 0 ? [{ severity: "warning" as const, where: "path", message: `${Math.round(pv.inside.outside_frac * 100)}% of the path is outside the captured volume.` }] : []),
    ...(pv && !stale && pv.stats.nonfinite_inputs > 0 ? [{ severity: "error" as const, where: "path", message: `${pv.stats.nonfinite_inputs} samples have undefined inputs (free fall or a singular yaw).` }] : []),
    ...(pvErr ? [{ severity: "error" as const, where: "preview", message: pvErr }] : []),
    ...(flyErr ? [{ severity: "error" as const, where: "fly", message: flyErr }] : []),
    ...(splat.err ? [{ severity: "error" as const, where: "splat", message: splat.err }] : []),
  ];

  // ── render ─────────────────────────────────────────────────────────────────
  const solvedSel = solvedKf(selKf);
  const moved = pv ? pv.keyframes.filter((k) => Math.abs(k.t_solved - k.t_file) > 0.005).length : 0;
  return (
    <DocActive refObj={docActive}>
      {msg && <div className={msg.ok ? "ok" : "err"} style={{ padding: "2px 8px", marginBottom: 4, background: msg.ok ? "var(--ok-bg)" : "var(--bad-bg)", border: "1px solid var(--line)", flex: "none" }}>
        {msg.text} <button className="lnk" onClick={() => setMsg(null)}>dismiss</button></div>}
      {!scene ? (
        <Tile title="Course workspace" icon="route"><p>Pick a scene in the ribbon (Course ▸ File ▸ Scene): the editor draws the course over its SfM point cloud and camera path.</p>
          <div className="row">{loadable.map((s) => <button key={s.scene} onClick={() => go(s.scene)}>{s.scene}</button>)}</div></Tile>
      ) : (
        <div className="cw" ref={wrap}>
          <div className="cw-top" style={max === "charts" ? { display: "none" } : max || !course || !pv ? { flex: "1 1 0" } : { flex: `0 0 ${lay.topPct}%` }}>
            <div className="cw-view" style={max === "kf" ? { display: "none" } : undefined}>
              <div className="viewport">
                {course || geo ? (
                  <Scene3D geo={geo} course={course ?? { Nco: 6, kfs: [], forces: null, goal: null, extra: {}, wpExtra: {} }}
                    sel={sel} onSelect={(i) => { setSel(i); if (i !== null) setGoalSel(false); }} preview={stale ? null : pv} cursor={cursor} tool={tool} opts={opts} drone={drone}
                    goalSelected={goalSel} onGoalSelect={setGoalSel}
                    onDragStart={() => course && hist.current.push(course)}
                    onMove={moveTo}
                    onYaw={(i, y) => setCell(i, 3, 0, round(y), false)}
                    onAdd={(v) => { if (!course) return; const i = sel ?? course.kfs.length - 2;
                      edit((c) => insertAfter(c, i, v)); setSel(Math.min(i, course.kfs.length - 2) + 1); setTool("move"); }}
                    onGoalMove={(v) => edit((c) => ({ ...c, goal: c.goal && withGoalAt(c.goal, v.map((x) => round(x)) as Vec3) }), false)}
                    keyNav={docIsActive} navSpeed={navSpeed}
                    splat={splat.on && sp.state === "ready" ? sp.data : null} onSplatReady={() => setSplatDrawn(true)} onSplatError={setSplatGpuErr} />
                ) : <p className="muted" style={{ padding: 16 }}>{geoErr ?? "Loading scene…"}</p>}
                <div className="vtag">{scene}{name ? ` / ${name}` : course ? ` / ${saveName} (unsaved)` : ""}{dirty ? " ●" : ""} · course frame (x, −y, −z): z down
                  {splat.busy && " · exporting the splat…"}{splat.on && splat.state === "loading" && !splat.busy && ` · ${sp.state === "downloading" ? `downloading the splat (${mb(splat.meta?.bytes)}, ${Math.round(sp.progress * 100)}%)` : "loading the splat"}…`}
                  {splat.on && splat.state === "ready" && ` · splat: ${(splat.meta?.n_written ?? 0).toLocaleString()} Gaussians`}</div>
                <div className="overlay">
                  <div className="seg">
                    <button className={`seg-b ${max === "view" ? "on" : ""}`} title={max === "view" ? "Restore the tiles (Esc)" : "Maximize the 3D view"} onClick={() => toggleMax("view")}>
                      <Icon name={max === "view" ? "restore" : "maximize"} size={18} /></button>
                  </div>
                  <div className="seg">
                    {([["move", "move", "Move (M)"], ["yaw", "yaw", "Yaw (R)"], ["add", "addpt", "Add (A)"]] as const).map(([k, ic, t]) => (
                      <button key={k} className={`seg-b ${tool === k ? "on" : ""}`} title={t} onClick={() => setTool(k)}><Icon name={ic} size={18} /></button>))}
                  </div>
                </div>
                <div className="legend">
                  <div><b>Keys:</b> arrows fly · PgUp/PgDn or E/Q up/down · Ctrl+arrows look · Shift faster · +/− closer/farther · F centre on selection · Home reset view</div>
                  {tool === "add" && <div><b>Add:</b> click the tinted plane to insert a keyframe after the selected one, at its altitude.</div>}
                  <div><span className="swatch" style={{ background: "#8a8f98" }} />camera box (where the camera went, not free space)
                    {" · "}<span className="swatch" style={{ background: "#2f9e6e" }} />waypoint box (inset {geo?.waypoint_box.margin ?? 0.5} m)
                    {emptyAxes(geo?.waypoint_box ?? null).length > 0 && <span className="warn"> · empty in {emptyAxes(geo!.waypoint_box).join(", ")}</span>}</div>
                  <div><span className="swatch" style={{ background: "linear-gradient(90deg,#3b82f6,#f59e0b)" }} />path, slow → fast
                    {pv && !stale ? ` (0–${pv.stats.v_max} m/s)` : ""} · <span className="swatch" style={{ background: "#e5484d" }} />too close / outside
                    {geo && ` · ${geo.n_points_sent.toLocaleString()} of ${geo.n_points.toLocaleString()} points`}</div>
                </div>
              </div>
            </div>
            {!max && <Splitter dir="v" onDrag={(dx) => setLay({ sideW: clampN(lay.sideW - dx, 280, (wrap.current?.clientWidth ?? 1600) - 320) })} onReset={() => setLay({ sideW: LAY0.sideW })} />}
            <div className="cw-side" style={max === "view" ? { display: "none" } : max === "kf" ? { flex: "1 1 0" } : { width: lay.sideW }}>
              <Tile title={course ? `Keyframes · ${name ?? saveName}` : "Keyframes"} icon="matrix" className="fill flush" meta={course ? `${course.kfs.length} keyframes · Nco ${course.Nco} · ${dirty ? "unsaved" : "saved"}` : undefined}
                actions={<MaxBtn on={max === "kf"} what="the keyframe table" onClick={() => toggleMax("kf")} />}>
                {course ? (
                  <>
                    <table className="kf">
                      <thead><tr><th style={{ width: 70 }}>name</th><th>t file</th>{pv?.mode === "expert" && !stale && <th>t solved</th>}<th>x</th><th>y</th><th>z</th><th>yaw</th></tr></thead>
                      <tbody>
                        {course.kfs.map((k, i) => {
                          const out = box ? !inside(pos0(k), box) : false;
                          const sk = solvedKf(k);
                          return (
                            <tr key={i} className={`${sel === i ? "sel" : ""} ${out ? "outside" : ""}`} onClick={() => { setSel(i); setGoalSel(false); }}
                              title={out ? "outside the captured volume" : undefined}>
                              <td><TextIn value={k.name} onCommit={(v) => setKf(i, { name: v })} /></td>
                              <td><NumIn value={k.t} onCommit={(v) => v !== null && setKf(i, { t: v })} /></td>
                              {pv?.mode === "expert" && !stale && <td className={`num ${sk && Math.abs(sk.t_solved - k.t) > 0.005 ? "warn" : "muted"}`}>{sk?.t_solved.toFixed(3) ?? "—"}</td>}
                              {[0, 1, 2, 3].map((r) => (
                                <td key={r}><NumIn value={k.fo[r][0] ?? null} nullable placeholder={r < 3 && sk
                                  ? `≈${displayPos(course, i, sk.pos)[r].toFixed(2)}` : "free"}
                                  onCommit={(v) => setCell(i, r, 0, v)} /></td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                    <p className="muted small" style={{ padding: "4px 6px" }}>Empty = free (the solver chooses; ≈ shows its value after a preview). The first and last keyframes fix x, y, z and yaw.
                      Select a row for Keyframe Tools; its derivative matrix is in Properties.{moved > 0 && pv?.mode === "expert" && !stale && <> The expert re-timed {moved} keyframe(s): <button className="lnk" onClick={applySolvedTimes}>write solved times</button>.</>}</p>
                  </>
                ) : <p className="empty pad">Load a course (Course ▸ File ▸ Course) or start a <button className="lnk" onClick={newLoop} disabled={!geo}>new loop</button>.</p>}
              </Tile>
              {course && (job || course.goal) && (
                <div className="tiles cols-2" style={{ flex: "none" }}>
                  <Tile title="Semantic goal" icon="goal" meta={course.goal?.query ? `“${course.goal.query}” · ${course.goal.backend ?? "?"}` : "thesis hook"}>
                    {course.goal ? (
                      <>
                        {course.goal.query && <GoalOrigin g={course.goal} scene={scene} compact />}
                        <label className="f">Label<input value={course.goal.label} placeholder="e.g. the red chair"
                          onChange={(e) => edit((c) => ({ ...c, goal: c.goal && { ...c.goal, label: e.target.value } }))} /></label>
                        <div className="row small" style={{ marginTop: 3 }}><span className="mono">{course.goal.position.map((v) => v.toFixed(2)).join(", ")}</span><span className="spacer" />
                          <button className="lnk" onClick={() => { setGoalSel(true); setSel(null); }}>select</button>
                          <button className="lnk" onClick={() => { edit((c) => ({ ...c, goal: null })); setGoalSel(false); }}>remove</button></div>
                      </>) : <p className="empty">None: Edit ▸ Goal places one, or query the splat (<a href={`#/splat/${scene}`}>splat editor</a>) and Send to course.</p>}
                  </Tile>
                  <Tile title="Last flight" icon="fly" meta={job ? <Pill s={job.status} label={`#${job.id}`} /> : undefined}>
                    {job ? (
                      <>
                        {!active(job.status) && res?.sim && res?.course?.name === name ? (
                          <table className="kv"><tbody>
                            <tr><td>Tracking</td><td className={res.sim.track_err_max_m > 0.5 ? "bad" : ""}>{res.sim.track_err_mean_m} / {res.sim.track_err_max_m} m</td></tr>
                            <tr><td>Render</td><td className={res.sim.dark_frames > 0 || res.sim.pixel_std < 5 ? "bad" : ""}>std {res.sim.pixel_std} · dark {res.sim.dark_frames}</td></tr>
                          </tbody></table>) : <p className="small">{job.label}</p>}
                        <div className="row small"><a href={`#/jobs/${job.id}`}>log</a>{!active(job.status) && <a href={flightUrl(scene)} target="_blank" rel="noreferrer">video</a>}<a href={`#/scene/${scene}`}>scene</a></div>
                      </>) : <p className="empty">Not flown in this session.</p>}
                  </Tile>
                </div>
              )}
            </div>
          </div>
          {course && pv && !max && <Splitter dir="h" onDrag={(dy) => setLay({ topPct: clampN(lay.topPct + (dy / (wrap.current?.clientHeight ?? 800)) * 100, 18, 88) })} onReset={() => setLay({ topPct: LAY0.topPct })} />}
          {course && (
            <div className="cw-bottom" style={max === "view" || max === "kf" ? { display: "none" } : !pv ? { flex: "none", minHeight: 0, marginTop: 6 } : undefined}>
              <div className="row" style={{ flexWrap: "nowrap", gap: 6, flex: "none" }}>
                <b style={{ color: "var(--navy)", whiteSpace: "nowrap" }}>Preview</b>
                <span className="muted small" style={{ whiteSpace: "nowrap" }}>{pvBusy === "expert" ? "re-timing like the expert (1–2 min)…" : pvBusy ? "solving…" : pv ? `${pv.mode === "expert" ? `re-timed like ${pv.pilot} (kT ${pv.kT})` : "minimum snap at the file's times"} · ${pv.hz} Hz · ${pv.solve_s} s${stale ? " · edited since" : ""}` : "none yet (F6)"}</span>
                {pv && <Kpis pv={pv} />}
                <span className="spacer" />
                {pv && <MaxBtn on={max === "charts"} what="the charts" onClick={() => toggleMax("charts")} />}
              </div>
              {pv && <Charts pv={pv} cursor={cursor} setCursor={setCursor} />}
            </div>
          )}
        </div>
      )}

      <ToProperties>
        {selKf && sel !== null && course ? (
          <>
            <div className="props-title"><Icon name="route" size={16} />Keyframe {selKf.name}<span className="spacer" /><span className="muted small">{sel + 1} of {course.kfs.length}</span></div>
            <PropSection title="Keyframe">
              <Prop k="Name" mono>{selKf.name}</Prop>
              <Prop k="t (file)" mono>{selKf.t.toFixed(3)} s</Prop>
              {solvedSel && <Prop k="t (solved)" mono tone={Math.abs(solvedSel.t_solved - selKf.t) > 0.005 ? "warn" : undefined}>{solvedSel.t_solved.toFixed(3)} s{pv?.mode === "expert" ? ` (${(solvedSel.t_solved - selKf.t >= 0 ? "+" : "")}${(solvedSel.t_solved - selKf.t).toFixed(3)})` : ""}</Prop>}
              <Prop k="Position" mono>{displayPos(course, sel, solvedSel?.pos as Vec3 | undefined).map((v, a) => `${selKf.fo[a][0] === null ? "≈" : ""}${v.toFixed(2)}`).join(", ")}</Prop>
              <Prop k="Yaw" mono>{selKf.fo[3][0] ?? "free"}</Prop>
              <Prop k="Captured volume" tone={box && !inside(pos0(selKf), box) ? "bad" : "ok"}>{box ? (inside(pos0(selKf), box) ? "inside" : "outside") : "—"}</Prop>
            </PropSection>
            {showMatrix && (
              <PropSection title="Derivative constraints (empty = free)">
                <div className="matrix">
                  <span />{ORDERS.map((o) => <span key={o} className="muted">{o}</span>)}
                  {AXES.map((ax, r) => <FragmentRow key={ax} label={ax} row={selKf.fo[r]} onCommit={(c, v) => setCell(sel, r, c, v)} />)}
                </div>
                <p className="muted small" style={{ padding: "0 8px" }}>m, m/s, m/s², … and rad for yaw. Endpoints normally pin velocity to 0. Keep consecutive yaw within π.</p>
              </PropSection>
            )}
            <CursorProps pv={pv && !stale ? pv : null} ci={cursorAt} />
          </>
        ) : goalSel && course?.goal ? (
          <>
            <div className="props-title"><Icon name="goal" size={16} />Semantic goal</div>
            <PropSection title="Goal">
              <div style={{ padding: "2px 8px" }}><label className="f">Label<input value={course.goal.label} placeholder="e.g. the red chair"
                onChange={(e) => edit((c) => ({ ...c, goal: c.goal && { ...c.goal, label: e.target.value } }))} /></label>
                <div className="fields" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginTop: 4 }}>{[0, 1, 2].map((a) => (
                  <label key={a} className="f">{"xyz"[a]}<NumIn value={course.goal!.position[a]} onCommit={(v) => v !== null &&
                    edit((c) => ({ ...c, goal: c.goal && withGoalAt(c.goal, c.goal.position.map((x, j) => (j === a ? v : x)) as Vec3) }))} /></label>))}</div></div>
              {course.goal.query && <GoalOrigin g={course.goal} scene={scene} />}
              <p className="muted small" style={{ padding: "4px 8px" }}>Saved as semantic_goal in the course file. FiGS and SousVide read only waypoints and forces, so it does not change the flight:
                the keyframes do (Send to course puts the approach point last). Dragging the goal keeps its query but drops the score, which belonged to the resolved position.</p>
            </PropSection>
          </>
        ) : (
          <>
            <div className="props-title"><Icon name="route" size={16} />{name ?? (course ? saveName : "Course")}</div>
            {course && <PropSection title="Course file">
              <Prop k="File" mono>configs/courses/{name ?? saveName}.json</Prop><Prop k="State" tone={dirty ? "warn" : "ok"}>{dirty ? "unsaved changes" : "saved"}</Prop>
              <Prop k="Keyframes">{course.kfs.length}</Prop>
              <Prop k="Nco"><input type="number" min={1} max={12} value={course.Nco} style={{ width: 60 }} onChange={(e) => edit((c) => ({ ...c, Nco: Math.max(1, Math.min(12, Number(e.target.value) || 6)) }))} /></Prop>
              <Prop k="Forces">{course.forces ? "custom (edit in Configs)" : "none"}</Prop>
              <Prop k="Integer cells" tone={intCells.length ? "warn" : "ok"}>{intCells.length || "none"}</Prop>
              <Prop k="Semantic goal">{course.goal ? course.goal.label || "(no label)" : "none"}</Prop>
            </PropSection>}
            {geo && <PropSection title="Scene">
              <Prop k="Scene">{scene}</Prop><Prop k="Sparse points">{geo.n_points.toLocaleString()}</Prop>
              <Prop k="Camera box" mono>{geo.camera_box.lo.map((v) => v.toFixed(1)).join(", ")} … {geo.camera_box.hi.map((v) => v.toFixed(1)).join(", ")}</Prop>
              <Prop k="Waypoint box" mono>inset {geo.waypoint_box.margin} m</Prop>
            </PropSection>}
            {splat.meta && <PropSection title="Gaussian splat">
              <Prop k="Shown" tone={splat.on ? "ok" : "muted"}>{splat.on ? (splat.state === "ready" ? "yes" : "loading…") : "no (Course ▸ Show ▸ Splat)"}</Prop>
              <Prop k="Run" mono>{splat.meta.run}</Prop><Prop k="Training step">{splat.meta.step ?? "—"}</Prop>
              <Prop k="Gaussians">{(splat.meta.n_written ?? 0).toLocaleString()} of {(splat.meta.n_total ?? 0).toLocaleString()}</Prop>
              <Prop k="Kept">opacity ≥ {splat.meta.min_opacity}, most visible first</Prop>
              <Prop k="Download">{mb(splat.meta.bytes)}{splat.meta.cached ? " (cached)" : splat.meta.seconds !== undefined ? `, exported in ${splat.meta.seconds} s` : ""}</Prop>
            </PropSection>}
            {drone && <PropSection title="Drone">
              <Prop k="Model">{drone.meta.name}</Prop><Prop k="Size">{cm(drone.meta.size[0])} × {cm(drone.meta.size[1])} × {cm(drone.meta.size[2])} cm</Prop>
              <Prop k="Sphere">{drone.meta.radius} m{drone.meta.guards ? " (guards included)" : ""}</Prop>
            </PropSection>}
            <CursorProps pv={pv && !stale ? pv : null} ci={cursorAt} />
            <PropSection title="Fly">
              <Prop k="Expert">{pilot}</Prop><Prop k="Frame">{frame}</Prop><Prop k="Method">{method}</Prop><Prop k="Allow outside">{allowOutside ? "yes" : "no"}</Prop>
            </PropSection>
          </>
        )}
      </ToProperties>
      <ToProblems items={paneProbs} />
    </DocActive>
  );
}

/** Where a semantic goal came from (splat editor ▸ Send to course): read-only, with a way back. */
function GoalOrigin({ g, scene, compact }: { g: SemanticGoal; scene?: string; compact?: boolean }) {
  const back = scene && g.query ? `#/splat/${encodeURIComponent(scene)}/${encodeURIComponent(g.query)}` : null;
  if (compact) return (
    <div className="small" style={{ marginBottom: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
      title={`query “${g.query}” · ${g.backend ?? "?"} · score ${g.score ?? "moved by hand"}${g.approach ? ` · approach ${g.approach.join(", ")}` : ""}`}>
      “{g.query}” · <span className="mono">{g.backend ?? "?"} · {g.score !== undefined ? g.score.toFixed(1) : "moved"}</span>{back && <> · <a href={back}>open in splat editor</a></>}
    </div>
  );
  return (
    <table className="kv small" style={{ margin: compact ? "0 0 3px" : "4px 0" }}><tbody>
      <tr><td>Query</td><td>“{g.query}”{back && <> · <a href={back}>open in splat editor</a></>}</td></tr>
      <tr><td>Backend · score</td><td className="mono">{g.backend ?? "—"} · {g.score !== undefined ? g.score.toFixed(1) : <span className="muted" title="moved by hand since it was resolved">moved</span>}</td></tr>
      {g.approach && <tr><td>Approach</td><td className="mono">{g.approach.map((v) => v.toFixed(2)).join(", ")}</td></tr>}
    </tbody></table>
  );
}

/** Tracks whether this document is the active one (for its local keyboard shortcuts). */
function DocActive({ refObj, children }: { refObj: React.MutableRefObject<boolean>; children: React.ReactNode }) {
  const { active: on } = useDoc();
  refObj.current = on;
  return <>{children}</>;
}

const nearest = (t: number[], x: number) => { let lo = 0, hi = t.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] < x) lo = m; else hi = m; } return x - t[lo] < t[hi] - x ? lo : hi; };

function CursorProps({ pv, ci }: { pv: Preview | null; ci: number | null }) {
  if (!pv || ci === null) return null;
  const u = pv.inputs;
  const thrust = u.lower[0] < 0 ? u.u[0][ci] / u.lower[0] : u.u[0][ci];
  const q = pv.quat?.[ci];
  let att = "";
  if (q) { const [x, y, z, w] = q; const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
    att = `roll ${Math.round(roll * 57.3)}° · pitch ${Math.round(pitch * 57.3)}°`; }
  return (
    <PropSection title={`At the cursor (t = ${pv.t[ci].toFixed(2)} s)`}>
      <Prop k="Position" mono>{pv.pos[ci].map((v) => v.toFixed(2)).join(", ")}</Prop>
      <Prop k="Speed" mono>{pv.speed[ci].toFixed(2)} m/s</Prop><Prop k="Acceleration" mono>{pv.acc_norm[ci].toFixed(2)} m/s²</Prop>
      <Prop k="Thrust" mono tone={thrust > 1 ? "bad" : undefined}>{Math.round(thrust * 100)} % of limit</Prop>
      {att && <Prop k="Attitude" mono>{att}</Prop>}
      {pv.clearance && <Prop k="Gap" mono tone={pv.clearance.d[ci] < pv.clearance.threshold ? "bad" : undefined}>{pv.clearance.d[ci].toFixed(2)} m</Prop>}
    </PropSection>
  );
}

function Kpis({ pv }: { pv: Preview }) {
  const u = pv.inputs;
  const viol = Object.keys(u.violations).length > 0;
  return (
    <div className="kpis" style={{ flexWrap: "nowrap", overflowX: "auto" }}>
      <Kpi v={`${pv.duration_solved} s`} l={pv.mode === "expert" ? (pv.timing?.retime_stalled ? "re-time stalled: times unchanged" : `duration (file ${pv.duration_file})`) : "duration"}
        bad={!!pv.timing?.retime_stalled} />
      <Kpi v={`${pv.stats.length_m} m`} l="path length" />
      <Kpi v={`${pv.stats.v_max} m/s`} l={`max speed (mean ${pv.stats.v_mean})`} />
      <Kpi v={`${pv.stats.a_max} m/s²`} l="max accel" />
      <Kpi v={pct(u.max_use[0])} l="thrust vs limit" bad={(u.max_use[0] ?? 0) > 1} />
      <Kpi v={pct(Math.max(...u.max_use.slice(1).map((x) => x ?? 0)))} l="body rate vs limit" bad={viol} />
      {pv.clearance && <Kpi v={`${pv.clearance.min} m`} l={`${(pv.clearance.body_radius ?? 0) > 0 ? "min gap" : "min clearance"} at ${pv.clearance.at_t} s`} bad={pv.clearance.min < pv.clearance.threshold} />}
      {pv.inside && <Kpi v={pct(pv.inside.outside_frac)} l="outside capture" bad={pv.inside.outside_frac > 0} />}
    </div>
  );
}

function Charts({ pv, cursor, setCursor }: { pv: Preview; cursor: number | null; setCursor: (t: number | null) => void }) {
  const u = pv.inputs;
  const thrust = u.u[0].map((v) => (u.lower[0] < 0 ? v / u.lower[0] : v));
  const rateLim = Math.max(...[1, 2, 3].map((i) => Math.max(Math.abs(u.lower[i]), Math.abs(u.upper[i]))));
  const rate = u.u[1].map((_, k) => Math.max(Math.abs(u.u[1][k]), Math.abs(u.u[2][k]), Math.abs(u.u[3][k])));
  const alt = pv.pos.map((q) => -q[2]);
  const charts: Omit<ChartProps, "cursor" | "onCursor">[] = [
    { t: pv.t, y: pv.speed, label: "Speed", unit: "m/s" },
    { t: pv.t, y: pv.acc_norm, label: "Acceleration", unit: "m/s²" },
    { t: pv.t, y: thrust, label: "Thrust (fraction of limit)", unit: "", refs: [{ y: 1, label: "limit" }], bad: u.violations.thrust ?? [] },
    { t: pv.t, y: rate, label: "Largest body rate |ω|", unit: "rad/s", refs: [{ y: rateLim, label: "limit" }], bad: [...(u.violations.wx ?? []), ...(u.violations.wy ?? []), ...(u.violations.wz ?? [])] },
    ...(pv.clearance ? [{ t: pv.t, y: pv.clearance.d, label: (pv.clearance.body_radius ?? 0) > 0 ? `Gap: drone sphere to ${pv.clearance.k > 1 ? `${pv.clearance.k}th` : "nearest"} point` : "Clearance", unit: "m",
      refs: [{ y: pv.clearance.threshold, label: `${pv.clearance.threshold} m` }], bad: pv.clearance.below }] : []),
    { t: pv.t, y: alt, label: "Altitude (−z)", unit: "m", zeroBased: false, bad: pv.inside?.outside_intervals ?? [] },
  ];
  // columns chosen from the tile's size; rows share its height (≥ 120 px each)
  const grid = useRef<HTMLDivElement>(null);
  const [cols, setCols] = useState(3);
  useLayoutEffect(() => {
    const el = grid.current; if (!el) return;
    const ro = new ResizeObserver(() => {     // the column count whose cells come closest to 2:1 at the largest size
      const n = charts.length, W = el.clientWidth, H = el.clientHeight;
      let best = 1, score = -1;
      for (let c = 1; c <= n; c++) {
        const r = Math.ceil(n / c), cw = (W - 6 * (c - 1)) / c, ch = (H - 6 * (r - 1)) / r;
        // a chart reads best near 2.2:1; cells under 110 px tall or 200 px wide are penalised
        const sc = Math.min(cw, 2.2 * ch) * Math.min(1, ch / 110) * (cw < 200 ? 0.6 : 1);
        if (sc >= score) { score = sc; best = c; }
      }
      setCols(best);
    });
    ro.observe(el); return () => ro.disconnect();
  }, [charts.length]);
  const rows = Math.ceil(charts.length / cols);
  return (
    <div className="chartgrid" ref={grid} style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${rows}, minmax(90px, 1fr))` }}>
      {charts.map((c) => <ChartCell key={c.label} {...c} cursor={cursor} onCursor={setCursor} />)}
    </div>
  );
}
type ChartProps = React.ComponentProps<typeof TimeChart>;
/** A chart that takes the height its grid cell has (the SVG's aspect follows the cell). */
function ChartCell(props: ChartProps) {
  const body = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  useLayoutEffect(() => {
    const el = body.current; if (!el) return;
    const ro = new ResizeObserver(() => {
      const w = Math.floor(el.clientWidth - 8), h = Math.floor(el.clientHeight - 24);   // padding; title line under the plot
      if (w > 40 && h > 20) setSize((s) => (s && s.w === w && s.h === Math.max(50, h) ? s : { w, h: Math.max(50, h) }));
    });
    ro.observe(el); return () => ro.disconnect();
  }, []);
  // drawn at the cell's pixel size, so text stays the same size however big the tile is
  return <div className="tile chartcell"><div className="tile-body" ref={body}>{size && <TimeChart {...props} width={size.w} height={size.h} />}</div></div>;
}
function MaxBtn({ on, what, onClick }: { on: boolean; what: string; onClick: () => void }) {
  return <button className="ph-btn" title={on ? "Restore the tiles (Esc)" : `Maximize ${what}`} onClick={onClick}><Icon name={on ? "restore" : "maximize"} size={14} /></button>;
}
interface CourseLayout { topPct: number; sideW: number }
const LAY_KEY = "galley.courseLayout";
const LAY0: CourseLayout = { topPct: 58, sideW: 460 };
const clampN = (v: number, a: number, b: number) => Math.max(a, Math.min(Math.max(a, b), v));

const cm = (m: number) => Math.round(m * 100);
const mb = (b?: number) => (b ? `${(b / 2 ** 20).toFixed(1)} MB` : "—");
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
