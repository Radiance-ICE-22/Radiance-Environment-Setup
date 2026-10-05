// Splat editor (#/splat/<scene>): ask the semantic features where something is, in words.
//
// Type a phrase → the CPU worker scores every Gaussian (LERF relevancy) and clusters the matches
// into candidates → the splat lights up by relevancy, the candidates are listed and boxed, the
// selected one shows its goal, approach point and the drone's sphere → Send to course writes a
// semantic_goal (and the approach point as the final keyframe) into a course and opens it for
// Save and fly. Clicking the splat picks the Gaussian under the cursor and asks which of a list
// of labels its features match; Annotate places ground-truth positions for the evaluation.
// The Semantics ribbon tab (Build, Query, View, Goal, Annotate) drives it.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  active, Annotation, Annotations, api, ApiError, Candidate, courseApi, Geometry, QueryReply, SEM_STEPS, SemBackend,
  semApi, SemanticRun, SemStatus, SemStep, SemTable,
} from "../api";
import { usePoll, useShowJob } from "../components";
import { Problem, ToProblems, ToProperties, useCommands, useDoc } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { Dialog, Pill, Prop, PropSection, Splitter, Tile } from "../shell/Panes";
import { useDrone } from "../course/Drone";
import { appendApproach, CourseFile, courseToGoal, fromFile, problems, SemanticGoal, toFile, Vec3 } from "../course/model";
import { buildColors, ColorMode, countLit, heatCss } from "../splat/recolor";
import { mb, useSplat } from "../splat/load";
import type { Pick } from "../splat/pick";
import SplatScene from "../splat/SplatScene";

const DEFAULT_LABELS = ["floor", "wall", "ceiling", "table", "chair", "shelf", "box", "door", "window", "cabinet", "bin", "light"];
const BODY_R = 0.19;            // the drone's bounding sphere (course_tools / Phase 1 gap), when no model is loaded
const r2 = (v: number) => Math.round(v * 100) / 100;
const fmt = (v: number[] | null | undefined, d = 2) => (v ? v.map((x) => x.toFixed(d)).join(", ") : "—");
const size = (c: Candidate) => [0, 1, 2].map((a) => (c.box.hi[a] - c.box.lo[a]).toFixed(2)).join(" × ");
const errMsg = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
/** Same rule as radiance_semantics.query.score_against (the gates): top box + 0.3 m holds it, or centroid within 0.75 m. */
export function isHit(c: Candidate | undefined, p: Vec3): { hit: boolean; err: number | null } {
  if (!c) return { hit: false, err: null };
  const err = Math.hypot(c.centroid[0] - p[0], c.centroid[1] - p[1], c.centroid[2] - p[2]);
  const inside = [0, 1, 2].every((a) => p[a] >= c.box.lo[a] - 0.3 && p[a] <= c.box.hi[a] + 0.3);
  return { hit: inside || err <= 0.75, err };
}
type BState = "none" | "running" | "ready" | "stale";
interface EvalRow { text: string; hit: boolean; err: number | null; ms: number; rank: number | null }

export default function SplatEditor({ scene, q }: { scene: string; q?: string }) {
  const d = useAppData();
  const { active: docActive } = useDoc();
  const show = useShowJob();
  const drone = useDrone();
  const bodyR = drone?.meta.radius ?? BODY_R;

  // ── splat, scene geometry, feature status ──────────────────────────────────
  const sp = useSplat(scene, true);
  const [drawn, setDrawn] = useState(false);
  const [gpuErr, setGpuErr] = useState<string | null>(null);
  useEffect(() => { setDrawn(false); setGpuErr(null); }, [sp.data]);
  const [geo, setGeo] = useState<Geometry | null>(null);
  const [geoErr, setGeoErr] = useState<string | null>(null);
  useEffect(() => {
    setGeo(null); setGeoErr(null); let alive = true;
    courseApi.geometry(scene).then((g) => alive && setGeo(g)).catch((e) => alive && setGeoErr(errMsg(e)));
    return () => { alive = false; };
  }, [scene]);
  const semJob = d.jobs.find((j) => j.kind === "semantics" && j.scene === scene && active(j.status)) ?? null;
  const status = usePoll(() => semApi.status(scene), semJob ? 4000 : 20000, [scene, semJob?.id, semJob?.status]);
  const st = status.data;
  const table = (b: SemBackend): SemTable | null => st?.tables.find((t) => t.active_run && t.backend === b) ?? null;
  const bstate = (b: SemBackend): BState => (b === "lift" && semJob ? "running" : table(b) ? (table(b)!.stale ? "stale" : "ready") : "none");

  // ── query ──────────────────────────────────────────────────────────────────
  const [text, setText] = useState(q ?? "");
  const [backend, setBackend] = useState<SemBackend>("lift");
  const [qs, setQs] = useState<{ standoff: number; top: number; threshold?: number; rel_alpha?: number; negatives: string }>({ standoff: 1.0, top: 5, negatives: "" });
  const [reply, setReply] = useState<QueryReply | null>(null);
  const [rel, setRel] = useState<Uint8Array | null>(null);
  const [busy, setBusy] = useState(false);
  const [qErr, setQErr] = useState<string | null>(null);
  const [took, setTook] = useState<number | null>(null);
  const [mode, setMode] = useState<ColorMode>("rgb");
  const [floor, setFloor] = useState(0.6);
  const [only, setOnly] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [selC, setSelC] = useState<number | null>(null);
  const [picked, setPicked] = useState<Pick | null>(null);
  const seq = useRef(0);
  const cands = reply?.result.candidates ?? [];
  const cand = selC !== null ? cands[selC] ?? null : null;

  const runQuery = useCallback(async (t0?: string) => {
    const t = (t0 ?? text).trim();
    if (!t) { setQErr("Type what to look for, e.g. “red tool chest”."); return; }
    const my = ++seq.current;
    setBusy(true); setQErr(null);
    const start = performance.now();
    try {
      const negatives = qs.negatives.split(",").map((s) => s.trim()).filter(Boolean);
      const r = await semApi.query(scene, { text: t, backend, standoff: qs.standoff, top: qs.top, threshold: qs.threshold, rel_alpha: qs.rel_alpha,
        negatives: negatives.length ? negatives : undefined, relevancy: true });
      if (my !== seq.current) return;
      const bytes = r.relevancy_id ? await semApi.relevancy(scene, r.relevancy_id) : null;
      if (my !== seq.current) return;
      setReply(r); setRel(bytes); setTook(performance.now() - start);
      setSelC(r.result.candidates.length ? 0 : null); setHover(null); setPicked(null);
      setFloor(r2(Math.min(0.95, Math.max(0.3, r.result.tau))));
      if (bytes) setMode("relevancy");
    } catch (e) { if (my === seq.current) setQErr(errMsg(e)); }
    finally { if (my === seq.current) setBusy(false); }
  }, [text, scene, backend, qs]);
  // #/splat/<scene>/<query> (Open in splat editor from a course): run it once
  const lastQ = useRef<string | null>(null);
  useEffect(() => {
    if (!q || lastQ.current === q || !st?.ready.length) return;
    lastQ.current = q; setText(q); runQuery(q);
  }, [q, st?.ready.length, runQuery]);

  // ── colours ────────────────────────────────────────────────────────────────
  const [pca, setPca] = useState<{ data: Uint8Array; stale: boolean } | null>(null);
  const [pcaErr, setPcaErr] = useState<string | null>(null);
  useEffect(() => { setPca(null); setPcaErr(null); }, [scene, backend, table(backend)?.key]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (mode !== "pca" || pca || pcaErr) return;
    semApi.pca(scene, backend).then(setPca).catch((e) => setPcaErr(errMsg(e)));
  }, [mode, pca, pcaErr, scene, backend]);
  const n = sp.data?.n ?? 0;
  const relOk = !!rel && !!reply && !reply.stale && rel.length === n;
  const pcaOk = !!pca && !pca.stale && pca.data.length === 3 * n;
  const colorMismatch =
    mode === "relevancy" && rel && reply && !reply.stale && rel.length !== n ? `The relevancy has ${rel.length.toLocaleString()} values but the splat has ${n.toLocaleString()} Gaussians: rebuild the features or the splat.` :
    mode === "pca" && pca && !pca.stale && pca.data.length !== 3 * n ? `The PCA table has ${(pca.data.length / 3).toLocaleString()} rows but the splat has ${n.toLocaleString()} Gaussians.` : null;
  const shownMode: ColorMode = mode === "relevancy" && relOk ? "relevancy" : mode === "pca" && pcaOk ? "pca" : "rgb";
  const onlyBox = only && cand ? { lo: cand.box.lo.map((v) => v - 0.15), hi: cand.box.hi.map((v) => v + 0.15) } : null;
  const [buildMs, setBuildMs] = useState(0);
  const colors = useMemo(() => {
    if (!sp.data || (shownMode === "rgb" && !onlyBox)) return null;
    const t0 = performance.now();
    const c = buildColors(sp.data.rgba, sp.data.n, { mode: shownMode, rel, pca: pca?.data, floor, only: onlyBox, pos: sp.data.pos });
    const ms = performance.now() - t0;
    setTimeout(() => setBuildMs(ms), 0);
    return c;
  }, [sp.data, shownMode, rel, pca, floor, onlyBox?.lo.join(), onlyBox?.hi.join()]);   // eslint-disable-line react-hooks/exhaustive-deps
  const [gpuMs, setGpuMs] = useState<{ cpu: number; frame: number } | null>(null);
  const lit = relOk ? countLit(rel!, floor) : null;

  // ── picking and labels ─────────────────────────────────────────────────────
  const [labelText, setLabelText] = useState(DEFAULT_LABELS.join(", "));
  const labelList = useMemo(() => {
    const l = labelText.split(",").map((s) => s.trim()).filter(Boolean);
    const t = reply?.result.text;
    return Array.from(new Set(t ? [t, ...l] : l)).slice(0, 64);
  }, [labelText, reply?.result.text]);
  const [labels, setLabels] = useState<{ index: number; scores: [string, number][]; seen: boolean; stale: boolean } | null>(null);
  const [labelsErr, setLabelsErr] = useState<string | null>(null);
  useEffect(() => {
    setLabels(null); setLabelsErr(null);
    if (!picked || annMode || !st?.ready.includes(backend)) return;
    let alive = true;
    const t = setTimeout(() => semApi.labels(scene, picked.index, labelList, backend)
      .then((r) => alive && setLabels({ index: picked.index, scores: r.scores, seen: r.seen, stale: r.stale }))
      .catch((e) => alive && setLabelsErr(errMsg(e))), 150);
    return () => { alive = false; clearTimeout(t); };
  }, [picked, labelList.join("|"), backend, scene, st?.ready.join()]);   // eslint-disable-line react-hooks/exhaustive-deps

  // ── annotations (queries.json) ─────────────────────────────────────────────
  const annSrv = usePoll(() => semApi.queries(scene), 0, [scene]);
  const [ann, setAnn] = useState<Annotations | null>(null);
  const [annDirty, setAnnDirty] = useState(false);
  useEffect(() => { if (annSrv.data && !annDirty) setAnn(annSrv.data); }, [annSrv.data]);   // eslint-disable-line react-hooks/exhaustive-deps
  const [annMode, setAnnMode] = useState(false);
  const [annLabel, setAnnLabel] = useState("");
  const [pinSel, setPinSel] = useState<number | null>(null);
  const [showPins, setShowPins] = useState(true);
  const [annMsg, setAnnMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const pins = ann?.queries ?? [];
  const setPins = (fn: (a: Annotation[]) => Annotation[]) => { setAnn((a) => ({ ...(a ?? { queries: [] }), queries: fn(a?.queries ?? []) })); setAnnDirty(true); };
  const placePin = (p: Pick) => {
    const label = (annLabel || text).trim();
    if (!label) { setAnnMsg({ ok: false, text: "Type the object's name in Annotate ▸ Label first." }); return; }
    const pos = p.point.map((v) => Math.round(v * 1000) / 1000) as Vec3;
    setPins((a) => {
      const i = a.findIndex((x) => x.text === label);
      if (i >= 0) { const c = [...a]; c[i] = { ...c[i], position: pos }; setPinSel(i); return c; }
      setPinSel(a.length); return [...a, { text: label, position: pos, set: "extra" }];
    });
    setAnnMsg({ ok: true, text: `“${label}” placed at ${fmt(pos)} (unsaved).` });
  };
  const saveAnn = async () => {
    if (!ann) return;
    try { const r = await semApi.saveQueries(scene, ann); setAnn(r); setAnnDirty(false); annSrv.reload(); status.reload(); setAnnMsg({ ok: true, text: `Saved ${r.queries.length} annotations.` }); }
    catch (e) { setAnnMsg({ ok: false, text: errMsg(e) }); }
  };
  const [evalRows, setEvalRows] = useState<EvalRow[] | null>(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const evaluate = async () => {
    const todo = pins.filter((a) => a.position);
    setEvalBusy(true); setEvalRows([]);
    const rows: EvalRow[] = [];
    for (const a of todo) {
      const t0 = performance.now();
      try {
        const r = await semApi.query(scene, { text: a.text, backend, standoff: qs.standoff, top: qs.top, relevancy: false });
        const c = r.result.candidates;
        const h = isHit(c[0], a.position as Vec3);
        const rank = c.findIndex((x) => isHit(x, a.position as Vec3).hit);
        rows.push({ text: a.text, hit: h.hit, err: h.err, ms: performance.now() - t0, rank: rank >= 0 ? rank + 1 : null });
      } catch { rows.push({ text: a.text, hit: false, err: null, ms: performance.now() - t0, rank: null }); }
      setEvalRows([...rows]);
    }
    setEvalBusy(false);
  };

  const onPick = (p: Pick | null) => {
    if (annMode) { if (p) placePin(p); return; }
    setPicked(p); setPinSel(null);
  };

  // ── dialogs ────────────────────────────────────────────────────────────────
  const [dlg, setDlg] = useState<null | "send" | "build">(null);
  const [approachKf, setApproachKf] = useState(true);

  // ── keyboard: Esc clears the pick / selection (document-local) ─────────────
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (!docActive || dlg) return;
      const el = e.target as HTMLElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return;
      if (e.key === "Escape") { if (annMode) setAnnMode(false); else if (picked) setPicked(null); else setSelC(null); }
    };
    addEventListener("keydown", h); return () => removeEventListener("keydown", h);
  });

  // ── layout ─────────────────────────────────────────────────────────────────
  const [sideW, setSideW] = useState(() => { try { return Number(localStorage.getItem("galley.splatSideW")) || 430; } catch { return 430; } });
  const setSide = (w: number) => { setSideW(w); try { localStorage.setItem("galley.splatSideW", String(w)); } catch { /* */ } };
  const wrap = useRef<HTMLDivElement>(null);
  const [navSpeed, setNavSpeed] = useState(() => { try { return Number(localStorage.getItem("galley.navSpeed")) || 1; } catch { return 1; } });
  const [camPath, setCamPath] = useState(false);
  const focusAt: Vec3 | null = picked?.point ?? (cand ? cand.centroid : pinSel !== null && pins[pinSel]?.position ? pins[pinSel].position as Vec3 : null);

  // ── ribbon ─────────────────────────────────────────────────────────────────
  const ready = st?.ready.includes(backend);
  const noTable = !st ? "Loading the feature status…" : ready ? false : bstate(backend) === "stale"
    ? `The ${backend} table was built for another checkpoint: Build ▸ Build features again.` : bstate(backend) === "running" ? "The features are being built (Output ▸ Log)." : `No ${backend} features yet: Build ▸ Build features.`;
  const noQuery = !reply ? "Run a query first (Query ▸ Query)." : false;
  const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
  useCommands({
    "ctx.splat": { checked: true },
    "sem.build": { run: () => setDlg("build"), disabled: !st ? "Loading…" : !st.script ? "figs/semantic_pipeline.py is not on the host." : !!semJob && `Job #${semJob.id} is ${semJob.status}.` },
    "sem.continue": { run: () => semApi.submit({ scene }).then(({ id }) => { show(id); status.reload(); }).catch((e) => setQErr(errMsg(e))),
      disabled: !st ? "Loading…" : !!semJob && `Job #${semJob.id} is ${semJob.status}.` },
    "sem.worker": { run: () => semApi.stopWorker().then(() => setQErr(null)).catch((e) => setQErr(errMsg(e))) },
    "sem.query": { run: () => runQuery(), disabled: noTable || (busy && "Querying…"), label: busy ? "Querying…" : undefined },
    "run": { run: () => runQuery(), disabled: noTable || (busy && "Querying…") },
    "sem.backend": { value: backend, options: [["lift", "lift"], ["fmgs", "fmgs (Phase 4)"]], set: (v) => setBackend(v === "fmgs" ? "fmgs" : "lift") },
    "sem.top": { value: qs.top, set: (v) => setQs({ ...qs, top: Math.min(20, Math.max(1, Math.round(Number(v) || 5))) }) },
    "sem.standoff": { value: qs.standoff, set: (v) => setQs({ ...qs, standoff: Math.min(5, Math.max(0, Number(v) || 0)) }) },
    "sem.threshold": { value: qs.threshold ?? "", set: (v) => setQs({ ...qs, threshold: num(v) === undefined ? undefined : Math.min(1, Math.max(0, num(v)!)) }) },
    "sem.relalpha": { value: qs.rel_alpha ?? "", set: (v) => setQs({ ...qs, rel_alpha: num(v) === undefined ? undefined : Math.min(1, Math.max(0, num(v)!)) }) },
    "sem.negatives": { value: qs.negatives, set: (v) => setQs({ ...qs, negatives: v }) },
    "sem.v.rgb": { checked: mode === "rgb", run: () => setMode("rgb") },
    "sem.v.rel": { checked: mode === "relevancy", run: () => setMode("relevancy"), disabled: !rel && "Run a query first: its relevancy colours the splat." },
    "sem.v.pca": { checked: mode === "pca", run: () => { setPcaErr(null); setMode("pca"); }, disabled: !table(backend) && `No ${backend} table.` },
    "sem.floor": { value: floor, set: (v) => setFloor(Math.min(0.99, Math.max(0, Number(v) || 0))) },
    "sem.only": { checked: only, set: (v) => setOnly(v === "true"), disabled: !cand && "Select a candidate." },
    "sem.pins": { checked: showPins, set: (v) => setShowPins(v === "true") },
    "view.cam": { checked: camPath, run: () => setCamPath(!camPath), disabled: !geo && "Loading the scene…" },
    "view.navspeed": { value: String(navSpeed), options: [["0.25", "0.25×"], ["0.5", "0.5×"], ["1", "1×"], ["2", "2×"], ["4", "4×"]],
      set: (v) => { setNavSpeed(Number(v)); try { localStorage.setItem("galley.navSpeed", v); } catch { /* */ } } },
    "sem.send": { run: () => setDlg("send"), disabled: noQuery || (!cand && "Select a candidate (Candidates tile).") },
    "sem.approachkf": { checked: approachKf, set: (v) => setApproachKf(v === "true") },
    "sem.annotate": { checked: annMode, run: () => { setAnnMode(!annMode); setPicked(null); setAnnMsg(null); }, disabled: !sp.data && "Wait for the splat to load." },
    "sem.annlabel": { value: annLabel, set: setAnnLabel },
    "sem.annsave": { run: saveAnn, disabled: !annDirty && "No unsaved annotations." },
    "sem.annquery": { run: evaluate, disabled: noTable || (!pins.some((a) => a.position) && "No placed annotations.") || (evalBusy && "Running…") },
    "sem.compare": { disabled: "Lift | FMGS side by side needs an FMGS table (Phase 4)." },
  });

  // ── problems ───────────────────────────────────────────────────────────────
  const probs: Problem[] = [
    ...(sp.err ? [{ severity: "error" as const, where: "splat", message: sp.err }] : []),
    ...(gpuErr ? [{ severity: "error" as const, where: "splat", message: `did not draw: ${gpuErr}` }] : []),
    ...(geoErr ? [{ severity: "warning" as const, where: `scene ${scene}`, message: geoErr }] : []),
    ...(status.err ? [{ severity: "error" as const, where: "features", message: status.err }] : []),
    ...(st && !ready ? [{ severity: "warning" as const, where: `${backend} features`, message: String(noTable) }] : []),
    ...(qErr ? [{ severity: "error" as const, where: "query", message: qErr }] : []),
    ...(reply?.stale ? [{ severity: "warning" as const, where: "query", message: "The table was built for another checkpoint, so its relevancy is not painted on this splat. Rebuild the features." }] : []),
    ...(reply?.result.ambiguous ? [{ severity: "warning" as const, where: `“${reply.result.text}”`, message: `Ambiguous: the runner-up scores ${Math.round((1 - (reply.result.margin ?? 0)) * 100)}% of the top candidate. Check both, or add a detail to the phrase.` }] : []),
    ...(cand && cand.gap_ok === false ? [{ severity: "warning" as const, where: `candidate #${cand.rank}`, message: `The approach point is ${cand.gap} m from the scenery (drone sphere ${bodyR} m): move it in the course editor or lower the standoff.` }] : []),
    ...(cand?.large ? [{ severity: "info" as const, where: `candidate #${cand.rank}`, message: "Large cluster (box diagonal over 4 m): probably a surface, not an object." }] : []),
    ...(colorMismatch ? [{ severity: "error" as const, where: "colours", message: colorMismatch }] : []),
    ...(pcaErr ? [{ severity: "error" as const, where: "PCA", message: pcaErr }] : []),
    ...(pca?.stale ? [{ severity: "warning" as const, where: "PCA", message: "The table was built for another checkpoint: not painted." }] : []),
    ...(labelsErr ? [{ severity: "warning" as const, where: "labels", message: labelsErr }] : []),
    ...(annMsg && !annMsg.ok ? [{ severity: "error" as const, where: "annotations", message: annMsg.text }] : []),
  ];

  // ── render ─────────────────────────────────────────────────────────────────
  const splatTag = sp.state === "ready" ? (drawn ? `${n.toLocaleString()} Gaussians` : "uploading to the GPU…")
    : sp.state === "downloading" ? `downloading the splat (${mb(sp.meta?.bytes)}, ${Math.round(sp.progress * 100)}%)…`
    : sp.state === "exporting" ? "exporting the splat from the checkpoint…" : sp.state === "parsing" ? "reading the splat…" : sp.state === "error" ? "splat failed (Output ▸ Problems)" : "";
  return (
    <div className="cw" ref={wrap}>
      <div className="cw-top" style={{ flex: "1 1 0" }}>
        <div className="cw-view">
          <div className="viewport" style={{ background: "#0d1117" }}>
            <SplatScene data={sp.state === "ready" ? sp.data : null} colors={colors}
              onSplatReady={() => setDrawn(true)} onSplatError={setGpuErr} onRecolor={setGpuMs}
              geo={geo} boxId={scene} candidates={cands} hover={hover} selected={selC}
              bodyRadius={bodyR} pins={pins} showPins={showPins || annMode} pinSel={pinSel}
              picked={annMode ? null : picked} onPick={onPick} minOpacity={0.05} cursor={annMode ? "place" : "pick"}
              keyNav={docActive && !dlg} navSpeed={navSpeed} showCamPath={camPath} focus={focusAt} />
            <form className="qbar" onSubmit={(e) => { e.preventDefault(); runQuery(); }}>
              <Icon name="search" size={16} />
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder={ready ? "Where is… (e.g. red tool chest)" : "Build the features first"} maxLength={200} aria-label="Query" />
              <select value={backend} onChange={(e) => setBackend(e.target.value as SemBackend)} title="Feature backend">
                <option value="lift">lift</option><option value="fmgs" disabled={!table("fmgs")}>fmgs</option></select>
              <button type="submit" className="push primary" disabled={!!noTable || busy}>{busy ? "…" : "Query"}</button>
            </form>
            <div className="vtag" style={{ top: 44 }}>{scene} · course frame (x, −y, −z): z down · {splatTag}
              {reply && ` · “${reply.result.text}” ${reply.worker_ms ?? "?"} ms worker${took !== null ? `, ${Math.round(took)} ms total` : ""}`}</div>
            {annMode && <div className="msgbar" style={{ position: "absolute", left: 8, right: 8, top: 70 }}><Icon name="pin" size={16} />
              <span><b>Annotate:</b> click the object to place “{(annLabel || text).trim() || "…"}” (Annotate ▸ Label). Esc leaves. {annMsg?.ok && annMsg.text}</span>
              {annDirty && <button className="push" onClick={saveAnn}>Save</button>}</div>}
            <div className="legend">
              {shownMode === "relevancy" && reply && <div className="row" style={{ gap: 5, alignItems: "center", flexWrap: "wrap" }}>
                <b>Relevancy</b><span className="mono">{floor.toFixed(2)}</span><span className="swatch" style={{ width: 90, height: 9, display: "inline-block", background: heatCss() }} /><span className="mono">1.0</span>
                <span className="muted">τ {reply.result.tau.toFixed(3)} · {lit?.toLocaleString()} lit · grey below</span></div>}
              {shownMode === "pca" && <div><b>PCA</b> of the {backend} features: similar colour, similar features</div>}
              {mode !== shownMode && <div className="warn">{mode === "relevancy" ? "Relevancy not shown (Problems)" : pcaErr ? "PCA not available (Problems)" : "Loading the PCA colours…"}</div>}
              <div className="muted"><span title={"Click a Gaussian: its labels in Properties.\nArrows fly · PgUp/PgDn or E/Q up/down · Ctrl+arrows look · Shift faster · +/− closer/farther · F centre on the selection · Home default view"}
                style={{ textDecoration: "underline dotted", cursor: "help" }}>click to inspect · keys</span>
                {colors && gpuMs && ` · recoloured in ${Math.round(buildMs + gpuMs.cpu)} ms + frame ${Math.round(gpuMs.frame)} ms`}</div>
            </div>
          </div>
        </div>
        <Splitter dir="v" onDrag={(dx) => setSide(Math.max(300, Math.min((wrap.current?.clientWidth ?? 1400) - 360, sideW - dx)))} onReset={() => setSide(430)} />
        <div className="cw-side" style={{ width: sideW, maxWidth: "46%", overflow: "auto" }}>
          <Tile title="Candidates" icon="search" className="flush" meta={reply ? `“${reply.result.text}” · ${reply.result.n_selected.toLocaleString()} selected · τ ${reply.result.tau.toFixed(3)}` : undefined}>
            {reply ? (cands.length ? (
              <table className="grid cands">
                <thead><tr><th>#</th><th className="num">score</th><th className="num">n</th><th>size (m)</th><th className="num">gap</th><th /></tr></thead>
                <tbody>{cands.map((c, i) => (
                  <tr key={i} className={`click ${selC === i ? "sel" : ""}`} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                    onClick={() => { setSelC(i); setPicked(null); }}>
                    <td>{c.rank}{i === 0 && reply.result.ambiguous && <span className="warn" title="ambiguous: close runner-up"> ?</span>}</td>
                    <td className="num">{c.score.toFixed(1)}</td><td className="num">{c.n.toLocaleString()}</td>
                    <td className="mono small">{size(c)}{c.large && <span className="warn"> large</span>}</td>
                    <td className={`num ${c.gap_ok === false ? "bad" : ""}`}>{c.gap === null ? "—" : c.gap.toFixed(2)}</td>
                    <td><button className="lnk" onClick={(e) => { e.stopPropagation(); setSelC(i); setDlg("send"); }}>send…</button></td>
                  </tr>))}</tbody>
              </table>) : <p className="empty pad">No Gaussian is above the threshold for “{reply.result.text}”. Try other words, or lower Query ▸ Threshold.</p>)
              : <p className="empty pad">{ready ? "Type a phrase above and press Enter." : String(noTable)}</p>}
            {/* margin = (top − runner-up) / top (radiance_semantics.query): 1 = no runner-up; < 0.25 = ambiguous */}
            {reply && cands.length > 0 && <p className="muted small" style={{ padding: "3px 8px" }}>{cands.length < 2 || reply.result.margin === null
              ? "Only one cluster above the threshold."
              : `Runner-up at ${Math.round((1 - reply.result.margin) * 100)}% of the top score${reply.result.ambiguous ? " — ambiguous" : ""}.`} Hover a row for its box; select it for the goal and approach point.</p>}
          </Tile>
          <FeaturesTile st={st} bstate={bstate} semJob={semJob} onBuild={() => setDlg("build")} />
          <Tile title="Annotations" icon="pin" className="flush" meta={`${pins.filter((a) => a.position).length} of ${pins.length} placed${annDirty ? " · unsaved" : ""}`}
            actions={<>{annDirty && <button onClick={saveAnn}>Save</button>}<button onClick={evaluate} disabled={!!noTable || evalBusy || !pins.some((a) => a.position)}>{evalBusy ? "Running…" : "Query all"}</button></>}>
            <table className="grid">
              <thead><tr><th>object</th><th>position (course)</th>{evalRows && <th>result</th>}<th /></tr></thead>
              <tbody>{pins.map((a, i) => {
                const ev = evalRows?.find((r) => r.text === a.text);
                return (
                  <tr key={i} className={`click ${pinSel === i ? "sel" : ""}`} onClick={() => { setPinSel(i); setPicked(null); }}>
                    <td>{a.text}{a.set && a.set !== "extra" && <span className="muted small"> · {a.set}</span>}</td>
                    <td className="mono small">{a.position ? fmt(a.position as number[]) : <span className="muted">not placed</span>}</td>
                    {evalRows && <td className={ev ? (ev.hit ? "ok" : "bad") : "muted"}>{ev ? `${ev.hit ? "hit" : "miss"}${ev.err !== null ? ` ${ev.err.toFixed(2)} m` : ""} · ${Math.round(ev.ms)} ms` : a.position ? "…" : ""}</td>}
                    <td style={{ whiteSpace: "nowrap" }}><button className="lnk" onClick={(e) => { e.stopPropagation(); setText(a.text); runQuery(a.text); }} disabled={!!noTable}>query</button>{" "}
                      <button className="lnk" onClick={(e) => { e.stopPropagation(); setAnnLabel(a.text); setAnnMode(true); }}>place</button>{" "}
                      <button className="lnk" onClick={(e) => { e.stopPropagation(); setPins((x) => x.filter((_, j) => j !== i)); setPinSel(null); }}>delete</button></td>
                  </tr>);
              })}{!pins.length && <tr><td colSpan={4} className="muted">None yet: Annotate, type the object's name, click it in the splat.</td></tr>}</tbody>
            </table>
            {evalRows && !evalBusy && evalRows.length > 0 && <p className="small" style={{ padding: "3px 8px" }}><b>{evalRows.filter((r) => r.hit).length} of {evalRows.length} hit</b> (top box + 0.3 m holds the annotation, or centroid within 0.75 m — the gates' rule) · slowest {Math.round(Math.max(...evalRows.map((r) => r.ms)))} ms</p>}
          </Tile>
        </div>
      </div>

      {dlg === "send" && cand && reply && <SendDialog scene={scene} cand={cand} text={reply.result.text} backend={backend} geo={geo}
        approachKf={approachKf} setApproachKf={setApproachKf} onClose={() => setDlg(null)} />}
      {dlg === "build" && <BuildDialog scene={scene} defaultsWidth={d.machine?.defaults.semantic_feat_width as number | undefined}
        onClose={() => setDlg(null)} onSubmitted={(id) => { setDlg(null); show(id); status.reload(); }} />}

      <ToProperties>
        {picked ? (
          <>
            <div className="props-title"><Icon name="splat" size={16} />Gaussian #{picked.index.toLocaleString()}<span className="spacer" /><button className="lnk" onClick={() => setPicked(null)}>clear</button></div>
            <PropSection title="Gaussian">
              <Prop k="Row" mono>{picked.index} (.splat record = table row)</Prop>
              <Prop k="Position" mono>{fmt(picked.point)}</Prop>
              {sp.data && <><Prop k="Opacity" mono>{sp.data.opacity[picked.index].toFixed(2)}</Prop><Prop k="Size" mono>{(sp.data.scale[picked.index] * 100).toFixed(1)} cm (largest axis)</Prop></>}
              <Prop k="Blend weight" mono>{picked.weight.toFixed(2)} at {picked.t.toFixed(2)} m</Prop>
              {relOk && <Prop k="Relevancy" mono tone={rel![picked.index] / 255 >= floor ? "ok" : "muted"}>{(rel![picked.index] / 255).toFixed(3)} for “{reply!.result.text}”</Prop>}
            </PropSection>
            <PropSection title={`Best labels (${backend})`}>
              {labels && labels.index === picked.index ? (
                <table className="grid"><tbody>{softmax(labels.scores).slice(0, 8).map(([l, cos, p], i) => (
                  <tr key={l}><td style={{ width: 18 }}>{i + 1}</td><td>{l}</td><td className="num mono">{cos.toFixed(3)}</td><td style={{ width: 90 }}>
                    <div className="bar"><span style={{ width: `${Math.round(p * 100)}%` }} /></div></td><td className="num mono">{Math.round(p * 100)}%</td></tr>))}</tbody></table>
              ) : <p className="muted small" style={{ padding: "2px 8px" }}>{labelsErr ?? (ready ? "Asking the worker…" : String(noTable))}</p>}
              {labels && !labels.seen && <p className="warn small" style={{ padding: "2px 8px" }}>No training view saw this Gaussian: its features are empty.</p>}
              <div style={{ padding: "2px 8px" }}><label className="f">Labels to compare (comma-separated; the query is added)
                <textarea rows={3} value={labelText} onChange={(e) => setLabelText(e.target.value)} /></label>
                <p className="muted small">Cosine similarity of the Gaussian's CLIP feature with each label's text embedding; % is a softmax at CLIP's scale (×100).</p></div>
            </PropSection>
          </>
        ) : cand && reply ? (
          <>
            <div className="props-title"><Icon name="goal" size={16} />Candidate #{cand.rank} · “{reply.result.text}”</div>
            <PropSection title="Candidate">
              <Prop k="Score" mono>{cand.score.toFixed(2)} (Σ (relevancy − threshold) · opacity)</Prop>
              <Prop k="Gaussians" mono>{cand.n.toLocaleString()}</Prop>
              <Prop k="Centroid" mono>{fmt(cand.centroid, 3)}</Prop>
              <Prop k="Box" mono>{fmt(cand.box.lo)} … {fmt(cand.box.hi)}</Prop>
              <Prop k="Size" mono>{size(cand)} m{cand.large ? " (large)" : ""}</Prop>
              <Prop k="Approach" mono>{fmt(cand.approach, 3)}</Prop>
              <Prop k="Gap" mono tone={cand.gap_ok === false ? "bad" : cand.gap_ok ? "ok" : undefined}>{cand.gap === null ? "—" : `${cand.gap.toFixed(2)} m`} (drone sphere {bodyR} m)</Prop>
              <Prop k="Seen by" mono>{cand.cameras} training cameras</Prop>
            </PropSection>
            <div style={{ padding: "4px 8px" }}><button className="push primary" onClick={() => setDlg("send")}>Send to course…</button></div>
          </>
        ) : (
          <>
            <div className="props-title"><Icon name="semantic" size={16} />{scene} · semantics</div>
            {reply && <PropSection title={`Query “${reply.result.text}”`}>
              <Prop k="Threshold τ" mono>{reply.result.tau.toFixed(3)} (fixed {reply.result.threshold}, relative {reply.result.rel_alpha})</Prop>
              <Prop k="Peak" mono>{reply.result.peak.toFixed(3)} (mean of the top 100)</Prop>
              <Prop k="Selected" mono>{reply.result.n_selected.toLocaleString()} Gaussians · voxel {reply.result.voxel} m</Prop>
              <Prop k="Percentiles" mono>{Object.entries(reply.result.rel_pct).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(" · ")}</Prop>
              <Prop k="Negatives" mono>{reply.result.negatives.join(", ")}</Prop>
              <Prop k="Time" mono>{Object.entries(reply.result.ms).map(([k, v]) => `${k} ${v}`).join(" · ")} ms</Prop>
            </PropSection>}
            {table(backend) && <PropSection title={`Table (${backend})`}>
              <Prop k="Run" mono>{table(backend)!.run}</Prop>
              <Prop k="State" tone={table(backend)!.stale ? "warn" : "ok"}>{table(backend)!.stale ? "stale (another checkpoint)" : "matches the active model"}</Prop>
              <Prop k="Rows" mono>{table(backend)!.rows?.toLocaleString()} ({table(backend)!.seen_rows?.toLocaleString() ?? "?"} seen)</Prop>
              <Prop k="Teachers" mono>{table(backend)!.teacher_tag ?? "—"}</Prop>
              <Prop k="Size" mono>{table(backend)!.mb} MB</Prop>
            </PropSection>}
            {sp.meta && <PropSection title="Splat">
              <Prop k="Gaussians" mono>{(sp.meta.n_written ?? n).toLocaleString()} of {(sp.meta.n_total ?? 0).toLocaleString()}</Prop>
              <Prop k="Download" mono>{mb(sp.meta.bytes)}</Prop>
              <Prop k="Run" mono>{sp.meta.run}</Prop>
            </PropSection>}
          </>
        )}
      </ToProperties>
      <ToProblems items={probs} />
    </div>
  );
}

function softmax(s: [string, number][]): [string, number, number][] {
  const m = Math.max(...s.map((x) => x[1]));
  const e = s.map(([, c]) => Math.exp(100 * (c - m))), z = e.reduce((a, b) => a + b, 0) || 1;
  return s.map(([l, c], i) => [l, c, e[i] / z]);
}

function FeaturesTile({ st, bstate, semJob, onBuild }: {
  st: SemStatus | null; bstate: (b: SemBackend) => BState;
  semJob: { id: number; status: string } | null; onBuild: () => void;
}) {
  const pill = (b: SemBackend) => {
    const s = bstate(b);
    return <Pill s={s === "ready" ? "succeeded" : s === "running" ? "running" : s === "stale" ? "warning" : "queued"} label={`${b}: ${s === "none" ? "not built" : s}`} />;
  };
  const lift = st?.tables.find((t) => t.active_run && t.backend === "lift");
  return (
    <Tile title="Features" icon="semantic" meta={st?.run ?? undefined} actions={<button onClick={onBuild}>Build…</button>}>
      <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>{pill("lift")}{pill("fmgs")}
        {semJob && <a href={`#/jobs/${semJob.id}`}>job #{semJob.id} {semJob.status}</a>}</div>
      {st && <div className="row small" style={{ gap: 4, marginTop: 4, flexWrap: "wrap" }}>{st.steps.map((x) => (
        <span key={x.step} className={`pill ${x.done ? "pill-ok" : "pill-q"}`} title={x.when ?? "not done"}>{x.step}</span>))}</div>}
      {lift && <p className="muted small" style={{ marginTop: 4 }}>lift: {lift.rows?.toLocaleString()} rows · {lift.mb} MB · {lift.lift.seconds ? `${Math.round(lift.lift.seconds)} s` : "—"}
        {lift.lift.peak_vram_mib ? ` · peak ${lift.lift.peak_vram_mib} MiB` : ""} · {lift.teacher_tag ?? ""}</p>}
      {st && st.queries.total > 0 && <p className="muted small">{st.queries.annotated} of {st.queries.total} annotated queries</p>}
    </Tile>
  );
}

// ── Send to course ────────────────────────────────────────────────────────────
function SendDialog({ scene, cand, text, backend, geo, approachKf, setApproachKf, onClose }: {
  scene: string; cand: Candidate; text: string; backend: SemBackend; geo: Geometry | null;
  approachKf: boolean; setApproachKf: (b: boolean) => void; onClose: () => void;
}) {
  const d = useAppData();
  const slug = `sem_${text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40) || "goal"}`;
  const [target, setTarget] = useState<string>("__new__");
  const [newName, setNewName] = useState(slug);
  const [label, setLabel] = useState(text);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const isNew = target === "__new__";
  const name = (isNew ? newName : target).trim();
  const goal: SemanticGoal = { label: label.trim() || text, position: cand.centroid as Vec3, query: text, backend, score: cand.score,
    extent: { lo: cand.box.lo as Vec3, hi: cand.box.hi as Vec3 }, approach: cand.approach as Vec3 };
  const send = async () => {
    setErr(null);
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) { setErr("Course name: letters, digits, _ and -."); return; }
    if (isNew && d.courses.some((c) => c.name === name) && !confirm(`A course “${name}” exists. Overwrite it?`)) return;
    setBusy(true);
    try {
      let course;
      if (isNew) {
        if (!geo) throw new Error("The scene geometry is still loading.");
        const start = (geo.camera_path[0] ?? [0, 0, -1]) as Vec3;
        course = courseToGoal(start, cand.approach as Vec3, cand.centroid as Vec3, geo.waypoint_box);
      } else {
        course = fromFile((await api.config("courses", name)) as CourseFile);
        if (approachKf) course = appendApproach(course, cand.approach as Vec3, cand.centroid as Vec3);
      }
      course = { ...course, goal };
      const pr = problems(course);
      if (pr.length) throw new Error(`The course would not be valid: ${pr.map((x) => x.msg).join("; ")}`);
      await api.saveConfig("courses", name, toFile(course), true);
      d.reload("courses");
      dispatchEvent(new CustomEvent("galley:course-saved", { detail: { name } }));
      onClose();
      location.hash = `#/course/${encodeURIComponent(scene)}/${encodeURIComponent(name)}`;
    } catch (e) { setErr(errMsg(e)); setBusy(false); }
  };
  return (
    <Dialog title="Send to course" onClose={onClose} footer={<button className="push primary" disabled={busy} onClick={send}>{busy ? "Saving…" : "Send and open"}</button>}>
      <p className="small">Writes <span className="mono">semantic_goal</span> for <b>“{text}”</b> (candidate #{cand.rank}, score {cand.score.toFixed(1)}) into a course,
        then opens it in the course editor: preview it and use <i>Save and fly</i> (F5).</p>
      <div className="fields" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <label className="f">Course<select value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="__new__">New course: start → approach point</option>
          {d.courses.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}</select></label>
        {isNew ? <label className="f">New course name<input value={newName} onChange={(e) => setNewName(e.target.value)} /></label>
          : <label className="f">Goal label<input value={label} onChange={(e) => setLabel(e.target.value)} /></label>}
      </div>
      {isNew ? (
        <p className="muted small" style={{ marginTop: 6 }}>Two keyframes, at rest at both ends: from where the capture started ({fmt(geo?.camera_path[0] ?? null)}, kept inside the waypoint box)
          to the approach point {fmt(cand.approach)}, facing the object. Add keyframes around obstacles in the course editor if the preview's gap check complains.</p>
      ) : (
        <label className="check small" style={{ marginTop: 6 }}><input type="checkbox" checked={approachKf} onChange={(e) => setApproachKf(e.target.checked)} />
          Append the approach point {fmt(cand.approach)} as the final keyframe, at rest and facing the object (the old last keyframe becomes a pass-through)</label>
      )}
      <table className="kv" style={{ marginTop: 6 }}><tbody>
        <tr><td>Goal (centroid)</td><td className="mono">{fmt(cand.centroid, 3)}</td></tr>
        <tr><td>Approach</td><td className="mono">{fmt(cand.approach, 3)} · gap {cand.gap === null ? "—" : `${cand.gap.toFixed(2)} m`}{cand.gap_ok === false && <span className="bad"> (too close)</span>}</td></tr>
        <tr><td>Box</td><td className="mono">{fmt(cand.box.lo)} … {fmt(cand.box.hi)}</td></tr>
      </tbody></table>
      {err && <p className="err">{err}</p>}
    </Dialog>
  );
}

// ── Build features (semantic_pipeline.py) ──────────────────────────────────────
function BuildDialog({ scene, defaultsWidth, onClose, onSubmitted }: { scene: string; defaultsWidth?: number; onClose: () => void; onSubmitted: (id: number) => void }) {
  const [dino, setDino] = useState(true);
  const [fw, setFw] = useState<number | undefined>(undefined);
  const [how, setHow] = useState<"continue" | "redo" | "only">("continue");
  const [step, setStep] = useState<SemStep>("lift");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const r: SemanticRun = { scene, teachers: dino ? ["clip", "dino"] : ["clip"], ...(fw ? { feat_width: fw } : {}),
    ...(how === "redo" ? { from_step: step, redo: [step] } : how === "only" ? { only: step, redo: [step] } : {}) };
  const submit = async () => { setBusy(true); setErr(null); try { onSubmitted((await semApi.submit(r)).id); } catch (e) { setErr(errMsg(e)); setBusy(false); } };
  return (
    <Dialog title={`Build semantic features · ${scene}`} onClose={onClose} footer={<button className="push primary" disabled={busy} onClick={submit}>{busy ? "Queueing…" : "Queue"}</button>}>
      <p className="small">Runs <span className="mono">figs/semantic_pipeline.py</span> as a GPU job: cameras (refined poses) → teachers (CLIP pyramid, DINOv2 per frame; the slow
        part, ~30 min on backroom) → lift (onto each Gaussian, ~3 min) → export. Finished steps are skipped, so Continue resumes where it stopped.</p>
      <div className="fields">
        <label className="check small"><input type="checkbox" checked={dino} onChange={(e) => setDino(e.target.checked)} />DINOv2 too (CLIP is always used)</label>
        <label className="f">Feature width (px)<input type="number" min={64} max={1920} placeholder={defaultsWidth ? `${defaultsWidth} (machine profile)` : "script default"}
          value={fw ?? ""} onChange={(e) => setFw(e.target.value === "" ? undefined : Number(e.target.value))} /></label>
        <label className="f">Steps<select value={how} onChange={(e) => setHow(e.target.value as typeof how)}>
          <option value="continue">Continue (skip finished steps)</option><option value="redo">Redo from a step</option><option value="only">Only one step</option></select></label>
        {how !== "continue" && <label className="f">Step<select value={step} onChange={(e) => setStep(e.target.value as SemStep)}>{SEM_STEPS.map((s) => <option key={s}>{s}</option>)}</select></label>}
      </div>
      <p className="muted small mono" style={{ marginTop: 6 }}>{JSON.stringify(r)}</p>
      {err && <p className="err">{err}</p>}
    </Dialog>
  );
}
