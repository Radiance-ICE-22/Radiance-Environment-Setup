// Galley's window: a Windows 7-style ribbon shell around tabbed documents.
//
//   title bar (Quick Access) · ribbon (tabs → groups → commands) · Explorer | documents | Properties
//   · Output (live log, queue, problems, GPU) · status bar
//
// Each hash route is a document. Opened documents stay mounted (hidden when not active), so
// switching between a scene, its course and a running job keeps their state. Documents bind
// the ribbon's commands with useCommands() (shell/core.tsx); the app binds the global ones here.
import { lazy, ReactNode, StrictMode, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { active, api, ApiError, FAMILIES, Family, getToken, Job, rerun, setToken } from "./api";
import { commands, DocCtx, PaneCtx, run, UiCtx, UiState, useAppCommands } from "./shell/core";
import { AppDataProvider, useAppData } from "./shell/data";
import { Icon } from "./shell/icons";
import { Dialog, DocDesc, DocTabs, Explorer, filterJobs, OutputPanel, OutTab, PropertiesPane, Splitter, StatusBar, TitleBar } from "./shell/Panes";
import { Tip } from "./shell/Tip";
import { Ribbon } from "./shell/Ribbon";
import { SHORTCUTS } from "./shell/ribbonSpec";
import Home from "./pages/Dashboard";
import ScenePage from "./pages/Scene";
import Monitor from "./pages/Jobs";
import JobPage from "./pages/Job";
import Configs from "./pages/Configs";
import NewCapture from "./pages/NewCapture";
import SvNetPage from "./pages/SvNet";

// three.js and the editor load only when the course editor is opened.
const CoursePage = lazy(() => import("./pages/Course"));

// ── routes → documents ──────────────────────────────────────────────────────
function parse(hash: string): string[] {
  return hash.replace(/^#?\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}
const enc = (r: string[]) => `#/${r.map(encodeURIComponent).join("/")}`;
function describe(r: string[]): DocDesc {
  const [top, a, b] = r;
  const href = enc(r);
  if (top === "scene" && a) return { key: `scene:${a}`, href, title: a, icon: "scene", route: r };
  if (top === "new") return { key: "new", href, title: "New capture", icon: "camera", route: r };
  if (top === "course") return { key: "course", href, title: a ? `${a} / ${b ?? "new course"}` : "Course editor", icon: "route", route: r };
  if (top === "svnet" && a) return { key: `svnet:${a}`, href, title: a, icon: "cohort", route: r };
  if (top === "svnet") return { key: "svnet", href, title: "SV-Net", icon: "network", route: r };
  if (top === "jobs" && a) return { key: `job:${a}`, href, title: `Job #${a}`, icon: "job", route: r };
  if (top === "jobs") return { key: "jobs", href, title: "Monitor", icon: "gauge", route: r };
  if (top === "configs") return { key: "configs", href, title: b ? `${b}.json` : a ? `Configs · ${a}` : "Configs", icon: "config", route: r };
  return { key: "home", href: "#/", title: "Home", icon: "home", route: [] };
}
const TAB_FOR: [RegExp, string][] = [[/^home$/, "Home"], [/^(scene:|new$)/, "Capture & Splat"], [/^course$/, "Course"], [/^svnet/, "SV-Net"], [/^(jobs$|job:)/, "Jobs"], [/^configs$/, "Configs"]];
const MAX_DOCS = 14;

function useHash() {
  const [h, setH] = useState(location.hash || "#/");
  useEffect(() => { const f = () => setH(location.hash || "#/"); addEventListener("hashchange", f); return () => removeEventListener("hashchange", f); }, []);
  return h;
}
function load<T>(k: string, d: T): T { try { const v = localStorage.getItem(k); return v ? { ...d, ...JSON.parse(v) } : d; } catch { return d; } }
function save(k: string, v: unknown) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } }

interface Layout { explorer: boolean; properties: boolean; output: boolean; status: boolean; minRibbon: boolean; density: "compact" | "comfortable" | "large"; outH: number; exW: number; prW: number }
const LAYOUT0: Layout = { explorer: true, properties: true, output: true, status: true, minRibbon: false, density: "compact", outH: 210, exW: 250, prW: 300 };

type DialogState = null | "help" | "keys" | "about" | "token" | { params: Job };

// ── the window ──────────────────────────────────────────────────────────────
function Window() {
  const d = useAppData();
  const hash = useHash();
  const cur = useMemo(() => describe(parse(hash)), [hash]);
  const [docs, setDocs] = useState<DocDesc[]>(() => {
    try { const l = JSON.parse(localStorage.getItem("galley.docs") ?? "[]") as string[]; return l.map((h) => describe(parse(h))); } catch { return []; }
  });
  // open (or update) the document for the current route
  useEffect(() => {
    setDocs((ds) => {
      const i = ds.findIndex((x) => x.key === cur.key);
      if (i >= 0) { if (ds[i].href === cur.href) return ds; const n = [...ds]; n[i] = cur; return n; }
      let n = [...ds, cur];
      while (n.length > MAX_DOCS) { const drop = n.findIndex((x) => x.key !== cur.key && x.key !== "home"); if (drop < 0) break; n = n.filter((_, j) => j !== drop); }
      return n;
    });
  }, [cur]);
  useEffect(() => save("galley.docs", docs.map((x) => x.href)), [docs]);
  const close = useCallback((key: string) => {
    setDocs((ds) => {
      const i = ds.findIndex((x) => x.key === key); if (i < 0) return ds;
      const n = ds.filter((x) => x.key !== key);
      if (key === cur.key) location.hash = (n[Math.min(i, n.length - 1)] ?? { href: "#/" }).href;
      return n;
    });
  }, [cur.key]);

  // active document → command layer, ribbon tab
  commands.setActive(cur.key);
  const [tab, setTab] = useState(() => TAB_FOR.find(([re]) => re.test(cur.key))?.[1] ?? "Home");
  const lastKey = useRef(cur.key);
  useEffect(() => {
    if (lastKey.current === cur.key) return; lastKey.current = cur.key;
    const t = TAB_FOR.find(([re]) => re.test(cur.key))?.[1]; if (t) setTab(t);
  }, [cur.key]);

  // layout, UI state, panes
  const [layout, setLayoutRaw] = useState<Layout>(() => load("galley.layout", LAYOUT0));
  const setLayout = (p: Partial<Layout>) => setLayoutRaw((l) => { const n = { ...l, ...p }; save("galley.layout", n); return n; });
  const [ui, setUiRaw] = useState<UiState>(() => ({ ...load("galley.ui", { jobKind: "all", jobScene: "all", jobStatus: "all", hideFinished: false, follow: true, wrap: true }), focus: null }));
  const setUi = useCallback((p: Partial<UiState>) => setUiRaw((u) => { const n = { ...u, ...p }; const { focus: _f, ...keep } = n; save("galley.ui", keep); return n; }), []);
  const [propsEl, setPropsEl] = useState<HTMLElement | null>(null);
  const [probEl, setProbEl] = useState<HTMLElement | null>(null);
  const [probCounts, setProbCounts] = useState<Record<string, number>>({});
  const setProblemCount = useCallback((k: string, n: number) => setProbCounts((c) => (c[k] === n ? c : { ...c, [k]: n })), []);
  const [outTab, setOutTab] = useState<OutTab>("log");
  const [dialog, setDialog] = useState<DialogState>(null);

  // full screen: the active document gets the whole screen (and the browser goes full screen)
  const [fs, setFs] = useState(false);
  const [fsPanes, setFsPanes] = useState({ properties: false, output: false });
  const enterFs = () => { setFs(true); document.documentElement.requestFullscreen?.().catch(() => {}); };
  const exitFs = () => { setFs(false); if (document.fullscreenElement) document.exitFullscreen().catch(() => {}); };
  useEffect(() => {
    const f = () => { if (!document.fullscreenElement) setFs(false); };   // Esc in browser full screen
    document.addEventListener("fullscreenchange", f);
    return () => document.removeEventListener("fullscreenchange", f);
  }, []);
  const showProps = fs ? fsPanes.properties : layout.properties;
  const showOut = fs ? fsPanes.output : layout.output;

  // the scene the Home and Course pickers default to
  const scene = cur.route[0] === "scene" || cur.route[0] === "course" ? cur.route[1] : undefined;
  const lastScene = useRef<string | undefined>(undefined);
  if (scene) lastScene.current = scene;
  const ctxScene = scene ?? lastScene.current ?? d.scenes.find((s) => s.loadable)?.scene ?? d.scenes[0]?.scene;
  const ctxCohort = cur.route[0] === "svnet" ? cur.route[1] : undefined;

  // ── app-wide command bindings ──────────────────────────────────────────────
  const focusJob = d.jobs.find((j) => j.id === ui.focus) ?? null;
  const cancelTarget = focusJob && active(focusJob.status) ? focusJob : d.jobs.find((j) => j.status === "running") ?? null;
  const rerunTarget = focusJob ?? d.jobs.find((j) => !active(j.status)) ?? null;
  const go = (h: string) => { location.hash = h; };
  const err = (e: unknown) => alert(e instanceof ApiError ? e.message : String(e));
  const sceneOpts = d.scenes.map((s) => [s.scene, s.loadable ? s.scene : `${s.scene} (no single model)`] as [string, string]);
  const NO_BACKEND = (what: string) => `${what} is not in the backend yet (ui/backend/galley): planned.`;
  const fam = cur.key === "configs" ? (cur.route[1] as Family | undefined) ?? "courses" : null;
  useAppCommands({
    "home.scene": { value: ctxScene ?? "", options: sceneOpts, set: (v) => v && go(`#/scene/${v}`) },
    "home.course": { value: cur.key === "course" ? cur.route[2] ?? "" : "", options: d.courses.map((c) => c.name), set: (v) => v && go(ctxScene ? `#/course/${ctxScene}/${v}` : "#/course"), disabled: !ctxScene && "Pick a scene first." },
    "home.cohort": { value: ctxCohort ?? "", options: d.cohorts.map((c) => c.cohort), set: (v) => v && go(`#/svnet/${v}`) },
    "run": { disabled: "This document has nothing to run. Open a scene, course, cohort or New capture." },
    "job.cancel": cancelTarget ? { run: () => api.cancel(cancelTarget.id).then(() => d.reload("jobs")).catch(err), label: undefined } : { disabled: "No job is queued or running." },
    "job.rerun": rerunTarget && !active(rerunTarget.status) ? { run: () => rerun(rerunTarget).then(({ id }) => { d.reload("jobs"); setUi({ focus: id }); }).catch(err) } : { disabled: "Select a finished job in Output ▸ Queue." },
    "queue.pause": { disabled: NO_BACKEND("Pausing the queue") },
    "queue.up": { disabled: NO_BACKEND("Reordering the queue") },
    "queue.clear": { checked: ui.hideFinished, run: () => setUi({ hideFinished: !ui.hideFinished }) },
    "m.gpu": { run: () => { setLayout({ output: true }); setOutTab("gpu"); } },
    "m.disk": { run: () => setDialog("about") },
    "m.profile": { run: () => setDialog("about") },
    "r.flight": d.runs[0]?.scene ? { run: () => go(`#/scene/${d.runs[0].scene}`) } : { disabled: "No flight run records yet." },
    "r.cohort": d.cohorts[0] ? { run: () => go(`#/svnet/${d.cohorts[0].cohort}`) } : { disabled: "No cohorts yet." },
    "r.job": d.jobs[0] ? { run: () => go(`#/jobs/${d.jobs[0].id}`) } : { disabled: "No jobs yet." },
    "help.docs": { run: () => setDialog("help") },
    "help.keys": { run: () => setDialog("keys") },
    "help.about": { run: () => setDialog("about") },
    "cap.new": { run: () => go("#/new") },
    "cap.upload": { disabled: NO_BACKEND("Uploading videos") + " Copy the video into video_captures/ for now." },
    "splat.viewer": { disabled: NO_BACKEND("Launching ns-viewer as a job") },
    "splat.export": { disabled: NO_BACKEND("ns-export") },
    "sv.new": { run: () => go("#/svnet") },
    "sv.open": { value: ctxCohort ?? "", options: d.cohorts.map((c) => c.cohort), set: (v) => v && go(`#/svnet/${v}`) },
    "sv.compare": { disabled: NO_BACKEND("Comparing cohorts") },
    "jobs.kind": { value: ui.jobKind, options: [["all", "all"], "figs", "svnet", "selftest"], set: (v) => setUi({ jobKind: v }) },
    "jobs.scene": { value: ui.jobScene, options: [["all", "all"], ...d.scenes.map((s) => s.scene), ...d.cohorts.map((c) => [c.cohort, `${c.cohort} (cohort)`] as [string, string])], set: (v) => setUi({ jobScene: v }) },
    "jobs.status": { value: ui.jobStatus, options: [["all", "all"], "queued", "running", "succeeded", "failed", "cancelled", "interrupted"], set: (v) => setUi({ jobStatus: v }) },
    "log.follow": { checked: ui.follow, run: () => setUi({ follow: !ui.follow }) },
    "log.wrap": { checked: ui.wrap, run: () => setUi({ wrap: !ui.wrap }) },
    "log.save": focusJob ?? d.jobs[0] ? { run: () => saveLog((focusJob ?? d.jobs[0])!) } : { disabled: "No job selected." },
    "job.selftest": { run: () => api.submitSelftest(20).then(({ id }) => { d.reload("jobs"); setUi({ focus: id }); setLayout({ output: true }); setOutTab("log"); }).catch(err) },
    "runs.diff": { disabled: NO_BACKEND("Diffing run records") },
    "runs.snapshot": focusJob ? { run: () => setDialog({ params: focusJob }) } : { disabled: "Select a job in Output ▸ Queue." },
    "runs.records": { run: () => go("#/") },
    ...Object.fromEntries(FAMILIES.map((f) => [`cfg.f.${f}`, { checked: fam === f, run: () => go(`#/configs/${f}`) }])),
    "cfg.diffup": { disabled: NO_BACKEND("Diffing against upstream") },
    "cfg.restore": { disabled: NO_BACKEND("Restoring from the overlay") },
    "view.explorer": { checked: layout.explorer, set: (v) => setLayout({ explorer: v === "true" }) },
    "view.properties": { checked: layout.properties, set: (v) => setLayout({ properties: v === "true" }) },
    "view.output": { checked: layout.output, set: (v) => setLayout({ output: v === "true" }) },
    "view.status": { checked: layout.status, set: (v) => setLayout({ status: v === "true" }) },
    "view.minribbon": { checked: layout.minRibbon, set: (v) => setLayout({ minRibbon: v === "true" }) },
    "view.reset": { run: () => setLayout(LAYOUT0) },
    "layout.edit": { run: () => setLayout({ explorer: true, properties: true, output: true }) },
    "layout.train": { run: () => setLayout({ explorer: false, properties: false, output: true }) },
    "layout.monitor": { run: () => { setLayout({ output: false }); go("#/jobs"); } },
    "density.compact": { checked: layout.density === "compact", run: () => setLayout({ density: "compact" }) },
    "density.comfortable": { checked: layout.density === "comfortable", run: () => setLayout({ density: "comfortable" }) },
    "density.large": { checked: layout.density === "large", run: () => setLayout({ density: "large" }) },
    "view.fullscreen": { checked: fs, run: () => (fs ? exitFs() : enterFs()) },
    "win.float": { run: () => window.open(location.href, "_blank", "noopener") },
    "win.closeall": docs.length > 1 ? { run: () => setDocs((ds) => ds.filter((x) => x.key === cur.key)) } : { disabled: "Only one document is open." },
    "win.split": { disabled: "Side-by-side documents are planned. Use View ▸ New window meanwhile." },
  });

  // ── keyboard ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const typing = el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
      const ctrl = e.ctrlKey || e.metaKey;
      let id: string | null = null;
      if (e.key === "Escape" && fs && !e.defaultPrevented && !typing && !document.fullscreenElement) { exitFs(); return; }
      if (ctrl && e.shiftKey && e.key.toLowerCase() === "f") id = "view.fullscreen";
      else if (e.key === "F5") id = e.shiftKey ? "job.cancel" : "run";
      else if (e.key === "F6") id = "prev.run";
      else if (e.key === "F7") id = "cfg.validate";
      else if (e.key === "F1" && ctrl) { e.preventDefault(); setLayout({ minRibbon: !layout.minRibbon }); return; }
      else if (e.key === "F1") id = "help.docs";
      else if (ctrl && e.key.toLowerCase() === "s") id = "file.save";
      else if (ctrl && !e.shiftKey && e.key.toLowerCase() === "z" && !typing) id = "edit.undo";
      if (!id) return;
      e.preventDefault();       // F5 must never reload the page
      run(id);
    };
    addEventListener("keydown", h);
    return () => removeEventListener("keydown", h);
  });

  // ── render ─────────────────────────────────────────────────────────────────
  const panes = useMemo(() => ({ properties: propsEl, problems: probEl, setProblemCount }), [propsEl, probEl, setProblemCount]);
  const uiCtx = useMemo(() => ({ ui, setUi }), [ui, setUi]);
  useEffect(() => { document.title = `${cur.title} — Galley`; }, [cur.title]);
  const filtered = filterJobs(d.jobs, ui);

  return (
    <UiCtx.Provider value={uiCtx}>
      <PaneCtx.Provider value={panes}>
        <div className={`window density-${layout.density} ${fs ? "fullscreen" : ""}`}>
          {!fs && <TitleBar title={`Galley — ${cur.title}`} />}
          <Ribbon tab={tab} setTab={setTab} minimized={layout.minRibbon || fs} setMinimized={(b) => { if (!fs) setLayout({ minRibbon: b }); }}
            appMenu={<AppMenu open={setDialog} closeOthers={() => setDocs((ds) => ds.filter((x) => x.key === cur.key))} />} />
          <div className="workspace">
            {layout.explorer && !fs && <div style={{ width: layout.exW, flex: "none", display: "flex" }}>
              <Explorer activeHref={cur.href} defaultScene={ctxScene ?? null} onClose={() => setLayout({ explorer: false })} /></div>}
            {layout.explorer && !fs && <Splitter dir="v" onDrag={(dx) => setLayout({ exW: clamp(layout.exW + dx, 170, 480) })} onReset={() => setLayout({ exW: LAYOUT0.exW })} />}
            <div className="center">
              {fs ? (
                <div className="fs-strip">
                  <Icon name={cur.icon} size={16} /><b>{cur.title}</b><span className="muted">full screen</span><span className="spacer" />
                  <Tip tip={{ title: "Properties", body: "Show the Properties pane beside the document while in full screen." }}>
                    <button className={`rb-small ${fsPanes.properties ? "on" : ""}`} onClick={() => setFsPanes((p) => ({ ...p, properties: !p.properties }))}><Icon name="panes" size={16} /><span>Properties</span></button></Tip>
                  <Tip tip={{ title: "Output", body: "Show the Output panel (live log, queue, problems, GPU) while in full screen." }}>
                    <button className={`rb-small ${fsPanes.output ? "on" : ""}`} onClick={() => setFsPanes((p) => ({ ...p, output: !p.output }))}><Icon name="log" size={16} /><span>Output</span></button></Tip>
                  <Tip tip={{ title: "Exit full screen", body: "Bring back the title bar, Explorer, document tabs, panes and status bar.", keyText: "Esc" }}>
                    <button className="rb-small" onClick={exitFs}><Icon name="exitfs" size={16} /><span>Exit full screen</span></button></Tip>
                </div>
              ) : <DocTabs docs={docs} active={cur.key} onClose={close} />}
              <div className="docs">
                {docs.map((doc) => (
                  <DocCtx.Provider key={doc.key} value={{ key: doc.key, active: doc.key === cur.key }}>
                    <div className={`doc ${doc.key === "course" ? "doc-fill" : doc.key === "configs" ? "doc-fill doc-cfg" : ""}`} style={{ display: doc.key === cur.key ? undefined : "none" }}>
                      <DocView doc={doc} />
                    </div>
                  </DocCtx.Provider>
                ))}
              </div>
              {showOut && <Splitter dir="h" onDrag={(dy) => setLayout({ outH: clamp(layout.outH - dy, 90, 600) })} onReset={() => setLayout({ outH: LAYOUT0.outH })} />}
              {showOut && <OutputPanel height={layout.outH} tab={outTab} setTab={setOutTab} problemsRef={setProbEl} problemCount={probCounts[cur.key] ?? 0}
                onClose={() => (fs ? setFsPanes((p) => ({ ...p, output: false })) : setLayout({ output: false }))} />}
              {!showOut && <div ref={setProbEl} style={{ display: "none" }} />}
            </div>
            {showProps && <Splitter dir="v" onDrag={(dx) => setLayout({ prW: clamp(layout.prW - dx, 220, 520) })} onReset={() => setLayout({ prW: LAYOUT0.prW })} />}
            {showProps && <div style={{ width: layout.prW, flex: "none", display: "flex" }}>
              <PropertiesPane setTarget={setPropsEl} onClose={() => (fs ? setFsPanes((p) => ({ ...p, properties: false })) : setLayout({ properties: false }))} /></div>}
          </div>
          {layout.status && !fs && <StatusBar />}
          {dialog && <Dialogs dialog={dialog} close={() => setDialog(null)} jobs={filtered} />}
        </div>
      </PaneCtx.Provider>
    </UiCtx.Provider>
  );
}
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

async function saveLog(j: Job) {
  const rows = await api.log(j.id);
  const blob = new Blob([rows.map((r) => r.line).join("\n") + "\n"], { type: "text/plain" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `galley_job${j.id}_${j.label.replace(/[^\w.-]+/g, "_")}.log`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function DocView({ doc }: { doc: DocDesc }) {
  const r = doc.route;
  switch (r[0]) {
    case "scene": return <ScenePage scene={r[1]} />;
    case "new": return <NewCapture />;
    case "course": return <Suspense fallback={<p className="muted pad">Loading the 3D editor…</p>}><CoursePage scene={r[1]} name={r[2]} /></Suspense>;
    case "svnet": return <SvNetPage cohort={r[1]} />;
    case "jobs": return r[1] ? <JobPage id={Number(r[1])} /> : <Monitor />;
    case "configs": return <Configs family={(r[1] as Family) ?? "courses"} name={r[2]} />;
    default: return <Home />;
  }
}

// ── application menu and dialogs ────────────────────────────────────────────
function AppMenu({ open, closeOthers }: { open: (d: DialogState) => void; closeOthers: () => void }) {
  const items: [string, string, string, () => void][] = [
    ["camera", "New capture…", "", () => (location.hash = "#/new")], ["cohort", "New cohort…", "", () => (location.hash = "#/svnet")],
    ["route", "Course editor", "", () => (location.hash = "#/course")], ["gauge", "Monitor", "", () => (location.hash = "#/jobs")],
    ["close", "Close other documents", "", closeOthers], ["token", "Access token…", "", () => open("token")],
    ["docs", "Keyboard shortcuts", "", () => open("keys")], ["help", "Help", "F1", () => open("help")], ["info", "About Galley", "", () => open("about")],
  ];
  return <>{items.map(([ic, l, k, f]) => <button key={l} className="am-item" onClick={f}><Icon name={ic} size={16} /><span>{l}</span>{k && <span className="am-key">{k}</span>}</button>)}</>;
}

function Dialogs({ dialog, close }: { dialog: Exclude<DialogState, null>; close: () => void; jobs: Job[] }) {
  const d = useAppData();
  if (dialog === "keys") return (
    <Dialog title="Keyboard shortcuts" onClose={close}>
      <table className="grid"><tbody>{SHORTCUTS.map(([k, v]) => <tr key={k}><td style={{ width: 120 }}><kbd>{k}</kbd></td><td>{v}</td></tr>)}</tbody></table>
    </Dialog>
  );
  if (dialog === "token") return <TokenDialog onClose={close} />;
  if (dialog === "about") {
    const m = d.machine;
    return (
      <Dialog title="About Galley" onClose={close}>
        <p><b>Galley</b> runs the FiGS / SOUS-VIDE pipeline of the Radiance project: phone video → Gaussian splat → course → expert flight → SV-Net student pilots.</p>
        <table className="grid"><tbody>
          <tr><td>Backend</td><td className="mono">{location.host} · {d.healthErr ? `not reachable (${d.healthErr})` : d.health?.ok ? "ok" : "…"}</td></tr>
          <tr><td>figs_env.sh</td><td>{d.health?.env_script ? "found" : "missing"}</td></tr>
          <tr><td>figs_pipeline.py</td><td>{d.health?.pipeline ? "found" : "missing"}</td></tr>
          <tr><td>GPU</td><td>{m?.gpu.name || "—"} · {m?.gpu.vram_mib ?? "?"} MiB</td></tr>
          <tr><td>Disk</td><td>{m?.disk ? `${m.disk.free_gb} GB free of ${m.disk.total_gb} GB` : "—"}</td></tr>
          {Object.entries(m?.paths ?? {}).map(([k, v]) => <tr key={k}><td>{k}</td><td className="mono">{v ?? "—"}</td></tr>)}
          {Object.entries(m?.defaults ?? {}).map(([k, v]) => <tr key={k}><td>default {k}</td><td className="mono">{String(v)}</td></tr>)}
        </tbody></table>
        <p className="muted small">Paths and defaults come from the machine profile, ui/machines/&lt;host&gt;.toml (or GALLEY_MACHINE).</p>
      </Dialog>
    );
  }
  if (typeof dialog === "object") {
    const j = dialog.params;
    return (
      <Dialog title={`Job #${j.id} parameters`} onClose={close}>
        <p className="mono small">{j.label}</p>
        <table className="grid"><tbody>{Object.entries(j.params).filter(([, v]) => v !== null).map(([k, v]) => <tr key={k}><td style={{ width: 140 }}>{k}</td><td className="mono">{JSON.stringify(v)}</td></tr>)}</tbody></table>
        <h4>Command</h4><pre className="mono small wrap">{j.argv.join(" ")}</pre>
      </Dialog>
    );
  }
  return (
    <Dialog title="Galley help" onClose={close}>
      <div className="help">
        <p><b>Ribbon.</b> Tabs group the commands by task: <i>Home</i> (context, queue, machine), <i>Capture &amp; Splat</i>, <i>Course</i>, <i>SV-Net</i>, <i>Jobs</i>, <i>Configs</i> and <i>View</i>.
          Commands act on the active document; a greyed command says in its tooltip what to open first. Rest the pointer on any command for its help and the script it runs.
          <i> Keyframe Tools</i> and <i>Model Tools</i> appear when a keyframe or a model is selected.</p>
        <p><b>Explorer</b> (left) lists scenes, courses, cohorts, config files and jobs. Click to open a document; documents stay open as tabs and keep their state.</p>
        <p><b>Properties</b> (right) shows details of what is selected in the active document. <b>Output</b> (bottom) has the live log of the running or selected job, the queue,
          the active document's problems and a GPU chart.</p>
        <p><b>One GPU job at a time.</b> Run (F5) queues the active document's job; Cancel (Shift+F5) stops it, and its step markers let Run resume where it stopped.</p>
        <p>Handoff documentation: <span className="mono">docs/GALLEY_UI.md</span>, <span className="mono">docs/FiGS_custom_video_guide.md</span>, <span className="mono">ui/README.md</span>.</p>
      </div>
    </Dialog>
  );
}

function TokenDialog({ onClose, required }: { onClose: () => void; required?: boolean }) {
  const [t, setT] = useState(getToken() ?? "");
  const ok = () => { setToken(t || null); onClose(); if (required) location.reload(); };
  return (
    <Dialog title="Galley access token" onClose={required ? () => {} : onClose} footer={<button className="push" onClick={ok}>OK</button>}>
      <p className="small">This server requires the token set in its machine profile ([server] token) or GALLEY_TOKEN.</p>
      <input type="password" autoFocus value={t} onChange={(e) => setT(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ok()} style={{ width: "100%" }} />
    </Dialog>
  );
}

function App() {
  const [needLogin, setNeedLogin] = useState(false);
  useEffect(() => { api.scenes().catch((e) => { if (e instanceof ApiError && e.status === 401) setNeedLogin(true); }); }, []);
  if (needLogin) return <div className="window"><TokenDialog required onClose={() => setNeedLogin(false)} /></div>;
  return <AppDataProvider><Window /></AppDataProvider>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);

export type { ReactNode };
