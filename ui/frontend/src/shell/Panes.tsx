// Docked parts of the window: title bar, Explorer, document tabs, Properties, Output, status bar.
import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { active, ago, api, duration, FAMILIES, Family, Job, streamJob } from "../api";
import { commands, useRegistryVersion, useUi } from "./core";
import { useAppData } from "./data";
import { Icon } from "./icons";
import { Tip } from "./Tip";
import { eta, fmtBytes, fmtDur, pending, useUploads } from "../uploads";
import { driveActive, useDrive } from "../drive";

// ── small shared widgets ────────────────────────────────────────────────────
const PILL: Record<string, [string, string]> = {
  succeeded: ["ok", "pill-ok"], running: ["run", "pill-run"], queued: ["queue", "pill-q"], failed: ["error", "pill-bad"],
  cancelled: ["pause", "pill-warn"], interrupted: ["pause", "pill-warn"], warning: ["warning", "pill-warn"],
};
export function Pill({ s, label }: { s: string; label?: string }) {
  const [ic, cls] = PILL[s] ?? ["info", "pill-q"];
  return <span className={`pill ${cls}`}><Icon name={ic} size={12} />{label ?? s}</span>;
}
export function Tile({ title, icon, meta, children, className, actions, id }: { title: ReactNode; icon?: string; meta?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode; id?: string }) {
  return (
    <section className={`tile ${className ?? ""}`} id={id}>
      <header>{icon && <Icon name={icon} size={16} />}<b>{title}</b><span className="spacer" />{meta && <span className="meta">{meta}</span>}{actions}</header>
      <div className="tile-body">{children}</div>
    </section>
  );
}
export function PaneHeader({ title, onClose }: { title: string; onClose?: () => void }) {
  return (
    <div className="pane-head"><b>{title}</b><span className="spacer" />
      {onClose && <Tip tip={{ title: `Close ${title}`, body: "View ▸ Panes brings it back." }}><button className="ph-btn" onClick={onClose}><Icon name="close" size={12} /></button></Tip>}
    </div>
  );
}
export function Dialog({ title, children, onClose, footer }: { title: string; children: ReactNode; onClose: () => void; footer?: ReactNode }) {
  useEffect(() => { const f = (e: KeyboardEvent) => e.key === "Escape" && onClose(); addEventListener("keydown", f); return () => removeEventListener("keydown", f); }, [onClose]);
  return (
    <div className="dlg-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dlg" role="dialog" aria-label={title}>
        <div className="dlg-title"><b>{title}</b><span className="spacer" /><button className="cap-close" onClick={onClose}><Icon name="close" size={12} /></button></div>
        <div className="dlg-body">{children}</div>
        <div className="dlg-foot">{footer}<button className="push primary" onClick={onClose}>Close</button></div>
      </div>
    </div>
  );
}

// ── title bar ───────────────────────────────────────────────────────────────
export function TitleBar({ title }: { title: string }) {
  useRegistryVersion();
  const qat: [string, string, string, string][] = [["file.save", "save", "Save", "Ctrl+S"], ["edit.undo", "undo", "Undo", "Ctrl+Z"], ["run", "run", "Run", "F5"], ["job.cancel", "cancel", "Cancel", "Shift+F5"]];
  return (
    <div className="titlebar">
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="1" y="1" width="16" height="16" rx="3" fill="#2B67B8" stroke="#1A4C8E" /><path d="M5 12c2-1 3-5 4-5s1.5 3 4 1" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" /><circle cx="5" cy="12" r="1.4" fill="#7CE08B" /><circle cx="13" cy="8" r="1.4" fill="#FFB347" /></svg>
      <div className="qat">
        {qat.map(([id, ic, label, key]) => { const b = commands.get(id); return (
          <Tip key={id} tip={{ title: label, body: `Quick Access: ${label.toLowerCase()} in the active document.`, keyText: key, note: !b ? "Nothing to do here in this document." : typeof b.disabled === "string" ? b.disabled : undefined }}>
            <button className="qat-btn" disabled={!b || !!b.disabled} onClick={() => b?.run?.()}><Icon name={ic} size={16} /></button>
          </Tip>); })}
      </div>
      <span className="spacer" /><span className="tb-title">{title}</span><span className="spacer" />
      <Tip tip={{ title: "Full screen", body: "Use the whole screen for Galley (Esc leaves full screen)." }}>
        <button className="cap-btn" onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen())}>
          <svg width="12" height="12" viewBox="0 0 12 12"><rect x="1.5" y="2.5" width="9" height="7" fill="none" stroke="#1E395B" strokeWidth="1.5" /></svg></button>
      </Tip>
    </div>
  );
}

// ── Explorer ────────────────────────────────────────────────────────────────
interface Node { id: string; label: string; icon: string; meta?: string; href?: string; kids?: Node[]; load?: () => Promise<Node[]>; bad?: boolean }
export function Explorer({ activeHref, defaultScene, onClose }: { activeHref: string; defaultScene: string | null; onClose: () => void }) {
  const d = useAppData();
  const [open, setOpen] = useState<Record<string, boolean>>(() => { try { return JSON.parse(localStorage.getItem("galley.tree") ?? "") } catch { return { scenes: true, cohorts: true, jobs: true, courses: true } } });
  const [q, setQ] = useState("");
  const [lazy, setLazy] = useState<Record<string, Node[]>>({});
  useEffect(() => { try { localStorage.setItem("galley.tree", JSON.stringify(open)); } catch { /* */ } }, [open]);
  const sc = defaultScene ?? d.scenes.find((s) => s.loadable)?.scene ?? d.scenes[0]?.scene;
  const tree: Node[] = useMemo(() => [
    { id: "home", label: "Home", icon: "home", href: "#/" },
    { id: "scenes", label: "Scenes", icon: "folder", meta: String(d.scenes.length), kids: [
      ...d.scenes.map((s) => ({ id: `scene:${s.scene}`, label: s.scene, icon: s.loadable ? "scene" : "warning", bad: !s.loadable,
        meta: s.loadable ? "model ✓" : s.models === 0 ? "no model" : `${s.models} models`, href: `#/scene/${s.scene}`,
        kids: [{ id: `sc:${s.scene}:course`, label: "Course editor", icon: "route", href: `#/course/${s.scene}` },
          { id: `sc:${s.scene}:splat`, label: "Semantics (splat editor)", icon: "semantic", href: `#/splat/${s.scene}` }] })),
      { id: "new", label: "New capture…", icon: "camera", href: "#/new" }] },
    { id: "courses", label: "Courses", icon: "folder", meta: String(d.courses.length), kids: d.courses.map((c) => ({ id: `course:${c.name}`, label: c.name, icon: "route", href: sc ? `#/course/${sc}/${c.name}` : `#/course` })) },
    { id: "cohorts", label: "Cohorts", icon: "folder", meta: String(d.cohorts.length), kids: [
      ...d.cohorts.map((c) => ({ id: `cohort:${c.cohort}`, label: c.cohort, icon: "cohort", href: `#/svnet/${c.cohort}`,
        meta: c.done.includes("deploy") ? "deployed" : `${c.done.length}/5`,
        kids: (c.roster ?? []).map((p) => ({ id: `cohort:${c.cohort}:${p}`, label: p, icon: "network", meta: c.students[p] !== undefined ? `${c.students[p]} m` : "student", href: `#/svnet/${c.cohort}` })) })),
      { id: "svnew", label: "New cohort…", icon: "cohort", href: "#/svnet" }] },
    { id: "configs", label: "Configs", icon: "folder", kids: FAMILIES.map((f) => ({ id: `cfg:${f}`, label: f, icon: { captures: "camera", courses: "route", pilots: "network", frames: "drone", methods: "steps", nnio: "json" }[f], href: `#/configs/${f}`,
      kids: lazy[`cfg:${f}`] ?? [], load: () => api.configs(f as Family).then((l) => l.map((c) => ({ id: `cfg:${f}:${c.name}`, label: c.name, icon: "json", meta: c.kind, href: `#/configs/${f}/${c.name}` }))) })) },
    { id: "jobs", label: "Jobs", icon: "folder", meta: String(d.jobs.length), kids: [
      { id: "monitor", label: "Monitor", icon: "gauge", href: "#/jobs" },
      ...d.jobs.slice(0, 12).map((j) => ({ id: `job:${j.id}`, label: `#${j.id} ${j.label}`, icon: ({ succeeded: "ok", running: "run", queued: "queue", failed: "error" } as Record<string, string>)[j.status] ?? "pause", meta: duration(j), href: `#/jobs/${j.id}` }))] },
  ], [d.scenes, d.courses, d.cohorts, d.jobs, lazy, sc]);
  const toggle = (n: Node) => {
    const now = !open[n.id]; setOpen({ ...open, [n.id]: now });
    if (now && n.load && !lazy[n.id]) n.load().then((k) => setLazy((l) => ({ ...l, [n.id]: k }))).catch(() => {});
  };
  const match = (n: Node): boolean => !q || n.label.toLowerCase().includes(q.toLowerCase()) || (n.kids ?? []).some(match);
  const row = (n: Node, depth: number): ReactNode => {
    if (!match(n)) return null;
    const hasKids = !!(n.kids?.length || n.load);
    const isOpen = !!q || open[n.id];
    const sel = n.href && n.href === activeHref;
    return (
      <div key={n.id}>
        <div className={`tree-row ${sel ? "sel" : ""}`} style={{ paddingLeft: 4 + depth * 14 }}
          onClick={() => { if (n.href) location.hash = n.href; else if (hasKids) toggle(n); }}
          onDoubleClick={() => hasKids && toggle(n)}>
          <span className="twisty" onClick={(e) => { e.stopPropagation(); if (hasKids) toggle(n); }}>{hasKids && <Icon name={isOpen ? "chevdown" : "chevron"} size={11} />}</span>
          <Icon name={n.icon} size={16} />
          <span className={`tl ${n.bad ? "warn" : ""}`}>{n.label}</span>
          {n.meta && <span className="tm">{n.meta}</span>}
        </div>
        {hasKids && isOpen && n.kids?.map((k) => row(k, depth + 1))}
      </div>
    );
  };
  return (
    <aside className="pane explorer">
      <PaneHeader title="Explorer" onClose={onClose} />
      <div className="pane-body">
        <Tip tip={{ title: "Search", body: "Filter the tree by name: scenes, courses, cohorts, config files, jobs." }} block>
          <div className="searchbox"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search scenes, courses, jobs…" /><Icon name="search" size={14} /></div>
        </Tip>
        <div className="tree">{tree.map((n) => row(n, 0))}</div>
      </div>
    </aside>
  );
}

// ── document tabs ───────────────────────────────────────────────────────────
export interface DocDesc { key: string; href: string; title: string; icon: string; route: string[] }
export function DocTabs({ docs, active, onClose }: { docs: DocDesc[]; active: string; onClose: (k: string) => void }) {
  return (
    <div className="doctabs">
      {docs.map((d) => (
        <div key={d.key} className={`doctab ${d.key === active ? "on" : ""}`} onClick={() => (location.hash = d.href)}
          onMouseDown={(e) => { if (e.button === 1) { e.preventDefault(); onClose(d.key); } }} title={d.href}>
          <Icon name={d.icon} size={16} /><span>{d.title}</span>
          <button className="dt-x" onClick={(e) => { e.stopPropagation(); onClose(d.key); }} aria-label={`Close ${d.title}`}><Icon name="close" size={11} /></button>
        </div>
      ))}
    </div>
  );
}

// ── Properties ──────────────────────────────────────────────────────────────
export function PropertiesPane({ setTarget, onClose }: { setTarget: (el: HTMLElement | null) => void; onClose: () => void }) {
  return (
    <aside className="pane properties">
      <PaneHeader title="Properties" onClose={onClose} />
      <div className="pane-body"><div className="props-target" ref={setTarget} /></div>
    </aside>
  );
}
export function PropSection({ title, children, icon }: { title: string; children: ReactNode; icon?: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="psec">
      <div className="psec-h" onClick={() => setOpen(!open)}><Icon name={open ? "chevdown" : "chevron"} size={10} />{icon && <Icon name={icon} size={14} />}<b>{title}</b></div>
      {open && <div className="psec-b">{children}</div>}
    </div>
  );
}
export function Prop({ k, children, mono, tone }: { k: string; children: ReactNode; mono?: boolean; tone?: "ok" | "bad" | "warn" | "muted" }) {
  return <div className="prow"><span className="pk">{k}</span><span className={`pv ${mono ? "mono" : ""} ${tone ?? ""}`}>{children}</span></div>;
}

// ── Output panel ────────────────────────────────────────────────────────────
export function filterJobs(jobs: Job[], ui: ReturnType<typeof useUi>["ui"]) {
  return jobs.filter((j) => (ui.jobKind === "all" || j.kind === ui.jobKind) && (ui.jobScene === "all" || j.scene === ui.jobScene || (j.params as any)?.cohort === ui.jobScene)
    && (ui.jobStatus === "all" || j.status === ui.jobStatus) && !(ui.hideFinished && (j.status === "succeeded" || j.status === "cancelled")));
}
export function QueueTable({ jobs, focus, setFocus, compact }: { jobs: Job[]; focus: number | null; setFocus: (id: number) => void; compact?: boolean }) {
  return (
    <table className="grid">
      <thead><tr><th style={{ width: 44 }}>#</th><th style={{ width: 64 }}>kind</th><th>what</th><th style={{ width: 112 }}>status</th><th style={{ width: 80, textAlign: "right" }}>time</th>{!compact && <th style={{ width: 70 }} />}</tr></thead>
      <tbody>{jobs.map((j) => (
        <tr key={j.id} className={`click ${j.id === focus ? "sel" : ""}`} onClick={() => setFocus(j.id)} onDoubleClick={() => (location.hash = `#/jobs/${j.id}`)}>
          <td>#{j.id}</td><td>{j.kind}</td><td className="ell">{j.label}</td><td><Pill s={j.status} /></td><td style={{ textAlign: "right" }}>{duration(j)}</td>
          {!compact && <td>{active(j.status) && <button className="push small" onClick={(e) => { e.stopPropagation(); api.cancel(j.id); }}>Cancel</button>}</td>}
        </tr>))}
        {jobs.length === 0 && <tr><td colSpan={6} className="muted">No jobs.</td></tr>}</tbody>
    </table>
  );
}
/** Streams a job's log; shared by the Output panel, the Monitor and job documents. */
export function useJobLog(id: number | null) {
  const [lines, setLines] = useState<string[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    setLines([]); setProgress(null); setJob(null);
    if (id === null) return;
    api.job(id).then(setJob).catch(() => {});
    let buf: string[] = []; let raf = 0;
    const flush = () => { raf = 0; if (buf.length) { const b = buf; buf = []; setLines((l) => (l.length > 5000 ? l.slice(-4000) : l).concat(b)); } };
    const close = streamJob(id, (m) => {
      if (m.type === "line") { buf.push(m.line); setProgress(null); if (!raf) raf = requestAnimationFrame(flush); }
      else if (m.type === "progress") setProgress(m.line);
      else { flush(); setProgress(null); api.job(id).then(setJob).catch(() => {}); }
    }, () => { flush(); api.job(id).then(setJob).catch(() => {}); });
    return () => { close(); if (raf) cancelAnimationFrame(raf); };
  }, [id]);
  return { lines, progress, job };
}
export function LogView({ lines, progress, height }: { lines: string[]; progress: string | null; height?: number | string }) {
  const { ui } = useUi(); const box = useRef<HTMLDivElement>(null);
  useEffect(() => { if (ui.follow && box.current) box.current.scrollTop = box.current.scrollHeight; }, [lines, progress, ui.follow]);
  return (
    <div className={`logview ${ui.wrap ? "wrap" : ""}`} ref={box} style={{ height }}>
      {lines.map((l, i) => <div key={i} className={/✔|✅|succeeded|\bok\b/.test(l) ? "lg-ok" : /✗|error|failed|Traceback/i.test(l) ? "lg-bad" : /^\s*(═|══|===|──)/.test(l) ? "lg-head" : /^\s*!/.test(l) ? "lg-warn" : ""}>{l || " "}</div>)}
      {progress && <div className="lg-prog">{progress}</div>}
      {lines.length === 0 && !progress && <div className="muted">No output yet.</div>}
    </div>
  );
}
export type OutTab = "log" | "queue" | "problems" | "gpu";
export function OutputPanel({ problemsRef, problemCount, onClose, tab, setTab, height }: {
  problemsRef: (el: HTMLElement | null) => void; problemCount: number; onClose: () => void; tab: OutTab; setTab: (t: OutTab) => void; height: number;
}) {
  const d = useAppData(); const { ui, setUi } = useUi();
  const focus = ui.focus; const setFocus = (id: number) => setUi({ focus: id });
  const jobs = filterJobs(d.jobs, ui);
  const fid = focus ?? d.jobs.find((j) => j.status === "running")?.id ?? d.jobs[0]?.id ?? null;
  const log = useJobLog(fid);
  const tabs: [typeof tab, string, string][] = [["log", "log", fid ? `Live log · #${fid}` : "Live log"], ["queue", "queue", `Queue (${d.jobs.filter((j) => active(j.status)).length})`],
    ["problems", problemCount ? "warning" : "ok", `Problems (${problemCount})`], ["gpu", "gpu", "GPU"]];
  return (
    <div className="output" style={{ height }}>
      <div className="out-head"><b>Output</b>
        {tabs.map(([k, ic, label]) => <button key={k} className={`rb-small ${tab === k ? "on" : ""}`} onClick={() => setTab(k)}><Icon name={ic} size={16} /><span>{label}</span></button>)}
        <span className="spacer" />
        <button className="ph-btn" onClick={onClose} title="Hide the Output panel (View ▸ Panes)"><Icon name="close" size={12} /></button>
      </div>
      <div className="out-body">
        <div className="out-split" style={{ display: tab === "log" ? "flex" : "none" }}>
          <div className="out-q"><QueueTable jobs={jobs.slice(0, 30)} focus={fid} setFocus={setFocus} compact /></div>
          <LogView lines={log.lines} progress={log.progress} />
        </div>
        <div style={{ display: tab === "queue" ? "block" : "none" }} className="out-scroll"><QueueTable jobs={jobs} focus={fid} setFocus={setFocus} /></div>
        <div style={{ display: tab === "problems" ? "block" : "none" }} className="out-scroll problems-host" ref={problemsRef} />
        <div style={{ display: tab === "gpu" ? "block" : "none" }} className="out-scroll"><GpuChart height={150} /></div>
      </div>
    </div>
  );
}
export function GpuChart({ height = 160 }: { height?: number }) {
  const d = useAppData(); const h = d.gpuHistory; const total = d.machine?.gpu.live?.total_mib ?? d.machine?.gpu.vram_mib ?? 1;
  const W = 600, H = height, L = 34, B = 14;
  if (h.length < 2) return <p className="muted small">Collecting nvidia-smi samples (every 5 s)…{!d.machine?.gpu.live && " nvidia-smi is not available on this host."}</p>;
  const t0 = h[0].t, t1 = h[h.length - 1].t;
  const X = (t: number) => L + (t - t0) / (t1 - t0 || 1) * (W - L - 4), Y = (f: number) => 4 + (1 - f) * (H - 4 - B);
  const mem = h.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)} ${Y(p.used / total).toFixed(1)}`).join("");
  const util = h.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)} ${Y(p.util / 100).toFixed(1)}`).join("");
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" className="gpuchart">
        <rect x={L} y={4} width={W - L - 4} height={H - 4 - B} fill="#fff" stroke="#D5DEEA" />
        <path d={`${mem}L${X(t1)} ${Y(0)}L${X(t0)} ${Y(0)}Z`} fill="#2F6FDD" fillOpacity=".12" /><path d={mem} stroke="#2F6FDD" strokeWidth="1.6" fill="none" />
        <path d={util} stroke="#3FA34D" strokeWidth="1" fill="none" />
        <text x={L - 4} y={12} textAnchor="end" className="axis">{(total / 1024).toFixed(0)}G</text><text x={L - 4} y={H - B} textAnchor="end" className="axis">0</text>
      </svg>
      <p className="muted small">memory (blue) and utilisation (green) of {d.machine?.gpu.name || "the GPU"} · last {Math.round((t1 - t0) / 60000)} min, sampled every 5 s while Galley is open</p>
    </div>
  );
}

// ── status bar ──────────────────────────────────────────────────────────────
export function StatusBar() {
  const d = useAppData();
  const live = d.machine?.gpu.live; const run = d.jobs.find((j) => j.status === "running");
  const queued = d.jobs.filter((j) => j.status === "queued").length;
  const root = d.machine?.paths?.project_root ?? "";
  const fields: [string, ReactNode, string, (() => void)?][] = [
    ["machine", root ? `project_root ${root}` : "machine profile", "Machine profile in use (ui/machines/<host>.toml): where FiGS and SousVide live.", () => commands.get("help.about")?.run?.()],
    ["gpu", live ? `${d.machine?.gpu.name.replace("NVIDIA GeForce ", "")} · ${(live.used_mib / 1024).toFixed(1)} / ${(live.total_mib / 1024).toFixed(0)} GB · ${live.util_pct}% · ${live.temp_c} °C` : `${d.machine?.gpu.name || "GPU"} · nvidia-smi unavailable`, "Live nvidia-smi, every 5 s."],
    ["disk", d.machine?.disk ? `${d.machine.disk.free_gb} GB free` : "disk —", "Free space under project_root."],
    [run ? "run" : "queue", run ? <>{`Job #${run.id} · ${run.label} · ${duration(run)}`}<span className="sb-prog"><span /></span></> : `Queue: idle${queued ? ` · ${queued} queued` : ""}`, "The job queue: one GPU job at a time. Click to open the Monitor.", () => (location.hash = "#/jobs")],
  ];
  const ups = useUploads();
  const up = ups.list.find((u) => u.state === "uploading" || u.state === "starting" || u.state === "finishing") ?? ups.list.find((u) => u.state === "paused" || u.state === "error");
  if (up) {
    const n = ups.list.filter(pending).length;
    const p = up.size ? Math.floor((100 * up.sent) / up.size) : 0;
    fields.push(["upload", <span className="sb-up">{up.state === "error" ? `Upload failed: ${up.name}` : up.state === "paused" ? `Upload paused: ${up.name} ${p} %`
      : `Uploading ${up.name} · ${p} %${up.rate ? ` · ${fmtBytes(up.rate)}/s · ${fmtDur(eta(up))} left` : ""}${n > 1 ? ` · ${n - 1} more` : ""}`}
      <span className={`upbar ${up.state === "error" ? "bad" : up.state === "paused" ? "paused" : ""}`}><span style={{ width: `${p}%` }} /></span></span>,
      "Video upload to the host's video_captures/. Click to open New capture, where it can be paused, resumed or cancelled.", () => (location.hash = "#/new")]);
  }
  const dr = useDrive();
  const di = dr.list.find(driveActive) ?? dr.list.find((i) => i.state === "error" || i.state === "interrupted");
  if (di) {
    const p = di.size ? Math.floor((100 * di.received) / di.size) : 0;
    const bad = !driveActive(di);
    fields.push(["cloud", <span className="sb-up">{bad ? `Drive import stopped: ${di.name}` : `Host ← Drive ${di.name} · ${p} %${di.rate ? ` · ${fmtBytes(di.rate)}/s · ${fmtDur((di.size - di.received) / di.rate)} left` : ""}`}
      <span className={`upbar ${bad ? "paused" : ""}`}><span style={{ width: `${p}%` }} /></span></span>,
      "The host is downloading a video you picked in Google Drive (it continues if this tab closes). Click to open New capture.", () => (location.hash = "#/new")]);
  }
  const ok = d.health?.ok && d.health.env_script && d.health.pipeline;
  return (
    <div className="statusbar">
      {fields.map(([ic, label, h, onClick], i) => (
        <Tip key={i} tip={{ title: typeof label === "string" ? label : ic === "upload" ? "Upload" : ic === "cloud" ? "Google Drive" : "Queue", body: h }}>
          <div className={`sb-f ${onClick ? "click" : ""}`} onClick={onClick}><Icon name={ic} size={14} />{label}</div>
        </Tip>
      ))}
      <span className="spacer" />
      <Tip tip={{ title: "Backend", body: d.healthErr ? `Not reachable: ${d.healthErr}` : `GET /api/health · figs_env.sh ${d.health?.env_script ? "found" : "missing"} · pipeline ${d.health?.pipeline ? "found" : "missing"}` }}>
        <div className="sb-f"><Icon name={d.healthErr ? "error" : ok ? "ok" : "warning"} size={14} />{d.healthErr ? "Disconnected" : `Connected · ${location.host}`}</div>
      </Tip>
    </div>
  );
}

/** Properties of one job (Monitor, job documents). */
export function JobProps({ job }: { job: Job }) {
  const ts = (t: number | null) => (t ? new Date(t * 1000).toLocaleString() : "—");
  return (
    <>
      <div className="props-title"><Icon name="job" size={16} />Job #{job.id}<span className="spacer" /><Pill s={job.status} /></div>
      <PropSection title="Job">
        <Prop k="Label">{job.label}</Prop><Prop k="Kind">{job.kind}</Prop><Prop k="Scene">{job.scene ?? "—"}</Prop>
        <Prop k="Created">{ts(job.created)} ({ago(job.created)})</Prop><Prop k="Started">{ts(job.started)}</Prop><Prop k="Finished">{ts(job.finished)}</Prop>
        <Prop k="Duration">{duration(job)}</Prop><Prop k="Exit code" tone={job.returncode ? "bad" : job.returncode === 0 ? "ok" : undefined}>{job.returncode ?? "—"}</Prop>
      </PropSection>
      <PropSection title="Parameters">
        {Object.entries(job.params ?? {}).filter(([, v]) => v !== null && v !== undefined && !(Array.isArray(v) && !v.length)).map(([k, v]) =>
          <Prop key={k} k={k} mono>{typeof v === "object" ? JSON.stringify(v) : String(v)}</Prop>)}
      </PropSection>
      <PropSection title="Command"><p className="mono small" style={{ overflowWrap: "anywhere", padding: "0 8px" }}>{job.argv.join(" ")}</p></PropSection>
      <div className="row" style={{ padding: 8 }}>
        {active(job.status) && <button className="push" onClick={() => api.cancel(job.id)}>Cancel</button>}
        <a href={`#/jobs/${job.id}`}>Open the full log</a>
      </div>
    </>
  );
}

/** Chevron strip of pipeline steps (scene steps, capture plan, cohort steps). */
export interface StepCell { name: string; state?: "done" | "run" | "bad" | ""; sub?: string; tip?: string; range?: boolean }
export function StepStrip({ steps, onPick }: { steps: StepCell[]; onPick?: (name: string) => void }) {
  return (
    <div className="stepstrip">
      {steps.map((s) => (
        <Tip key={s.name} tip={s.tip ? { title: `Step ${s.name}`, body: s.tip } : null} block>
          <div className={`chev ${s.state ?? ""} ${s.range ? "range" : ""}`} onClick={() => onPick?.(s.name)} style={{ cursor: onPick ? "pointer" : undefined }}>
            <b>{s.state === "done" ? "✓ " : s.state === "run" ? "▶ " : ""}{s.name}</b><span>{s.sub ?? ""}</span>
          </div>
        </Tip>
      ))}
    </div>
  );
}

/** Drag handle between two panes or tiles: "v" resizes widths, "h" heights. */
export function Splitter({ dir, onDrag, onReset }: { dir: "h" | "v"; onDrag: (d: number) => void; onReset?: () => void }) {
  const ref = useRef(onDrag); ref.current = onDrag;
  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    let last = dir === "v" ? e.clientX : e.clientY;
    const move = (m: PointerEvent) => { const p = dir === "v" ? m.clientX : m.clientY; if (p !== last) { ref.current(p - last); last = p; } };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); document.body.classList.remove("dragging"); };
    addEventListener("pointermove", move); addEventListener("pointerup", up); document.body.classList.add("dragging");
  };
  return <div className={`splitter splitter-${dir}`} onPointerDown={down} onDoubleClick={onReset} title={onReset ? "Drag to resize; double-click to reset" : undefined} />;
}

