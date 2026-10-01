// Home document: everything at a glance. Scenes, queue, cohorts, flight records, GPU.
import { useState } from "react";
import { ago, duration } from "../api";
import { Problem, ToProblems, ToProperties, useUi } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { GpuChart, Pill, Prop, PropSection, QueueTable, Tile } from "../shell/Panes";

export default function Home() {
  const d = useAppData();
  const { ui, setUi } = useUi();
  const [selScene, setSelScene] = useState<string | null>(null);
  const [selRun, setSelRun] = useState<string | null>(null);
  const running = d.jobs.find((j) => j.status === "running");
  const queued = d.jobs.filter((j) => j.status === "queued").length;
  const sc = d.scenes.find((s) => s.scene === selScene);
  const rec = d.runs.find((r) => r._file === selRun);

  const probs: Problem[] = [
    ...(d.healthErr ? [{ severity: "error" as const, where: "backend", message: `Not reachable: ${d.healthErr}` }] : []),
    ...(d.health && !d.health.env_script ? [{ severity: "error" as const, where: "machine profile", message: "figs_env.sh not found: jobs cannot start." }] : []),
    ...(d.health && !d.health.pipeline ? [{ severity: "error" as const, where: "machine profile", message: "figs_pipeline.py not found." }] : []),
    ...d.scenes.filter((s) => !s.loadable).map((s) => ({ severity: "warning" as const, where: `scene ${s.scene}`,
      message: s.models === 0 ? "No active model: FiGS cannot load it. Train, or promote an archived run." : `${s.models} active models: FiGS refuses to guess. Archive all but one.` })),
    ...d.jobs.slice(0, 20).filter((j) => j.status === "failed").map((j) => ({ severity: "info" as const, where: `job #${j.id}`, message: `${j.label} failed (exit ${j.returncode ?? "?"}).` })),
    ...(d.machine?.disk && d.machine.disk.free_gb < 100 ? [{ severity: "warning" as const, where: "disk", message: `${d.machine.disk.free_gb} GB free: data_beta needs ~88 GB per course.` }] : []),
  ];

  return (
    <>
      <div className="tiles cols-4" style={{ marginBottom: 6 }}>
        <Kpi icon={running ? "run" : "queue"} v={running ? `#${running.id} · ${duration(running)}` : "idle"} l={running ? running.label : `queue${queued ? ` · ${queued} waiting` : " empty"}`} />
        <Kpi icon="scene" v={`${d.scenes.filter((s) => s.loadable).length} / ${d.scenes.length}`} l="scenes FiGS can load" />
        <Kpi icon="cohort" v={`${d.cohorts.length}`} l={`cohorts · ${d.cohorts.filter((c) => c.done.includes("deploy")).length} deployed`} />
        <Kpi icon="gpu" v={d.machine?.gpu.live ? `${(d.machine.gpu.live.used_mib / 1024).toFixed(1)} / ${(d.machine.gpu.live.total_mib / 1024).toFixed(0)} GB` : "—"}
          l={`${d.machine?.gpu.name || "GPU"}${d.machine?.disk ? ` · disk ${d.machine.disk.free_gb} GB free` : ""}`} />
      </div>

      <div className="tiles cols-3">
        <Tile title="Scenes" icon="scene" meta="double-click to open" className="flush">
          <table>
            <thead><tr><th>scene</th><th className="num">models</th><th>FiGS can load</th><th /></tr></thead>
            <tbody>{d.scenes.map((s) => (
              <tr key={s.scene} className={`click ${selScene === s.scene ? "sel" : ""}`} onClick={() => { setSelScene(s.scene); setSelRun(null); }} onDoubleClick={() => (location.hash = `#/scene/${s.scene}`)}>
                <td><b>{s.scene}</b>{!s.has_workspace && <span className="muted small"> · no workspace</span>}</td>
                <td className="num">{s.models}</td>
                <td>{s.loadable ? <Pill s="succeeded" label="yes" /> : <Pill s="warning" label={s.models === 0 ? "no model" : "more than one"} />}</td>
                <td><a href={`#/course/${s.scene}`} title="Course editor"><Icon name="route" size={14} /></a></td>
              </tr>))}
              {d.scenes.length === 0 && <tr><td colSpan={4} className="muted">No scenes yet: Capture &amp; Splat ▸ New capture.</td></tr>}</tbody>
          </table>
        </Tile>
        <Tile title="Queue and recent jobs" icon="queue" meta="one GPU job at a time" className="flush"
          actions={<a href="#/jobs" className="small">Monitor</a>}>
          <QueueTable jobs={d.jobs.slice(0, 10)} focus={ui.focus} setFocus={(id) => setUi({ focus: id })} compact />
        </Tile>
        <Tile title="Cohorts" icon="cohort" meta="SV-Net" className="flush" actions={<a href="#/svnet" className="small">New cohort</a>}>
          <table>
            <thead><tr><th>cohort</th><th>scene</th><th>steps</th><th>student error</th></tr></thead>
            <tbody>{d.cohorts.map((c) => (
              <tr key={c.cohort} className="click" onDoubleClick={() => (location.hash = `#/svnet/${c.cohort}`)} onClick={() => (location.hash = `#/svnet/${c.cohort}`)}>
                <td><b>{c.cohort}</b></td><td>{c.scene ?? "—"}</td><td>{c.done.length}/5</td>
                <td className="mono">{Object.entries(c.students).map(([k, v]) => `${k} ${v} m`).join(", ") || "—"}</td>
              </tr>))}
              {d.cohorts.length === 0 && <tr><td colSpan={4} className="muted">None yet.</td></tr>}</tbody>
          </table>
        </Tile>
      </div>

      <div className="tiles cols-2-1">
        <Tile title="Flight run records" icon="log" meta="runs/<scene>_<time>.json, newest first" className="flush">
          <table>
            <thead><tr><th>record</th><th>course</th><th className="num">frames</th><th className="num">tracking max (m)</th><th className="num">sim VRAM</th><th className="num">dark frames</th></tr></thead>
            <tbody>{d.runs.slice(0, 12).map((r) => (
              <tr key={r._file} className={`click ${selRun === r._file ? "sel" : ""}`} onClick={() => { setSelRun(r._file); setSelScene(null); }} onDoubleClick={() => (location.hash = `#/scene/${r.scene}`)}>
                <td className="mono">{r._file}</td><td>{r.course?.name ?? "—"}</td><td className="num">{r.sim?.frames ?? "—"}</td>
                <td className={`num ${r.sim?.track_err_max_m > 0.5 ? "bad" : ""}`}>{r.sim?.track_err_max_m ?? "—"}</td>
                <td className="num">{r.sim?.peak_vram_mib ?? "—"}</td><td className={`num ${r.sim?.dark_frames > 0 ? "bad" : ""}`}>{r.sim?.dark_frames ?? "—"}</td>
              </tr>))}
              {d.runs.length === 0 && <tr><td colSpan={6} className="muted">No flights recorded yet.</td></tr>}</tbody>
          </table>
        </Tile>
        <Tile title="GPU" icon="gpu" meta={d.machine?.gpu.name}><GpuChart height={150} /></Tile>
      </div>

      <ToProperties>
        {sc ? (
          <>
            <div className="props-title"><Icon name="scene" size={16} />{sc.scene}</div>
            <PropSection title="Scene">
              <Prop k="Workspace">{sc.has_workspace ? "yes" : "none"}</Prop>
              <Prop k="Active models" tone={sc.loadable ? "ok" : "bad"}>{sc.models}{sc.loadable ? " (FiGS can load it)" : ""}</Prop>
              <Prop k="Pipeline state">{sc.has_state ? "yes" : "none"}</Prop>
              <Prop k="Jobs">{d.jobs.filter((j) => j.scene === sc.scene).length}</Prop>
            </PropSection>
            <div className="pad-x row" style={{ padding: 8 }}>
              <button className="push" onClick={() => (location.hash = `#/scene/${sc.scene}`)}>Open scene</button>
              <button className="push" onClick={() => (location.hash = `#/course/${sc.scene}`)}>Course editor</button>
            </div>
          </>
        ) : rec ? (
          <>
            <div className="props-title"><Icon name="log" size={16} />{rec._file}</div>
            <PropSection title="Flight">
              <Prop k="Scene">{rec.scene}</Prop><Prop k="Course">{rec.course?.name ?? "—"}</Prop>
              {Object.entries(rec.sim ?? {}).map(([k, v]) => <Prop key={k} k={k} mono>{typeof v === "object" ? JSON.stringify(v) : String(v)}</Prop>)}
            </PropSection>
          </>
        ) : (
          <>
            <div className="props-title"><Icon name="machine" size={16} />Machine</div>
            <PropSection title="GPU">
              <Prop k="Name">{d.machine?.gpu.name || "—"}</Prop><Prop k="VRAM">{d.machine?.gpu.vram_mib ?? "—"} MiB</Prop>
              {d.machine?.gpu.live && <><Prop k="In use">{Math.round(d.machine.gpu.live.used_mib)} MiB</Prop><Prop k="Utilisation">{d.machine.gpu.live.util_pct} %</Prop><Prop k="Temperature">{d.machine.gpu.live.temp_c} °C</Prop></>}
            </PropSection>
            <PropSection title="Paths">{Object.entries(d.machine?.paths ?? {}).map(([k, v]) => <Prop key={k} k={k} mono>{v ?? "—"}</Prop>)}</PropSection>
            <PropSection title="Defaults">{Object.entries(d.machine?.defaults ?? {}).map(([k, v]) => <Prop key={k} k={k} mono>{String(v)}</Prop>)}</PropSection>
            <PropSection title="Backend">
              <Prop k="Health" tone={d.healthErr ? "bad" : "ok"}>{d.healthErr ? "not reachable" : d.health?.ok ? "ok" : "…"}</Prop>
              <Prop k="figs_env.sh" tone={d.health?.env_script ? "ok" : "bad"}>{d.health?.env_script ? "found" : "missing"}</Prop>
              <Prop k="Current job">{d.health?.current_job ? `#${d.health.current_job}` : "none"}</Prop>
              {d.jobs[0] && <Prop k="Last job">#{d.jobs[0].id} · {ago(d.jobs[0].created)}</Prop>}
            </PropSection>
          </>
        )}
      </ToProperties>
      <ToProblems items={probs} />
    </>
  );
}

function Kpi({ icon, v, l }: { icon: string; v: string; l: string }) {
  return (
    <div className="tile" style={{ flexDirection: "row", alignItems: "center", gap: 8, padding: "5px 8px" }}>
      <Icon name={icon} size={32} />
      <div style={{ minWidth: 0 }}><div style={{ fontSize: 15, fontWeight: 600, color: "var(--navy)" }}>{v}</div><div className="muted small ell" style={{ maxWidth: "100%" }}>{l}</div></div>
    </div>
  );
}
