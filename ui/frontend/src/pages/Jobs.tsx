// Monitor document: the queue (filtered from the Jobs tab), the selected job's log, GPU.
import { ago, duration } from "../api";
import { Problem, ToProblems, ToProperties, useUi } from "../shell/core";
import { useAppData } from "../shell/data";
import { filterJobs, GpuChart, JobProps, LogView, Pill, QueueTable, Tile, useJobLog } from "../shell/Panes";

export default function Monitor() {
  const d = useAppData();
  const { ui, setUi } = useUi();
  const jobs = filterJobs(d.jobs, ui);
  const fid = ui.focus ?? d.jobs.find((j) => j.status === "running")?.id ?? d.jobs[0]?.id ?? null;
  const log = useJobLog(fid);
  const j = d.jobs.find((x) => x.id === fid) ?? log.job;
  const probs: Problem[] = d.jobs.filter((x) => x.status === "failed" || x.status === "interrupted").slice(0, 30)
    .map((x) => ({ severity: x.status === "failed" ? "error" as const : "warning" as const, where: `job #${x.id}`, message: `${x.label}: ${x.status}${x.returncode !== null ? ` (exit ${x.returncode})` : ""} · ${ago(x.finished)}` }));
  const filtered = ui.jobKind !== "all" || ui.jobScene !== "all" || ui.jobStatus !== "all" || ui.hideFinished;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 6 }}>
      <div className="tiles cols-1-2" style={{ flex: 1, minHeight: 0, alignItems: "stretch" }}>
        <Tile title="Queue" icon="queue" className="fill flush" meta={`${jobs.length} job${jobs.length === 1 ? "" : "s"}${filtered ? " (filtered: Jobs ▸ Filter)" : ""} · click to select, double-click to open`}>
          <QueueTable jobs={jobs} focus={fid} setFocus={(id) => setUi({ focus: id })} />
        </Tile>
        <Tile title={j ? `#${j.id} ${j.label}` : "Log"} icon="log" className="fill flush"
          meta={j ? <><Pill s={j.status} /> {duration(j)}{j.returncode !== null ? ` · exit ${j.returncode}` : ""} · {log.lines.length} lines</> : undefined}
          actions={j && <a href={`#/jobs/${j.id}`} className="small" style={{ marginLeft: 6 }}>open</a>}>
          <div style={{ display: "flex", flex: 1, minHeight: 200 }}><LogView lines={log.lines} progress={log.progress} /></div>
        </Tile>
      </div>
      <Tile title="GPU" icon="gpu" meta={`${d.machine?.gpu.name ?? ""} · one GPU job at a time; queued jobs start in order`}><GpuChart height={110} /></Tile>
      <ToProperties>{j ? <JobProps job={j} /> : null}</ToProperties>
      <ToProblems items={probs} />
    </div>
  );
}
