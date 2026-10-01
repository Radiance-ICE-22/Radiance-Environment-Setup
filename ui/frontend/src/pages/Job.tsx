// Job document: one job's full log, its parameters and command line.
import { useEffect, useState } from "react";
import { active, api, ApiError, duration, rerun } from "../api";
import { ToProblems, ToProperties, useCommands, useUi } from "../shell/core";
import { useAppData } from "../shell/data";
import { JobProps, LogView, Pill, Tile, useJobLog } from "../shell/Panes";

export default function JobPage({ id }: { id: number }) {
  const d = useAppData();
  const { setUi } = useUi();
  const log = useJobLog(id);
  const [, tick] = useState(0);
  const job = d.jobs.find((j) => j.id === id) ?? log.job;
  useEffect(() => {        // keep the duration ticking while running
    if (!job || !active(job.status)) return;
    const t = setInterval(() => tick((x) => x + 1), 1000);
    return () => clearInterval(t);
  }, [job?.status]);
  const err = (e: unknown) => alert(e instanceof ApiError ? e.message : String(e));
  useCommands({
    "job.cancel": job && active(job.status) ? { run: () => api.cancel(id).then(() => d.reload("jobs")).catch(err) } : { disabled: "This job is not queued or running." },
    "job.rerun": job && !active(job.status) ? { run: () => rerun(job).then(({ id: n }) => { d.reload("jobs"); location.hash = `#/jobs/${n}`; }).catch(err) } : { disabled: "The job is still queued or running." },
  });
  useEffect(() => { setUi({ focus: id }); }, [id, setUi]);
  if (!job) return <p className="muted pad">Loading job #{id}…</p>;
  const failLine = job.status === "failed" ? [...log.lines].reverse().find((l) => /failed|error|✗/i.test(l)) : undefined;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 6 }}>
      <Tile title={`#${job.id} ${job.label}`} icon="job" className="fill flush"
        meta={<><Pill s={job.status} /> {job.kind} · {duration(job)}{job.returncode !== null ? ` · exit ${job.returncode}` : ""} · {log.lines.length} lines</>}
        actions={job.scene ? <a className="small" style={{ marginLeft: 6 }} href={`#/scene/${job.scene}`}>scene {job.scene}</a> : undefined}>
        <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
          {failLine && <p className="err" style={{ padding: "3px 8px", background: "var(--bad-bg)", borderBottom: "1px solid #EBB0A9", margin: 0 }}>{failLine}</p>}
          <LogView lines={log.lines} progress={log.progress} />
        </div>
      </Tile>
      <ToProperties><JobProps job={job} /></ToProperties>
      <ToProblems items={failLine ? [{ severity: "error", where: `job #${job.id}`, message: failLine }] : []} />
    </div>
  );
}
