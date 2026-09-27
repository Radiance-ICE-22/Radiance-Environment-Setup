import { useEffect, useRef, useState } from "react";
import { active, api, duration, Job, streamJob } from "../api";
import { Badge } from "../components";

export default function JobPage({ id }: { id: number }) {
  const [job, setJob] = useState<Job | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLines([]); setProgress(null);
    api.job(id).then(setJob).catch((e) => setErr(String(e)));
    let buf: string[] = [];
    let raf = 0;
    const flush = () => { raf = 0; if (buf.length) { const b = buf; buf = []; setLines((l) => l.concat(b)); } };
    const close = streamJob(id, (m) => {
      if (m.type === "line") { buf.push(m.line); setProgress(null); if (!raf) raf = requestAnimationFrame(flush); }
      else if (m.type === "progress") setProgress(m.line);
      else { flush(); setProgress(null); api.job(id).then(setJob); }
    }, () => { flush(); api.job(id).then(setJob).catch(() => {}); });
    return () => { close(); if (raf) cancelAnimationFrame(raf); };
  }, [id]);

  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight; }, [lines, progress, follow]);
  useEffect(() => {        // keep the duration ticking while running
    if (!job || !active(job.status)) return;
    const t = setInterval(() => setJob((j) => (j ? { ...j } : j)), 1000);
    return () => clearInterval(t);
  }, [job?.status]);

  if (err) return <p className="err">{err}</p>;
  if (!job) return <p className="muted">Loading…</p>;
  const failLine = job.status === "failed" ? [...lines].reverse().find((l) => /failed|error|✗/i.test(l)) : undefined;

  return (
    <>
      <div className="row">
        <h1>#{job.id} {job.label}</h1><Badge s={job.status} />
        <span className="spacer" />
        {job.scene && <a href={`#/scene/${job.scene}`}>Scene {job.scene}</a>}
        {active(job.status) && <button className="danger" onClick={() => api.cancel(job.id).then(setJob)}>Cancel</button>}
      </div>
      <p className="muted small">
        {duration(job)}{job.returncode !== null && ` · exit ${job.returncode}`} · {lines.length} lines
        <label style={{ marginLeft: 12 }}><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow</label>
      </p>
      {failLine && <p className="err">{failLine}</p>}
      <div className="log" ref={box}>
        {lines.join("\n")}
        {progress && <div className="progress">{progress}</div>}
      </div>
      <details style={{ marginTop: 10 }}>
        <summary className="muted small">Command</summary>
        <pre className="mono small" style={{ whiteSpace: "pre-wrap" }}>{job.argv.join(" ")}</pre>
      </details>
    </>
  );
}
