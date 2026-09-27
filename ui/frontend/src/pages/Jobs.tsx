import { useState } from "react";
import { active, api, ago, duration } from "../api";
import { Badge, usePoll } from "../components";

export default function Jobs() {
  const jobs = usePoll(() => api.jobs(), 3000);
  const [busy, setBusy] = useState(false);
  return (
    <>
      <div className="row">
        <h1>Jobs</h1><span className="spacer" />
        <button disabled={busy} title="Harmless 20 s job for checking the queue, streaming and cancel"
          onClick={async () => { setBusy(true); const { id } = await api.submitSelftest(20); location.hash = `#/jobs/${id}`; }}>
          Run self-test
        </button>
      </div>
      <p className="muted small">One job runs at a time; the GPU is treated as exclusive. Queued jobs start in order.</p>
      {jobs.err && <p className="err">{jobs.err}</p>}
      <div className="panel">
        <table>
          <thead><tr><th>#</th><th>Job</th><th>Scene</th><th>Status</th><th>Created</th><th>Duration</th><th /></tr></thead>
          <tbody>
            {jobs.data?.map((j) => (
              <tr key={j.id} className="click" onClick={() => (location.hash = `#/jobs/${j.id}`)}>
                <td>{j.id}</td><td>{j.label}</td><td>{j.scene ?? "—"}</td><td><Badge s={j.status} /></td>
                <td className="muted">{ago(j.created)}</td><td>{duration(j)}</td>
                <td>{active(j.status) && (
                  <button className="danger" onClick={(e) => { e.stopPropagation(); api.cancel(j.id).then(jobs.reload); }}>Cancel</button>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
