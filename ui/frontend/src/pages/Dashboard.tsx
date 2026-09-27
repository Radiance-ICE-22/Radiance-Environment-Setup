import { api, ago } from "../api";
import { Badge, usePoll } from "../components";

export default function Dashboard() {
  const scenes = usePoll(api.scenes, 15000);
  const jobs = usePoll(() => api.jobs(), 3000);
  const runs = usePoll(() => api.runs(), 15000);
  const current = jobs.data?.find((j) => j.status === "running");

  return (
    <>
      <h1>Overview</h1>
      {current && (
        <div className="panel row">
          <Badge s="running" /> <a href={`#/jobs/${current.id}`}>{current.label}</a>
          <span className="muted small">started {ago(current.started)}</span>
        </div>
      )}
      <div className="grid2">
        <div className="panel">
          <h2>Scenes</h2>
          {scenes.err && <p className="err">{scenes.err}</p>}
          <table>
            <thead><tr><th>Scene</th><th>Trained models</th><th>FiGS can load</th></tr></thead>
            <tbody>
              {scenes.data?.map((s) => (
                <tr key={s.scene} className="click" onClick={() => (location.hash = `#/scene/${s.scene}`)}>
                  <td><a href={`#/scene/${s.scene}`}>{s.scene}</a>{!s.has_workspace && <span className="muted small"> · no workspace</span>}</td>
                  <td>{s.models}</td>
                  <td className={s.loadable ? "ok" : "bad"}>{s.loadable ? "yes" : s.models === 0 ? "no model" : "more than one"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>Recent jobs</h2>
          <table>
            <thead><tr><th>#</th><th>Job</th><th>Status</th><th>Created</th></tr></thead>
            <tbody>
              {jobs.data?.slice(0, 8).map((j) => (
                <tr key={j.id} className="click" onClick={() => (location.hash = `#/jobs/${j.id}`)}>
                  <td>{j.id}</td><td>{j.label}</td><td><Badge s={j.status} /></td><td className="muted">{ago(j.created)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small"><a href="#/jobs">All jobs</a></p>
        </div>
      </div>
      <div className="panel">
        <h2>Flight run records</h2>
        <table>
          <thead><tr><th>Record</th><th>Course</th><th>Frames</th><th>Tracking error max (m)</th><th>Sim VRAM (MiB)</th><th>Dark frames</th></tr></thead>
          <tbody>
            {runs.data?.slice(0, 10).map((r) => (
              <tr key={r._file} className="click" onClick={() => (location.hash = `#/scene/${r.scene}`)}>
                <td className="mono">{r._file}</td><td>{r.course?.name ?? "—"}</td><td>{r.sim?.frames ?? "—"}</td>
                <td>{r.sim?.track_err_max_m ?? "—"}</td><td>{r.sim?.peak_vram_mib ?? "—"}</td><td>{r.sim?.dark_frames ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
