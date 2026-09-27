import { lazy, StrictMode, Suspense, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { api, ApiError, Machine, setToken } from "./api";
import Dashboard from "./pages/Dashboard";
import ScenePage from "./pages/Scene";
import Jobs from "./pages/Jobs";
import JobPage from "./pages/Job";
import Configs from "./pages/Configs";
import NewCapture from "./pages/NewCapture";
import SvNetPage from "./pages/SvNet";

// three.js and the editor load only when the course editor is opened.
const CoursePage = lazy(() => import("./pages/Course"));

// Hash routes keep the static build servable from any path with no server rewrites.
function useRoute(): string[] {
  const [h, setH] = useState(location.hash.slice(1) || "/");
  useEffect(() => {
    const f = () => setH(location.hash.slice(1) || "/");
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return h.split("/").filter(Boolean).map(decodeURIComponent);
}

function GpuBox() {
  const [m, setM] = useState<Machine | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = () => api.machine().then((x) => alive && setM(x)).catch(() => {});
    tick();
    const t = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  if (!m) return null;
  const live = m.gpu.live;
  return (
    <div className="gpu">
      <div>{m.gpu.name || "GPU"}</div>
      {live ? <div>{Math.round(live.used_mib)} / {Math.round(live.total_mib)} MiB · {live.util_pct}% · {live.temp_c} °C</div>
        : <div>nvidia-smi unavailable</div>}
      {m.disk && <div>disk {m.disk.free_gb} GB free</div>}
    </div>
  );
}

function Login({ onDone }: { onDone: () => void }) {
  const [t, setT] = useState("");
  return (
    <div className="panel" style={{ maxWidth: 420, margin: "80px auto" }}>
      <h2>Galley access token</h2>
      <p className="muted small">This server requires the token set in its machine profile or GALLEY_TOKEN.</p>
      <div className="row">
        <input type="password" value={t} onChange={(e) => setT(e.target.value)} style={{ flex: 1 }} />
        <button className="primary" onClick={() => { setToken(t); onDone(); }}>Continue</button>
      </div>
    </div>
  );
}

function App() {
  const r = useRoute();
  const [needLogin, setNeedLogin] = useState(false);
  const [, force] = useState(0);
  useEffect(() => {
    api.scenes().catch((e) => { if (e instanceof ApiError && e.status === 401) setNeedLogin(true); });
  }, []);
  if (needLogin) return <Login onDone={() => { setNeedLogin(false); force((x) => x + 1); }} />;

  const top = r[0] ?? "";
  const nav: [string, string][] = [["", "Overview"], ["new", "New capture"], ["course", "Course editor"], ["svnet", "SV-Net"], ["jobs", "Jobs"], ["configs", "Configs"]];
  let page;
  if (top === "scene" && r[1]) page = <ScenePage scene={r[1]} />;
  else if (top === "jobs" && r[1]) page = <JobPage id={Number(r[1])} />;
  else if (top === "jobs") page = <Jobs />;
  else if (top === "configs") page = <Configs family={r[1] as any} name={r[2]} />;
  else if (top === "new") page = <NewCapture />;
  else if (top === "svnet") page = <SvNetPage cohort={r[1]} />;
  else if (top === "course") page = <Suspense fallback={<p className="muted">Loading the 3D editor…</p>}><CoursePage scene={r[1]} name={r[2]} /></Suspense>;
  else page = <Dashboard />;

  return (
    <div className="shell">
      <nav className="side">
        <div className="brand">Galley</div>
        {nav.map(([k, label]) => (
          <a key={k} href={`#/${k}`} className={top === k || (k === "" && top === "scene") ? "on" : ""}>{label}</a>
        ))}
        <GpuBox />
      </nav>
      <main>{page}</main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
