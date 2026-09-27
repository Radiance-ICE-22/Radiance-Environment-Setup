// SV-Net (Phase 4): SOUS-VIDE's learning half through figs/svnet_pipeline.py.
// rollout → observe → train_hist → train_comm → deploy, one cohort at a time.
import { useEffect, useMemo, useState } from "react";
import {
  active, api, ago, ApiError, CohortStatus, cohortVideoUrl, DeployRow, LossLog, SV_STEPS, SvnetRun, SvStep, svApi,
} from "../api";
import { LineChart } from "../charts";
import { Badge, NumField, Select, usePoll } from "../components";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const STEP_HELP: Record<SvStep, string> = {
  preflight: "environment, configs, scene, disk estimate",
  rollout: "expert flies the courses many times, randomised (long, GPU)",
  observe: "turn rollouts into each pilot's network inputs",
  train_hist: "histNet: flight history → drone parameters",
  train_comm: "regenerate observations, then commNet: image + state → command",
  deploy: "fly expert and students in FiGS; metrics and videos",
};

function useSubmitSv() {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const submit = async (r: SvnetRun) => {
    setBusy(true); setErr(null);
    try {
      const clean = Object.fromEntries(Object.entries(r).filter(([, v]) =>
        v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0))) as unknown as SvnetRun;
      const { id } = await svApi.submit(clean);
      location.hash = `#/jobs/${id}`;
    } catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); } finally { setBusy(false); }
  };
  return { busy, err, submit };
}

export default function SvNetPage({ cohort }: { cohort?: string }) {
  return cohort ? <CohortPage cohort={cohort} /> : <CohortList />;
}

// ── list + new cohort ────────────────────────────────────────────────────────
function CohortList() {
  const list = usePoll(svApi.cohorts, 10000);
  return (
    <>
      <h1>SV-Net</h1>
      <p className="muted">SOUS-VIDE's learning half, run unchanged from the upstream code through <span className="mono">figs/svnet_pipeline.py</span>:
        the expert's rollouts through a splat become training data for a student pilot (histNet + commNet), which is then flown in FiGS.
        A <b>cohort</b> is one experiment: its rollouts, observations and trained pilots live in <span className="mono">SousVide/cohorts/&lt;cohort&gt;/</span>.</p>
      <div className="panel">
        <h2>Cohorts</h2>
        {list.err && <p className="err">{list.err}</p>}
        {list.data?.length === 0 && <p className="muted">None yet.</p>}
        {!!list.data?.length && (
          <table>
            <thead><tr><th>Cohort</th><th>Scene</th><th>Courses</th><th>Method</th><th>Roster</th><th>Steps</th><th>Student tracking error</th></tr></thead>
            <tbody>
              {list.data.map((c) => (
                <tr key={c.cohort} className="click" onClick={() => (location.hash = `#/svnet/${c.cohort}`)}>
                  <td><a href={`#/svnet/${c.cohort}`}>{c.cohort}</a>{!c.managed && <span className="muted small"> · not made here</span>}</td>
                  <td>{c.scene ?? "—"}</td><td>{c.courses?.join(", ") ?? "—"}</td><td>{c.method ?? "—"}</td>
                  <td>{c.roster?.join(", ") ?? "—"}</td><td>{c.done.length}/5</td>
                  <td>{Object.entries(c.students).map(([k, v]) => `${k} ${v} m`).join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <NewCohort existing={list.data?.map((c) => c.cohort) ?? []} />
    </>
  );
}

function NewCohort({ existing }: { existing: string[] }) {
  const scenes = usePoll(api.scenes, 0);
  const courses = usePoll(() => api.configs("courses"), 0);
  const methods = usePoll(() => api.configs("methods"), 0);
  const pilots = usePoll(() => api.configs("pilots"), 0);
  const machine = usePoll(api.machine, 0);
  const d = machine.data?.defaults ?? {};
  const [r, setR] = useState<SvnetRun>({ cohort: "", courses: [], roster: ["Maverick"], method: "data_alpha" });
  const [touchedEval, setTouchedEval] = useState(false);
  const set = (p: Partial<SvnetRun>) => setR({ ...r, ...p });
  const { busy, err, submit } = useSubmitSv();
  useEffect(() => {
    if (!touchedEval && d.svnet_comm_eval) setR((x) => ({ ...x, comm_eval: String(d.svnet_comm_eval) }));
  }, [d.svnet_comm_eval, touchedEval]);

  const students = (pilots.data ?? []).filter((p) => p.kind === "student").map((p) => p.name);
  const experts = (pilots.data ?? []).filter((p) => p.kind === "expert").map((p) => p.name);
  const dataMethods = (methods.data ?? []).map((m) => m.name).filter((m) => m.startsWith("data"));
  const evalMethods = (methods.data ?? []).map((m) => m.name).filter((m) => m.startsWith("eval"));
  const loadable = (scenes.data ?? []).filter((s) => s.loadable).map((s) => s.scene);
  const nameOk = NAME_RE.test(r.cohort) && !existing.includes(r.cohort);
  const ready = nameOk && !!r.scene && !!r.courses?.length && !!r.roster?.length;
  const toggle = (k: "courses" | "roster", v: string) =>
    set({ [k]: (r[k] ?? []).includes(v) ? (r[k] ?? []).filter((x) => x !== v) : [...(r[k] ?? []), v] });
  const maxData = d.max_data_method as string | undefined;
  const tooBig = maxData && r.method && dataMethods.indexOf(r.method) > dataMethods.indexOf(maxData) && dataMethods.includes(maxData);

  return (
    <div className="panel">
      <h2>New cohort</h2>
      <div className="fields">
        <label className="f">Cohort name<input value={r.cohort} placeholder="first" onChange={(e) => set({ cohort: e.target.value.trim() })} />
          <span className={`hint ${r.cohort && !nameOk ? "bad" : ""}`}>{!r.cohort ? "letters, digits, _ and -" : existing.includes(r.cohort) ? "already exists" : NAME_RE.test(r.cohort) ? "ok" : "letters, digits, _ and - only"}</span></label>
        <Select label="Scene (one trained model)" value={r.scene} options={loadable} allowDefault={false} onChange={(v) => set({ scene: v })} />
        <Select label="Rollout method" value={r.method} options={dataMethods.length ? dataMethods : ["data_alpha"]} allowDefault={false}
          onChange={(v) => set({ method: v })} hint={tooBig ? `larger than this machine's ${maxData}` : "alpha ≈ 8 GB, beta ≈ 88 GB, gamma ≈ 265 GB per 16 s course"} />
        <Select label="Expert" value={r.expert} options={experts} onChange={(v) => set({ expert: v })} hint="default Viper" />
      </div>
      <h3>Courses</h3>
      <div className="checks">
        {courses.data?.map((c) => (
          <label key={c.name}><input type="checkbox" checked={r.courses?.includes(c.name) ?? false} onChange={() => toggle("courses", c.name)} />
            <span className="mono">{c.name}</span></label>
        ))}
      </div>
      <p className="muted small">Pick courses that have flown cleanly in this scene (Scene page or course editor): rollouts that end more than
        tol_select (5 cm for data_*) off the path are discarded.</p>
      <h3>Student pilots</h3>
      <div className="checks">
        {students.map((p) => (
          <label key={p}><input type="checkbox" checked={r.roster?.includes(p) ?? false} onChange={() => toggle("roster", p)} />
            <span className="mono">{p}</span></label>
        ))}
      </div>
      <h3>Training and evaluation</h3>
      <div className="fields">
        <NumField label="histNet epochs" value={r.hist_epochs} placeholder="200" min={1} onChange={(v) => set({ hist_epochs: v })} />
        <NumField label="commNet epochs" value={r.comm_epochs} placeholder="300" min={1} onChange={(v) => set({ comm_epochs: v })} />
        <NumField label="Save every (epochs)" value={r.lim_sv} placeholder="50" min={1} onChange={(v) => set({ lim_sv: v })} />
        <Select label="commNet in-loop evaluation" value={r.comm_eval} options={["none", ...evalMethods]}
          onChange={(v) => { setTouchedEval(true); set({ comm_eval: v }); }} hint="notebook: eval_nominal (10 full-course flights per save)" />
        <Select label="Final evaluation" value={r.deploy_method} options={evalMethods} onChange={(v) => set({ deploy_method: v })} hint="default eval_nominal" />
      </div>
      {err && <p className="err">{err}</p>}
      <div className="row" style={{ marginTop: 12 }}>
        <button disabled={busy || !ready} title="Checks the environment and prints the data-size estimate; flies nothing"
          onClick={() => submit({ ...r, only: "preflight" })}>Preflight only (size estimate)</button>
        <button className="primary" disabled={busy || !ready} onClick={() => submit(r)}>Queue the full run</button>
        <span className="muted small">The cohort remembers these settings; later runs need only the cohort name.</span>
      </div>
    </div>
  );
}

// ── one cohort ───────────────────────────────────────────────────────────────
function CohortPage({ cohort }: { cohort: string }) {
  const jobs = usePoll(() => svApi.jobs(cohort), 4000, [cohort]);
  const running = jobs.data?.some((j) => active(j.status)) ?? false;
  const st = usePoll(() => svApi.cohort(cohort), running ? 4000 : 15000, [cohort, running]);
  const c = st.data;
  const res = c?.results ?? {};
  const cfg = c?.config ?? {};

  return (
    <>
      <div className="row"><h1>{cohort}</h1><span className="spacer" />
        {cfg.scene && <a href={`#/scene/${cfg.scene}`}>Scene {cfg.scene} →</a>}<a href="#/svnet">← SV-Net</a></div>
      {st.err && <p className="err">{st.err}</p>}
      {c && (
        <div className="panel">
          <div className="row"><h2 style={{ margin: 0 }}>Steps</h2><span className="spacer" />
            <span className="muted small mono">{c.data_dir}</span></div>
          <div className="steps steps6" style={{ marginTop: 10 }}>
            {c.steps.map((x) => (
              <div key={x.step} className={`step ${x.done ? "done" : ""}`} title={STEP_HELP[x.step]}>
                <b>{x.step}</b>
                <span className={x.done ? "ok" : "muted"}>{x.step === "preflight" ? "runs every time" : x.done ? (x.when ?? "done").replace("T", " ") : "not done"}</span>
              </div>
            ))}
          </div>
          <p className="small" style={{ marginTop: 10 }}>
            <span className="mono">{cfg.scene}</span> · courses <span className="mono">{(cfg.courses ?? []).join(", ")}</span> ·
            {" "}<span className="mono">{cfg.method}</span> · expert <span className="mono">{cfg.expert}</span> · roster <span className="mono">{(cfg.roster ?? []).join(", ")}</span> ·
            {" "}histNet {cfg.hist_epochs} / commNet {cfg.comm_epochs} epochs · in-loop eval <span className="mono">{cfg.comm_eval}</span> · final eval <span className="mono">{cfg.deploy_method}</span> on <span className="mono">{cfg.deploy_course}</span>
          </p>
          <p className="muted small">Disk: {!c.exists ? "no data yet" : Object.entries(c.disk_gb).filter(([k, v]) => v > 0 || k !== "_archive").map(([k, v]) => `${k} ${v < 0.01 ? "<0.01" : v} GB`).join(" · ")}
            {res.estimate && ` · estimate for rollouts ~${res.estimate.rollout_gb} GB (${res.estimate.free_gb} GB free when checked)`}</p>
        </div>
      )}

      <RunPanel cohort={cohort} cfg={cfg} disabled={running} />

      {res.rollout && (
        <div className="panel">
          <h2>Rollouts</h2>
          <table>
            <thead><tr><th>Course</th><th>Rollouts kept</th><th>Samples</th><th>Files</th><th>Size</th></tr></thead>
            <tbody>{Object.entries(res.rollout.courses).map(([k, v]) => (
              <tr key={k}><td className="mono">{k}</td><td>{v.rollouts}</td><td>{v.samples.toLocaleString()}</td><td>{v.files}</td><td>{v.gb} GB</td></tr>))}</tbody>
          </table>
          <p className="muted small">{res.rollout.wallclock} · peak {vram(res.rollout.peak_vram_mib)}
            {res.estimate && ` · estimate was ~${res.estimate.rollouts} rollouts; the difference is what tol_select discarded (plus the expert's re-timing)`}</p>
        </div>
      )}

      {c && (Object.keys(c.live).length > 0 || res.train_histNet || res.train_commNet) && (
        <div className="grid2">
          {(["histNet", "commNet"] as const).map((net) => (
            <TrainingPanel key={net} net={net} status={c} running={running} />
          ))}
        </div>
      )}

      {res.deploy && <DeployPanel cohort={cohort} d={res.deploy} />}

      <div className="panel">
        <h2>Jobs for this cohort</h2>
        <table>
          <thead><tr><th>#</th><th>Job</th><th>Status</th><th>Created</th></tr></thead>
          <tbody>
            {jobs.data?.map((j) => (
              <tr key={j.id} className="click" onClick={() => (location.hash = `#/jobs/${j.id}`)}>
                <td>{j.id}</td><td>{j.label}</td><td><Badge s={j.status} /></td><td className="muted">{ago(j.created)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function RunPanel({ cohort, cfg, disabled }: { cohort: string; cfg: Record<string, any>; disabled: boolean }) {
  const methods = usePoll(() => api.configs("methods"), 0);
  const evalMethods = (methods.data ?? []).map((m) => m.name).filter((m) => m.startsWith("eval"));
  const [r, setR] = useState<SvnetRun>({ cohort });
  const set = (p: Partial<SvnetRun>) => setR({ ...r, ...p });
  const { busy, err, submit } = useSubmitSv();
  const [open, setOpen] = useState(false);
  return (
    <div className="panel">
      <div className="row">
        <h2 style={{ margin: 0 }}>Run</h2><span className="spacer" />
        <button className="primary" disabled={busy || disabled} onClick={() => submit({ cohort })}
          title="Runs every step that is not done yet (or whose inputs changed)">Continue</button>
        <button disabled={busy || disabled} onClick={() => setOpen(!open)}>{open ? "Hide options" : "Steps and options…"}</button>
      </div>
      {disabled && <p className="muted small">A job for this cohort is queued or running.</p>}
      {open && (
        <>
          <div className="fields" style={{ marginTop: 10 }}>
            <Select label="Only" value={r.only} options={SV_STEPS} onChange={(v) => set({ only: v, from_step: undefined, stop_after: undefined })} />
            <Select label="From" value={r.from_step} options={SV_STEPS} onChange={(v) => set({ from_step: v, only: undefined })} />
            <Select label="Stop after" value={r.stop_after} options={SV_STEPS} onChange={(v) => set({ stop_after: v, only: undefined })} />
            <NumField label="histNet epochs" value={r.hist_epochs} placeholder={String(cfg.hist_epochs ?? 200)} onChange={(v) => set({ hist_epochs: v })} />
            <NumField label="commNet epochs" value={r.comm_epochs} placeholder={String(cfg.comm_epochs ?? 300)} onChange={(v) => set({ comm_epochs: v })} />
            <Select label="commNet in-loop evaluation" value={r.comm_eval} options={["none", ...evalMethods]} onChange={(v) => set({ comm_eval: v })}
              hint={`now ${cfg.comm_eval ?? "eval_nominal"}`} />
            <Select label="Final evaluation" value={r.deploy_method} options={evalMethods} onChange={(v) => set({ deploy_method: v })}
              hint={`now ${cfg.deploy_method ?? "eval_nominal"}`} />
          </div>
          <h3>Force redo</h3>
          <div className="checks">
            {SV_STEPS.filter((x) => x !== "preflight").map((x) => (
              <label key={x}><input type="checkbox" checked={r.redo?.includes(x) ?? false}
                onChange={(e) => set({ redo: e.target.checked ? [...(r.redo ?? []), x] : (r.redo ?? []).filter((y) => y !== x) })} />
                <span className="mono">{x}</span></label>
            ))}
          </div>
          <h3>Start a network from scratch</h3>
          <div className="checks">
            {(["histNet", "commNet"] as const).map((n) => (
              <label key={n}><input type="checkbox" checked={r.fresh?.includes(n) ?? false}
                onChange={(e) => set({ fresh: e.target.checked ? [...(r.fresh ?? []), n] : (r.fresh ?? []).filter((y) => y !== n) })} />
                <span className="mono">{n}</span></label>
            ))}
          </div>
          <p className="muted small">Upstream keeps training a network that already exists (its .pt is loaded). “From scratch” moves it to
            <span className="mono"> roster/&lt;pilot&gt;/_archive/</span> first. Re-running <span className="mono">rollout</span> or <span className="mono">observe</span> moves the old
            data to <span className="mono">cohorts/{cohort}/_archive/</span>; nothing is deleted. Re-running a step re-runs everything after it.</p>
          {err && <p className="err">{err}</p>}
          <button className="primary" disabled={busy || disabled} onClick={() => submit({ ...r, cohort })}>Queue</button>
        </>
      )}
      {!open && err && <p className="err">{err}</p>}
    </div>
  );
}

function TrainingPanel({ net, status, running }: { net: "histNet" | "commNet"; status: CohortStatus; running: boolean }) {
  const saved = status.results[`train_${net}`]?.pilots ?? {};
  const pilots = [...new Set([...Object.keys(saved), ...Object.keys(status.live).filter((p) => status.live[p][net])])];
  const [pick, setPick] = useState<string | null>(null);
  const [log, setLog] = useState(true);
  const p = pick && pilots.includes(pick) ? pick : pilots[0];
  const s: LossLog | undefined = p ? saved[p] : undefined;
  const live = p ? status.live[p]?.[net] ?? [] : [];
  // after a run, prefer the saved log (train + test); while training, the live per-epoch file
  const useSaved = !!s && !(running && live.length > (s.loss_train?.length ?? 0));
  const train = useSaved ? s!.loss_train : live;
  const r = status.results[`train_${net}`];
  const evalPts = s?.eval_tte_upstream ?? [];
  return (
    <div className="panel">
      <div className="row">
        <h2 style={{ margin: 0 }}>{net}</h2>
        <span className="muted small">{net === "histNet" ? "history → [m, kt]" : "image + state + histNet features → command"}</span>
        <span className="spacer" />
        {pilots.length > 1 && <select value={p} onChange={(e) => setPick(e.target.value)}>{pilots.map((x) => <option key={x}>{x}</option>)}</select>}
        <label className="check small"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log</label>
      </div>
      {!p ? <p className="muted small">Not trained yet.</p> : (
        <>
          <LineChart points={train} xLabel="epoch" yLabel="loss" log={log}
            second={useSaved && s!.loss_test.length ? { points: s!.loss_test, label: "test loss", firstLabel: "train loss" } : undefined} />
          <p className="small">
            {useSaved
              ? <>{s!.epochs} epochs · train {fmtL(s!.loss_train)} · test {fmtL(s!.loss_test)} · {s!.n_train?.toLocaleString()} / {s!.n_test?.toLocaleString()} samples (train / test){s!.n_logs > 1 && ` · ${s!.n_logs} training runs logged; showing the latest`}</>
              : <>live: epoch {live.length ? live[live.length - 1][0] : 0}, train loss {live.length ? live[live.length - 1][1].toPrecision(4) : "—"} (test loss is logged every save)</>}
          </p>
          {r && <p className="muted small">{r.wallclock} · peak {vram(r.peak_vram_mib)}{net === "commNet" && (r as any).deployment ? ` · in-loop evaluation ${(r as any).deployment[2]} on ${(r as any).deployment[0]}` : ""}</p>}
          {evalPts.length > 0 && (
            <p className="muted small">Upstream in-loop evaluation (its TTE, which picks the kept checkpoint): {evalPts.map(([e, v]) => `${e}: ${v.toFixed(2)}`).join(" · ")}</p>
          )}
        </>
      )}
    </div>
  );
}
const vram = (m: number) => (m >= 0 ? `${m} MiB` : "— (no nvidia-smi)");
const fmtL = (pts: [number, number][]) => (pts.length ? pts[pts.length - 1][1].toPrecision(4) : "—");

function DeployPanel({ cohort, d }: { cohort: string; d: NonNullable<CohortStatus["results"]["deploy"]> }) {
  const rows = useMemo(() => Object.entries(d.pilots) as [string, DeployRow][], [d]);
  return (
    <div className="panel">
      <h2>Evaluation in FiGS</h2>
      <p className="muted small"><span className="mono">{d.method}</span> on <span className="mono">{d.course}</span> in <span className="mono">{d.scene}</span> ·
        {" "}{d.finished?.replace("T", " ")} · {d.wallclock} · peak {vram(d.peak_vram_mib)}</p>
      <table>
        <thead><tr><th>Pilot</th><th>Tracking error mean</th><th>max</th><th>within 0.3 m</th><th>final</th><th>Rollouts</th>
          <th className="muted">upstream TTE</th><th className="muted">upstream PP</th><th>Hz mean / worst</th></tr></thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name}>
              <td><b>{name}</b> <span className="muted small">{r.role}</span></td>
              <td>{r.tte.mean_m} m</td><td>{r.tte.max_m} m</td><td>{Math.round(r.tte["within_0.3m"] * 100)}%</td>
              <td>{r.tte.final_mean_m} m</td><td>{r.tte.rollouts}</td>
              <td className="muted">{r.upstream_tte_mean}</td><td className="muted">{r.upstream_pp}</td>
              <td>{r.hz_mean} / {r.hz_worst}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">Tracking error: for every step, the distance from the drone to the nearest point of the expert's planned path
        (as rollout_generator's tol_select check measures it). The upstream TTE/PP columns come from sousvide's compute_flight_metrics, which takes
        the norm over the wrong axis (whole path per axis, then the smallest axis): they are shown for comparison with the paper's tables, not as distances.</p>
      <div className="videos">
        {rows.filter(([, r]) => r.video).map(([name, r]) => (
          <figure key={name}>
            <video controls src={cohortVideoUrl(cohort, r.video!.split("/").pop()!)} />
            <figcaption>{name} ({r.role}) · last rollout</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
