// SV-Net (Phase 4): SOUS-VIDE's learning half through figs/svnet_pipeline.py.
// rollout → observe → train_hist → train_comm → deploy, one cohort at a time.
// The "SV-Net" document is the new-cohort form; each cohort is its own document.
import { useEffect, useMemo, useState } from "react";
import {
  active, api, ApiError, CohortStatus, cohortVideoUrl, DeployRow, LossLog, SV_STEPS, SvnetRun, SvStep, svApi,
} from "../api";
import { LineChart } from "../charts";
import { NumField, Select, usePoll, useShowJob } from "../components";
import { Problem, ToProblems, ToProperties, useCommands, useUi } from "../shell/core";
import { useAppData } from "../shell/data";
import { Icon } from "../shell/icons";
import { Pill, Prop, PropSection, QueueTable, StepStrip, Tile } from "../shell/Panes";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const DUP_KEY = "galley.svdup";
const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
const STEP_HELP: Record<SvStep, string> = {
  preflight: "environment, configs, scene, disk estimate (runs every time)",
  rollout: "expert flies the courses many times, randomised (long, GPU)",
  observe: "turn rollouts into each pilot's network inputs",
  train_hist: "histNet: flight history → drone parameters",
  train_comm: "regenerate observations, then commNet: image + state → command",
  deploy: "fly expert and students in FiGS; metrics and videos",
};
const GB: Record<string, string> = { data_alpha: "~6 GB", data_beta: "~88 GB", data_gamma: "~265 GB" };

function useSubmitSv() {
  const show = useShowJob();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const submit = async (r: SvnetRun) => {
    setBusy(true); setErr(null);
    try {
      const clean = Object.fromEntries(Object.entries(r).filter(([, v]) =>
        v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0))) as unknown as SvnetRun;
      const { id } = await svApi.submit(clean);
      show(id);
      return id;
    } catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); return null; } finally { setBusy(false); }
  };
  return { busy, err, submit };
}

export default function SvNetPage({ cohort }: { cohort?: string }) {
  return cohort ? <CohortPage cohort={cohort} /> : <NewCohort />;
}

// ── new cohort (with the cohort list) ────────────────────────────────────────
function NewCohort() {
  const d = useAppData();
  const methods = usePoll(() => api.configs("methods"), 0);
  const pilots = usePoll(() => api.configs("pilots"), 0);
  const def = d.machine?.defaults ?? {};
  const [r, setR] = useState<SvnetRun>(() => {
    try { const dup = sessionStorage.getItem(DUP_KEY); if (dup) { sessionStorage.removeItem(DUP_KEY); return JSON.parse(dup); } } catch { /* */ }
    return { cohort: "", courses: [], roster: ["Maverick"], method: "data_alpha" };
  });
  const [touchedEval, setTouchedEval] = useState(!!r.comm_eval);
  const set = (p: Partial<SvnetRun>) => setR((x) => ({ ...x, ...p }));
  const { busy, err, submit } = useSubmitSv();
  useEffect(() => {
    if (!touchedEval && def.svnet_comm_eval) setR((x) => ({ ...x, comm_eval: String(def.svnet_comm_eval) }));
  }, [def.svnet_comm_eval, touchedEval]);

  const existing = d.cohorts.map((c) => c.cohort);
  const students = (pilots.data ?? []).filter((p) => p.kind === "student").map((p) => p.name);
  const experts = (pilots.data ?? []).filter((p) => p.kind === "expert").map((p) => p.name);
  const dataMethods = (methods.data ?? []).map((m) => m.name).filter((m) => m.startsWith("data"));
  const evalMethods = (methods.data ?? []).map((m) => m.name).filter((m) => m.startsWith("eval"));
  const loadable = d.scenes.filter((s) => s.loadable).map((s) => s.scene);
  const nameOk = NAME_RE.test(r.cohort) && !existing.includes(r.cohort);
  const ready = nameOk && !!r.scene && !!r.courses?.length && !!r.roster?.length;
  const why = !r.cohort ? "Name the cohort first." : !nameOk ? "Fix the cohort name." : !r.scene ? "Choose a scene." : !r.courses?.length ? "Pick at least one course." : !r.roster?.length ? "Pick at least one student." : false;
  const toggle = (k: "courses" | "roster", v: string) =>
    set({ [k]: (r[k] ?? []).includes(v) ? (r[k] ?? []).filter((x) => x !== v) : [...(r[k] ?? []), v] });
  const maxData = def.max_data_method as string | undefined;
  const tooBig = !!(maxData && r.method && dataMethods.includes(maxData) && dataMethods.indexOf(r.method) > dataMethods.indexOf(maxData));

  useCommands({
    "run": { run: () => submit(r), disabled: busy ? "Submitting…" : why },
    "sv.preflight": { run: () => submit({ ...r, only: "preflight" }), disabled: why },
    "sv.estimate": { run: () => submit({ ...r, only: "preflight" }), disabled: why },
    "sv.method": { value: r.method ?? "", options: (dataMethods.length ? dataMethods : ["data_alpha"]).map((m) => [m, `${m} ${GB[m] ?? ""}`] as [string, string]), set: (v) => set({ method: v }) },
    "sv.scene": { value: r.scene ?? "", options: loadable, set: (v) => set({ scene: v || undefined }) },
    "sv.nro": { value: r.nro_ds ?? "", set: (v) => set({ nro_ds: num(v) }) },
    "sv.compress": { checked: !!r.use_compress, set: (v) => set({ use_compress: v === "true" }) },
    "sv.subsample": { value: r.subsample ?? "", set: (v) => set({ subsample: num(v) }) },
    "sv.hist": { value: r.hist_epochs ?? "", set: (v) => set({ hist_epochs: num(v) }) },
    "sv.comm": { value: r.comm_epochs ?? "", set: (v) => set({ comm_epochs: num(v) }) },
    "sv.batch": { value: r.batch_size ?? "", set: (v) => set({ batch_size: num(v) }) },
    "sv.lr": { value: r.lr ?? "", set: (v) => set({ lr: num(v) }) },
    "sv.eval": { value: r.comm_eval ?? "", options: [["", "default"], "none", ...evalMethods], set: (v) => { setTouchedEval(true); set({ comm_eval: v || undefined }); } },
    "sv.fresh": { disabled: "A new cohort has no networks yet." },
  });

  const probs: Problem[] = [
    ...(r.cohort && !nameOk ? [{ severity: "error" as const, where: "cohort name", message: existing.includes(r.cohort) ? "A cohort with this name exists already." : "Letters, digits, _ and - only." }] : []),
    ...(loadable.length === 0 ? [{ severity: "error" as const, where: "scene", message: "No scene has exactly one trained model." }] : []),
    ...(tooBig ? [{ severity: "warning" as const, where: "method", message: `${r.method} is larger than this machine's ${maxData} (machine profile max_data_method).` }] : []),
    ...(r.method && GB[r.method] && d.machine?.disk ? [{ severity: "info" as const, where: "disk", message: `${r.method} needs ${GB[r.method]} per course; ${d.machine.disk.free_gb} GB free.` }] : []),
    ...(why && r.cohort ? [{ severity: "info" as const, where: "new cohort", message: why }] : []),
  ];

  return (
    <>
      <Tile title="Cohorts" icon="cohort" meta="SousVide/cohorts/<cohort>/ · click to open" className="flush">
        <table>
          <thead><tr><th>cohort</th><th>scene</th><th>courses</th><th>method</th><th>roster</th><th>steps</th><th>student tracking error</th></tr></thead>
          <tbody>
            {d.cohorts.map((c) => (
              <tr key={c.cohort} className="click" onClick={() => (location.hash = `#/svnet/${c.cohort}`)}>
                <td><b>{c.cohort}</b>{!c.managed && <span className="muted small"> · not made here</span>}</td>
                <td>{c.scene ?? "—"}</td><td>{c.courses?.join(", ") ?? "—"}</td><td>{c.method ?? "—"}</td>
                <td>{c.roster?.join(", ") ?? "—"}</td><td>{c.done.length}/5</td>
                <td className="mono">{Object.entries(c.students).map(([k, v]) => `${k} ${v} m`).join(", ") || "—"}</td>
              </tr>))}
            {d.cohorts.length === 0 && <tr><td colSpan={7} className="muted">None yet.</td></tr>}
          </tbody>
        </table>
      </Tile>

      <div className="tiles cols-3">
        <Tile title="New cohort" icon="cohort" meta="a cohort is one experiment">
          <div className="fields" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <label className="f">Cohort name<input value={r.cohort} placeholder="p4_beta" onChange={(e) => set({ cohort: e.target.value.trim() })} />
              <span className={`hint ${r.cohort && !nameOk ? "bad" : ""}`}>{!r.cohort ? "letters, digits, _ and -" : existing.includes(r.cohort) ? "already exists" : NAME_RE.test(r.cohort) ? "ok" : "letters, digits, _ and - only"}</span></label>
            <Select label="Scene (one trained model)" value={r.scene} options={loadable} allowDefault={false} onChange={(v) => set({ scene: v })} />
            <Select label="Rollout method" value={r.method} options={dataMethods.length ? dataMethods : ["data_alpha"]} allowDefault={false}
              onChange={(v) => set({ method: v })} hint={tooBig ? `larger than this machine's ${maxData}` : "alpha ~6 GB, beta ~88 GB, gamma ~265 GB"} />
            <Select label="Expert" value={r.expert} options={experts} onChange={(v) => set({ expert: v })} hint="default Viper" />
          </div>
        </Tile>
        <Tile title="Courses" icon="route" meta={`${r.courses?.length ?? 0} selected`}>
          <div className="checks">
            {d.courses.map((c) => (
              <label key={c.name}><input type="checkbox" checked={r.courses?.includes(c.name) ?? false} onChange={() => toggle("courses", c.name)} /><span className="mono">{c.name}</span></label>))}
          </div>
          <p className="muted small" style={{ marginTop: 6 }}>Pick courses that flew cleanly in this scene: rollouts that end more than tol_select (5 cm for data_*) off the path are discarded.</p>
          <h4>Student pilots</h4>
          <div className="checks">
            {students.map((p) => (
              <label key={p}><input type="checkbox" checked={r.roster?.includes(p) ?? false} onChange={() => toggle("roster", p)} /><span className="mono">{p}</span></label>))}
          </div>
        </Tile>
        <Tile title="Training and evaluation" icon="chart">
          <div className="fields">
            <NumField label="histNet epochs" value={r.hist_epochs} placeholder="200" min={1} onChange={(v) => set({ hist_epochs: v })} />
            <NumField label="commNet epochs" value={r.comm_epochs} placeholder="300" min={1} onChange={(v) => set({ comm_epochs: v })} />
            <NumField label="Save every (epochs)" value={r.lim_sv} placeholder="50" min={1} onChange={(v) => set({ lim_sv: v })} />
            <Select label="commNet in-loop evaluation" value={r.comm_eval} options={["none", ...evalMethods]}
              onChange={(v) => { setTouchedEval(true); set({ comm_eval: v }); }} hint="eval_nominal: 10 flights per save" />
            <Select label="Final evaluation" value={r.deploy_method} options={evalMethods} onChange={(v) => set({ deploy_method: v })} hint="default eval_nominal" />
          </div>
          {err && <p className="err">{err}</p>}
          <div className="row" style={{ marginTop: 8 }}>
            <button disabled={busy || !ready} onClick={() => submit({ ...r, only: "preflight" })}>Preflight only</button>
            <button className="primary" disabled={busy || !ready} onClick={async () => { const id = await submit(r); if (id) setTimeout(() => (location.hash = `#/svnet/${r.cohort}`), 300); }}>Queue the full run (F5)</button>
          </div>
        </Tile>
      </div>

      <ToProperties>
        <div className="props-title"><Icon name="cohort" size={16} />New cohort{r.cohort ? `: ${r.cohort}` : ""}</div>
        <PropSection title="Data">
          <Prop k="Scene">{r.scene ?? "—"}</Prop><Prop k="Courses">{r.courses?.join(", ") || "—"}</Prop>
          <Prop k="Method" tone={tooBig ? "warn" : undefined}>{r.method} {GB[r.method ?? ""] ?? ""}</Prop>
          <Prop k="Expert">{r.expert ?? "Viper (default)"}</Prop><Prop k="Rollouts/save">{r.nro_ds ?? "default"}</Prop>
          <Prop k="Compress">{r.use_compress ? "yes" : "default"}</Prop><Prop k="Subsample">{r.subsample ?? "default"}</Prop>
        </PropSection>
        <PropSection title="Training">
          <Prop k="Roster">{r.roster?.join(", ") || "—"}</Prop><Prop k="histNet epochs">{r.hist_epochs ?? "200"}</Prop><Prop k="commNet epochs">{r.comm_epochs ?? "300"}</Prop>
          <Prop k="Batch">{r.batch_size ?? "default"}</Prop><Prop k="LR">{r.lr ?? "default"}</Prop>
          <Prop k="In-loop eval">{r.comm_eval ?? "default"}</Prop><Prop k="Final eval">{r.deploy_method ?? "eval_nominal"}</Prop>
        </PropSection>
        <PropSection title="Machine"><Prop k="Disk free">{d.machine?.disk?.free_gb ?? "—"} GB</Prop><Prop k="Max data method">{maxData ?? "—"}</Prop></PropSection>
      </ToProperties>
      <ToProblems items={probs} />
    </>
  );
}

// ── one cohort ───────────────────────────────────────────────────────────────
function CohortPage({ cohort }: { cohort: string }) {
  const d = useAppData();
  const { ui, setUi } = useUi();
  const jobs = d.jobs.filter((j) => j.kind === "svnet" && (j.params as any)?.cohort === cohort);
  const running = jobs.some((j) => active(j.status));
  const st = usePoll(() => svApi.cohort(cohort), running ? 4000 : 15000, [cohort, running]);
  const methods = usePoll(() => api.configs("methods"), 0);
  const evalMethods = (methods.data ?? []).map((m) => m.name).filter((m) => m.startsWith("eval"));
  const c = st.data;
  const res = c?.results ?? {};
  const cfg = c?.config ?? {};
  const { busy, err, submit } = useSubmitSv();
  const [o, setO] = useState<SvnetRun & { redoFrom?: boolean }>({ cohort });
  const setOpt = (p: Partial<typeof o>) => setO((x) => ({ ...x, ...p }));
  const lock = running ? "A job for this cohort is queued or running." : false;
  const go = (extra: Partial<SvnetRun> = {}) => {
    const { redoFrom, ...rest } = o;
    const from = extra.from_step ?? rest.from_step;
    submit({ ...rest, ...extra, cohort, redo: redoFrom && from ? [from] : extra.redo ?? rest.redo });
  };
  const deploy = res.deploy;
  const rows = useMemo(() => (deploy ? (Object.entries(deploy.pilots) as [string, DeployRow][]) : []), [deploy]);
  const vid = (role: "expert" | "student") => rows.find(([, r]) => r.role === role && r.video)?.[1].video;
  const openVid = (v?: string | null) => v && window.open(cohortVideoUrl(cohort, v.split("/").pop()!), "_blank");
  const csv = () => {
    const head = ["pilot", "role", "tte_mean_m", "tte_max_m", "within_0.3m", "final_mean_m", "rollouts", "upstream_tte_mean", "upstream_pp", "hz_mean", "hz_worst"];
    const body = rows.map(([n, r]) => [n, r.role, r.tte.mean_m, r.tte.max_m, r.tte["within_0.3m"], r.tte.final_mean_m, r.tte.rollouts, r.upstream_tte_mean, r.upstream_pp, r.hz_mean, r.hz_worst].join(","));
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([[head.join(","), ...body].join("\n") + "\n"], { type: "text/csv" })); a.download = `${cohort}_deploy.csv`; a.click();
  };

  useCommands({
    "run": { run: () => go(), disabled: lock || (busy && "Submitting…") },
    "sv.continue": { run: () => submit({ cohort }), disabled: lock || (busy && "Submitting…") },
    "sv.preflight": { run: () => submit({ cohort, only: "preflight" }), disabled: lock },
    "sv.duplicate": { run: () => {
      const n = prompt("Name of the new cohort (same settings):", `${cohort}_2`)?.trim(); if (!n) return;
      const keep: SvnetRun = { cohort: n, scene: cfg.scene, courses: cfg.courses, method: cfg.method, roster: cfg.roster, expert: cfg.expert, hist_epochs: cfg.hist_epochs,
        comm_epochs: cfg.comm_epochs, comm_eval: cfg.comm_eval, deploy_method: cfg.deploy_method, lim_sv: cfg.lim_sv };
      try { sessionStorage.setItem(DUP_KEY, JSON.stringify(keep)); } catch { /* */ }
      location.hash = "#/svnet";
    }, disabled: !c && "Loading…" },
    "sv.from": { value: o.from_step ?? "", options: [["", "first unfinished"], ...SV_STEPS], set: (v) => setOpt({ from_step: (v || undefined) as SvStep }) },
    "sv.to": { value: o.stop_after ?? "", options: [["", "last"], ...SV_STEPS], set: (v) => setOpt({ stop_after: (v || undefined) as SvStep }) },
    "sv.redo": { checked: !!o.redoFrom, set: (v) => setOpt({ redoFrom: v === "true" }) },
    "sv.hist": { value: o.hist_epochs ?? "", set: (v) => setOpt({ hist_epochs: num(v) }) },
    "sv.comm": { value: o.comm_epochs ?? "", set: (v) => setOpt({ comm_epochs: num(v) }) },
    "sv.batch": { value: o.batch_size ?? "", set: (v) => setOpt({ batch_size: num(v) }) },
    "sv.lr": { value: o.lr ?? "", set: (v) => setOpt({ lr: num(v) }) },
    "sv.eval": { value: o.comm_eval ?? "", options: [["", `now ${cfg.comm_eval ?? "default"}`], "none", ...evalMethods], set: (v) => setOpt({ comm_eval: v || undefined }) },
    "sv.fresh": { checked: !!o.fresh?.length, set: (v) => setOpt({ fresh: v === "true" ? ["histNet", "commNet"] : undefined }) },
    "sv.deploy": { run: () => submit({ cohort, only: "deploy", redo: ["deploy"] }), disabled: lock || (!c?.steps.find((x) => x.step === "train_comm")?.done && "Train commNet first.") },
    "sv.vexpert": vid("expert") ? { run: () => openVid(vid("expert")) } : { disabled: "No expert deployment video yet." },
    "sv.vstudent": vid("student") ? { run: () => openVid(vid("student")) } : { disabled: "No student deployment video yet." },
    "sv.csv": rows.length ? { run: csv } : { disabled: "Deploy first." },
    "sv.method": { value: cfg.method ?? "", disabled: "Fixed for this cohort: duplicate it to change the data." },
    "sv.scene": { value: cfg.scene ?? "", disabled: "Fixed for this cohort." },
  });

  const probs: Problem[] = [
    ...(st.err ? [{ severity: "error" as const, where: cohort, message: st.err }] : []),
    ...(res.estimate && res.estimate.rollout_gb > res.estimate.free_gb ? [{ severity: "error" as const, where: "disk", message: `Rollouts need ~${res.estimate.rollout_gb} GB; ${res.estimate.free_gb} GB free.` }] : []),
    ...rows.filter(([, r]) => r.role === "student" && r.tte.mean_m > 0.5).map(([n, r]) => ({ severity: "warning" as const, where: `deploy ${n}`, message: `Student tracking error ${r.tte.mean_m} m mean (${Math.round(r.tte["within_0.3m"] * 100)}% within 0.3 m). Check the video; more data (data_beta) usually helps.` })),
    ...jobs.filter((j) => j.status === "failed").slice(0, 5).map((j) => ({ severity: "info" as const, where: `job #${j.id}`, message: `${j.label} failed.` })),
  ];
  const runningJob = jobs.find((j) => j.status === "running");

  return (
    <>
      <Tile title="Steps" icon="steps" meta={<>{c?.data_dir}{runningJob && <> · <Pill s="running" label={`#${runningJob.id}`} /></>}</>}>
        {c ? <StepStrip steps={c.steps.map((x) => ({ name: x.step, state: x.done ? "done" : "", tip: STEP_HELP[x.step],
          sub: x.step === "preflight" ? "every run" : x.done ? (x.when ?? "done").replace("T", " ").slice(5, 16) : "not done",
          range: o.from_step ? SV_STEPS.indexOf(x.step) >= SV_STEPS.indexOf(o.from_step) && (!o.stop_after || SV_STEPS.indexOf(x.step) <= SV_STEPS.indexOf(o.stop_after)) : false }))}
          onPick={(n) => setOpt({ from_step: n as SvStep })} /> : <p className="muted">{st.err ?? "Loading…"}</p>}
        {c && <p className="small" style={{ marginTop: 4 }}>
          <span className="mono">{cfg.scene}</span> · courses <span className="mono">{(cfg.courses ?? []).join(", ")}</span> · <span className="mono">{cfg.method}</span> ·
          {" "}expert <span className="mono">{cfg.expert}</span> · roster <span className="mono">{(cfg.roster ?? []).join(", ")}</span> · histNet {cfg.hist_epochs} / commNet {cfg.comm_epochs} epochs ·
          {" "}disk {!c.exists ? "no data yet" : Object.entries(c.disk_gb).filter(([k, v]) => v > 0 || k !== "_archive").map(([k, v]) => `${k} ${v < 0.01 ? "<0.01" : v} GB`).join(" · ")}
        </p>}
        {err && <p className="err">{err}</p>}
      </Tile>

      <div className="tiles cols-3">
        <Tile title="Rollouts" icon="data" className="flush" meta={res.rollout ? `${res.rollout.wallclock} · peak ${vram(res.rollout.peak_vram_mib)}` : undefined}>
          {res.rollout ? (
            <table>
              <thead><tr><th>course</th><th className="num">kept</th><th className="num">samples</th><th className="num">files</th><th className="num">GB</th></tr></thead>
              <tbody>{Object.entries(res.rollout.courses).map(([k, v]) => (
                <tr key={k}><td className="mono">{k}</td><td className="num">{v.rollouts}</td><td className="num">{v.samples.toLocaleString()}</td><td className="num">{v.files}</td><td className="num">{v.gb}</td></tr>))}</tbody>
            </table>) : <p className="empty pad">No rollouts yet.</p>}
          {res.estimate && <p className="muted small" style={{ padding: "4px 8px" }}>Estimate ~{res.estimate.rollouts} rollouts, ~{res.estimate.rollout_gb} GB; the difference is what tol_select discarded.</p>}
        </Tile>
        {c && (["histNet", "commNet"] as const).map((net) => <TrainingPanel key={net} net={net} status={c} running={running} />)}
      </div>

      {deploy && <DeployPanel cohort={cohort} d={deploy} rows={rows} />}

      <Tile title="Jobs for this cohort" icon="job" className="flush">
        <QueueTable jobs={jobs} focus={ui.focus} setFocus={(id) => setUi({ focus: id })} compact />
      </Tile>

      <ToProperties>
        <div className="props-title"><Icon name="cohort" size={16} />{cohort}</div>
        <PropSection title="Cohort">{Object.entries(cfg).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) =>
          <Prop key={k} k={k} mono>{Array.isArray(v) ? v.join(", ") : String(v)}</Prop>)}</PropSection>
        <PropSection title="Next run (SV-Net tab ▸ Train / Run)">
          <Prop k="From">{o.from_step ?? "first unfinished"}</Prop><Prop k="To">{o.stop_after ?? "last"}</Prop><Prop k="Force redo">{o.redoFrom ? "yes" : "no"}</Prop>
          <Prop k="Overrides">{Object.entries(o).filter(([k, v]) => !["cohort", "from_step", "stop_after", "redoFrom"].includes(k) && v !== undefined).map(([k, v]) => `${k}=${v}`).join(" ") || "none"}</Prop>
        </PropSection>
        {c && <PropSection title="Disk">{Object.entries(c.disk_gb).map(([k, v]) => <Prop key={k} k={k}>{v} GB</Prop>)}</PropSection>}
        <p className="muted small" style={{ padding: "4px 8px" }}>Upstream keeps training a network that exists; “Fresh networks” archives it first. Re-running rollout or observe moves the old data to
          cohorts/{cohort}/_archive/: nothing is deleted.</p>
      </ToProperties>
      <ToProblems items={probs} />
    </>
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
  const useSaved = !!s && !(running && live.length > (s.loss_train?.length ?? 0));
  const train = useSaved ? s!.loss_train : live;
  const r = status.results[`train_${net}`];
  return (
    <Tile title={net} icon="chart" meta={net === "histNet" ? "history → [m, kt]" : "image + state → command"}
      actions={<span className="row" style={{ marginLeft: 6, flexWrap: "nowrap" }}>
        {pilots.length > 1 && <select value={p} onChange={(e) => setPick(e.target.value)}>{pilots.map((x) => <option key={x}>{x}</option>)}</select>}
        <label className="check small"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} />log</label></span>}>
      {!p ? <p className="empty">Not trained yet.</p> : (
        <>
          <LineChart points={train} xLabel="epoch" yLabel="loss" log={log}
            second={useSaved && s!.loss_test.length ? { points: s!.loss_test, label: "test loss", firstLabel: "train loss" } : undefined} />
          <p className="small">
            {useSaved
              ? <>{s!.epochs} epochs · train {fmtL(s!.loss_train)} · test {fmtL(s!.loss_test)} · {s!.n_train?.toLocaleString()} / {s!.n_test?.toLocaleString()} samples</>
              : <>live: epoch {live.length ? live[live.length - 1][0] : 0}, train loss {live.length ? live[live.length - 1][1].toPrecision(4) : "—"}</>}
            {r && <span className="muted"> · {r.wallclock} · peak {vram(r.peak_vram_mib)}</span>}
          </p>
          {(s?.eval_tte_upstream ?? []).length > 0 && <p className="muted small">Upstream in-loop TTE: {s!.eval_tte_upstream.map(([e, v]) => `${e}: ${v.toFixed(2)}`).join(" · ")}</p>}
        </>
      )}
    </Tile>
  );
}
const vram = (m: number) => (m >= 0 ? `${m} MiB` : "— (no nvidia-smi)");
const fmtL = (pts: [number, number][]) => (pts.length ? pts[pts.length - 1][1].toPrecision(4) : "—");

function DeployPanel({ cohort, d, rows }: { cohort: string; d: NonNullable<CohortStatus["results"]["deploy"]>; rows: [string, DeployRow][] }) {
  return (
    <div className="tiles cols-2-1">
      <Tile title="Evaluation in FiGS" icon="deploy" className="flush" meta={`${d.method} on ${d.course} in ${d.scene} · ${d.finished?.replace("T", " ")} · ${d.wallclock} · peak ${vram(d.peak_vram_mib)}`}>
        <table>
          <thead><tr><th>pilot</th><th className="num">tracking mean</th><th className="num">max</th><th className="num">within 0.3 m</th><th className="num">final</th><th className="num">rollouts</th>
            <th className="num">upstream TTE</th><th className="num">upstream PP</th><th className="num">Hz mean / worst</th></tr></thead>
          <tbody>{rows.map(([name, r]) => (
            <tr key={name}>
              <td><b>{name}</b> <span className="muted small">{r.role}</span></td>
              <td className={`num ${r.role === "student" && r.tte.mean_m > 0.5 ? "bad" : ""}`}>{r.tte.mean_m} m</td><td className="num">{r.tte.max_m} m</td>
              <td className="num">{Math.round(r.tte["within_0.3m"] * 100)}%</td><td className="num">{r.tte.final_mean_m} m</td><td className="num">{r.tte.rollouts}</td>
              <td className="num muted">{r.upstream_tte_mean}</td><td className="num muted">{r.upstream_pp}</td><td className="num">{r.hz_mean} / {r.hz_worst}</td>
            </tr>))}</tbody>
        </table>
        <p className="muted small" style={{ padding: "4px 8px" }}>Tracking error: per step, the distance from the drone to the nearest point of the expert's planned path (as tol_select measures it).
          The upstream TTE/PP columns come from sousvide's compute_flight_metrics, which takes the norm over the wrong axis: shown for comparison with the paper's tables, not as distances.</p>
      </Tile>
      <Tile title="Videos" icon="video" meta="last rollout per pilot">
        <div className="videos">
          {rows.filter(([, r]) => r.video).map(([name, r]) => (
            <figure key={name}><video controls src={cohortVideoUrl(cohort, r.video!.split("/").pop()!)} /><figcaption>{name} ({r.role})</figcaption></figure>))}
        </div>
      </Tile>
    </div>
  );
}
