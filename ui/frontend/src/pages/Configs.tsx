import { useEffect, useState } from "react";
import { api, ApiError, ago, Family, FAMILIES } from "../api";
import { usePoll } from "../components";

const HELP: Record<Family, string> = {
  captures: "Camera intrinsics (or null to use SfM's) and ArUco extractor settings. marker_length is metres, black square only.",
  courses: "Keyframes: t (s) and a 4-row fo matrix (x, y, z, yaw) by derivative order; null = free. Course frame is z-down: altitude is negative z.",
  pilots: "Experts (Viper: MPC plan/track) and students (Maverick, Iceman: histNet/commNet networks).",
  frames: "Drone physical parameters and onboard camera.",
  methods: "data_* drive rollout synthesis; eval_* drive evaluation.",
  nnio: "Network input/output field layouts.",
};

export default function Configs({ family = "courses", name }: { family?: Family; name?: string }) {
  const list = usePoll(() => api.configs(family), 0, [family]);
  return (
    <>
      <h1>Configs</h1>
      <div className="row" style={{ marginBottom: 12 }}>
        {FAMILIES.map((f) => (
          <a key={f} className="btn" href={`#/configs/${f}`} style={f === family ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}>{f}</a>
        ))}
      </div>
      <p className="muted small">{HELP[family]} Files live in <span className="mono">SousVide/configs/{family}/</span>; saved captures, courses and pilots are also copied into the FYP-Radiance overlay.</p>
      <div className="grid2" style={{ gridTemplateColumns: "240px 1fr" }}>
        <div className="panel">
          {list.data?.map((c) => (
            <div key={c.name}>
              <a href={`#/configs/${family}/${c.name}`} style={c.name === name ? { fontWeight: 700 } : undefined}>{c.name}</a>
              {c.kind && <span className="muted small"> · {c.kind}</span>}
            </div>
          ))}
          {list.data?.length === 0 && <p className="muted">none</p>}
        </div>
        {name ? <Editor key={`${family}/${name}`} family={family} name={name} onSaved={list.reload} />
          : <div className="panel muted">Pick a config to view or edit.</div>}
      </div>
    </>
  );
}

function Editor({ family, name, onSaved }: { family: Family; name: string; onSaved: () => void }) {
  const [text, setText] = useState("");
  const [orig, setOrig] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [saveAs, setSaveAs] = useState(name);
  const [mtime, setMtime] = useState<number | null>(null);

  useEffect(() => {
    api.config(family, name).then((d) => { const t = JSON.stringify(d, null, 4); setText(t); setOrig(t); })
      .catch((e) => setMsg({ ok: false, text: String(e.message ?? e) }));
    api.configs(family).then((l) => setMtime(l.find((c) => c.name === name)?.modified ?? null)).catch(() => {});
  }, [family, name]);

  const parse = (): unknown | undefined => {
    try { return JSON.parse(text); } catch (e) { setMsg({ ok: false, text: `Not valid JSON: ${(e as Error).message}` }); return undefined; }
  };
  const run = async (fn: () => Promise<string>) => {
    try { setMsg({ ok: true, text: await fn() }); }
    catch (e) { setMsg({ ok: false, text: e instanceof ApiError ? e.message : String(e) }); }
  };

  return (
    <div className="panel">
      <div className="row">
        <h2 className="mono" style={{ margin: 0 }}>{family}/{name}.json</h2>
        <span className="muted small">modified {ago(mtime)}</span>
        <span className="spacer" />
        {text !== orig && <span className="warn small">unsaved changes</span>}
      </div>
      <textarea rows={28} spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} style={{ marginTop: 10 }} />
      {msg && <p className={msg.ok ? "ok" : "err"}>{msg.text}</p>}
      <div className="row">
        <button onClick={() => { const d = parse(); if (d !== undefined) run(async () => { await api.validateConfig(family, d); return "Valid."; }); }}>Validate</button>
        <button className="primary" disabled={text === orig} onClick={() => {
          const d = parse();
          if (d !== undefined) run(async () => { const r = await api.saveConfig(family, name, d); setOrig(text); onSaved();
            return `Saved ${r.path}${r.mirrored ? ` and ${r.mirrored}` : ""}`; });
        }}>Save</button>
        <button onClick={() => setText(orig)} disabled={text === orig}>Revert</button>
        <span className="spacer" />
        <input value={saveAs} onChange={(e) => setSaveAs(e.target.value)} style={{ width: 160 }} />
        <button disabled={!saveAs || saveAs === name} onClick={() => {
          const d = parse();
          if (d !== undefined) run(async () => { await api.saveConfig(family, saveAs, d, false); onSaved(); location.hash = `#/configs/${family}/${saveAs}`; return "Created."; });
        }}>Save as new</button>
      </div>
    </div>
  );
}
