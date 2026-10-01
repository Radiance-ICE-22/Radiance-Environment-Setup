// Configs document: SousVide config files by family, a JSON editor validated by the backend's
// Pydantic models. Family buttons, Validate/Save/Revert/Duplicate/Format live on the Configs tab.
import { useEffect, useState } from "react";
import { api, ApiError, ago, Family } from "../api";
import { usePoll } from "../components";
import { Problem, ToProblems, ToProperties, useCommands } from "../shell/core";
import { Icon } from "../shell/icons";
import { Prop, PropSection, Tile } from "../shell/Panes";

const HELP: Record<Family, string> = {
  captures: "Camera intrinsics (or null to use SfM's) and ArUco extractor settings. marker_length is metres, black square only.",
  courses: "Keyframes: t (s) and a 4-row fo matrix (x, y, z, yaw) by derivative order; null = free. Course frame is z-down: altitude is negative z.",
  pilots: "Experts (Viper: MPC plan/track) and students (Maverick, Iceman: histNet/commNet networks).",
  frames: "Drone physical parameters and onboard camera.",
  methods: "data_* drive rollout synthesis; eval_* drive evaluation.",
  nnio: "Network input/output field layouts.",
};
const MIRRORED: Family[] = ["captures", "courses", "pilots"];

export default function Configs({ family = "courses", name }: { family?: Family; name?: string }) {
  const list = usePoll(() => api.configs(family), 0, [family]);
  const [q, setQ] = useState("");
  const items = (list.data ?? []).filter((c) => !q || c.name.toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="cfg-split">
      <Tile title={`configs/${family}`} icon="folder" className="fill flush" meta={`${list.data?.length ?? 0} files`}>
        <div className="searchbox"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter…" /><Icon name="search" size={14} /></div>
        <div className="tree">
          {items.map((c) => (
            <div key={c.name} className={`tree-row ${c.name === name ? "sel" : ""}`} onClick={() => (location.hash = `#/configs/${family}/${c.name}`)}>
              <span className="twisty" /><Icon name="json" size={16} /><span className="tl">{c.name}</span>{c.kind && <span className="tm">{c.kind}</span>}
            </div>))}
          {list.data?.length === 0 && <p className="muted pad">none</p>}
        </div>
      </Tile>
      {name ? <Editor key={`${family}/${name}`} family={family} name={name} onSaved={list.reload} />
        : <Tile title="No file open" icon="config" className="fill"><p className="muted">{HELP[family]}</p><p className="muted">Pick a file on the left (or in Explorer ▸ Configs).
          Files live in <span className="mono">SousVide/configs/{family}/</span>{MIRRORED.includes(family) && "; saved files are also copied into the FYP-Radiance overlay"}.</p>
          <ToProperties><FamilyProps family={family} /></ToProperties></Tile>}
    </div>
  );
}

function FamilyProps({ family }: { family: Family }) {
  return (
    <>
      <div className="props-title"><Icon name="folder" size={16} />configs/{family}</div>
      <PropSection title="Family"><p className="small" style={{ padding: "0 8px" }}>{HELP[family]}</p>
        <Prop k="Folder" mono>SousVide/configs/{family}/</Prop><Prop k="Overlay">{MIRRORED.includes(family) ? "mirrored on save" : "not mirrored"}</Prop></PropSection>
    </>
  );
}

function Editor({ family, name, onSaved }: { family: Family; name: string; onSaved: () => void }) {
  const [text, setText] = useState("");
  const [orig, setOrig] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [mtime, setMtime] = useState<number | null>(null);
  const [kind, setKind] = useState<string | undefined>();
  const [lastSave, setLastSave] = useState<{ path: string; mirrored: string | null } | null>(null);

  useEffect(() => {
    api.config(family, name).then((d) => { const t = JSON.stringify(d, null, 4); setText(t); setOrig(t); })
      .catch((e) => setMsg({ ok: false, text: String(e.message ?? e) }));
    api.configs(family).then((l) => { const c = l.find((x) => x.name === name); setMtime(c?.modified ?? null); setKind(c?.kind); }).catch(() => {});
  }, [family, name]);

  const parse = (): unknown | undefined => {
    try { return JSON.parse(text); } catch (e) { setMsg({ ok: false, text: `Not valid JSON: ${(e as Error).message}` }); return undefined; }
  };
  const act = async (fn: () => Promise<string>) => {
    try { setMsg({ ok: true, text: await fn() }); } catch (e) { setMsg({ ok: false, text: e instanceof ApiError ? e.message : String(e) }); }
  };
  const validate = () => { const d = parse(); if (d !== undefined) act(async () => { await api.validateConfig(family, d); return "Valid: matches the Pydantic model and round-trips."; }); };
  const save = () => {
    const d = parse();
    if (d !== undefined) act(async () => { const r = await api.saveConfig(family, name, d); setOrig(text); setLastSave(r); setMtime(Date.now() / 1000); onSaved();
      return `Saved ${r.path}${r.mirrored ? ` and ${r.mirrored}` : ""}.`; });
  };
  const duplicate = () => {
    const n = prompt(`Save ${family}/${name} as a new file named:`, `${name}_copy`)?.trim();
    if (!n) return;
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(n)) { setMsg({ ok: false, text: "Name: letters, digits, _ and - only." }); return; }
    const d = parse();
    if (d !== undefined) act(async () => { await api.saveConfig(family, n, d, false); onSaved(); location.hash = `#/configs/${family}/${n}`; return `Created ${n}.`; });
  };
  const dirty = text !== orig;
  let jsonErr: string | null = null;
  try { JSON.parse(text || "null"); } catch (e) { jsonErr = (e as Error).message; }
  useCommands({
    "cfg.validate": { run: validate, disabled: !!jsonErr && "Not valid JSON." },
    "file.save": { run: save, disabled: !dirty && "No unsaved changes." },
    "cfg.revert": { run: () => { setText(orig); setMsg(null); }, disabled: !dirty && "No unsaved changes." },
    "cfg.duplicate": { run: duplicate },
    "cfg.format": { run: () => { const d = parse(); if (d !== undefined) setText(JSON.stringify(d, null, 4)); }, disabled: !!jsonErr && "Not valid JSON." },
    "cfg.mirror": MIRRORED.includes(family) ? { run: save, disabled: !dirty && "Saved already: the overlay copy is current." } : { disabled: `${family} are not mirrored to the overlay.` },
  });

  const probs: Problem[] = [
    ...(jsonErr ? [{ severity: "error" as const, where: `${family}/${name}.json`, message: `Not valid JSON: ${jsonErr}` }] : []),
    ...(msg && !msg.ok ? [{ severity: "error" as const, where: `${family}/${name}.json`, message: msg.text }] : []),
    ...(dirty ? [{ severity: "info" as const, where: `${family}/${name}.json`, message: "Unsaved changes." }] : []),
  ];
  const lines = text.split("\n").length;

  return (
    <Tile title={`${family}/${name}.json`} icon="json" className="fill flush cfg-editor"
      meta={<>{dirty ? <b className="warn">unsaved · </b> : null}{lines} lines · modified {ago(mtime)}{kind ? ` · ${kind}` : ""}</>}>
      {msg && <p className={msg.ok ? "ok" : "err"} style={{ margin: 0, padding: "3px 8px", background: msg.ok ? "var(--ok-bg)" : "var(--bad-bg)" }}>{msg.text}</p>}
      <textarea spellCheck={false} value={text} onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Tab") { e.preventDefault(); const t = e.currentTarget, a = t.selectionStart; setText(text.slice(0, a) + "    " + text.slice(t.selectionEnd)); requestAnimationFrame(() => { t.selectionStart = t.selectionEnd = a + 4; }); } }} />
      <ToProperties>
        <div className="props-title"><Icon name="json" size={16} />{name}.json</div>
        <PropSection title="File">
          <Prop k="Family">{family}</Prop><Prop k="Kind">{kind ?? "—"}</Prop><Prop k="Path" mono>SousVide/configs/{family}/{name}.json</Prop>
          <Prop k="Modified">{ago(mtime)}</Prop><Prop k="State" tone={dirty ? "warn" : "ok"}>{dirty ? "unsaved changes" : "saved"}</Prop>
          <Prop k="JSON" tone={jsonErr ? "bad" : "ok"}>{jsonErr ? "invalid" : "valid syntax"}</Prop>
          <Prop k="Overlay">{MIRRORED.includes(family) ? (lastSave?.mirrored ? <span className="mono">{lastSave.mirrored}</span> : "mirrored on save") : "not mirrored"}</Prop>
        </PropSection>
        <PropSection title="About this family"><p className="small" style={{ padding: "0 8px" }}>{HELP[family]}</p></PropSection>
        {family === "courses" && <PropSection title="Edit visually"><p style={{ padding: "0 8px" }}>
          <a href="#/course">Open the course editor</a> for 3D editing and preview.</p></PropSection>}
      </ToProperties>
      <ToProblems items={probs} />
    </Tile>
  );
}
