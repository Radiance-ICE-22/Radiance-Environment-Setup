import { useEffect, useState } from "react";
import { Binding, commands, useRegistryVersion } from "./core";
import { Icon } from "./icons";
import { Item, TABS } from "./ribbonSpec";
import { Tip } from "./Tip";

const tipFor = (it: Item, b: Binding | undefined, when?: string) => ({
  title: it.label, body: it.h, runs: "r" in it ? it.r || undefined : undefined, keyText: "key" in it ? it.key : undefined,
  note: !b ? (when ?? "Not available in this context.") : typeof b.disabled === "string" ? b.disabled : undefined,
});

function Cmd({ it, when }: { it: Item; when?: string }) {
  const b = commands.get(it.id);
  const disabled = !b || !!b.disabled;
  if (b?.hidden) return null;
  if (it.t === "L" || it.t === "S") {
    const label = b?.label ?? it.label;
    return (
      <Tip tip={tipFor(it, b, when)}>
        <button className={`rb-${it.t === "L" ? "large" : "small"} ${b?.checked ? "on" : ""}`} disabled={disabled}
          onClick={() => b?.run?.()} aria-pressed={b?.checked} data-cmd={it.id}>
          <Icon name={it.icon} size={it.t === "L" ? 32 : 16} />
          <span>{label}</span>
        </button>
      </Tip>
    );
  }
  if (it.t === "C") {
    return (
      <Tip tip={tipFor(it, b, when)}>
        <label className={`rb-check ${disabled ? "dis" : ""}`} data-cmd={it.id}>
          <input type="checkbox" disabled={disabled} checked={!!b?.checked} onChange={(e) => b?.set?.(String(e.target.checked))} />{it.label}
        </label>
      </Tip>
    );
  }
  // field
  const val = b?.value ?? "";
  const f = it as Extract<Item, { t: "F" }>;
  return (
    <Tip tip={tipFor(it, b, when)}>
      <label className={`rb-field ${disabled ? "dis" : ""}`} data-cmd={it.id}>
        <span>{it.label}</span>
        {f.kind === "select" ? (
          <select disabled={disabled} value={String(val)} style={{ width: f.w ?? 96 }} onChange={(e) => b?.set?.(e.target.value)}>
            {!(b?.options ?? []).some((o) => (Array.isArray(o) ? o[0] : o) === String(val)) && <option value={String(val)}>{String(val) || "—"}</option>}
            {(b?.options ?? []).map((o) => { const [v, l] = Array.isArray(o) ? o : [o, o]; return <option key={v} value={v}>{l}</option>; })}
          </select>
        ) : (
          <FieldInput kind={f.kind} step={f.step} width={f.w ?? 80} value={String(val)} disabled={disabled} onCommit={(v) => b?.set?.(v)} />
        )}
      </label>
    </Tip>
  );
}
/** Commits on Enter, blur or spinner click, so typing "0.1" does not fire three changes. */
function FieldInput({ kind, step, width, value, disabled, onCommit }: { kind: string; step?: number; width: number; value: string; disabled: boolean; onCommit: (v: string) => void }) {
  const [t, setT] = useState(value); const [focus, setFocus] = useState(false);
  useEffect(() => { if (!focus) setT(value); }, [value, focus]);
  return <input type={kind === "spin" ? "number" : "text"} step={step ?? 1} disabled={disabled} style={{ width }} value={t}
    onFocus={() => setFocus(true)} onBlur={() => { setFocus(false); if (t !== value) onCommit(t); }}
    onChange={(e) => { setT(e.target.value); if (!(e.nativeEvent as InputEvent).inputType) onCommit(e.target.value); /* spinner arrows */ }}
    onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />;
}

export function Ribbon({ tab, setTab, minimized, setMinimized, appMenu }: {
  tab: string; setTab: (t: string) => void; minimized: boolean; setMinimized: (b: boolean) => void; appMenu: React.ReactNode;
}) {
  useRegistryVersion();
  const [menu, setMenu] = useState(false);
  const [peek, setPeek] = useState(false);
  const visible = TABS.filter((t) => !t.contextual || commands.get(t.contextual)?.checked);
  const cur = visible.find((t) => t.name === tab) ?? visible[0];
  // fall back to a visible tab, except while a contextual tab's document has not registered its
  // context yet (the splat editor is lazy-loaded): its tab appears as soon as it does
  const wanted = TABS.find((t) => t.name === tab);
  const pendingCtx = !!wanted?.contextual && commands.get(wanted.contextual) === undefined;
  useEffect(() => { if (cur.name !== tab && !pendingCtx) setTab(cur.name); }, [cur.name, tab, setTab, pendingCtx]);
  const showBody = !minimized || peek;
  return (
    <div className={`ribbon ${minimized ? "min" : ""}`} onMouseLeave={() => setPeek(false)}>
      <div className="rb-tabs">
        <div className="rb-app-wrap">
          <button className="rb-app" onClick={() => setMenu(!menu)}>Galley <Icon name="dropdown" size={10} /></button>
          {menu && <div className="rb-appmenu" onMouseLeave={() => setMenu(false)} onClick={() => setMenu(false)}>{appMenu}</div>}
        </div>
        {visible.map((t) => (
          <button key={t.name} className={`rb-tab ${t.name === cur.name ? "on" : ""} ${t.contextual ? "ctx" : ""}`}
            onClick={() => { setTab(t.name); if (minimized) setPeek(true); }} onDoubleClick={() => setMinimized(!minimized)}>{t.name}</button>
        ))}
        <span className="spacer" />
        <Tip tip={{ title: minimized ? "Expand the ribbon" : "Minimize the ribbon", body: "Show only the tab names; click a tab to open it for one command.", keyText: "Ctrl+F1" }}>
          <button className="rb-tool" onClick={() => setMinimized(!minimized)}><Icon name={minimized ? "chevdown" : "chevron"} size={14} style={{ transform: minimized ? undefined : "rotate(-90deg)" }} /></button>
        </Tip>
        <Tip tip={{ title: "Help", body: "What each part of the window does.", keyText: "F1" }}>
          <button className="rb-tool" onClick={() => commands.get("help.docs")?.run?.()}><Icon name="help" size={16} /></button>
        </Tip>
      </div>
      {showBody && (
        <div className={`rb-body ${minimized ? "peek" : ""}`}>
          {cur.groups.map((g, gi) => (
            <div className="rb-group" key={g.name + gi}>
              <div className="rb-content">
                {g.items.map((it, i) => Array.isArray(it)
                  ? <div className="rb-col" key={i}>{it.map((x) => <Cmd key={x.id} it={x} when={g.when} />)}</div>
                  : <Cmd key={it.id} it={it} when={g.when} />)}
              </div>
              <div className="rb-glabel">{g.name}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
