// The right half of the splat editor's Compare view (Semantics ▸ View ▸ Compare): the same splat,
// the same camera (linked), the same query on the OTHER feature backend (lift | fmgs). It keeps its
// own result — relevancy colours, candidates, a picked Gaussian's labels — and can send its own
// candidate to a course. The main view (left) stays the editor's primary one.
import { useEffect, useMemo, useState } from "react";
import { ApiError, Candidate, Geometry, QueryReply, SemBackend, semApi } from "../api";
import { buildColors, ColorMode, countLit, heatCss } from "../splat/recolor";
import type { SplatData } from "../splat/format";
import type { Pick } from "../splat/pick";
import SplatScene, { CamLink } from "../splat/SplatScene";

const errMsg = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
const size = (c: Candidate) => [0, 1, 2].map((a) => (c.box.hi[a] - c.box.lo[a]).toFixed(2)).join(" × ");

export interface CompareQuery { text: string; nonce: number; standoff: number; top: number; threshold?: number; rel_alpha?: number; negatives?: string[] }

export default function SplatCompare({ scene, data, geo, backend, q, mode, link, bodyR, onSend }: {
  scene: string; data: SplatData | null; geo: Geometry | null; backend: SemBackend; q: CompareQuery | null; mode: ColorMode;
  link: CamLink; bodyR: number; onSend: (c: Candidate, text: string, backend: SemBackend) => void;
}) {
  const [reply, setReply] = useState<QueryReply | null>(null);
  const [rel, setRel] = useState<Uint8Array | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [floor, setFloor] = useState(0.6);
  const [sel, setSel] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [picked, setPicked] = useState<Pick | null>(null);
  const [labels, setLabels] = useState<[string, number][] | null>(null);
  const [pca, setPca] = useState<Uint8Array | null>(null);

  useEffect(() => {
    if (!q?.text) return;
    let alive = true;
    setBusy(true); setErr(null);
    (async () => {
      try {
        const r = await semApi.query(scene, { text: q.text, backend, standoff: q.standoff, top: q.top, threshold: q.threshold,
          rel_alpha: q.rel_alpha, negatives: q.negatives?.length ? q.negatives : undefined, relevancy: true });
        const b = r.relevancy_id ? await semApi.relevancy(scene, r.relevancy_id) : null;
        if (!alive) return;
        setReply(r); setRel(b); setSel(r.result.candidates.length ? 0 : null); setPicked(null);
        setFloor(Math.round(Math.min(0.95, Math.max(0.3, r.result.tau)) * 100) / 100);
      } catch (e) { if (alive) { setErr(errMsg(e)); setReply(null); setRel(null); } }
      finally { if (alive) setBusy(false); }
    })();
    return () => { alive = false; };
  }, [scene, backend, q?.text, q?.nonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (mode !== "pca" || pca) return;
    semApi.pca(scene, backend).then((r) => setPca(r.stale ? null : r.data)).catch((e) => setErr(errMsg(e)));
  }, [mode, pca, scene, backend]);

  useEffect(() => {
    setLabels(null);
    if (!picked) return;
    const l = Array.from(new Set([q?.text, "floor", "wall", "ceiling", "table", "chair", "shelf", "box", "door"].filter(Boolean) as string[]));
    let alive = true;
    semApi.labels(scene, picked.index, l, backend).then((r) => alive && setLabels(r.scores)).catch(() => {});
    return () => { alive = false; };
  }, [picked, scene, backend, q?.text]);

  const n = data?.n ?? 0;
  const relOk = !!rel && !!reply && !reply.stale && rel.length === n;
  const shown: ColorMode = mode === "relevancy" && relOk ? "relevancy" : mode === "pca" && pca?.length === 3 * n ? "pca" : "rgb";
  const colors = useMemo(() => (data && shown !== "rgb" ? buildColors(data.rgba, n, { mode: shown, rel, pca, floor }) : null),
    [data, n, shown, rel, pca, floor]);
  const cands = reply?.result.candidates ?? [];

  return (
    <div className="viewport" style={{ background: "#0d1117" }}>
      <SplatScene data={data} colors={colors} geo={geo} boxId={scene} candidates={cands} hover={hover} selected={sel}
        bodyRadius={bodyR} pins={[]} showPins={false} pinSel={null} picked={picked} onPick={setPicked} minOpacity={0.05} cursor="pick"
        keyNav={false} navSpeed={1} showCamPath={false} focus={null} link={link} linkId="compare" />
      <div className="vtag" style={{ top: 8 }}><b>{backend}</b>{busy ? " · querying…" : reply ? ` · “${reply.result.text}” · ${reply.worker_ms ?? "?"} ms · τ ${reply.result.tau.toFixed(3)}` : q?.text ? "" : " · run a query"}
        {err && <span className="bad"> · {err}</span>}</div>
      {cands.length > 0 && (
        <div className="cmp-cands">
          <table className="grid cands"><thead><tr><th>#</th><th className="num">score</th><th className="num">n</th><th>size</th><th className="num">gap</th><th /></tr></thead>
            <tbody>{cands.map((c, i) => (
              <tr key={i} className={`click ${sel === i ? "sel" : ""}`} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onClick={() => { setSel(i); setPicked(null); }}>
                <td>{c.rank}</td><td className="num">{c.score.toFixed(1)}</td><td className="num">{c.n.toLocaleString()}</td><td className="mono small">{size(c)}</td>
                <td className={`num ${c.gap_ok === false ? "bad" : ""}`}>{c.gap === null ? "—" : c.gap.toFixed(2)}</td>
                <td><button className="lnk" onClick={(e) => { e.stopPropagation(); onSend(c, reply!.result.text, backend); }}>send…</button></td></tr>))}</tbody></table>
        </div>
      )}
      {reply && cands.length === 0 && <div className="cmp-cands"><p className="muted small" style={{ padding: 4 }}>No cluster above the threshold.</p></div>}
      <div className="legend">
        {shown === "relevancy" && reply && <div className="row" style={{ gap: 5, alignItems: "center", flexWrap: "wrap" }}>
          <b>Relevancy</b><span className="mono">{floor.toFixed(2)}</span><span className="swatch" style={{ width: 70, height: 9, display: "inline-block", background: heatCss() }} />
          <span className="mono">1.0</span><span className="muted">{countLit(rel!, floor).toLocaleString()} lit</span></div>}
        {shown === "pca" && <div><b>PCA</b> of the {backend} features</div>}
        {picked && <div><b>#{picked.index.toLocaleString()}</b>{labels ? ` · ${labels.slice(0, 3).map(([l, c]) => `${l} ${c.toFixed(3)}`).join(" · ")}` : " · …"}</div>}
        <div className="muted">camera linked to the left view</div>
      </div>
    </div>
  );
}
