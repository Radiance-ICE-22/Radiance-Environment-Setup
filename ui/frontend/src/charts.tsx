// Two small SVG charts: a single-series line (training curves) and a single-series bar
// chart (ArUco detections per 10 s window). Single series → no legend; the title names it.
// Marks: 2 px line, bars with 4 px rounded tops and 2 px gaps, recessive grid, hover tooltip.
import { useMemo, useRef, useState } from "react";

const W = 640, H = 220, PAD = { l: 52, r: 16, t: 12, b: 30 };

function ticks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
const fmt = (v: number) => v === 0 ? "0" : Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  : Math.abs(v) >= 1 ? v.toFixed(2).replace(/\.?0+$/, "") : String(+v.toPrecision(3));

function Tooltip({ x, y, children }: { x: number; y: number; children: React.ReactNode }) {
  return (
    <div className="tip" style={{ left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%` }}>{children}</div>
  );
}

export function LineChart({ points, xLabel, yLabel, log = false }:
  { points: [number, number][]; xLabel: string; yLabel: string; log?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const pts = useMemo(() => points.filter(([, v]) => Number.isFinite(v) && (!log || v > 0)), [points, log]);
  if (pts.length < 2) return <p className="muted small">Not enough points to plot.</p>;
  const tf = (v: number) => (log ? Math.log10(v) : v);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => tf(p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  const padY = (y1 - y0) * 0.06; y0 -= log ? 0 : padY; y1 += padY;
  if (!log && Math.min(...ys) >= 0 && y0 < 0) y0 = 0;
  const X = (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * (W - PAD.l - PAD.r);
  const Y = (v: number) => PAD.t + (1 - (tf(v) - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  // log axis: 1-2-5 steps per decade, kept inside the plotted range
  const yt = log
    ? Array.from({ length: Math.ceil(y1) - Math.floor(y0) + 1 }, (_, i) => Math.floor(y0) + i)
        .flatMap((e) => [1, 2, 5].map((m) => m * 10 ** e)).filter((v) => tf(v) >= y0 && tf(v) <= y1)
    : ticks(y0, y1);
  const d = pts.map(([s, v], i) => `${i ? "L" : "M"}${X(s).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    for (let i = 1; i < pts.length; i++) if (Math.abs(X(pts[i][0]) - sx) < Math.abs(X(pts[best][0]) - sx)) best = i;
    setHover(best);
  };
  const h = hover !== null ? pts[hover] : null;
  return (
    <div className="chart">
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} by ${xLabel}`}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {yt.map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={Y(v)} y2={Y(v)} className="grid" />
            <text x={PAD.l - 6} y={Y(v) + 4} textAnchor="end" className="axis">{fmt(v)}</text>
          </g>
        ))}
        {ticks(x0, x1).map((v) => (
          <text key={v} x={X(v)} y={H - 10} textAnchor="middle" className="axis">{fmt(v)}</text>
        ))}
        <path d={d} className="series" fill="none" />
        {h && (
          <g>
            <line x1={X(h[0])} x2={X(h[0])} y1={PAD.t} y2={H - PAD.b} className="crosshair" />
            <circle cx={X(h[0])} cy={Y(h[1])} r={4} className="dot" />
          </g>
        )}
      </svg>
      {h && <Tooltip x={X(h[0])} y={Y(h[1])}><b>{fmt(h[1])}</b> {yLabel}<br />{xLabel} {fmt(h[0])}</Tooltip>}
    </div>
  );
}

export function BarChart({ bars, xLabel, yLabel, detail }:
  { bars: { label: string; value: number }[]; xLabel: string; yLabel: string; detail?: (i: number) => string }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!bars.length) return <p className="muted small">No data.</p>;
  const max = Math.max(1, ...bars.map((b) => b.value));
  const yt = ticks(0, max, 4);
  const top = Math.max(max, yt[yt.length - 1]);
  const bw = (W - PAD.l - PAD.r) / bars.length;
  const Y = (v: number) => PAD.t + (1 - v / top) * (H - PAD.t - PAD.b);
  const every = Math.ceil(bars.length / 12);
  const h = hover !== null ? bars[hover] : null;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} by ${xLabel}`} onMouseLeave={() => setHover(null)}>
        {yt.map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={Y(v)} y2={Y(v)} className="grid" />
            <text x={PAD.l - 6} y={Y(v) + 4} textAnchor="end" className="axis">{fmt(v)}</text>
          </g>
        ))}
        {bars.map((b, i) => {
          const x = PAD.l + i * bw + 1, w = Math.max(1, bw - 2), y = Y(b.value), hgt = H - PAD.b - y;
          const r = Math.min(4, w / 2, hgt);
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={PAD.l + i * bw} y={PAD.t} width={bw} height={H - PAD.t - PAD.b} fill="transparent" />
              {b.value > 0 && <path className={`bar ${hover === i ? "on" : ""}`}
                d={`M${x},${H - PAD.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${H - PAD.b}Z`} />}
              {i % every === 0 && <text x={x + w / 2} y={H - 10} textAnchor="middle" className="axis">{b.label}</text>}
            </g>
          );
        })}
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="baseline" />
      </svg>
      {h && hover !== null && (
        <Tooltip x={PAD.l + (hover + 0.5) * bw} y={Y(h.value)}>
          <b>{h.value}</b> {yLabel}<br />{xLabel} {h.label}{detail ? <><br />{detail(hover)}</> : null}
        </Tooltip>
      )}
    </div>
  );
}
