// Win7-style "super tooltip": shown 600 ms after the pointer rests on a command.
import { ReactNode, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Icon } from "./icons";

export interface TipText { title: string; body?: string; runs?: string; keyText?: string; note?: string }

export function Tip({ tip, children, block, delay = 600 }: { tip: TipText | null; children: ReactNode; block?: boolean; delay?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const timer = useRef<number>(0);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  if (!tip) return <>{children}</>;
  const show = () => {
    clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      const r = ref.current?.getBoundingClientRect(); if (!r) return;
      const x = Math.min(r.left, window.innerWidth - 330), y = r.bottom + 4 + 300 > window.innerHeight ? Math.max(4, r.top - 8) : r.bottom + 4;
      setPos({ x, y: y });
    }, delay);
  };
  const hide = () => { clearTimeout(timer.current); setPos(null); };
  const Tag = block ? "div" : "span";
  return (
    <Tag ref={ref as any} className={block ? "tipwrap block" : "tipwrap"} onMouseEnter={show} onMouseLeave={hide} onMouseDown={hide}>
      {children}
      {pos && createPortal(
        <div className="supertip" style={{ left: pos.x, top: pos.y }} role="tooltip">
          <div className="st-head"><b>{tip.title}</b>{tip.keyText && <span>{tip.keyText}</span>}</div>
          {tip.body && <div className="st-body">{tip.body}</div>}
          {tip.note && <div className="st-note">{tip.note}</div>}
          {tip.runs && <div className="st-runs">Runs: {tip.runs}</div>}
          <div className="st-foot"><Icon name="help" size={14} /> Press F1 for more help</div>
        </div>, document.body)}
    </Tag>
  );
}
