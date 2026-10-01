// ───────────────────────── helpers (plain Figma Plugin API) ─────────────────────────
const F = "Noto Sans";
const hex = h => ({ r: parseInt(h.slice(1,3),16)/255, g: parseInt(h.slice(3,5),16)/255, b: parseInt(h.slice(5,7),16)/255 });
const solid = (h, o = 1) => ({ type: 'SOLID', color: hex(h), opacity: o });
const vgrad = (...stops) => ({ type: 'GRADIENT_LINEAR', gradientTransform: [[0,1,0],[-1,0,1]], gradientStops: stops.map(([p, h, a = 1]) => ({ position: p, color: Object.assign({}, hex(h), { a }) })) });
const txt = (str, size = 12, style = "Regular", color = "#1E1E1E", fam = F) => { const t = figma.createText(); t.fontName = { family: fam, style }; t.fontSize = size; t.characters = String(str); t.fills = [solid(color)]; return t; };
const mono = (str, size = 11, color = "#1E1E1E") => txt(str, size, "Regular", color, "Source Code Pro");
const wrapTxt = (t, w) => { t.resize(w, Math.max(14, t.height)); t.textAutoResize = 'HEIGHT'; return t; };
// auto-layout frame that hugs its content
const al = (dir = 'HORIZONTAL', p = {}) => {
  const f = figma.createFrame(); f.layoutMode = dir; f.primaryAxisSizingMode = 'AUTO'; f.counterAxisSizingMode = 'AUTO';
  f.fills = []; f.clipsContent = false; Object.assign(f, p); return f;
};
// auto-layout frame with a fixed size
const box = (dir, w, h, p = {}) => {
  const f = figma.createFrame(); f.layoutMode = dir; f.resize(w, h); f.primaryAxisSizingMode = 'FIXED'; f.counterAxisSizingMode = 'FIXED';
  f.fills = []; f.clipsContent = true; Object.assign(f, p); return f;
};
const add = (parent, child, opts = {}) => { parent.appendChild(child); if (opts.fillW) child.layoutSizingHorizontal = 'FILL'; if (opts.fillH) child.layoutSizingVertical = 'FILL'; if (opts.grow) child.layoutGrow = 1; return child; };
const spacer = (parent) => { const s = al('HORIZONTAL'); parent.appendChild(s); s.layoutSizingHorizontal = 'FILL'; return s; };
const rect = (w, h, fill, p = {}) => { const r = figma.createRectangle(); r.resize(w, h); r.fills = fill ? [typeof fill === 'string' ? solid(fill) : fill] : []; Object.assign(r, p); return r; };
const border = (n, color = "#A0B3CC", sides = {}) => { n.strokes = [solid(color)]; n.strokeTopWeight = sides.t ?? 1; n.strokeBottomWeight = sides.b ?? 1; n.strokeLeftWeight = sides.l ?? 1; n.strokeRightWeight = sides.r ?? 1; return n; };
const svg = (s) => figma.createNodeFromSvg(s);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
let icons = {}, sets = {}, TOOLTIP, PANEHDR, RIBBONS = {};
const ico = (name, size) => { const c = icons[name] || icons["info"]; const i = c.createInstance(); i.resize(size, size); return i; };
const variant = (set, name) => set.children.find(v => v.name === name) || set.children[0];
const props = (inst, map) => { const p = {}; const keys = Object.keys(inst.componentProperties); for (const [k, v] of Object.entries(map)) { const key = keys.find(x => x.split('#')[0] === k); if (key !== undefined) p[key] = v; } inst.setProperties(p); return inst; };
const help = (node, h, r, key) => { if (!h) return; try { node.annotations = [{ labelMarkdown: `**Help:** ${h}${r ? `\n\n**Runs:** \`${r}\`` : ''}${key ? `\n\n**Shortcut:** ${key}` : ''}` }]; } catch (e) {} };
const tooltip = (title, body, runs, key) => { const i = TOOLTIP.createInstance(); props(i, { "Title": title, "Body": body, "Runs": runs ? `Runs: ${runs}` : "", "Show runs": !!runs, "Shortcut": key || "", "Show shortcut": !!key }); return i; };
const paneHeader = (title, w) => { const i = PANEHDR.createInstance(); props(i, { "Title": title }); i.resize(w, 24); return i; };
const pill = (kind) => variant(sets.pill, `Kind=${kind}`).createInstance();
const pushBtn = (label, st = "Default") => props(variant(sets.push, `State=${st}`).createInstance(), { "Label": label });
const smallBtn = (label, icon, state = "Default", h) => { const i = props(variant(sets.small, `State=${state}`).createInstance(), { "Label": label, "Icon": (icons[icon] || icons.info).id, "Show label": label !== "", "Has menu": false }); help(i, h); return i; };
const field = (kind, label, value, h) => { const i = props(variant(sets.field, `Kind=${kind}`).createInstance(), { "Label": label, "Value": value }); help(i, h); return i; };
const checkbox = (label, on) => props(variant(sets.cb, `Checked=${!!on}`).createInstance(), { "Label": label });
const treeItem = (label, icon, meta = "", state = "Default", kids = true) => props(variant(sets.tree, `State=${state}`).createInstance(), { "Label": label, "Meta": meta, "Icon": (icons[icon] || icons.info).id, "Has children": kids });
const docTab = (label, icon, active) => props(variant(sets.doctab, `State=${active ? 'Active' : 'Inactive'}`).createInstance(), { "Label": label, "Icon": (icons[icon] || icons.info).id });
const vsep = (h, color = "#B6CBE6") => rect(1, h, color);

// ───────────────────────── window chrome ─────────────────────────
const titleBar = (title, W) => {
  const t = box('HORIZONTAL', W, 30, { name: "title bar", itemSpacing: 8, counterAxisAlignItems: 'CENTER', paddingLeft: 8, paddingRight: 6 });
  t.fills = [vgrad([0,"#C4DCF5"],[0.45,"#A9C9EE"],[0.5,"#9DBFE8"],[1,"#B6D1F0"])];
  const logo = svg('<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="16" height="16" rx="3" fill="#2B67B8" stroke="#1A4C8E"/><path d="M5 12c2-1 3-5 4-5s1.5 3 4 1" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/><circle cx="5" cy="12" r="1.4" fill="#7CE08B"/><circle cx="13" cy="8" r="1.4" fill="#FFB347"/></svg>'); logo.name = "app icon"; t.appendChild(logo);
  const qat = al('HORIZONTAL', { name: "quick access toolbar", itemSpacing: 1, counterAxisAlignItems: 'CENTER' }); t.appendChild(qat);
  for (const [lab, ic, h] of [["Save","save","Save the active document (Ctrl+S)."],["Undo","undo","Undo (Ctrl+Z)."],["Run","run","Run the active document's job (F5)."],["Cancel","cancel","Cancel the running job (Shift+F5)."]]) { const b = smallBtn("", ic, "Default", h); b.name = `qat: ${lab}`; qat.appendChild(b); }
  qat.appendChild(ico("dropdown", 10));
  spacer(t); t.appendChild(txt(title, 12, "Regular", "#1E395B")); spacer(t);
  const caps = svg('<svg width="100" height="20" viewBox="0 0 100 20" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x=".5" y=".5" width="27" height="18" rx="3" fill="#D3E3F6" stroke="#5B7AA8"/><path d="M9 12h10" stroke="#1E395B" stroke-width="2"/><rect x="27.5" y=".5" width="27" height="18" rx="3" fill="#D3E3F6" stroke="#5B7AA8"/><rect x="36" y="5" width="10" height="8" stroke="#1E395B" stroke-width="1.5"/><rect x="54.5" y=".5" width="45" height="18" rx="3" fill="#D9604C" stroke="#8C2A1C"/><path d="M73 5.5l8 8M81 5.5l-8 8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>'); caps.name = "caption buttons"; t.appendChild(caps);
  return t;
};
const statusBar = (fields, W) => {
  const s = box('HORIZONTAL', W, 24, { name: "status bar", itemSpacing: 0, counterAxisAlignItems: 'CENTER', paddingLeft: 6, paddingRight: 6, clipsContent: false });
  s.fills = [vgrad([0,"#EEF3FA"],[1,"#D5E1F0"])]; border(s, "#A0B3CC", { t: 1, b: 0, l: 0, r: 0 });
  fields.forEach((fd, i) => {
    if (i) s.appendChild(vsep(16));
    const f2 = al('HORIZONTAL', { name: `status: ${fd.label}`, itemSpacing: 5, counterAxisAlignItems: 'CENTER', paddingLeft: 8, paddingRight: 8 }); s.appendChild(f2);
    if (fd.icon) f2.appendChild(ico(fd.icon, 14));
    f2.appendChild(txt(fd.label, 11, "Regular", "#1E395B"));
    if (fd.progress !== undefined) f2.appendChild(progress(120, fd.progress));
    help(f2, fd.h);
    if (fd.grow) f2.layoutSizingHorizontal = 'FILL';
  });
  return s;
};
function progress(w, frac, label) {
  const p = al('HORIZONTAL', { name: "progress", itemSpacing: 6, counterAxisAlignItems: 'CENTER' });
  const track = figma.createFrame(); track.name = "track"; track.resize(w, 12); track.cornerRadius = 2; track.fills = [vgrad([0,"#E6E6E6"],[1,"#F6F6F6"])]; track.strokes = [solid("#A0A0A0")]; track.strokeWeight = 1; track.clipsContent = true;
  const fill = rect(Math.max(1, w * frac), 12, vgrad([0,"#7CE08B"],[0.45,"#2DC24A"],[0.5,"#06B025"],[1,"#5CD872"])); track.appendChild(fill);
  p.appendChild(track); if (label) p.appendChild(txt(label, 11, "Regular", "#1E395B"));
  return p;
}
// docked pane: header + body
const pane = (title, w, h, p = {}) => {
  const f = box('VERTICAL', w, h, Object.assign({ name: `pane: ${title}`, itemSpacing: 0 }, p));
  f.fills = [solid("#FFFFFF")]; border(f, "#A0B3CC");
  const hd = paneHeader(title, w); f.appendChild(hd); hd.layoutSizingHorizontal = 'FILL';
  const body = al('VERTICAL', { name: "body", itemSpacing: 0, paddingTop: 4, paddingBottom: 4, paddingLeft: 4, paddingRight: 4 }); f.appendChild(body);
  body.layoutSizingHorizontal = 'FILL'; body.layoutSizingVertical = 'FILL'; body.clipsContent = true;
  return { frame: f, body };
};
// a titled tile inside a document
const tile = (title, w, h, iconName, meta) => {
  const f = box('VERTICAL', w, h, { name: `tile: ${title}`, itemSpacing: 6, paddingTop: 0, paddingBottom: 8, paddingLeft: 0, paddingRight: 0 });
  f.fills = [solid("#FFFFFF")]; border(f, "#B9C9DE"); f.cornerRadius = 3;
  const hd = box('HORIZONTAL', w, 24, { name: "tile header", itemSpacing: 6, counterAxisAlignItems: 'CENTER', paddingLeft: 8, paddingRight: 8 });
  hd.fills = [vgrad([0,"#FAFCFF"],[1,"#E7EFF9"])]; border(hd, "#C9D7EA", { t: 0, l: 0, r: 0, b: 1 });
  if (iconName) hd.appendChild(ico(iconName, 16));
  hd.appendChild(txt(title, 12, "SemiBold", "#1E395B")); spacer(hd);
  if (meta) hd.appendChild(txt(meta, 11, "Regular", "#5B6B7F"));
  f.appendChild(hd); hd.layoutSizingHorizontal = 'FILL';
  const body = al('VERTICAL', { name: "tile body", itemSpacing: 6, paddingLeft: 10, paddingRight: 10 }); f.appendChild(body);
  body.layoutSizingHorizontal = 'FILL'; body.layoutSizingVertical = 'FILL';
  return { frame: f, body };
};
// property grid (Properties pane)
const propSection = (title) => { const h = al('HORIZONTAL', { name: `section: ${title}`, paddingLeft: 6, paddingTop: 3, paddingBottom: 3, counterAxisAlignItems: 'CENTER', itemSpacing: 4 }); h.fills = [vgrad([0,"#EEF3FA"],[1,"#E1EAF5"])]; h.appendChild(ico("chevron", 10)); h.appendChild(txt(title, 11, "Bold", "#1E395B")); return h; };
const propRow = (k, v, opts = {}) => {
  const r = al('HORIZONTAL', { name: `prop: ${k}`, itemSpacing: 0, counterAxisAlignItems: 'CENTER' });
  const kc = box('HORIZONTAL', opts.kw || 110, 20, { paddingLeft: 14, counterAxisAlignItems: 'CENTER' }); border(kc, "#E3E9F1", { t: 0, l: 0, r: 1, b: 1 });
  kc.appendChild(txt(k, 11, "Regular", "#3A4A5E")); r.appendChild(kc);
  const vc = al('HORIZONTAL', { paddingLeft: 6, counterAxisAlignItems: 'CENTER', itemSpacing: 4 });
  vc.resize(10, 20); vc.counterAxisSizingMode = 'FIXED'; r.appendChild(vc); vc.layoutSizingHorizontal = 'FILL'; border(vc, "#E3E9F1", { t: 0, l: 0, r: 0, b: 1 });
  if (opts.icon) vc.appendChild(ico(opts.icon, 12));
  vc.appendChild(opts.mono ? mono(v, 11, opts.color || "#1E1E1E") : txt(v, 11, opts.bold ? "SemiBold" : "Regular", opts.color || "#1E1E1E"));
  help(r, opts.h);
  return r;
};
const propGrid = (parent, sections) => {
  for (const [title, rows] of sections) {
    add(parent, propSection(title), { fillW: true });
    for (const row of rows) add(parent, propRow(row[0], row[1], row[2] || {}), { fillW: true });
  }
};
// table
const table = (cols, rows, opts = {}) => {
  const t = al('VERTICAL', { name: opts.name || "table", itemSpacing: 0 });
  const mk = (cells, head, rowOpts = {}) => {
    const r = al('HORIZONTAL', { itemSpacing: 0, counterAxisAlignItems: 'CENTER' });
    if (head) r.fills = [vgrad([0,"#FAFCFF"],[1,"#E9F0F8"])];
    else if (rowOpts.sel) r.fills = [vgrad([0,"#DCEBFC"],[1,"#C1DBFC"])];
    else if (rowOpts.bad) r.fills = [solid("#FDF0EE")];
    cells.forEach((cv, i) => {
      const w = cols[i][1];
      const cell = box('HORIZONTAL', w, opts.rowH || 20, { paddingLeft: 5, paddingRight: 5, counterAxisAlignItems: 'CENTER', itemSpacing: 4, primaryAxisAlignItems: cols[i][2] === 'R' ? 'MAX' : 'MIN' });
      border(cell, head ? "#C9D7EA" : "#EDF1F6", { t: 0, l: 0, r: 1, b: 1 });
      if (cv && typeof cv === 'object' && cv.node) cell.appendChild(cv.node);
      else if (cv && typeof cv === 'object' && cv.icon) { cell.appendChild(ico(cv.icon, 12)); cell.appendChild(txt(cv.text, 11, "Regular", cv.color || "#1E1E1E")); }
      else cell.appendChild(head ? txt(cv, 11, "SemiBold", "#1E395B") : (cols[i][3] === 'mono' ? mono(cv, 11, (rowOpts.colors && rowOpts.colors[i]) || "#1E1E1E") : txt(cv, 11, "Regular", (rowOpts.colors && rowOpts.colors[i]) || "#1E1E1E")));
      r.appendChild(cell);
    });
    return r;
  };
  t.appendChild(mk(cols.map(c => c[0]), true));
  for (const row of rows) t.appendChild(mk(row.cells || row, false, row.cells ? row : {}));
  border(t, "#C9D7EA");
  return t;
};
// ───────────────────────── charts (SVG) ─────────────────────────
const lineChart = (w, h, series, o = {}) => {
  const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1]));
  const x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs);
  const y0 = o.ymin ?? Math.min(0, ...ys), y1 = o.ymax ?? Math.max(...ys) * 1.08;
  const L = 30, R = 6, T = 6, B = 16;
  const X = x => L + (x - x0) / ((x1 - x0) || 1) * (w - L - R), Y = y => T + (1 - (y - y0) / ((y1 - y0) || 1)) * (h - T - B);
  let g = `<rect x="${L}" y="${T}" width="${w - L - R}" height="${h - T - B}" fill="#FFFFFF" stroke="#D5DEEA"/>`;
  for (let i = 1; i < 4; i++) { const yy = T + i * (h - T - B) / 4; g += `<path d="M${L} ${yy}H${w - R}" stroke="#EEF2F7"/>`; }
  for (const b of (o.bad || [])) g += `<rect x="${X(b[0])}" y="${T}" width="${Math.max(2, X(b[1]) - X(b[0]))}" height="${h - T - B}" fill="#E5484D" fill-opacity=".12"/>`;
  for (const r of (o.refs || [])) g += `<path d="M${L} ${Y(r)}H${w - R}" stroke="#C42B1C" stroke-dasharray="4 3"/>`;
  for (const s of series) {
    const d = s.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join('');
    if (s.area) g += `<path d="${d}L${X(s.pts[s.pts.length - 1][0])} ${Y(y0)}L${X(s.pts[0][0])} ${Y(y0)}Z" fill="${s.color}" fill-opacity=".12"/>`;
    g += `<path d="${d}" stroke="${s.color}" stroke-width="${s.width || 1.6}" ${s.dash ? 'stroke-dasharray="4 3"' : ''} fill="none" stroke-linejoin="round"/>`;
    if (s.dots) for (const p of s.pts) g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="#fff" stroke="${s.color}" stroke-width="1.5"/>`;
  }
  if (o.cursor !== undefined) g += `<path d="M${X(o.cursor)} ${T}V${h - B}" stroke="#EC4899" stroke-width="1.5"/>`;
  g += `<path d="M${L} ${h - B}H${w - R}" stroke="#8A94A6"/>`;
  const n = svg(`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" fill="none" xmlns="http://www.w3.org/2000/svg">${g}</svg>`);
  n.name = o.name || "chart";
  // axis labels as real text
  const wrap = figma.createFrame(); wrap.name = (o.name || "chart") + " (labelled)"; wrap.resize(w, h); wrap.fills = []; wrap.clipsContent = false;
  wrap.appendChild(n); n.x = 0; n.y = 0;
  const lab = (s, x, y, align) => { const t = txt(s, 9, "Regular", "#5B6B7F"); wrap.appendChild(t); t.x = align === 'R' ? x - t.width : x; t.y = y; };
  lab(fmt(y1), L - 3, T - 2, 'R'); lab(fmt(y0), L - 3, h - B - 10, 'R');
  lab(fmt(x0) + (o.xunit || ''), L, h - B + 2); lab(fmt(x1) + (o.xunit || ''), w - R, h - B + 2, 'R');
  return wrap;
};
const fmt = v => Math.abs(v) >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + 'k' : Math.abs(v) >= 10 ? Math.round(v).toString() : Math.abs(v) >= 1 ? v.toFixed(1) : v === 0 ? '0' : v.toFixed(2);
const barChart = (w, h, vals, color, o = {}) => {
  const L = 26, B = 14, T = 6, R = 4, max = o.max || Math.max(...vals) * 1.1;
  const bw = (w - L - R) / vals.length;
  let g = `<rect x="${L}" y="${T}" width="${w - L - R}" height="${h - T - B}" fill="#fff" stroke="#D5DEEA"/>`;
  if (o.ref) { const yy = T + (1 - o.ref / max) * (h - T - B); g += `<path d="M${L} ${yy}H${w - R}" stroke="#C42B1C" stroke-dasharray="4 3"/>`; }
  vals.forEach((v, i) => { const bh = v / max * (h - T - B); g += `<rect x="${L + i * bw + 1}" y="${h - B - bh}" width="${Math.max(1, bw - 2)}" height="${bh}" fill="${v === 0 ? '#E5484D' : color}"/>`; });
  const n = svg(`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" fill="none" xmlns="http://www.w3.org/2000/svg">${g}</svg>`); n.name = o.name || "bars";
  return n;
};
// ───────────────────────── data (real values from the project where known) ─────────────────────────
const KF = [
  // name, t_file, t_solved, x, y, z (null = free; solved shown with ≈), yaw (null = free)
  ["fo0", 0, 0, 1.5, 0, -0.7, -1.57, false],
  ["fo0a", 0.816, 0.839, 1.5, -1.01, -0.70, -1.8, true],
  ["fo1", 3.453, 3.506, 1.5, -6.5, -0.7, -2.8, false],
  ["fo2", 3.924, 3.998, 1.0, -7.0, -0.7, null, false],
  ["fo3", 5.443, 5.584, -1.0, -7.0, -0.7, -4.72, false],
  ["fo4", 5.898, 6.064, -1.4, -6.5, -0.7, -5.0, false],
  ["fo4a", 8.851, 9.025, -1.4, -0.98, -0.70, null, true],
  ["fo5", 9.322, 9.521, -1.4, -0.5, -0.7, -5.0, false],
  ["fo6", 10.472, 10.692, -1.0, 0, -0.7, null, false],
  ["fo7", 12.107, 12.344, 0, 0, -0.7, -6.28, false],
];
const SEL_KF = 3; // fo2 selected
// smooth path through keyframes (Catmull-Rom in time), sampled at 20 Hz
const PATH = (() => {
  const P = KF.map(k => [k[3], k[4], k[5]]), T = KF.map(k => k[1]);
  const out = [];
  for (let t = 0; t <= T[T.length - 1] + 1e-9; t += 0.05) {
    let i = 0; while (i < T.length - 2 && t > T[i + 1]) i++;
    const u = (t - T[i]) / (T[i + 1] - T[i]);
    const dt = T[i + 1] - T[i];
    const tan = (j, ax) => (j === 0 || j === P.length - 1) ? 0 : (P[j + 1][ax] - P[j - 1][ax]) / (T[j + 1] - T[j - 1]);
    const h = (ax) => { const p0 = P[i][ax], p1 = P[i + 1][ax], m0 = tan(i, ax) * dt, m1 = tan(i + 1, ax) * dt, u2 = u * u, u3 = u2 * u;
      return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1; };
    out.push([t, h(0), h(1), -0.7]);
  }
  return out;
})();
const SPEED = PATH.map((p, i) => { const q = PATH[Math.min(i + 1, PATH.length - 1)], r = PATH[Math.max(i - 1, 0)]; const dt = (q[0] - r[0]) || 0.05; return [p[0], Math.hypot(q[1] - r[1], q[2] - r[2]) / dt]; });
const vmaxRaw = Math.max(...SPEED.map(s => s[1]));
const SPEEDN = SPEED.map(([t, v]) => [t, v / vmaxRaw * 2.416]);
const smooth = (arr, k) => arr.map((p, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - k); j <= Math.min(arr.length - 1, i + k); j++) { s += arr[j][1]; n++; } return [p[0], s / n]; });
const ACC0 = PATH.map((p, i) => { const a = PATH[Math.max(i - 1, 0)], c = PATH[Math.min(i + 1, PATH.length - 1)]; return [p[0], Math.hypot(c[1] - 2 * p[1] + a[1], c[2] - 2 * p[2] + a[2]) / 0.0025]; });
const ACCS = smooth(ACC0, 3); const amax = Math.max(...ACCS.map(a => a[1]));
const ACC = ACCS.map(([t, a]) => [t, a / amax * 2.279]);
const THRUST = SPEEDN.map(([t], i) => [t, 0.36 + 0.06 * ACC[i][1] / 2.279]);
const RATE = SPEEDN.map(([t], i) => [t, 0.05 + 0.63 * 3 * (ACC[i][1] / 2.279) * 0.33]);
// gap = distance from the drone's 0.19 m sphere to the 5th-nearest sparse point: a pillar near the
// fo0a → fo1 straight (x 1.0, y −3.4) and a cabinet near the return leg
const GAP = PATH.map(([t]) => [t, 0.9 - 0.83 * Math.exp(-Math.pow((t - 1.95) / 0.35, 2)) - 0.45 * Math.exp(-Math.pow((t - 7.4) / 0.5, 2)) + 0.05 * Math.sin(t)]);
const CURSOR_T = 5.4;
const cursorIdx = Math.round(CURSOR_T / 0.05);
// SV-Net p4_smoke (intellisense08, 1 Oct)
const COMM_TTE = [[0, 227.6], [50, 127.0], [100, 76.9], [150, 74.9], [200, 339.1], [250, 61.0], [300, 30.3]];
const HIST_LOSS = [[1, 23.6], [10, 6.1], [25, 2.4], [50, 1.433], [75, 1.31], [100, 1.224], [125, 1.15], [150, 1.097], [175, 1.07], [200, 1.055]];
const HIST_TEST = HIST_LOSS.map(([e, v]) => [e, v * (e < 20 ? 1.05 : 0.976)]);
const COMM_LOSS = [[1, 0.068], [25, 0.026], [50, 0.0172], [75, 0.0115], [100, 0.0082], [150, 0.005], [200, 0.004], [250, 0.0032], [300, 0.00267]];
const COMM_TEST = [[1, 0.071], [25, 0.031], [50, 0.024], [75, 0.021], [100, 0.0195], [150, 0.0188], [200, 0.0186], [250, 0.0185], [300, 0.0185]];
const LOG8 = [
  "══ train_comm — regenerate observations, train commNet (GPU)",
  "    in-loop evaluation every 50 epochs on circuit with eval_single",
  "  Maverick > [histNet,commNet] Data Count:(4440 dpts)  3/3 datasets  (49.9s)",
  "  Maverick > commNet  epoch 1/300 epochs  train loss 0.06805",
  "  Maverick > commNet  epoch 50/300 epochs  train loss 0.01723",
  "  Maverick > commNet  epoch 100/300 epochs  train loss 0.00823",
  "  Maverick > commNet  epoch 150/300 epochs  train loss 0.00499",
  "  Maverick > commNet  epoch 200/300 epochs  train loss 0.00402",
  "  Maverick > commNet  epoch 250/300 epochs  train loss 0.00317",
  "  Maverick > commNet  epoch 300/300 epochs  train loss 0.00267",
  "Maverick > commNet : Best checkpoint is commNet_ckpt300.",
  "    VRAM baseline 215 MiB → peak 6785 MiB",
  "  ✔ Maverick commNet: 300 epochs, train 0.0026688, test 0.018516, 2311.7 s",
  "    'train_comm' completed in 40m 49s",
  "══ deploy — fly expert and students in FiGS, metrics and videos",
  "✅ Done loading checkpoint from outputs/backroom/splatfacto/2026-09-26_190524/…/step-000029999.ckpt",
  "  ✔ Viper      per-point tracking error mean 0.001 m, max 0.0172 m, 100% within 0.3 m",
  "  ✔ Maverick   per-point tracking error mean 9.4337 m, max 79.2759 m, 16% within 0.3 m",
  "    'deploy' completed in 1m 32s",
  "══ Done — total 49m 15s",
];
const JOBS = [
  ["#8", "svnet", "p4_smoke · preflight to deploy", "Succeeded", "49m 15s", "20:33"],
  ["#7", "figs", "backroom · course to record (bringup)", "Succeeded", "46.0s", "20:22"],
  ["#6", "selftest", "selftest 60s", "Interrupted", "4s", "20:15"],
  ["#5", "figs", "backroom · preflight", "Succeeded", "6.8s", "20:14"],
  ["#4", "figs", "backroom_t4 · course to record", "Succeeded", "34.0s", "27 Sep"],
  ["#3", "figs", "backroom_t4 · train (cache cpu)", "Succeeded", "60m 24s", "27 Sep"],
  ["#2", "svnet", "p4_smoke · train_comm (dummy 4 GB)", "Failed", "3m 10s", "27 Sep"],
];
// ───────────────────────── 3D course view (projected to SVG) ─────────────────────────
// course frame: x, y, z down. Camera looks from above one corner, like the editor's default view.
const viewport3D = (W, H) => {
  const cam = { yaw: -0.62, pitch: 0.78, dist: 13.5, cx: 0.05, cy: -3.5, cz: -0.4, f: 900 };
  const proj = ([x, y, z]) => {
    let dx = x - cam.cx, dy = y - cam.cy, dz = -(z - cam.cz); // up = -z
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    let X = dx * cy - dy * sy, Yd = dx * sy + dy * cy;           // rotate about up
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const depth = Yd * cp + dz * sp + cam.dist, Up = -Yd * sp + dz * cp;
    return [W / 2 + X * cam.f / depth, H * 0.5 - Up * cam.f / depth, depth];
  };
  let rnd = 7; const R = () => { rnd = (rnd * 16807) % 2147483647; return rnd / 2147483647; };
  let g = `<rect width="${W}" height="${H}" fill="#F7F9FC"/>`;
  // floor grid at z = 0
  for (let x = -4; x <= 4; x += 0.5) { const a = proj([x, -10, 0]), b = proj([x, 3, 0]); g += `<path d="M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}" stroke="${x % 1 === 0 ? '#C7CED8' : '#E1E6ED'}"/>`; }
  for (let y = -10; y <= 3; y += 0.5) { const a = proj([-4, y, 0]), b = proj([4, y, 0]); g += `<path d="M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}" stroke="${y % 1 === 0 ? '#C7CED8' : '#E1E6ED'}"/>`; }
  // sparse points: floor clutter, walls, furniture blobs, a pillar
  const pts = [];
  for (let i = 0; i < 1400; i++) pts.push([-2.7 + R() * 5.5, -8.4 + R() * 9.8, -R() * 0.05, [150, 140, 128]]);
  for (let i = 0; i < 900; i++) { const side = Math.floor(R() * 3); const h = -R() * 2.5; if (side === 0) pts.push([-2.75, -8.4 + R() * 9.8, h, [120, 142, 168]]); else if (side === 1) pts.push([-2.7 + R() * 5.5, -8.45, h, [132, 150, 172]]); else pts.push([2.8, -8.4 + R() * 9.8, h, [140, 128, 118]]); }
  for (let i = 0; i < 260; i++) pts.push([-2.6 + R() * 0.7, -2.6 + R() * 1.4, -R() * 1.2, [176, 96, 70]]);
  for (let i = 0; i < 180; i++) { const a = R() * 6.283; pts.push([1.0 + 0.18 * Math.cos(a), -3.4 + 0.18 * Math.sin(a), -R() * 2.5, [196, 90, 86]]); }
  pts.sort((a, b) => proj(b)[2] - proj(a)[2]);
  for (const p of pts) { const q = proj(p); g += `<circle cx="${q[0].toFixed(1)}" cy="${q[1].toFixed(1)}" r="1.6" fill="rgb(${p[3].join(',')})" fill-opacity=".85"/>`; }
  // camera box and waypoint box
  const boxEdges = (lo, hi) => { const v = []; for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) v.push([x, y, z]); const e = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]; return e.map(([a, b]) => { const p = proj(v[a]), q = proj(v[b]); return `M${p[0].toFixed(1)} ${p[1].toFixed(1)}L${q[0].toFixed(1)} ${q[1].toFixed(1)}`; }).join(''); };
  g += `<path d="${boxEdges([-2.1, -7.9, -0.25], [2.2, 0.9, -1.45])}" stroke="#8A8F98" stroke-width="1"/>`;
  g += `<path d="${boxEdges([-1.6, -7.4, -0.75], [1.7, 0.4, -0.95])}" stroke="#2F9E6E" stroke-width="1.4" stroke-dasharray="6 4"/>`;
  // camera path
  let cp = ""; for (let i = 0; i <= 60; i++) { const t = i / 60; const q = proj([1.9 * Math.cos(t * 6.283) * 0.9, -3.5 + 4.2 * Math.sin(t * 6.283), -1.2 - 0.1 * Math.sin(t * 12)]); cp += `${i ? 'L' : 'M'}${q[0].toFixed(1)} ${q[1].toFixed(1)}`; }
  g += `<path d="${cp}" stroke="#8A8F98" stroke-opacity=".7" stroke-width="1"/>`;
  // path coloured by speed, red where the gap is under the threshold
  const vmax = 2.416;
  for (let i = 1; i < PATH.length; i++) {
    const a = proj([PATH[i - 1][1], PATH[i - 1][2], PATH[i - 1][3]]), b = proj([PATH[i][1], PATH[i][2], PATH[i][3]]);
    const u = Math.min(1, SPEEDN[i][1] / vmax);
    const col = GAP[i][1] < 0.15 ? '#E5484D' : `rgb(${Math.round(59 + (245 - 59) * u)},${Math.round(130 + (158 - 130) * u)},${Math.round(246 + (11 - 246) * u)})`;
    g += `<path d="M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}" stroke="${col}" stroke-width="3.2" stroke-linecap="round"/>`;
  }
  // keyframes with yaw arrows
  const labels = [];
  KF.forEach((k, i) => {
    const q = proj([k[3], k[4], k[5]]); const sel = i === SEL_KF; const col = sel ? '#F59E0B' : '#2F6FDD';
    if (k[6] !== null) { const e = proj([k[3] + 0.45 * Math.cos(k[6]), k[4] + 0.45 * Math.sin(k[6]), k[5]]); g += `<path d="M${q[0].toFixed(1)} ${q[1].toFixed(1)}L${e[0].toFixed(1)} ${e[1].toFixed(1)}" stroke="${col}" stroke-width="2.4"/>`; }
    g += `<circle cx="${q[0].toFixed(1)}" cy="${q[1].toFixed(1)}" r="${sel ? 7 : 5.5}" fill="${col}" fill-opacity="${k[7] ? 0.45 : 1}" stroke="#fff" stroke-width="1.5"/>`;
    labels.push([k[0], q[0] + 8, q[1] - 16, sel]);
  });
  // goal marker
  const gq = proj([0.7, -3.6, -1.0]); g += `<path d="M${gq[0]} ${gq[1] - 9}l8 9-8 9-8-9z" fill="none" stroke="#A855F7" stroke-width="1.6"/>`; labels.push(["goal: the red chair", gq[0] + 10, gq[1] - 8, 'goal']);
  // min-gap marker with drone sphere
  const mi = GAP.reduce((m, p, i) => p[1] < GAP[m][1] ? i : m, 0); const mq = proj([PATH[mi][1], PATH[mi][2], PATH[mi][3]]);
  const r0 = Math.abs(proj([PATH[mi][1] + 0.19, PATH[mi][2], PATH[mi][3]])[0] - mq[0]);
  g += `<circle cx="${mq[0].toFixed(1)}" cy="${mq[1].toFixed(1)}" r="${Math.max(r0, 9).toFixed(1)}" fill="#E5484D" fill-opacity=".1" stroke="#E5484D" stroke-dasharray="3 2"/>`;
  labels.push([`${GAP[mi][1].toFixed(2)} m`, mq[0] + 12, mq[1] + 2, 'gap']);
  // cursor + velocity
  const cq = proj([PATH[cursorIdx][1], PATH[cursorIdx][2], PATH[cursorIdx][3]]);
  const nx = PATH[cursorIdx + 1], vq = proj([nx[1] + (nx[1] - PATH[cursorIdx][1]) * 6, nx[2] + (nx[2] - PATH[cursorIdx][2]) * 6, nx[3]]);
  g += `<path d="M${cq[0].toFixed(1)} ${cq[1].toFixed(1)}L${vq[0].toFixed(1)} ${vq[1].toFixed(1)}" stroke="#EC4899" stroke-width="2.4"/>`;
  const node = svg(`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" fill="none" xmlns="http://www.w3.org/2000/svg">${g}</svg>`);
  node.name = "3D view (course frame, z down)";
  return { node, labels, cursor: cq };
};
// ───────────────────────── window shell shared by all screens ─────────────────────────
const W = 1920, H = 1080, EXW = 250, PRW = 300;
const shadow = { type: 'DROP_SHADOW', color: { r: 0, g: 0, b: 0, a: 0.35 }, offset: { x: 0, y: 6 }, radius: 18, spread: 0, visible: true, blendMode: 'NORMAL' };
const STATUS = (queue, prog) => [
  { icon: "machine", label: "intellisense08 · ui/machines/intellisense08.toml", h: "Machine profile in use. Click to see paths and defaults." },
  { icon: "gpu", label: "RTX 2080 · 3.2 / 8 GB · 61 °C", h: "Live nvidia-smi. Amber when a process outside Galley holds the GPU." },
  { icon: "disk", label: "568 GB free", h: "Free space under project_root." },
  { icon: prog !== undefined ? "run" : "queue", label: queue, progress: prog, h: "The job queue: one GPU job at a time. Click to open the Queue." },
  { icon: null, label: "", grow: true },
  { icon: "ok", label: "Connected · 127.0.0.1:8800 (VS Code port forward)", h: "Backend health (/api/health): env_script and pipeline found." },
];
const windowShell = (name, title, ribbonKey, statusFields) => {
  const win = box('VERTICAL', W, H, { name: `Screen · ${name}`, itemSpacing: 0, clipsContent: false });
  win.fills = [solid("#DCE6F4")]; win.strokes = [solid("#5A7EB0")]; win.strokeWeight = 1; win.effects = [shadow];
  add(win, titleBar(title, W));
  const rib = RIBBONS[ribbonKey].createInstance(); rib.name = `ribbon (${ribbonKey})`; add(win, rib);
  const main = box('HORIZONTAL', W, H - 30 - rib.height - 24, { name: "workspace", itemSpacing: 4, paddingTop: 4, paddingBottom: 4, paddingLeft: 4, paddingRight: 4 });
  add(win, main);
  add(win, statusBar(statusFields, W));
  return { win, rib, main, mainH: H - 30 - rib.height - 24 - 8 };
};
// Explorer tree
const EXPLORER = (sel) => [
  [0, "Scenes", "folder", "5", true],
  [1, "backroom", "scene", "active model", true],
  [2, "Models", "splat", "1 active · 1 archived", true, sel === "Models"],
  [2, "Courses", "folder", "3", true],
  [3, "circuit", "route", "12.1 s", false, sel === "circuit"],
  [3, "backroom_loop", "route", "8.6 s", false],
  [3, "square_loop", "route", "", false],
  [2, "Flights", "video", "4", true],
  [1, "flightroom", "scene", "", true],
  [1, "mid_gate", "scene", "", true],
  [1, "src_open", "scene", "", true],
  [1, "backroom1", "warning", "not loadable", false],
  [0, "Cohorts", "folder", "1", true],
  [1, "p4_smoke", "cohort", "deploy ✓", true, sel === "p4_smoke"],
  [2, "Maverick", "network", "student", false],
  [1, "p4_beta (draft)", "cohort", "data_beta", false],
  [0, "Configs", "folder", "", true],
  [1, "courses", "route", "6", true, sel === "courses"],
  [1, "pilots", "network", "Viper · Maverick · Iceman", true],
  [1, "frames", "drone", "carl", true],
  [1, "methods", "steps", "data_* · eval_*", true],
  [0, "Jobs", "folder", "8", true],
  [1, "#8 svnet p4_smoke", "ok", "49 min", false, sel === "#8"],
  [1, "#7 figs backroom", "ok", "46 s", false],
  [1, "#6 selftest", "pause", "interrupted", false],
];
const explorerPane = (h, sel) => {
  const { frame, body } = pane("Explorer", EXW, h);
  const search = box('HORIZONTAL', EXW - 10, 22, { paddingLeft: 6, paddingRight: 4, itemSpacing: 4, counterAxisAlignItems: 'CENTER' });
  search.fills = [solid("#FFFFFF")]; border(search, "#ABADB3"); search.appendChild(txt("Search scenes, courses, jobs…", 11, "Regular", "#8A94A6")); spacer(search); search.appendChild(ico("search", 14));
  help(search, "Filter the tree (Ctrl+E). Matches names, run ids and job numbers.");
  add(body, search); body.itemSpacing = 1;
  const gap = rect(10, 4, null); add(body, gap);
  for (const [d, label, icon, meta, kids, isSel] of EXPLORER(sel)) {
    const row = al('HORIZONTAL', { paddingLeft: d * 14, name: `tree: ${label}` });
    const it = treeItem(label, icon, meta, isSel ? "Selected" : "Default", kids);
    it.resize(EXW - 12 - d * 14, 22);
    if (d === 0) { const lab = it.findOne(n => n.name === "Label"); if (lab && lab.type === 'TEXT') { /* keep instance text style */ } }
    row.appendChild(it); body.appendChild(row);
  }
  return frame;
};
// Properties pane
const propertiesPane = (h, title, sections, extra) => {
  const { frame, body } = pane("Properties", PRW, h);
  const head = al('HORIZONTAL', { paddingLeft: 4, paddingTop: 2, paddingBottom: 6, itemSpacing: 6, counterAxisAlignItems: 'CENTER' });
  head.appendChild(ico(title[1], 16)); head.appendChild(txt(title[0], 12, "Bold", "#1E395B")); add(body, head, { fillW: true });
  propGrid(body, sections);
  if (extra) for (const e of extra) add(body, e, { fillW: !e.name.endsWith("(no-fill)") });
  return frame;
};
// document tabs + document body + output panel
const centerColumn = (w, h, tabs, outputH) => {
  const col = box('VERTICAL', w, h, { name: "documents", itemSpacing: 0 });
  const strip = box('HORIZONTAL', w, 27, { name: "document tabs", itemSpacing: 2, counterAxisAlignItems: 'MAX', paddingLeft: 4 });
  for (const [label, icon, active] of tabs) strip.appendChild(docTab(label, icon, active));
  spacer(strip);
  const split = smallBtn("", "panes", "Default", "Split the document area (View ▸ Window ▸ Split right)."); strip.appendChild(split);
  add(col, strip, { fillW: true });
  const doc = box('VERTICAL', w, h - 27 - (outputH ? outputH + 4 : 0), { name: "document", itemSpacing: 6, paddingTop: 6, paddingBottom: 6, paddingLeft: 6, paddingRight: 6 });
  doc.fills = [solid("#FFFFFF")]; border(doc, "#A0B3CC", { t: 1, l: 1, r: 1, b: 1 });
  add(col, doc, { fillW: true });
  let out = null;
  if (outputH) { add(col, rect(10, 4, null)); out = box('VERTICAL', w, outputH, { name: "output panel", itemSpacing: 0 }); out.fills = [solid("#FFFFFF")]; border(out, "#A0B3CC"); add(col, out, { fillW: true }); }
  return { col, doc, out, docW: w - 12, docH: h - 27 - (outputH ? outputH + 4 : 0) - 12 };
};
const outputTabs = (out, tabs, w) => {
  const hd = box('HORIZONTAL', w, 26, { name: "output tabs", itemSpacing: 2, paddingLeft: 6, paddingRight: 6, counterAxisAlignItems: 'CENTER' });
  hd.fills = [vgrad([0,"#F7FAFE"],[1,"#DDE7F4"])]; border(hd, "#A0B3CC", { t: 0, l: 0, r: 0, b: 1 });
  hd.appendChild(txt("Output", 12, "SemiBold", "#1E395B")); hd.appendChild(rect(12, 2, null));
  for (const [label, icon, active] of tabs) {
    const b = smallBtn(label, icon, active ? "Checked" : "Default"); hd.appendChild(b);
  }
  spacer(hd); hd.appendChild(ico("pin", 14));
  add(out, hd, { fillW: true });
  const body = al('HORIZONTAL', { name: "output body", itemSpacing: 8, paddingTop: 6, paddingLeft: 8, paddingRight: 8, paddingBottom: 6 });
  add(out, body, { fillW: true }); body.layoutSizingVertical = 'FILL';
  return body;
};
const logBox = (lines, w, h, title) => {
  const f = box('VERTICAL', w, h, { name: title || "log", itemSpacing: 1, paddingTop: 4, paddingLeft: 6, paddingRight: 6 });
  f.fills = [solid("#FBFCFE")]; border(f, "#D5DEEA");
  for (const l of lines) {
    const col = l.includes("✔") || l.includes("✅") ? "#2E8B3D" : l.includes("✗") || l.includes("Maverick   per-point") ? "#C42B1C" : l.startsWith("══") ? "#2B67B8" : "#2A2F38";
    f.appendChild(mono(l, 10.5, col));
  }
  return f;
};
const queueList = (w, rows, selIdx = 0) => table([["#", 34], ["kind", 56], ["what", w - 34 - 56 - 96 - 70 - 2], ["status", 96], ["time", 70, 'R']],
  rows.map((r, i) => ({ cells: [r[0], r[1], r[2], { node: pill(r[3]) }, r[4]], sel: i === selIdx })), { name: "queue" });
const kpi = (value, label, bad) => {
  const k = al('VERTICAL', { name: `kpi: ${label}`, itemSpacing: 0, paddingTop: 4, paddingBottom: 4, paddingLeft: 8, paddingRight: 10 });
  k.fills = [solid(bad ? "#FDF0EE" : "#F5F8FC")]; border(k, bad ? "#E8A39B" : "#D5DEEA"); k.cornerRadius = 3;
  k.appendChild(txt(value, 14, "Bold", bad ? "#C42B1C" : "#1E395B")); k.appendChild(txt(label, 10, "Regular", "#5B6B7F"));
  return k;
};
const hoverTip = (win, rib, cmdName, tipArgs) => {
  const target = rib.findOne(n => n.name === `cmd: ${cmdName}`);
  const tip = tooltip(...tipArgs); tip.name = `hover tooltip · ${cmdName}`;
  win.appendChild(tip); tip.layoutPositioning = 'ABSOLUTE';
  if (target) { const wx = win.absoluteTransform[0][2], wy = win.absoluteTransform[1][2]; tip.x = target.absoluteTransform[0][2] - wx + 8; tip.y = target.absoluteTransform[1][2] - wy + target.height + 6; }
  else { tip.x = 600; tip.y = 160; }
  const cursor = svg('<svg width="14" height="20" viewBox="0 0 14 20" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 1v15l4-4 3 7 2.5-1-3-6.5H13z" fill="#fff" stroke="#000" stroke-linejoin="round"/></svg>');
  cursor.name = "mouse pointer"; win.appendChild(cursor); cursor.layoutPositioning = 'ABSOLUTE'; cursor.x = tip.x + 10; cursor.y = tip.y - 22;
  return tip;
};
// ───────────────────────── Screen 1: Course workspace ─────────────────────────
const screenCourse = () => {
  const S = windowShell("Course workspace", "Galley — backroom / circuit  ·  intellisense08", "Course", STATUS("Queue: idle · last job #8 succeeded"));
  const { win, rib, main, mainH } = S;
  add(main, explorerPane(mainH, "circuit"));
  const CW = W - 8 - 8 - EXW - PRW;
  const C = centerColumn(CW, mainH, [["backroom / circuit", "route", true], ["p4_smoke", "cohort", false], ["Job #8", "job", false], ["Viper.json", "config", false]], 186);
  add(main, C.col);
  // top: 3D view + keyframe grid
  const top = al('HORIZONTAL', { name: "view + keyframes", itemSpacing: 6 }); add(C.doc, top, { fillW: true });
  const VW = 836, VH = 412;
  const vf = figma.createFrame(); vf.name = "3D view"; vf.resize(VW, VH); vf.clipsContent = true; vf.fills = [solid("#F7F9FC")]; border(vf, "#B9C9DE");
  const V = viewport3D(VW, VH); vf.appendChild(V.node);
  for (const [s, x, y, kind] of V.labels) {
    const col = kind === true ? "#B45309" : kind === 'goal' ? "#7E3BC4" : kind === 'gap' ? "#C42B1C" : "#1E395B";
    const t = txt(s, 10.5, kind === true || kind === 'gap' ? "Bold" : "SemiBold", col); vf.appendChild(t); t.x = x; t.y = y;
  }
  const dr = ico("drone", 30); vf.appendChild(dr); dr.x = V.cursor[0] - 15; dr.y = V.cursor[1] - 15; dr.name = "drone at cursor (true scale in the app: 3D model)";
  const ctag = txt("t = 5.40 s · 2.31 m/s · drone banked 13°", 10.5, "SemiBold", "#BE185D"); vf.appendChild(ctag); ctag.x = V.cursor[0] + 18; ctag.y = V.cursor[1] + 8;
  const legend = al('VERTICAL', { name: "legend", itemSpacing: 2, paddingTop: 5, paddingBottom: 5, paddingLeft: 8, paddingRight: 8 }); legend.fills = [solid("#FFFFFF", 0.88)]; border(legend, "#D5DEEA"); legend.cornerRadius = 3;
  for (const l of ["grey box: camera box (where the camera went, not free space)", "green dashed box: waypoint box (inset 0.5 m)", "path: blue slow, amber fast (0–2.42 m/s) · red = gap < 0.15 m", "53,286 sparse points (all shown) · red ring = drone's 0.19 m sphere at the closest point"]) legend.appendChild(txt(l, 10, "Regular", "#3A4A5E"));
  vf.appendChild(legend); legend.x = 8; legend.y = VH - legend.height - 8;
  const gz = svg('<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="30" fill="#fff" fill-opacity=".7"/><path d="M32 32L54 40" stroke="#E5484D" stroke-width="2.5"/><path d="M32 32L14 42" stroke="#2F9E6E" stroke-width="2.5"/><path d="M32 32V58" stroke="#2F6FDD" stroke-width="2.5"/><circle cx="54" cy="40" r="5" fill="#E5484D"/><circle cx="14" cy="42" r="5" fill="#2F9E6E"/><circle cx="32" cy="58" r="5" fill="#2F6FDD"/></svg>');
  gz.name = "axis gizmo (x, y, z↓)"; vf.appendChild(gz); gz.x = VW - 74; gz.y = VH - 74;
  const vtool = al('HORIZONTAL', { name: "view toolbar", itemSpacing: 1, paddingLeft: 2, paddingRight: 2, paddingTop: 2, paddingBottom: 2 }); vtool.fills = [solid("#FFFFFF", 0.9)]; border(vtool, "#C9D7EA"); vtool.cornerRadius = 3;
  for (const [ic, h] of [["scene", "Frame the scene (F)."], ["search", "Zoom to selection (Z)."], ["boxes", "Top view (T)."], ["campath", "Follow the drone while playing."]]) vtool.appendChild(smallBtn("", ic, "Default", h));
  vf.appendChild(vtool); vtool.x = VW - vtool.width - 8; vtool.y = 8;
  top.appendChild(vf);
  // keyframe grid
  const KW = C.docW - VW - 6;
  const kt = tile("Keyframes · circuit", KW, VH, "matrix", "10 keyframes · Nco 6 · saved");
  const rows = KF.map((k, i) => ({
    cells: [{ node: ico(k[7] ? "outlier" : "route", 12) }, k[0], k[1].toFixed(3), k[2].toFixed(3), k[3].toFixed(2), (k[7] ? "≈" : "") + k[4].toFixed(2), (k[7] ? "≈" : "") + k[5].toFixed(2), k[6] === null ? "free" : k[6].toFixed(2), { node: ico("ok", 12) }],
    sel: i === SEL_KF, colors: [null, null, null, Math.abs(k[2] - k[1]) > 0.1 ? "#B45309" : "#1E1E1E", null, k[7] ? "#8A94A6" : null, k[7] ? "#8A94A6" : null, k[6] === null ? "#8A94A6" : null],
  }));
  const ktab = table([["", 22], ["name", 52], ["t file", 50, 'R', 'mono'], ["t solved", 58, 'R', 'mono'], ["x", 46, 'R', 'mono'], ["y", 50, 'R', 'mono'], ["z", 50, 'R', 'mono'], ["yaw", 48, 'R', 'mono'], ["in", 26]], rows, { name: "keyframe grid" });
  add(kt.body, ktab);
  const note = wrapTxt(txt("≈ = free cell, value from the solver. Amber t solved = the expert re-timed it (Viper kT = 10): Keyframe Tools ▸ Write solved times makes file and flight agree.", 11, "Regular", "#5B6B7F"), KW - 24); add(kt.body, note);
  const kbtns = al('HORIZONTAL', { itemSpacing: 4 }); for (const [l, ic, h] of [["Insert after", "insert", "New keyframe after fo2 (Ins)."], ["Delete", "delete", "Delete fo2 (Del)."], ["Matrix…", "matrix", "Edit fo2's derivative constraints."], ["Write solved times", "retime", "Copy re-timed t values into the file."]]) kbtns.appendChild(smallBtn(l, ic, "Default", h));
  add(kt.body, kbtns);
  top.appendChild(kt.frame);
  // KPIs
  const kp = al('HORIZONTAL', { name: "preview KPIs", itemSpacing: 6, counterAxisAlignItems: 'CENTER' }); add(C.doc, kp, { fillW: true });
  kp.appendChild(txt("Preview", 12, "Bold", "#1E395B")); kp.appendChild(txt("re-timed like Viper · kT 10 · 20 Hz · solved in 19.2 s", 11, "Regular", "#5B6B7F"));
  for (const [v, l, bad] of [["12.344 s", "duration (file 12.107)"], ["17.84 m", "path length"], ["2.416 m/s", "max speed"], ["2.279 m/s²", "max accel"], ["42 %", "thrust vs limit"], ["21 %", "body rate vs limit"], ["0.12 m", "min gap at 1.95 s", true], ["0 %", "outside capture"]]) kp.appendChild(kpi(v, l, bad));
  // charts
  const ch = al('HORIZONTAL', { name: "preview charts", itemSpacing: 6 }); add(C.doc, ch, { fillW: true });
  const cw = Math.floor((C.docW - 5 * 6) / 6), chh = 150;
  const gapBad = []; let st = null; GAP.forEach(([t, v]) => { if (v < 0.15 && st === null) st = t; if (v >= 0.15 && st !== null) { gapBad.push([st, t]); st = null; } });
  for (const [title, series, o] of [
    ["Speed (m/s)", [{ pts: SPEEDN, color: "#2F6FDD", area: true }], {}],
    ["Acceleration (m/s²)", [{ pts: ACC, color: "#F0A030" }], {}],
    ["Thrust (fraction of limit)", [{ pts: THRUST, color: "#8E5CC9" }], { ymin: 0, ymax: 1.05, refs: [1] }],
    ["Largest body rate |ω| (rad/s)", [{ pts: RATE, color: "#2AA7A0" }], { ymin: 0, ymax: 3.2, refs: [3] }],
    ["Gap: drone sphere to 5th point (m)", [{ pts: GAP, color: "#3FA34D", area: true }], { ymin: 0, refs: [0.15], bad: gapBad }],
    ["Altitude −z (m)", [{ pts: PATH.map(p => [p[0], -p[3]]), color: "#5B6475" }], { ymin: 0, ymax: 1.2 }],
  ]) {
    const t = tile(title, cw, chh, null); t.body.paddingLeft = 4; t.body.paddingRight = 4;
    t.body.appendChild(lineChart(cw - 10, chh - 42, series, Object.assign({ cursor: CURSOR_T, xunit: " s", name: title }, o)));
    help(t.frame, `${title}. Hover to move the cursor; the drone in the 3D view follows. Red bands: limit exceeded.`);
    ch.appendChild(t.frame);
  }
  // output panel
  const ob = outputTabs(C.out, [["Live log", "log", true], ["Queue", "queue", false], ["Problems (1)", "warning", false], ["GPU", "gpu", false]], CW);
  ob.appendChild(queueList(520, JOBS.slice(0, 5), 0));
  ob.appendChild(logBox([
    "$ course_tools.py preview --scene backroom --pilot Viper --frame carl --mode expert --body-radius 0.19",
    "  MinTimeSnap(kT=10, use_l2_time) · 10 keyframes · Nco 6",
    "  circuit - Ideal Time Steps: [0. 0.839 3.506 3.998 5.584 6.064 9.025 9.521 10.692 12.344]",
    "  TsFO_to_tXU · 247 samples · inputs within Viper bounds",
    "  clearance: k=5, body radius 0.19 m → min gap 0.12 m at t = 1.95 s",
    "  ✔ solved in 19.2 s",
    "! Problems: gap under 0.15 m between fo0a and fo1 (pillar at x 1.0, y −3.4)",
  ], CW - 520 - 26, 140, "preview log"));
  // properties
  const mtx = table([["", 34], ["pos", 46, 'R', 'mono'], ["vel", 42, 'R', 'mono'], ["acc", 42, 'R', 'mono'], ["jerk", 42, 'R', 'mono'], ["snap", 46, 'R', 'mono']],
    [["x", "1.00", "free", "free", "free", "—"], ["y", "-7.00", "free", "free", "free", "—"], ["z", "-0.70", "free", "free", "free", "—"], ["yaw", "free", "free", "free", "free", "—"]].map(r => ({ cells: r, colors: r.map(v => v === "free" || v === "—" ? "#8A94A6" : "#1E1E1E") })), { name: "derivative matrix" });
  const mtxWrap = al('VERTICAL', { paddingLeft: 6, paddingTop: 4, paddingBottom: 6, itemSpacing: 4 }); mtxWrap.appendChild(txt("Derivative constraints (empty = free)", 10, "SemiBold", "#3E6AAA")); mtxWrap.appendChild(mtx); mtxWrap.name = "derivative matrix (no-fill)";
  add(main, propertiesPane(mainH, ["Keyframe fo2", "route"], [
    ["Keyframe", [["Name", "fo2", { mono: true }], ["t (file)", "3.924 s", { mono: true }], ["t (solved)", "3.998 s  (+0.074)", { mono: true, color: "#B45309", h: "With Viper's kT the expert re-optimises times." }], ["Position", "1.00, -7.00, -0.70", { mono: true }], ["Yaw", "free", { color: "#8A94A6" }], ["Inside capture", "yes", { icon: "ok" }]]],
  ], [mtxWrap,
    propSection("At the cursor  (t = 5.40 s)"), propRow("Speed", "2.31 m/s"), propRow("Acceleration", "1.42 m/s²"), propRow("Thrust", "39 % of limit"), propRow("Attitude", "roll 13° · pitch −4° · yaw −4.6", { mono: true }), propRow("Gap", "0.86 m", { icon: "ok" }),
    propSection("Drone"), propRow("Model", "Drone_5_2205 (CAD)"), propRow("Size", "29 × 33 × 10 cm"), propRow("Sphere", "0.19 m (guards included)"),
    propSection("Course file"), propRow("File", "configs/courses/circuit.json", { mono: true }), propRow("Overlay", "mirrored", { icon: "ok" }), propRow("Integer cells", "none", { icon: "ok" }),
  ]));
  hoverTip(win, rib, "Re-time", ["Re-time", "Re-time like the expert: Viper's kT re-optimises segment times, as the real flight does. 20 s – 2 min.", "course_tools.py preview --mode expert", ""]);
  return win;
};

// ───────────────────────── Screen 2: Scene & Splat ─────────────────────────
const screenScene = () => {
  const S = windowShell("Scene & Splat", "Galley — backroom  ·  intellisense08", "Capture & Splat", STATUS("Queue: idle"));
  const { win, rib, main, mainH } = S;
  add(main, explorerPane(mainH, "Models"));
  const CW = W - 8 - 8 - EXW - PRW;
  const C = centerColumn(CW, mainH, [["backroom", "scene", true], ["backroom / circuit", "route", false], ["New capture", "camera", false]], 186);
  add(main, C.col);
  // step strip
  const steps = ["preflight", "probe", "transcode", "aruco", "config", "patch", "sfm", "train", "verify", "bounds", "course", "simulate", "validate", "record"];
  const strip = al('HORIZONTAL', { name: "pipeline steps", itemSpacing: 0, counterAxisAlignItems: 'CENTER' }); add(C.doc, strip, { fillW: true });
  const sw = Math.floor(C.docW / steps.length);
  steps.forEach((s, i) => {
    const done = true;
    const node = svg(`<svg width="${sw}" height="34" viewBox="0 0 ${sw} 34" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M${i ? 0 : 1} 1H${sw - 12}L${sw - 1} 17L${sw - 12} 33H${i ? 0 : 1}${i ? `L11 17Z` : 'Z'}" fill="${done ? '#E3F4E5' : '#F2F5F9'}" stroke="${done ? '#7CC48A' : '#C9D1DC'}"/></svg>`);
    const cell = figma.createFrame(); cell.resize(sw, 34); cell.fills = []; cell.clipsContent = false; cell.name = `step: ${s}`; cell.appendChild(node);
    const t = txt(s, 11, "SemiBold", "#2E6B37"); cell.appendChild(t); t.x = (i ? 16 : 8); t.y = 3;
    const t2 = txt(["4.6 s", "0.4 s", "12 s", "3.1 s", "0.1 s", "0.2 s", "38 m", "60 m", "8 s", "1 s", "0.3 s", "46 s", "2 s", "0.1 s"][i], 10, "Regular", "#5B6B7F"); cell.appendChild(t2); t2.x = t.x; t2.y = 18;
    help(cell, `Step ${s}: done (marker .figs_pipeline_state/backroom/${s}.done). Right-click ▸ Redo from here.`);
    strip.appendChild(cell);
  });
  // row 1
  const r1 = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, r1, { fillW: true });
  const tw1 = 330, tw2 = Math.floor((C.docW - tw1 - 12) / 2), th = 230;
  const rec = tile("Reconstruction", tw1, th, "sfm", "hloc + COLMAP");
  propGrid(rec.body, [["", [["Registered", "300 / 300 (100 %)", { icon: "ok" }], ["Sparse points", "53,286"], ["Images", "300 @ 1920×1080"], ["Match pairs", "44,850 (exhaustive)"], ["Marker scale", "0.150 m · id 0"], ["Checkpoint", "375 MB"], ["Active run", "2026-09-26_190524", { mono: true }]]]]);
  rec.body.children[0].visible = false;
  r1.appendChild(rec.frame);
  const tr = tile("Training (splatfacto)", tw2, th, "chart", "30k steps · cache cpu · 2360 MiB peak");
  const loss = []; for (let s = 0; s <= 30000; s += 500) loss.push([s, 0.032 + 0.18 * Math.exp(-s / 2500) + 0.004 * Math.sin(s / 900)]);
  const psnr = []; for (let s = 0; s <= 30000; s += 500) psnr.push([s, (17 + 11 * (1 - Math.exp(-s / 4000))) / 1000 * 8]);
  tr.body.appendChild(lineChart(tw2 - 20, th - 64, [{ pts: loss, color: "#2F6FDD", area: true }], { ymin: 0, name: "train loss" }));
  tr.body.appendChild(txt("train loss (blue) from the run's TensorBoard event files · hover for step, loss, PSNR", 10, "Regular", "#5B6B7F"));
  r1.appendChild(tr.frame);
  const ar = tile("ArUco detections per 10 s", tw2, th, "marker", "marker 0 · 0.150 m");
  ar.body.appendChild(barChart(tw2 - 20, th - 82, [14, 22, 30, 41, 38, 26, 9, 0, 12, 33, 45, 40, 28, 17, 21, 35, 30, 11], "#2F6FDD", { ref: 6 }));
  ar.body.appendChild(wrapTxt(txt("total 452 detections vs 120 needed (3 × 40 marked frames) · 17 / 18 windows · median marker 74 px", 11, "Regular", "#1E395B"), tw2 - 24));
  ar.body.appendChild(txt("red bar = window with no detections (the phone faced away from the marker)", 10, "Regular", "#5B6B7F"));
  r1.appendChild(ar.frame);
  // row 2
  const r2 = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, r2, { fillW: true });
  const mw = 640, fw = C.docW - mw - 6, th2 = C.docH - 34 - 6 - th - 6 - 6;
  const md = tile("Models", mw, th2, "splat", "FiGS needs exactly one active run");
  md.body.appendChild(table([["", 24], ["run", 150, 'L', 'mono'], ["status", 80], ["trained on", 110], ["steps", 56, 'R'], ["peak VRAM", 76, 'R'], ["size", 60, 'R'], ["", 60]], [
    { cells: [{ node: ico("promote", 12) }, "2026-09-26_190524", "active", "intellisense08", "30k", "5203 MiB", "371 MB", { node: smallBtn("", "archive", "Default", "Archive this run.") }], sel: true },
    { cells: [{ node: ico("archive", 12) }, "2025-01-16_122349", "archived", "upstream", "30k", "—", "368 MB", { node: smallBtn("", "promote", "Default", "Promote this run.") }] },
  ], { name: "models" }));
  const mb = al('HORIZONTAL', { itemSpacing: 4 }); for (const [l, ic, h] of [["Promote", "promote", "Make the selected archived run active."], ["Archive", "archive", "Archive the active run."], ["Retrain…", "retrain", "Train a new run from the cached SfM."], ["Viewer", "viewer", "ns-viewer (queued GPU job)."]]) mb.appendChild(smallBtn(l, ic, "Default", h)); md.body.appendChild(mb);
  md.body.appendChild(wrapTxt(txt("Archive/Promote are refused while a job for backroom is queued or running, and clear the verify/simulate/validate markers because they describe the old model.", 11, "Regular", "#5B6B7F"), mw - 24));
  r2.appendChild(md.frame);
  const fl = tile("Last flight · circuit", fw, th2, "video", "runs/backroom_2026-10-01_2023.json");
  const fr = al('HORIZONTAL', { itemSpacing: 10 }); fl.body.appendChild(fr);
  const vid = figma.createFrame(); vid.resize(240, 135); vid.fills = [vgrad([0,"#3A4A5E"],[1,"#1E2733"])]; vid.cornerRadius = 2; vid.name = "flight video";
  const play = ico("play", 28); vid.appendChild(play); play.x = 106; play.y = 53;
  const vt = txt("backroom_flight.mp4 · 640×360 · 12.35 s", 10, "Regular", "#DCE6F4"); vid.appendChild(vt); vt.x = 8; vt.y = 116;
  fr.appendChild(vid);
  const fs = al('VERTICAL', { itemSpacing: 0 }); fs.resize(fw - 290, 10); fs.counterAxisSizingMode = 'FIXED'; fs.primaryAxisSizingMode = 'AUTO'; fr.appendChild(fs);
  propGrid(fs, [["Expert flight", [["Tracking error", "mean 0.002 m · max 0.034 m", { icon: "ok" }], ["Frames", "247 · no near-black"], ["Pixel std", "52.6"], ["Peak VRAM", "739 MiB"], ["Wall-clock", "46.0 s"]]]]);
  r2.appendChild(fl.frame);
  // output
  const ob = outputTabs(C.out, [["Live log", "log", true], ["Queue", "queue", false], ["Problems", "warning", false], ["GPU", "gpu", false]], CW);
  ob.appendChild(queueList(520, JOBS.slice(1, 6), 0));
  ob.appendChild(logBox([
    "=== 2026-10-01 20:22:43 fly backroom + circuit (course → simulate → validate → record)",
    "  ✔ all 10 keyframes inside the captured volume",
    "    VRAM baseline 215 MiB → peak 739 MiB",
    "  ✔ outputs/flights/backroom_flight.mp4 (247 frames) in 46.0s",
    "    tracking error: mean 0.002 m, max 0.034 m",
    "    640x360  247 frames  12.35s  pixel mean 88.1  std 52.6  near-black 0/247",
    "  ✔ run record: runs/backroom_2026-10-01_2023.json",
  ], CW - 520 - 26, 140, "flight log"));
  add(main, propertiesPane(mainH, ["backroom · active model", "splat"], [
    ["Scene", [["Name", "backroom", { mono: true }], ["Capture", "backroom.MOV · 1920×1080", {}], ["Workspace", "gsplats/workspace/backroom", { mono: true }], ["Loadable", "yes (one active run)", { icon: "ok" }]]],
    ["Active run", [["Run", "2026-09-26_190524", { mono: true }], ["Steps", "30,000"], ["Downscale", "auto (2×) · 960×540"], ["Image cache", "gpu (lab default)"], ["Peak VRAM", "5203 MiB"], ["Checkpoint", "step-000029999.ckpt", { mono: true }]]],
    ["Bounds (course frame)", [["Camera box x", "-2.1 … 2.2 m", { mono: true }], ["Camera box y", "-7.9 … 0.9 m", { mono: true }], ["Camera box z", "-1.45 … -0.25 m", { mono: true }], ["Waypoint box", "inset 0.5 m", {}]]],
    ["Markers (.done)", [["Steps done", "14 / 14", { icon: "ok" }], ["State", ".figs_pipeline_state/backroom/", { mono: true }]]],
  ]));
  hoverTip(win, rib, "Train", ["Train", "Train the Gaussian splat (nerfstudio splatfacto) from the SfM result.", "figs_pipeline.py --only train --cache-images cpu", ""]);
  return win;
};

// ───────────────────────── Screen 3: SV-Net cohort ─────────────────────────
const screenSvnet = () => {
  const S = windowShell("SV-Net cohort", "Galley — p4_smoke (SV-Net)  ·  intellisense08", "SV-Net", STATUS("Queue: idle · job #8 succeeded in 49 min"));
  const { win, rib, main, mainH } = S;
  add(main, explorerPane(mainH, "p4_smoke"));
  const CW = W - 8 - 8 - EXW - PRW;
  const C = centerColumn(CW, mainH, [["p4_smoke", "cohort", true], ["p4_beta (draft)", "cohort", false], ["backroom / circuit", "route", false]], 186);
  add(main, C.col);
  const steps = [["preflight", "7.4 s", "—", "preflight"], ["rollout", "3m 06s", "739 MiB", "data"], ["observe", "42 s", "3207 MiB", "network"], ["train_hist", "2m 59s", "413 MiB", "chart"], ["train_comm", "40m 49s", "6785 MiB", "network"], ["deploy", "1m 32s", "753 MiB", "deploy"]];
  const strip = al('HORIZONTAL', { name: "cohort steps", itemSpacing: 6, counterAxisAlignItems: 'CENTER' }); add(C.doc, strip, { fillW: true });
  const scw = Math.floor((C.docW - 5 * 6) / 6);
  steps.forEach(([n, t, v, ic], i) => {
    const c = box('HORIZONTAL', scw, 44, { name: `step: ${n}`, itemSpacing: 8, paddingLeft: 8, paddingRight: 8, counterAxisAlignItems: 'CENTER' });
    c.fills = [vgrad([0,"#F1FAF2"],[1,"#E0F2E3"])]; border(c, "#8FCB9A"); c.cornerRadius = 3;
    c.appendChild(ico(ic, 24)); const tx = al('VERTICAL', { itemSpacing: 0 }); tx.appendChild(txt(`${i + 1}. ${n}`, 12, "SemiBold", "#1E395B")); tx.appendChild(txt(`${t} · peak ${v}`, 10, "Regular", "#5B6B7F")); c.appendChild(tx); spacer(c); c.appendChild(ico("ok", 16));
    help(c, `${n}: done. Each step runs in its own process; re-running it re-runs everything after it.`);
    strip.appendChild(c);
  });
  const th = 270, r1 = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, r1, { fillW: true });
  const lw = 640, ew = 360, rw = C.docW - lw - ew - 12;
  const lc = tile("Losses", lw, th, "chart", "train ── test ┅");
  const lr = al('HORIZONTAL', { itemSpacing: 8 }); lc.body.appendChild(lr);
  const half = Math.floor((lw - 28) / 2);
  const hcol = al('VERTICAL', { itemSpacing: 2 }); hcol.appendChild(txt("histNet · 200 epochs · train 1.055 / test 1.029", 11, "SemiBold", "#1E395B")); hcol.appendChild(lineChart(half, th - 70, [{ pts: HIST_LOSS, color: "#2F6FDD" }, { pts: HIST_TEST, color: "#F0A030", dash: true }], { ymin: 0, ymax: 25, name: "histNet loss" })); lr.appendChild(hcol);
  const ccol = al('VERTICAL', { itemSpacing: 2 }); ccol.appendChild(txt("commNet · 300 epochs · train 0.0027 / test 0.0185", 11, "SemiBold", "#C42B1C")); ccol.appendChild(lineChart(half, th - 70, [{ pts: COMM_LOSS, color: "#2F6FDD" }, { pts: COMM_TEST, color: "#F0A030", dash: true }], { ymin: 0, ymax: 0.075, name: "commNet loss" })); lr.appendChild(ccol);
  r1.appendChild(lc.frame);
  const ev = tile("In-loop evaluation (upstream TTE)", ew, th, "deploy", "eval_single · every 50 epochs");
  ev.body.appendChild(lineChart(ew - 20, th - 84, [{ pts: COMM_TTE, color: "#C42B1C", dots: true }], { ymin: 0, ymax: 360, name: "in-loop TTE" }));
  ev.body.appendChild(wrapTxt(txt("228, 127, 77, 75, 339, 61, 30: erratic; best = last checkpoint. Upstream's TTE is not a distance (norm over axis 0).", 10.5, "Regular", "#5B6B7F"), ew - 24));
  r1.appendChild(ev.frame);
  const ro = tile("Data", rw, th, "data", "data_alpha · Nro_ds 50");
  propGrid(ro.body, [["Rollouts", [["Kept (tol_select)", "111 · circuit"], ["Samples", "4,440 (train 4,096 / test 448)"], ["Rollout data", "5.41 GB"], ["Observations", "2.68 GB (commNet)"], ["Estimate was", "108 · 4,320 · 5.97 GB"]]], ["Disk", [["Free", "568 GB"], ["data_beta", "~88 GB (fits)"], ["data_gamma", "~265 GB (fits once)"]]]]);
  r1.appendChild(ro.frame);
  const r2 = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, r2, { fillW: true });
  const th2 = C.docH - 44 - 6 - th - 6 - 6, tw = 700, vw = C.docW - tw - 6;
  const et = tile("Evaluation · circuit", tw, th2, "compare", "per-point tracking error vs upstream TTE/PP");
  et.body.appendChild(table([["pilot", 90], ["role", 60], ["track mean", 80, 'R', 'mono'], ["track max", 80, 'R', 'mono'], ["within 0.3", 70, 'R', 'mono'], ["TTE (up.)", 70, 'R', 'mono'], ["PP (up.)", 62, 'R', 'mono'], ["Hz", 40, 'R', 'mono'], ["", 92]], [
    { cells: ["Viper", "expert", "0.001 m", "0.017 m", "100 %", "0.0016", "1.00", "0.4", { node: pill("Succeeded") }] },
    { cells: ["Maverick", "student", "9.434 m", "79.28 m", "16 %", "30.31", "0.03", "1.1", { node: pill("Failed") }], bad: true, colors: [null, null, "#C42B1C", "#C42B1C", "#C42B1C", null, null, null] },
  ], { name: "evaluation" }));
  const diag = al('VERTICAL', { itemSpacing: 4, paddingTop: 4 }); et.body.appendChild(diag);
  diag.appendChild(txt("Diagnosis", 11, "Bold", "#1E395B"));
  for (const l of ["The pipeline works end to end; the student leaves the course.", "commNet test loss is 7× its train loss on 4,096 samples: likely too little data (data_alpha).", "Next: watch the student video (where it diverges), then cohort p4_beta = same settings with data_beta."]) diag.appendChild(wrapTxt(txt("• " + l, 11, "Regular", "#3A3A3A"), tw - 30));
  const nb = al('HORIZONTAL', { itemSpacing: 4 }); nb.appendChild(smallBtn("Duplicate as p4_beta…", "copy", "Default", "Same settings, method data_beta.")); nb.appendChild(smallBtn("Compare cohorts", "compare", "Default", "Side by side with p4_beta.")); diag.appendChild(nb);
  r2.appendChild(et.frame);
  const vv = tile("Deployment videos", vw, th2, "video", "cohorts/p4_smoke/deployment_data/");
  const vr = al('HORIZONTAL', { itemSpacing: 8 }); vv.body.appendChild(vr);
  for (const [lab, col] of [["expert · Viper", "#2E8B3D"], ["student · Maverick", "#C42B1C"]]) {
    const vc = al('VERTICAL', { itemSpacing: 3 });
    const v = figma.createFrame(); v.resize(Math.floor((vw - 36) / 2), Math.floor((vw - 36) / 2 * 9 / 16)); v.fills = [vgrad([0,"#3A4A5E"],[1,"#1E2733"])]; v.cornerRadius = 2;
    const p = ico("play", 26); v.appendChild(p); p.x = v.width / 2 - 13; p.y = v.height / 2 - 13;
    vc.appendChild(v); vc.appendChild(txt(lab, 11, "SemiBold", col)); vr.appendChild(vc);
  }
  vv.body.appendChild(txt("▶ plays both in sync · sim_circuit_expert_rgb.mp4 / sim_circuit_Maverick_rgb.mp4", 10, "Regular", "#5B6B7F"));
  r2.appendChild(vv.frame);
  const ob = outputTabs(C.out, [["Live log", "log", true], ["Queue", "queue", false], ["Problems", "warning", false], ["GPU", "gpu", false]], CW);
  ob.appendChild(queueList(520, JOBS.slice(0, 5), 0));
  ob.appendChild(logBox(LOG8.slice(9), CW - 520 - 26, 140, "job #8 log"));
  add(main, propertiesPane(mainH, ["Cohort p4_smoke", "cohort"], [
    ["Setup", [["Scene", "backroom (2026-09-26_190524)"], ["Courses", "circuit"], ["Method", "data_alpha"], ["Expert / frame", "Viper / carl"], ["Roster", "Maverick"], ["Nro_ds", "50"], ["Compress", "off"]]],
    ["Training", [["histNet epochs", "200"], ["commNet epochs", "300"], ["Batch / LR", "64 / 1e-4"], ["In-loop eval", "eval_single"], ["Final eval", "eval_single"]]],
    ["Run", [["Job", "#8 · 49m 15s", { icon: "ok" }], ["Host", "intellisense08 · RTX 2080"], ["State", ".svnet_pipeline_state/p4_smoke/", { mono: true }], ["SousVide", "a2400aa + 2 notebook commits", {}]]],
  ]));
  hoverTip(win, rib, "Continue", ["Continue", "Resume the cohort from its first unfinished step; settings come from the cohort.", "svnet_pipeline.py --cohort p4_smoke", "F5"]);
  return win;
};

// ───────────────────────── Screen 4: Monitor ─────────────────────────
const screenMonitor = () => {
  const S = windowShell("Monitor", "Galley — Monitor  ·  job #8 running  ·  intellisense08", "Jobs", STATUS("Job #8 · train_comm · epoch 150/300", 0.62));
  const { win, rib, main, mainH } = S;
  add(main, explorerPane(mainH, "#8"));
  const CW = W - 8 - 8 - EXW - PRW;
  const C = centerColumn(CW, mainH, [["Monitor", "gauge", true], ["Job #8", "job", false], ["p4_smoke", "cohort", false]], 0);
  add(main, C.col);
  const r1 = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, r1, { fillW: true });
  const qw = 720, gw = C.docW - qw - 6, qh = 250;
  const q = tile("Queue", qw, qh, "queue", "one GPU job at a time");
  const rows = JOBS.map((j, i) => i === 0 ? ["#8", "svnet", "p4_smoke · train_comm (epoch 150/300)", "Running", "24m 10s", "20:33"] : j);
  rows.splice(1, 0, ["#9", "svnet", "p4_beta · preflight to deploy (data_beta)", "Queued", "—", "—"]);
  q.body.appendChild(queueList(qw - 20, rows.slice(0, 7), 0));
  r1.appendChild(q.frame);
  const g = tile("GPU · RTX 2080 (8 GB)", gw, qh, "gpu", "nvidia-smi every 2 s");
  const vram = []; for (let t = 0; t <= 24; t += 0.25) vram.push([t, t < 3 ? 0.74 : t < 4 ? 3.2 : t < 7 ? 0.41 : 5.1 + 1.6 * Math.abs(Math.sin(t * 1.3)) * (Math.floor(t * 4) % 13 === 0 ? 1 : 0.4)]);
  const util = vram.map(([t, v]) => [t, Math.min(8, v * 1.2 + 1.5 * Math.abs(Math.sin(t * 3)))]);
  g.body.appendChild(lineChart(gw - 20, qh - 70, [{ pts: vram, color: "#2F6FDD", area: true }, { pts: util.map(([t, v]) => [t, v * 0.9]), color: "#3FA34D", width: 1 }], { ymin: 0, ymax: 8.2, refs: [8], xunit: " min", name: "GPU memory" }));
  g.body.appendChild(txt("memory GB (blue) · utilisation scaled (green) · peaks = in-loop evaluation flights (6.8 GB)", 10, "Regular", "#5B6B7F"));
  r1.appendChild(g.frame);
  const r2 = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, r2, { fillW: true });
  const lh = C.docH - qh - 6, lw = 900, dw = C.docW - lw - 6;
  const lg = tile("Job #8 · live log", lw, lh, "follow", "following · rich bars shown as plain lines");
  const prog = al('HORIZONTAL', { itemSpacing: 8, counterAxisAlignItems: 'CENTER' }); prog.appendChild(txt("train_comm", 11, "SemiBold", "#1E395B")); prog.appendChild(progress(420, 0.5, "epoch 150 / 300 · ETA 20 min")); lg.body.appendChild(prog);
  lg.body.appendChild(logBox(LOG8.slice(0, 7).concat(["  Maverick > commNet  epoch 151/300 …"]), lw - 20, lh - 70, "live log"));
  r2.appendChild(lg.frame);
  const df = tile("Diff runs", dw, lh, "diff", "runs/backroom_… · bringup vs 27 Sep");
  df.body.appendChild(table([["field", 120], ["27 Sep (dummy)", 130, 'R', 'mono'], ["1 Oct (lab)", 130, 'R', 'mono']], [
    ["host", "dummy", "intellisense08"], ["GPU", "3050 Ti 4 GB", "RTX 2080 8 GB"], ["model", "backroom_t4", "backroom"],
    { cells: ["peak VRAM", "614 MiB", "739 MiB"], colors: [null, null, "#B45309"] }, ["frames", "247", "247"], ["track mean", "0.002 m", "0.002 m"], ["track max", "0.034 m", "0.034 m"], ["pixel std", "52.7", "52.6"], ["wall-clock", "34.0 s", "46.0 s"],
  ].map(r => Array.isArray(r) ? { cells: r } : r), { name: "run diff" }));
  df.body.appendChild(wrapTxt(txt("Same flight on both hosts: the policy-independent half of the pipeline reproduces exactly.", 11, "Regular", "#5B6B7F"), dw - 24));
  r2.appendChild(df.frame);
  add(main, propertiesPane(mainH, ["Job #8", "job"], [
    ["Job", [["Kind", "svnet"], ["Cohort", "p4_smoke"], ["Status", "running", { icon: "run" }], ["Started", "20:33:40"], ["Elapsed", "24m 10s"], ["PID / group", "48211 (own session)", { mono: true }]]],
    ["Command", [["Script", "svnet_pipeline.py", { mono: true }], ["Steps", "preflight to deploy"], ["Env", "source figs_env.sh", { mono: true }]]],
    ["Resources", [["GPU now", "5.9 GB · 97 %"], ["Peak so far", "6785 MiB"], ["Disk written", "8.1 GB"]]],
  ], [al('VERTICAL', { itemSpacing: 4, paddingTop: 8, paddingLeft: 6 })]));
  const pb = main.findOne(n => n.name === "pane: Properties"); const lastBody = pb.findOne(n => n.name === "body");
  const btns = al('HORIZONTAL', { itemSpacing: 6, paddingTop: 8, paddingLeft: 6 }); btns.appendChild(pushBtn("Cancel job")); btns.appendChild(pushBtn("Open cohort", "Primary")); lastBody.appendChild(btns);
  hoverTip(win, rib, "Diff runs", ["Diff runs", "Diff two run records (runs/*.json) or two jobs' parameters.", "", ""]);
  return win;
};

// ───────────────────────── Screen 5: Configs ─────────────────────────
const screenConfigs = () => {
  const S = windowShell("Configs", "Galley — configs / courses / circuit.json  ·  intellisense08", "Configs", STATUS("Queue: idle"));
  const { win, rib, main, mainH } = S;
  add(main, explorerPane(mainH, "courses"));
  const CW = W - 8 - 8 - EXW - PRW;
  const C = centerColumn(CW, mainH, [["circuit.json", "json", true], ["Viper.json", "network", false], ["carl.json", "drone", false]], 186);
  add(main, C.col);
  const row = al('HORIZONTAL', { itemSpacing: 6 }); add(C.doc, row, { fillW: true });
  const gw = 560, jw = C.docW - gw - 6, h = C.docH;
  const pg = tile("Property grid", gw, h, "matrix", "edits validate as you type");
  const sections = [["waypoints", [["Nco", "6", { mono: true }], ["keyframes", "10 (fo0 … fo7)", {}]]]];
  for (const k of KF.slice(0, 5)) sections.push([`keyframes.${k[0]}`, [["t", k[1].toFixed(3), { mono: true }], ["fo[x,y,z] pos", k[7] ? `${k[3].toFixed(1)}, null, null` : `${k[3].toFixed(1)}, ${k[4].toFixed(1)}, ${k[5].toFixed(1)}`, { mono: true, color: k[7] ? "#8A94A6" : "#1E1E1E" }], ["fo[yaw] pos", k[6] === null ? "null" : k[6].toFixed(2), { mono: true, color: k[6] === null ? "#8A94A6" : "#1E1E1E" }]]]);
  sections.push(["keyframes.fo4 … fo7", [["", "5 more (expand)", { color: "#5B6B7F" }]]]);
  sections.push(["forces", [["value", "null", { mono: true, color: "#8A94A6" }]]]);
  propGrid(pg.body, sections);
  row.appendChild(pg.frame);
  const js = tile("JSON · circuit.json", jw, h, "json", "UTF-8 · 2-space · floats");
  const code = [
    '{', '  "waypoints": {', '    "Nco": 6,', '    "keyframes": {',
    '      "fo0":  { "t": 0.0,   "fo": [[1.5, 0.0], [0.0, 0.0], [-0.7, 0.0], [-1.57, 0.0]] },',
    '      "fo0a": { "t": 0.816, "fo": [[1.5], [null], [null], [-1.8]] },',
    '      "fo1":  { "t": 3.453, "fo": [[1.5], [-6.5], [-0.7], [-2.8]] },',
    '      "fo2":  { "t": 3.924, "fo": [[1.0], [-7.0], [-0.7], [null]] },',
    '      "fo3":  { "t": 5.443, "fo": [[-1.0], [-7.0], [-0.7], [-4.72]] },',
    '      "fo4":  { "t": 5.898, "fo": [[-1.4], [-6.5], [-0.7], [-5.0]] },',
    '      "fo4a": { "t": 8.851, "fo": [[-1.4], [null], [null], [null]] },',
    '      "fo5":  { "t": 9.322, "fo": [[-1.4], [-0.5], [-0.7], [-5.0]] },',
    '      "fo6":  { "t": 10.472, "fo": [[-1.0], [0.0], [-0.7], [null]] },',
    '      "fo7":  { "t": 12.107, "fo": [[0.0, 0.0], [0.0, 0.0], [-0.7, 0.0], [-6.28, 0.0]] }',
    '    }', '  },', '  "forces": null', '}',
  ];
  const ed = box('HORIZONTAL', jw - 20, Math.min(h - 120, 22 + code.length * 16), { itemSpacing: 0 }); ed.fills = [solid("#FFFFFF")]; border(ed, "#D5DEEA");
  const gut = al('VERTICAL', { itemSpacing: 1.6, paddingTop: 4, paddingLeft: 6, paddingRight: 6 }); gut.fills = [solid("#F2F5F9")];
  const src = al('VERTICAL', { itemSpacing: 1.6, paddingTop: 4, paddingLeft: 8 });
  code.forEach((l, i) => { gut.appendChild(mono(String(i + 1).padStart(2, ' '), 11, "#8A94A6")); src.appendChild(mono(l, 11, l.includes('null') ? "#7E3BC4" : l.includes('"t"') ? "#1E395B" : "#2A2F38")); });
  ed.appendChild(gut); gut.layoutSizingVertical = 'FILL'; ed.appendChild(src);
  js.body.appendChild(ed);
  const vb = al('HORIZONTAL', { itemSpacing: 6, counterAxisAlignItems: 'CENTER' });
  vb.appendChild(ico("ok", 16)); vb.appendChild(txt("Valid · round-trips · 0 integer cells · mirrored to figs/sousvide_overlay/configs/courses/circuit.json", 11, "SemiBold", "#2E8B3D"));
  js.body.appendChild(vb);
  const bb = al('HORIZONTAL', { itemSpacing: 6 }); bb.appendChild(pushBtn("Revert")); bb.appendChild(pushBtn("Validate")); bb.appendChild(pushBtn("Save", "Primary")); js.body.appendChild(bb);
  row.appendChild(js.frame);
  const ob = outputTabs(C.out, [["Problems (0)", "ok", true], ["Live log", "log", false], ["Queue", "queue", false]], CW);
  ob.appendChild(table([["", 24], ["severity", 80], ["where", 220, 'L', 'mono'], ["message", 700]], [
    { cells: [{ node: ico("info", 12) }, "info", "keyframes.fo0a.fo[1..2]", "free cells (null): the solver chooses y and z"] },
    { cells: [{ node: ico("info", 12) }, "info", "keyframes.fo2.fo[3]", "yaw free"] },
    { cells: [{ node: ico("ok", 12) }, "ok", "file", "no JSON integers in fo (FiGS would read 0 as the previous cell's value)"] },
    { cells: [{ node: ico("warning", 12) }, "note", "name", "circuit is an upstream course: a SousVide re-clone restores the original; the overlay keeps yours"] },
  ], { name: "problems" }));
  add(main, propertiesPane(mainH, ["circuit.json", "json"], [
    ["File", [["Family", "courses"], ["Path", "SousVide/configs/courses/circuit.json", { mono: true }], ["Overlay copy", "in sync", { icon: "ok" }], ["Upstream", "a2400aa · modified", { color: "#B45309" }]]],
    ["Model", [["Schema", "Course (Pydantic)"], ["Keyframes", "10"], ["First / last", "fully fixed + vel 0", { icon: "ok" }], ["Times", "strictly increasing", { icon: "ok" }]]],
    ["Used by", [["Flights", "backroom ×3"], ["Cohorts", "p4_smoke"], ["Editor", "Open in Course editor ›", { color: "#0066CC" }]]],
  ]));
  hoverTip(win, rib, "Validate", ["Validate", "Check against the Pydantic model without saving; refuses anything that would not round-trip.", "POST /api/configs/{family}/validate", "F7"]);
  return win;
};
// ───────────────────────── entry point ─────────────────────────
const RIBBON_IDS = { "Home": "10:311", "Capture & Splat": "10:627", "Course": "10:1013", "SV-Net": "10:1311", "Jobs": "10:1532", "Configs": "10:1753", "View": "10:1956", "Keyframe Tools": "10:2145", "Model Tools": "10:2295" };
async function main() {
  await Promise.all(["Regular", "SemiBold", "Bold"].map(s => figma.loadFontAsync({ family: F, style: s })));
  await figma.loadFontAsync({ family: "Source Code Pro", style: "Regular" });
  const DS = await figma.getNodeByIdAsync("0:1"); await DS.loadAsync();
  for (const c of DS.findAllWithCriteria({ types: ['COMPONENT'] })) if (c.name.startsWith('icon/')) icons[c.name.slice(5)] = c;
  const SETID = { large: "8:38", small: "8:74", field: "8:93", cb: "8:103", tab: "8:112", tree: "8:153", doctab: "8:172", pill: "8:214", push: "8:221" };
  for (const [k, id] of Object.entries(SETID)) { sets[k] = await figma.getNodeByIdAsync(id); if (!sets[k]) throw new Error(`component set ${k} (${id}) not found — is this the "Galley — Win7 Ribbon UI" file?`); }
  TOOLTIP = await figma.getNodeByIdAsync("8:113"); PANEHDR = await figma.getNodeByIdAsync("8:173");
  const RP = await figma.getNodeByIdAsync("6:2"); await RP.loadAsync();
  for (const [k, id] of Object.entries(RIBBON_IDS)) RIBBONS[k] = await figma.getNodeByIdAsync(id);
  // fix: ribbon strip/body backgrounds rendered only ~1375 px wide in the first build; use solid fills
  for (const r of Object.values(RIBBONS)) {
    for (const ch of r.children) {
      if (ch.name === "tab strip") ch.fills = [solid("#D7E4F4")];
      if (ch.name === "ribbon body") ch.fills = [solid("#EEF4FB")];
    }
  }
  // fix: ribbon field values wrapped onto two lines ("auto (2×)", "eval_single"): one line, ellipsis, wider box
  for (const v of sets.field.children) {
    const bx = v.findOne(n => n.name === "Box"); if (bx) { bx.resize(92, 20); }
    const val = v.findOne(n => n.name === "Value");
    if (val && val.type === 'TEXT') { await figma.loadFontAsync(val.fontName); val.textAutoResize = 'NONE'; val.resize(Math.max(20, val.width), 16); val.textTruncation = 'ENDING'; val.maxLines = 1; val.layoutSizingHorizontal = 'FILL'; }
  }
  const page = await figma.getNodeByIdAsync("6:3"); await figma.setCurrentPageAsync(page);
  // re-runnable: remove what a previous run of this plugin made
  for (const n of [...page.children]) if (n.name.startsWith("Screen · ") || n.name === "Screens · header" || n.name === "Plugin errors") n.remove();
  const head = al('VERTICAL', { name: "Screens · header", itemSpacing: 6 });
  head.appendChild(txt("Galley — screens (1920 × 1080)", 28, "Bold", "#1E395B"));
  head.appendChild(wrapTxt(txt("One window, five working layouts. Ribbon = toolkit + navigation; Explorer replaces page switching; documents open as tabs; the Properties pane and the Output panel (log, queue, problems, GPU) stay docked, so similar tasks never leave the screen. Each screen shows one hover super-tooltip (pointer drawn). All numbers are the project's real values from 27 Sep – 1 Oct where known.", 13, "Regular", "#5B6B7F"), 1800));
  page.appendChild(head); head.x = 0; head.y = -160;
  const errors = []; const made = [];
  const builders = [["Course workspace", screenCourse], ["Scene & Splat", screenScene], ["SV-Net cohort", screenSvnet], ["Monitor", screenMonitor], ["Configs", screenConfigs]];
  let i = 0;
  for (const [name, fn] of builders) {
    try { const w = fn(); page.appendChild(w); w.x = (i % 2) * 2040; w.y = Math.floor(i / 2) * 1200; made.push(name); }
    catch (e) { errors.push(`${name}: ${e && e.message ? e.message : e}`); }
    i++;
  }
  if (errors.length) {
    const eb = al('VERTICAL', { name: "Plugin errors", itemSpacing: 4, paddingTop: 12, paddingBottom: 12, paddingLeft: 12, paddingRight: 12 });
    eb.fills = [solid("#FDF0EE")]; eb.appendChild(txt("Errors (send these to Claude)", 16, "Bold", "#C42B1C"));
    for (const e of errors) eb.appendChild(mono(e, 12, "#C42B1C"));
    page.appendChild(eb); eb.x = 0; eb.y = 3700;
  }
  figma.viewport.scrollAndZoomIntoView(page.children.filter(n => n.name.startsWith("Screen · ")));
  return { made, errors };
}
main().then(r => figma.closePlugin(r.errors.length ? `Built ${r.made.length}/5 screens — ${r.errors.length} error(s), see the red box on the Screens page` : `Built all ${r.made.length} screens on the Screens page`))
  .catch(e => figma.closePlugin(`Galley screens failed: ${e && e.message ? e.message : e}`));
