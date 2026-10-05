// The browser .splat file (figs/course_tools.py splat): one 32-byte record per Gaussian, in
// splat_order() — the same order as the semantic tables, so record i IS table row i.
//
//   bytes  0–11  position x, y, z   float32, SPLAT frame (nerfstudio world)
//   bytes 12–23  scale sx, sy, sz   float32, linear (exp of the log scales)
//   bytes 24–27  r, g, b, a         uint8 (colour from SH degree 0, a = opacity)
//   bytes 28–31  rotation w, x, y, z  uint8, (q + 1) · 128
//
// Everything the editor draws or picks is in the COURSE frame: (x, −y, −z) of the splat.
// No three.js here, so the node tests can import it.

export const RECORD = 32;

export interface SplatData {
  n: number;
  buf: ArrayBuffer;            // the file as downloaded (never modified)
  pos: Float32Array;           // n × 3, course frame
  scale: Float32Array;         // n, largest of the three scales (m)
  opacity: Float32Array;       // n, 0..1
  rgba: Uint8Array;            // n × 4, the file's colours (a view into buf's records is not contiguous, so a copy)
  box: { lo: [number, number, number]; hi: [number, number, number] };   // 1st–99th percentile, course frame
}

export function parseSplat(buf: ArrayBuffer): SplatData {
  if (buf.byteLength % RECORD) throw new Error(`.splat size ${buf.byteLength} is not a multiple of ${RECORD} bytes`);
  const n = buf.byteLength / RECORD;
  const f = new Float32Array(buf), u = new Uint8Array(buf);
  const pos = new Float32Array(n * 3), scale = new Float32Array(n), opacity = new Float32Array(n), rgba = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 8;
    pos[i * 3] = f[o]; pos[i * 3 + 1] = -f[o + 1]; pos[i * 3 + 2] = -f[o + 2];
    scale[i] = Math.max(f[o + 3], f[o + 4], f[o + 5]);
    const b = i * RECORD + 24;
    rgba[i * 4] = u[b]; rgba[i * 4 + 1] = u[b + 1]; rgba[i * 4 + 2] = u[b + 2]; rgba[i * 4 + 3] = u[b + 3];
    opacity[i] = u[b + 3] / 255;
  }
  return { n, buf, pos, scale, opacity, rgba, box: percentileBox(pos, n) };
}

/** 1st–99th percentile of each axis (a sample of up to 20 000 centres): framing that ignores floaters. */
export function percentileBox(pos: Float32Array, n: number): SplatData["box"] {
  const lo: [number, number, number] = [0, 0, 0], hi: [number, number, number] = [0, 0, 0];
  if (!n) return { lo, hi };
  const step = Math.max(1, Math.floor(n / 20000));
  for (let a = 0; a < 3; a++) {
    const s: number[] = [];
    for (let i = 0; i < n; i += step) s.push(pos[i * 3 + a]);
    s.sort((x, y) => x - y);
    lo[a] = s[Math.floor(0.01 * (s.length - 1))]; hi[a] = s[Math.ceil(0.99 * (s.length - 1))];
  }
  return { lo, hi };
}

/** Synthetic .splat for tests: records at the given splat-frame positions. */
export function makeSplat(items: { p: [number, number, number]; s?: number; c?: [number, number, number, number] }[]): ArrayBuffer {
  const buf = new ArrayBuffer(items.length * RECORD);
  const f = new Float32Array(buf), u = new Uint8Array(buf);
  items.forEach((it, i) => {
    const o = i * 8, s = it.s ?? 0.02, c = it.c ?? [200, 200, 200, 255];
    f[o] = it.p[0]; f[o + 1] = it.p[1]; f[o + 2] = it.p[2]; f[o + 3] = s; f[o + 4] = s; f[o + 5] = s;
    u.set(c, i * RECORD + 24); u.set([255, 128, 128, 128], i * RECORD + 28);   // identity rotation
  });
  return buf;
}
