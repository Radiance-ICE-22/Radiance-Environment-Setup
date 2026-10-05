// test.splat: 3000 anisotropic, rotated, coloured Gaussians in a 2 m cube (seeded)
import { writeFileSync, mkdirSync } from "node:fs";
let s = 7; const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
const n = 3000, buf = new ArrayBuffer(n * 32), f = new Float32Array(buf), u = new Uint8Array(buf);
for (let i = 0; i < n; i++) {
  const o = i * 8;
  f[o] = r() * 2 - 1; f[o + 1] = r() * 2 - 1; f[o + 2] = r() * 2 - 1;
  f[o + 3] = 0.01 + r() * 0.08; f[o + 4] = 0.005 + r() * 0.03; f[o + 5] = 0.005 + r() * 0.02;
  u.set([r() * 255, r() * 255, r() * 255, 120 + r() * 135], i * 32 + 24);
  let q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5]; const l = Math.hypot(...q); q = q.map((v) => v / l);
  u.set(q.map((v) => Math.max(0, Math.min(255, Math.round(v * 128 + 128)))), i * 32 + 28);
}
mkdirSync("public", { recursive: true }); writeFileSync("public/test.splat", new Uint8Array(buf));
