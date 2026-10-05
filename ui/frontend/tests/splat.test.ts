// Unit tests for the splat editor's pure modules: node --test tests/ (Node ≥ 22.6, type stripping).
import { test } from "node:test";
import assert from "node:assert/strict";
import { makeSplat, parseSplat, RECORD } from "../src/splat/format.ts";
import { buildColors, countLit, heat, HEAT_STOPS, relevancyLut } from "../src/splat/recolor.ts";
import { pickGaussian } from "../src/splat/pick.ts";

test("parseSplat: byte offsets and the splat → course flip (x, −y, −z)", () => {
  const buf = makeSplat([{ p: [1, 2, 3], s: 0.05, c: [10, 20, 30, 128] }, { p: [-1, -2, -3], s: 0.2 }]);
  assert.equal(buf.byteLength, 2 * RECORD);
  const d = parseSplat(buf);
  assert.equal(d.n, 2);
  assert.deepEqual(Array.from(d.pos.slice(0, 3)), [1, -2, -3]);
  assert.deepEqual(Array.from(d.pos.slice(3, 6)), [-1, 2, 3]);
  assert.ok(Math.abs(d.scale[0] - 0.05) < 1e-7 && Math.abs(d.scale[1] - 0.2) < 1e-7);
  assert.deepEqual(Array.from(d.rgba.slice(0, 4)), [10, 20, 30, 128]);   // bytes 24–27 of record 0
  assert.ok(Math.abs(d.opacity[0] - 128 / 255) < 1e-7);
  assert.throws(() => parseSplat(new ArrayBuffer(33)), /multiple of 32/);
});

test("heat map: endpoints, monotone brightness", () => {
  assert.deepEqual(heat(0), HEAT_STOPS[0]);
  assert.deepEqual(heat(1), HEAT_STOPS[HEAT_STOPS.length - 1]);
  assert.deepEqual(heat(-3), HEAT_STOPS[0]);
  const lum = (c: number[]) => 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
  for (let u = 0; u < 1; u += 0.05) assert.ok(lum(heat(u + 0.05)) >= lum(heat(u)) - 1e-9);
});

test("relevancy LUT: grey below the floor, heat from the floor up", () => {
  const lut = relevancyLut(0.6);
  assert.equal(lut[152], null);
  assert.deepEqual(lut[153], heat(0));
  assert.deepEqual(lut[255], heat(1));
});

test("buildColors: rgb copies, relevancy lights the matches, opacity untouched, cap checks", () => {
  const n = 3;
  const rgba = new Uint8Array([200, 0, 0, 255, 0, 200, 0, 100, 0, 0, 200, 50]);
  assert.deepEqual(Array.from(buildColors(rgba, n, { mode: "rgb" })), Array.from(rgba));
  const rel = new Uint8Array([255, 0, 200]);
  const out = buildColors(rgba, n, { mode: "relevancy", rel, floor: 0.7 });
  assert.deepEqual(Array.from(out.slice(0, 3)), heat(1));
  assert.deepEqual(Array.from(out.slice(8, 11)), heat((200 / 255 - 0.7) / 0.3));
  assert.equal(out[4], out[5]);                            // greyed: r = g
  assert.deepEqual([out[3], out[7], out[11]], [255, 100, 50]);
  assert.throws(() => buildColors(rgba, n, { mode: "relevancy", rel: new Uint8Array(2) }), /2 values for 3/);
  assert.throws(() => buildColors(rgba, n, { mode: "pca", pca: new Uint8Array(6) }), /2 rows for 3/);
  const pca = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(Array.from(buildColors(rgba, n, { mode: "pca", pca })), [1, 2, 3, 255, 4, 5, 6, 100, 7, 8, 9, 50]);
  assert.equal(countLit(rel, 0.7), 2);
});

test("buildColors: candidates only greys outside the box", () => {
  const rgba = new Uint8Array([255, 0, 0, 255, 255, 0, 0, 255]);
  const pos = new Float32Array([0, 0, 0, 5, 5, 5]);
  const out = buildColors(rgba, 2, { mode: "rgb", only: { lo: [-1, -1, -1], hi: [1, 1, 1] }, pos });
  assert.deepEqual(Array.from(out.slice(0, 3)), [255, 0, 0]);
  assert.equal(out[4], out[5]);
});

test("buildColors: 1 M Gaussians recolour in well under a second", () => {
  const n = 1_000_000, rgba = new Uint8Array(n * 4), rel = new Uint8Array(n).map((_, i) => i % 256);
  const t0 = performance.now();
  buildColors(rgba, n, { mode: "relevancy", rel, floor: 0.6 });
  const ms = performance.now() - t0;
  assert.ok(ms < 1000, `${ms.toFixed(0)} ms`);
});

// synthetic scene for picking: a wall of Gaussians at x = 5 behind a single one at x = 2
function scene() {
  const items: { p: [number, number, number]; s: number; c?: [number, number, number, number] }[] = [];
  for (let y = -1; y <= 1; y += 0.05) for (let z = -1; z <= 1; z += 0.05) items.push({ p: [5, y, z], s: 0.03 });
  items.push({ p: [2, 0, 0], s: 0.05 });                      // index n−1, in front
  items.push({ p: [2, 0.3, 0], s: 0.05, c: [0, 0, 0, 3] });   // nearly transparent (course y = −0.3)
  return parseSplat(makeSplat(items));
}
const PA = (2 * Math.tan((50 * Math.PI) / 360)) / 800;

test("pick: the front Gaussian occludes the wall", () => {
  const d = scene();
  const p = pickGaussian(d.pos, d.scale, d.opacity, d.n, { o: [0, 0, 0], d: [1, 0, 0] }, { pixelAngle: PA });
  assert.ok(p);
  assert.equal(p!.index, d.n - 2);
  assert.ok(Math.abs(p!.t - 2) < 1e-6);
});

test("pick: off the front Gaussian, the wall Gaussian on the ray; transparent ones are skipped", () => {
  const d = scene();
  const dir = (y: number) => { const l = Math.hypot(5, y); return [5 / l, y / l, 0] as [number, number, number]; };
  // rays are in the course frame: towards course y = −0.5 on the wall
  const p = pickGaussian(d.pos, d.scale, d.opacity, d.n, { o: [0, 0, 0], d: dir(-0.5) }, { pixelAngle: PA });
  assert.ok(p);
  assert.ok(Math.abs(p!.point[0] - 5) < 1e-6 && Math.abs(p!.point[1] + 0.5) < 0.03, JSON.stringify(p));
  // through the near-transparent one at (2, −0.3, 0): skipped, the wall at y = −0.75 behind it is picked
  const q = pickGaussian(d.pos, d.scale, d.opacity, d.n, { o: [0, 0, 0], d: dir(-0.75) }, { pixelAngle: PA });
  assert.ok(q && Math.abs(q.point[0] - 5) < 1e-6 && Math.abs(q.point[1] + 0.75) < 0.03, JSON.stringify(q));
  assert.equal(pickGaussian(d.pos, d.scale, d.opacity, d.n, { o: [0, 0, 0], d: [0, 0, 1] }, { pixelAngle: PA }), null);
});

test("pick: a mask limits the candidates", () => {
  const d = scene();
  const mask = new Uint8Array(d.n).fill(1); mask[d.n - 2] = 0;
  const p = pickGaussian(d.pos, d.scale, d.opacity, d.n, { o: [0, 0, 0], d: [1, 0, 0] }, { pixelAngle: PA, mask });
  assert.ok(p && Math.abs(p.point[0] - 5) < 1e-6);
});

test("pick: 1 M Gaussians in well under 200 ms", () => {
  const n = 1_000_000, pos = new Float32Array(n * 3), scale = new Float32Array(n).fill(0.02), op = new Float32Array(n).fill(0.8);
  for (let i = 0; i < n * 3; i++) pos[i] = Math.sin(i * 12.9898) * 5;
  const t0 = performance.now();
  pickGaussian(pos, scale, op, n, { o: [0, 0, 0], d: [1, 0, 0] }, { pixelAngle: PA });
  const ms = performance.now() - t0;
  assert.ok(ms < 200, `${ms.toFixed(0)} ms`);
});
