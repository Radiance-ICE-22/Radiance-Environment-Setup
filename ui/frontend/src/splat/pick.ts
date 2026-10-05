// Which Gaussian is under the cursor? The renderer's own answer: walk the Gaussians the ray
// passes through front to back and take the one with the largest blend weight
//   w_i = T_i · α_i,   α_i = opacity_i · exp(−d_i² / 2σ_i²),   T_{i+1} = T_i (1 − α_i)
// (d_i = distance from the ray to the centre, σ_i = the largest scale widened by a couple of
// pixels at that depth, as the rasteriser dilates tiny splats). That is the Gaussian whose
// colour the pixel mostly shows, so it is the row whose features the labels describe.
//
// A plain pass over all centres: ~10 ms for 500 k in a browser, no index to build or keep.

export interface Ray { o: [number, number, number]; d: [number, number, number] }   // d unit length
export interface PickOpts {
  pixelAngle: number;          // radians per pixel at the view centre (2 tan(fov/2) / height)
  dilatePx?: number;           // splat dilation in pixels (default 2)
  minOpacity?: number;         // ignore fainter Gaussians (the view's opacity floor)
  near?: number;               // ignore centres closer than this along the ray
  mask?: Uint8Array | null;    // optional: only rows with mask[i] != 0 are pickable
}
export interface Pick { index: number; t: number; d: number; weight: number; point: [number, number, number] }

export function pickGaussian(pos: Float32Array, scale: Float32Array, opacity: Float32Array, n: number, ray: Ray, o: PickOpts): Pick | null {
  const [ox, oy, oz] = ray.o, [dx, dy, dz] = ray.d;
  const dil = (o.dilatePx ?? 2) * o.pixelAngle, minOp = o.minOpacity ?? 0.02, near = o.near ?? 0.05;
  const hitI: number[] = [], hitT: number[] = [], hitA: number[] = [], hitD: number[] = [];
  for (let i = 0; i < n; i++) {
    const op = opacity[i];
    if (op < minOp || (o.mask && !o.mask[i])) continue;
    const vx = pos[i * 3] - ox, vy = pos[i * 3 + 1] - oy, vz = pos[i * 3 + 2] - oz;
    const t = vx * dx + vy * dy + vz * dz;
    if (t < near) continue;
    const d2 = vx * vx + vy * vy + vz * vz - t * t;
    const s = scale[i], px = dil * t;
    const s2 = s * s + px * px;
    if (d2 > 9 * s2) continue;                     // beyond 3 σ: contributes < 1 %
    const a = Math.min(0.99, op * Math.exp(-0.5 * d2 / s2));
    if (a < 1 / 255) continue;
    hitI.push(i); hitT.push(t); hitA.push(a); hitD.push(Math.sqrt(Math.max(0, d2)));
  }
  if (!hitI.length) return null;
  const order = hitI.map((_, k) => k).sort((a, b) => hitT[a] - hitT[b]);
  let T = 1, best = -1, bestW = 0;
  for (const k of order) {
    const w = T * hitA[k];
    if (w > bestW) { bestW = w; best = k; }
    T *= 1 - hitA[k];
    if (T < 1e-3) break;
  }
  if (best < 0) return null;
  const i = hitI[best];
  return { index: i, t: hitT[best], d: hitD[best], weight: bestW, point: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]] };
}
