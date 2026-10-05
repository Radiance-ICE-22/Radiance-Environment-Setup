// Colours for the splat editor's view modes, written as sRGB RGBA bytes per .splat record
// (SplatMesh converts them to linear and uploads only the colour words; positions and shapes
// never change). Pure functions over typed arrays, so the node tests can check them.
//
//   rgb        the splat's own colours
//   relevancy  Gaussians at or above the floor take the heat map (floor → 1), the rest are a dimmed
//              grey version of the scene, so the matches light up and the room stays readable
//   pca        the lift table's first three principal components as RGB (pca_rgb.u8)
//
// "Candidates only" greys everything outside a box (the selected candidate's, plus a margin).

export type ColorMode = "rgb" | "relevancy" | "pca";
export interface ColorOpts {
  mode: ColorMode;
  rel?: Uint8Array | null;        // n bytes, round(relevancy · 255)
  pca?: Uint8Array | null;        // n × 3 bytes
  floor?: number;                 // relevancy shown from here up (0..1)
  only?: { lo: number[]; hi: number[] } | null;   // course frame; needs pos
  pos?: Float32Array | null;      // n × 3, course frame
  dim?: number;                   // brightness of the greyed scene (0..1)
}

// heat map: a perceptual dark-red → orange → yellow → white ramp ("inferno" from its first
// quarter up), so the lowest shown value is still clearly coloured against the grey scene
export const HEAT_STOPS: [number, number, number][] = [
  [120, 28, 109], [165, 44, 96], [207, 68, 70], [237, 105, 37], [251, 155, 6], [247, 209, 61], [252, 255, 164],
];
export function heat(u: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, u)) * (HEAT_STOPS.length - 1);
  const i = Math.min(HEAT_STOPS.length - 2, Math.floor(x)), t = x - i;
  const a = HEAT_STOPS[i], b = HEAT_STOPS[i + 1];
  return [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)];
}
export const heatCss = () => `linear-gradient(90deg, ${HEAT_STOPS.map((c, i) => `rgb(${c.join(",")}) ${Math.round((i / (HEAT_STOPS.length - 1)) * 100)}%`).join(", ")})`;

/** Colour for each relevancy byte: null below the floor (→ dimmed grey), else the heat colour. */
export function relevancyLut(floor: number): ([number, number, number] | null)[] {
  const f = Math.min(0.999, Math.max(0, floor));
  return Array.from({ length: 256 }, (_, b) => { const v = b / 255; return v + 1e-9 < f ? null : heat((v - f) / (1 - f)); });
}

export function buildColors(rgba: Uint8Array, n: number, o: ColorOpts, out = new Uint8Array(n * 4)): Uint8Array {
  if (rgba.length < n * 4) throw new Error("rgba is shorter than n records");
  const dim = o.dim ?? 0.55;
  const grey = (i: number) => {
    const y = (0.3 * rgba[i * 4] + 0.59 * rgba[i * 4 + 1] + 0.11 * rgba[i * 4 + 2]) * dim + 18;
    out[i * 4] = y; out[i * 4 + 1] = y; out[i * 4 + 2] = y + 4;
  };
  if (o.mode === "relevancy" && (!o.rel || o.rel.length !== n)) throw new Error(`relevancy has ${o.rel?.length ?? 0} values for ${n} Gaussians`);
  if (o.mode === "pca" && (!o.pca || o.pca.length !== n * 3)) throw new Error(`PCA has ${(o.pca?.length ?? 0) / 3} rows for ${n} Gaussians`);
  const lut = o.mode === "relevancy" ? relevancyLut(o.floor ?? 0.5) : null;
  const only = o.only && o.pos ? o.only : null;
  for (let i = 0; i < n; i++) {
    out[i * 4 + 3] = rgba[i * 4 + 3];                  // opacity never changes
    if (only) {
      const x = o.pos![i * 3], y = o.pos![i * 3 + 1], z = o.pos![i * 3 + 2];
      if (x < only.lo[0] || x > only.hi[0] || y < only.lo[1] || y > only.hi[1] || z < only.lo[2] || z > only.hi[2]) { grey(i); continue; }
    }
    if (o.mode === "relevancy") {
      const c = lut![o.rel![i]];
      if (c) { out[i * 4] = c[0]; out[i * 4 + 1] = c[1]; out[i * 4 + 2] = c[2]; } else grey(i);
    } else if (o.mode === "pca") {
      out[i * 4] = o.pca![i * 3]; out[i * 4 + 1] = o.pca![i * 3 + 1]; out[i * 4 + 2] = o.pca![i * 3 + 2];
    } else {
      out[i * 4] = rgba[i * 4]; out[i * 4 + 1] = rgba[i * 4 + 1]; out[i * 4 + 2] = rgba[i * 4 + 2];
    }
  }
  return out;
}

/** How many relevancy bytes are at or above the floor (shown lit). */
export function countLit(rel: Uint8Array, floor: number): number {
  const t = Math.ceil(Math.min(0.999, Math.max(0, floor)) * 255 - 1e-6);
  let k = 0; for (let i = 0; i < rel.length; i++) if (rel[i] >= t) k++;
  return k;
}
