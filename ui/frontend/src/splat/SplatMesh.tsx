// Gaussian splat renderer that takes a parsed .splat buffer and can be recoloured in place.
//
// Vendored from @react-three/drei 10.7.9 (core/Splat.js, MIT): same shaders, same covariance
// packing, same back-to-front counting sort in a worker. What changes:
//   - it takes the buffer (splat/format.ts) instead of a URL: one download serves the course
//     editor and the splat editor, and the semantic overlays need the bytes anyway;
//   - `colors` (n × 4 sRGB bytes, splat/recolor.ts) rewrites only the colour word of each
//     Gaussian and re-uploads that texture — milliseconds, where drei would re-download and
//     re-parse the whole file under a new URL (and keep every old copy cached);
//   - centres are kept in the course frame directly (drei's (x, −y, −z) flip is done once in
//     format.ts), and the worker sorts only when the view changed, asking for one frame when the
//     order is ready (the canvases run frameloop="demand").
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import type { SplatData } from "./format";

const VERT = /* glsl */ `
  precision highp sampler2D;
  precision highp usampler2D;
  out vec4 vColor;
  out vec3 vPosition;
  uniform vec2 viewport;
  uniform float focal;
  attribute uint splatIndex;
  uniform sampler2D centerAndScaleTexture;
  uniform usampler2D covAndColorTexture;

  vec2 unpackInt16(in uint value) {
    int v = int(value);
    int v0 = v >> 16;
    int v1 = (v & 0xFFFF);
    if ((v & 0x8000) != 0) v1 |= 0xFFFF0000;
    return vec2(float(v1), float(v0));
  }

  void main () {
    ivec2 texSize = textureSize(centerAndScaleTexture, 0);
    ivec2 texPos = ivec2(splatIndex % uint(texSize.x), splatIndex / uint(texSize.x));
    vec4 centerAndScaleData = texelFetch(centerAndScaleTexture, texPos, 0);
    vec4 center = vec4(centerAndScaleData.xyz, 1);
    vec4 camspace = modelViewMatrix * center;
    vec4 pos2d = projectionMatrix * camspace;

    float bounds = 1.2 * pos2d.w;
    if (pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }

    uvec4 covAndColorData = texelFetch(covAndColorTexture, texPos, 0);
    vec2 cov3D_M11_M12 = unpackInt16(covAndColorData.x) * centerAndScaleData.w;
    vec2 cov3D_M13_M22 = unpackInt16(covAndColorData.y) * centerAndScaleData.w;
    vec2 cov3D_M23_M33 = unpackInt16(covAndColorData.z) * centerAndScaleData.w;
    mat3 Vrk = mat3(
      cov3D_M11_M12.x, cov3D_M11_M12.y, cov3D_M13_M22.x,
      cov3D_M11_M12.y, cov3D_M13_M22.y, cov3D_M23_M33.x,
      cov3D_M13_M22.x, cov3D_M23_M33.x, cov3D_M23_M33.y
    );
    mat3 J = mat3(
      focal / camspace.z, 0., -(focal * camspace.x) / (camspace.z * camspace.z),
      0., focal / camspace.z, -(focal * camspace.y) / (camspace.z * camspace.z),
      0., 0., 0.
    );
    mat3 W = transpose(mat3(modelViewMatrix));
    mat3 T = W * J;
    mat3 cov = transpose(T) * Vrk * T;
    vec2 vCenter = vec2(pos2d) / pos2d.w;
    float diagonal1 = cov[0][0] + 0.3;
    float offDiagonal = cov[0][1];
    float diagonal2 = cov[1][1] + 0.3;
    float mid = 0.5 * (diagonal1 + diagonal2);
    float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
    vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);
    uint colorUint = covAndColorData.w;
    vColor = vec4(
      float(colorUint & uint(0xFF)) / 255.0,
      float((colorUint >> uint(8)) & uint(0xFF)) / 255.0,
      float((colorUint >> uint(16)) & uint(0xFF)) / 255.0,
      float(colorUint >> uint(24)) / 255.0
    );
    vPosition = position;
    gl_Position = vec4(vCenter + position.x * v2 / viewport * 2.0 + position.y * v1 / viewport * 2.0, pos2d.z / pos2d.w, 1.0);
  }`;

const FRAG = /* glsl */ `
  #include <alphatest_pars_fragment>
  #include <alphahash_pars_fragment>
  in vec4 vColor;
  in vec3 vPosition;
  void main () {
    float A = -dot(vPosition.xy, vPosition.xy);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    vec4 diffuseColor = vec4(vColor.rgb, B);
    #include <alphatest_fragment>
    #include <alphahash_fragment>
    gl_FragColor = diffuseColor;
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

// back-to-front 16-bit counting sort (drei's), on centres + a cull weight per Gaussian
function sortWorker(self: any) {
  let c: Float32Array | null = null;
  self.onmessage = (e: MessageEvent) => {
    if (e.data.method === "init") { c = new Float32Array(e.data.centers); return; }
    if (e.data.method !== "sort" || !c) return;
    const v = new Float32Array(e.data.view), n = c.length / 4, threshold = -0.0001;
    let maxD = -Infinity, minD = Infinity, k = 0;
    const depth = new Float32Array(n), valid = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const d = v[0] * c[i * 4] + v[1] * c[i * 4 + 1] + v[2] * c[i * 4 + 2] + v[3];
      if (d < 0 && c[i * 4 + 3] > threshold * d) {     // in front of the camera, not vanishingly small
        depth[k] = d; valid[k] = i; k++;
        if (d > maxD) maxD = d; if (d < minD) minD = d;
      }
    }
    const inv = (256 * 256 - 1) / (maxD - minD || 1), size = new Int32Array(k), counts = new Uint32Array(65536), starts = new Uint32Array(65536);
    for (let i = 0; i < k; i++) { size[i] = ((depth[i] - minD) * inv) | 0; counts[size[i]]++; }
    for (let i = 1; i < 65536; i++) starts[i] = starts[i - 1] + counts[i - 1];
    const out = new Uint32Array(k);
    for (let i = 0; i < k; i++) out[starts[size[i]]++] = valid[i];
    self.postMessage({ indices: out, key: e.data.key }, [out.buffer]);
  };
}

/** sRGB byte → linear byte, exactly three.js Color.convertSRGBToLinear() then a Uint8 store (as drei does). */
const LIN = (() => {
  const t = new Uint8Array(256);
  for (let v = 0; v < 256; v++) { const c = v / 255; t[v] = (c < 0.04045 ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4)) * 255; }
  return t;
})();

interface Gpu { n: number; w: number; h: number; cs: Float32Array; cc: Uint32Array; centers: Float32Array; csTex: THREE.DataTexture; ccTex: THREE.DataTexture }

export const TEX_WIDTH = 2048;
/** Textures and sort inputs for a parsed .splat (exported for tests). */
export function buildGpu(d: SplatData): Omit<Gpu, "csTex" | "ccTex"> {
  const n = d.n, w = TEX_WIDTH, h = Math.max(1, Math.ceil(n / w));
  const f = new Float32Array(d.buf), u = new Uint8Array(d.buf);
  const cs = new Float32Array(w * h * 4), cc = new Uint32Array(w * h * 4);
  const c16 = new Int16Array(cc.buffer), c8 = new Uint8Array(cc.buffer), centers = new Float32Array(n * 4);
  const S = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const o = i * 8, b = i * 32;
    const x = f[o], y = -f[o + 1], z = -f[o + 2];      // course frame
    S[0] = f[o + 3]; S[1] = f[o + 4]; S[2] = f[o + 5];
    // rotation in the course frame: (w, x, y, z) → (w, x, −y, −z)
    let qw = (u[b + 28] - 128) / 128, qx = (u[b + 29] - 128) / 128, qy = -(u[b + 30] - 128) / 128, qz = -(u[b + 31] - 128) / 128;
    const ql = Math.hypot(qw, qx, qy, qz) || 1; qw /= ql; qx /= ql; qy /= ql; qz /= ql;
    const R = [
      1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
      2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
      2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
    ];
    // Σ = R S² Rᵀ, upper triangle in drei's order: 00, 01, 02, 11, 12, 22
    const sig = (a: number, c: number) => R[a * 3] * S[0] * S[0] * R[c * 3] + R[a * 3 + 1] * S[1] * S[1] * R[c * 3 + 1] + R[a * 3 + 2] * S[2] * S[2] * R[c * 3 + 2];
    const e = [sig(0, 0), sig(0, 1), sig(0, 2), sig(1, 1), sig(1, 2), sig(2, 2)];
    let m = 0; for (const v of e) m = Math.max(m, Math.abs(v));
    m = m || 1e-12;
    cs[i * 4] = x; cs[i * 4 + 1] = y; cs[i * 4 + 2] = z; cs[i * 4 + 3] = m / 32767;
    for (let j = 0; j < 6; j++) c16[i * 8 + j] = (e[j] * 32767) / m;
    c8[i * 16 + 12] = LIN[u[b + 24]]; c8[i * 16 + 13] = LIN[u[b + 25]]; c8[i * 16 + 14] = LIN[u[b + 26]]; c8[i * 16 + 15] = u[b + 27];
    centers[i * 4] = x; centers[i * 4 + 1] = y; centers[i * 4 + 2] = z; centers[i * 4 + 3] = (Math.max(S[0], S[1], S[2]) * u[b + 27]) / 255;
  }
  return { n, w, h, cs, cc, centers };
}

/** Write sRGB RGBA bytes (n × 4) into the colour words; null restores the file's colours. */
export function writeColors(g: { n: number; cc: Uint32Array }, d: SplatData, colors: Uint8Array | null) {
  const c8 = new Uint8Array(g.cc.buffer), src = colors ?? d.rgba;
  for (let i = 0; i < g.n; i++) {
    const k = i * 16 + 12, s = i * 4;
    c8[k] = LIN[src[s]]; c8[k + 1] = LIN[src[s + 1]]; c8[k + 2] = LIN[src[s + 2]]; c8[k + 3] = src[s + 3];
  }
}

export function SplatMesh({ data, colors = null, alphaTest = 0.02, onReady, onRecolor }: {
  data: SplatData; colors?: Uint8Array | null; alphaTest?: number;
  onReady?: () => void; onRecolor?: (t: { cpu: number; frame: number }) => void;
}) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const invalidate = useThree((s) => s.invalidate);
  const mesh = useRef<THREE.Mesh>(null);

  const gpu = useMemo<Gpu>(() => {
    const g = buildGpu(data);
    const csTex = new THREE.DataTexture(g.cs, g.w, g.h, THREE.RGBAFormat, THREE.FloatType);
    const ccTex = new THREE.DataTexture(g.cc, g.w, g.h, THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
    ccTex.internalFormat = "RGBA32UI";
    csTex.needsUpdate = true; ccTex.needsUpdate = true;
    return { ...g, csTex, ccTex };
  }, [data]);

  const geometry = useMemo(() => {
    const g = new THREE.InstancedBufferGeometry();
    const p = new THREE.BufferAttribute(new Float32Array(18), 3);
    p.setXYZ(0, -2, -2, 0); p.setXYZ(1, 2, 2, 0); p.setXYZ(2, -2, 2, 0);
    p.setXYZ(3, 2, -2, 0); p.setXYZ(4, 2, 2, 0); p.setXYZ(5, -2, -2, 0);
    g.setAttribute("position", p);
    const idx = new THREE.InstancedBufferAttribute(new Uint32Array(gpu.n), 1, false);
    idx.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("splatIndex", idx);
    g.instanceCount = 0;
    return g;
  }, [gpu]);

  const material = useMemo(() => {
    const m = new THREE.ShaderMaterial({
      uniforms: {
        viewport: { value: new THREE.Vector2(1980, 1080) }, focal: { value: 1000 },
        centerAndScaleTexture: { value: gpu.csTex }, covAndColorTexture: { value: gpu.ccTex }, alphaTest: { value: alphaTest },
      },
      vertexShader: VERT, fragmentShader: FRAG,
      transparent: true, depthTest: true, depthWrite: alphaTest > 0,
      blending: THREE.CustomBlending, blendSrcAlpha: THREE.OneFactor, toneMapped: false,
    });
    m.alphaTest = alphaTest;
    return m;
  }, [gpu, alphaTest]);

  const onReadyRef = useRef(onReady); onReadyRef.current = onReady;
  // the sorting worker
  const sort = useRef({ worker: null as Worker | null, pending: false, last: null as Float32Array | null, ready: false });
  useEffect(() => {
    const url = URL.createObjectURL(new Blob([`(${sortWorker.toString()})(self)`], { type: "application/javascript" }));
    const w = new Worker(url);
    URL.revokeObjectURL(url);
    const st = sort.current; st.worker = w; st.pending = false; st.last = null; st.ready = false;
    const centers = gpu.centers.slice();
    w.postMessage({ method: "init", centers: centers.buffer }, [centers.buffer]);
    w.onmessage = (e) => {
      const ind = new Uint32Array(e.data.indices);
      const a = geometry.getAttribute("splatIndex") as THREE.InstancedBufferAttribute;
      (a.array as Uint32Array).set(ind); a.needsUpdate = true;
      a.clearUpdateRanges(); a.addUpdateRange(0, ind.length);
      geometry.instanceCount = ind.length;
      st.pending = false;
      if (!st.ready) { st.ready = true; onReadyRef.current?.(); }
      invalidate();
    };
    invalidate();
    return () => { w.terminate(); st.worker = null; };
  }, [gpu, geometry, invalidate]);

  useEffect(() => () => { gpu.csTex.dispose(); gpu.ccTex.dispose(); }, [gpu]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => () => material.dispose(), [material]);

  // recolour: rewrite the colour words and upload that texture again
  const onRecolorRef = useRef(onRecolor); onRecolorRef.current = onRecolor;
  const first = useRef(true);
  useEffect(() => {
    if (first.current && !colors) { first.current = false; return; }   // built with the file's colours
    first.current = false;
    const t0 = performance.now();
    writeColors(gpu, data, colors);
    gpu.ccTex.needsUpdate = true;
    const t1 = performance.now();
    invalidate();
    // the upload happens in the next frame; report the CPU part and that frame separately
    requestAnimationFrame(() => requestAnimationFrame(() => onRecolorRef.current?.({ cpu: t1 - t0, frame: performance.now() - t1 })));
  }, [colors, gpu, data, invalidate]);

  const vp = useMemo(() => new THREE.Vector4(), []);
  useFrame(() => {
    const m = mesh.current; if (!m) return;
    gl.getCurrentViewport(vp);
    material.uniforms.viewport.value.set(vp.z, vp.w);
    material.uniforms.focal.value = (vp.w / 2) * Math.abs(camera.projectionMatrix.elements[5]);
    const st = sort.current;
    if (!st.worker || st.pending) return;
    camera.updateMatrixWorld();
    m.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, m.matrixWorld);
    const e = m.modelViewMatrix.elements;
    const view = new Float32Array([e[2], e[6], e[10], e[14]]);
    const l = st.last;
    if (l && Math.abs(l[0] - view[0]) < 1e-4 && Math.abs(l[1] - view[1]) < 1e-4 && Math.abs(l[2] - view[2]) < 1e-4 && Math.abs(l[3] - view[3]) < 1e-3) return;
    st.last = view.slice(); st.pending = true;
    st.worker.postMessage({ method: "sort", view: view.buffer, key: 0 }, [view.buffer]);
  });

  return <mesh ref={mesh} frustumCulled={false} geometry={geometry} material={material} raycast={() => null} />;
}
