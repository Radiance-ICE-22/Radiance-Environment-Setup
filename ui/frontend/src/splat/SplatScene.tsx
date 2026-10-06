// 3D view of the splat editor (pages/SplatEditor.tsx), in the COURSE frame like the course
// editor: the splat (recoloured by relevancy or PCA), the query's candidate boxes, the selected
// goal with its approach point and the drone's sphere, annotation pins, the picked Gaussian.
// A click that is not a drag picks the Gaussian under the cursor (splat/pick.ts).
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, Grid, Line, OrbitControls } from "@react-three/drei";
import type { Annotation, Candidate, Geometry } from "../api";
import { annPositions } from "../api";

/** One pin per annotated position (every instance of a class); queries placed at the same spot (synonyms such as
 *  "clock" and "grandfather clock") share a pin and one label instead of overprinting each other. */
function pinGroups(pins: Annotation[]) {
  const m = new Map<string, { key: string; pos: V3; texts: string[]; idx: number[] }>();
  pins.forEach((a, i) => annPositions(a).forEach((p) => {
    const key = p.map((v) => Math.round(v * 20)).join(",");          // 5 cm cells
    const g = m.get(key) ?? { key, pos: p as V3, texts: [], idx: [] };
    if (!g.texts.includes(a.text)) g.texts.push(a.text);
    g.idx.push(i);
    m.set(key, g);
  }));
  return [...m.values()];
}
import { boxEdges, Frame, KeyNav, Label, V3 } from "../three/common";
import type { SplatData } from "./format";
import { SplatLayer } from "./load";
import { Pick, pickGaussian } from "./pick";

export const SC = { box: "#f5c842", boxSel: "#ffffff", goal: "#a855f7", approach: "#22c55e", body: "#22c55e", pin: "#38bdf8", pick: "#ec4899", path: "#8a8f98" };

export interface SceneProps {
  data: SplatData | null; colors: Uint8Array | null;
  onSplatReady?: () => void; onSplatError?: (m: string) => void; onRecolor?: (t: { cpu: number; frame: number }) => void;
  geo: Geometry | null; boxId: string;
  candidates: Candidate[]; hover: number | null; selected: number | null;
  bodyRadius: number;
  pins: Annotation[]; showPins: boolean; pinSel: number | null;
  picked: Pick | null; onPick: (p: Pick | null) => void; minOpacity: number; cursor: "pick" | "place";
  keyNav: boolean; navSpeed: number; showCamPath: boolean; focus: V3 | null;
  link?: CamLink | null; linkId?: string;        // Compare: two views share one camera
}

/** Two canvases following one camera (splat editor ▸ Compare). Whichever view moves publishes its
 *  camera position and orbit target; the other copies them. A view that joins takes the current one. */
export interface CamLink { last: { from: string; p: number[]; t: number[] } | null; subs: Set<(from: string, p: number[], t: number[]) => void> }
export const makeLink = (): CamLink => ({ last: null, subs: new Set() });
function LinkCam({ link, id }: { link: CamLink; id: string }) {
  const { camera, controls, invalidate } = useThree() as unknown as { camera: THREE.PerspectiveCamera; controls: { target: THREE.Vector3; update: () => void; addEventListener: (e: string, f: () => void) => void; removeEventListener: (e: string, f: () => void) => void } | null; invalidate: () => void };
  useEffect(() => {
    if (!controls) return;
    let applying = false;
    const onChange = () => {
      if (applying) return;
      const p = camera.position.toArray(), t = controls.target.toArray();
      link.last = { from: id, p, t };
      link.subs.forEach((f) => f(id, p, t));
    };
    const apply = (from: string, p: number[], t: number[]) => {
      if (from === id) return;
      applying = true;
      camera.position.fromArray(p); controls.target.fromArray(t); controls.update();
      applying = false;
      invalidate();
    };
    controls.addEventListener("change", onChange);
    link.subs.add(apply);
    if (link.last && link.last.from !== id) apply(link.last.from, link.last.p, link.last.t);   // join at the current view
    else onChange();                                                                            // or announce ours
    return () => { controls.removeEventListener("change", onChange); link.subs.delete(apply); };
  }, [link, id, controls, camera, invalidate]);
  return null;
}

/** Click (not drag) → ray through the cursor → the Gaussian the pixel mostly shows. */
function Picker({ data, onPick, minOpacity }: { data: SplatData | null; onPick: (p: Pick | null) => void; minOpacity: number }) {
  const { gl, camera } = useThree();
  const live = useRef({ data, onPick, minOpacity }); live.current = { data, onPick, minOpacity };
  useEffect(() => {
    const el = gl.domElement;
    let down: { x: number; y: number } | null = null;
    const pd = (e: PointerEvent) => { if (e.button === 0) down = { x: e.clientX, y: e.clientY }; };
    const pu = (e: PointerEvent) => {
      const d = down; down = null;
      if (!d || e.button !== 0 || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return;
      const { data: sd, onPick: cb, minOpacity: mo } = live.current;
      if (!sd) return;
      const r = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      const rc = new THREE.Raycaster(); rc.setFromCamera(ndc, camera);
      const cam = camera as THREE.PerspectiveCamera;
      const pixelAngle = (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov ?? 50) / 2)) / r.height;
      const o = rc.ray.origin, dir = rc.ray.direction;
      cb(pickGaussian(sd.pos, sd.scale, sd.opacity, sd.n, { o: [o.x, o.y, o.z], d: [dir.x, dir.y, dir.z] }, { pixelAngle, minOpacity: mo }));
    };
    el.addEventListener("pointerdown", pd); el.addEventListener("pointerup", pu);
    return () => { el.removeEventListener("pointerdown", pd); el.removeEventListener("pointerup", pu); };
  }, [gl, camera]);
  return null;
}

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

export default function SplatScene(p: SceneProps) {
  const { geo } = p;
  const frameBox = p.data?.box ?? (geo ? geo.points_box ?? geo.camera_box : null);
  const floorZ = p.data?.box.hi[2] ?? geo?.points_box?.hi[2] ?? (geo ? geo.camera_box.hi[2] + 1.2 : 0);
  const sel = p.selected !== null ? p.candidates[p.selected] : null;
  return (
    <Canvas frameloop="demand" camera={{ up: [0, 0, -1], position: [-6, 6, -5], fov: 50, near: 0.02, far: 400 }}
      style={{ cursor: p.cursor === "place" ? "crosshair" : "default" }} gl={{ preserveDrawingBuffer: true }}>
      <OrbitControls makeDefault enableDamping={false} />
      {/* a follower view (Compare) never frames itself: it takes the main view's camera */}
      <Frame box={frameBox} id={p.link && p.linkId && p.linkId !== "main" ? undefined : p.data ? `${p.boxId}:splat` : geo ? p.boxId : undefined} />
      <KeyNav enabled={p.keyNav} speed={p.navSpeed} box={frameBox} focus={p.focus} />
      <Picker data={p.data} onPick={p.onPick} minOpacity={p.minOpacity} />
      {p.link && <LinkCam link={p.link} id={p.linkId ?? "main"} />}
      <Grid position={[0, 0, floorZ]} rotation={[Math.PI / 2, 0, 0]} args={[40, 40]} cellSize={0.5} sectionSize={1}
        cellColor="#8a8f98" sectionColor="#6b7079" cellThickness={0.6} sectionThickness={1} fadeDistance={35} infiniteGrid side={THREE.DoubleSide} />
      {p.data && <SplatLayer data={p.data} colors={p.colors} onReady={p.onSplatReady} onError={p.onSplatError} onRecolor={p.onRecolor} />}
      {geo && p.showCamPath && geo.camera_path.length > 1 && <Line points={geo.camera_path} color={SC.path} lineWidth={1} transparent opacity={0.7} />}

      {p.candidates.map((c, i) => (i === p.hover || i === p.selected) && (
        <group key={`box${i}`}>
          <Line points={boxEdges(c.box)} segments color={i === p.selected ? SC.boxSel : SC.box} lineWidth={i === p.selected ? 2.5 : 1.5} depthTest={false} renderOrder={5} />
          <group position={[c.box.lo[0], c.box.lo[1], c.box.lo[2]]}><Label text={`#${c.rank} · ${c.score.toFixed(1)}`} color={i === p.selected ? "#ffffff" : "#fde68a"} bold /></group>
        </group>
      ))}
      {sel && (
        <>
          <group position={sel.centroid}>
            <mesh raycast={() => null} renderOrder={6}><octahedronGeometry args={[0.12]} /><meshBasicMaterial color={SC.goal} depthTest={false} transparent /></mesh>
          </group>
          <group position={sel.approach}>
            <mesh raycast={() => null} renderOrder={6}><sphereGeometry args={[0.05, 16, 12]} /><meshBasicMaterial color={SC.approach} depthTest={false} transparent /></mesh>
            {p.bodyRadius > 0 && <mesh raycast={() => null}><sphereGeometry args={[p.bodyRadius, 24, 16]} />
              <meshBasicMaterial color={sel.gap_ok === false ? "#e5484d" : SC.body} wireframe transparent opacity={0.45} /></mesh>}
            <Line points={[[0, 0, 0], sub(sel.centroid, sel.approach)]} color={SC.approach} lineWidth={2} dashed dashSize={0.08} gapSize={0.05} depthTest={false} />
            <Label text={`approach${sel.gap !== null ? ` · gap ${sel.gap.toFixed(2)} m` : ""}`} color={sel.gap_ok === false ? "#ff6b6b" : "#86efac"} bold />
          </group>
        </>
      )}
      {p.showPins && pinGroups(p.pins).map((g) => {
        const sel = p.pinSel !== null && g.idx.includes(p.pinSel);
        return (
          <group key={g.key} position={g.pos}>
            <mesh raycast={() => null} renderOrder={6}><sphereGeometry args={[sel ? 0.06 : 0.04, 14, 10]} /><meshBasicMaterial color={sel ? "#ffffff" : SC.pin} depthTest={false} transparent /></mesh>
            <Line points={[[0, 0, 0], [0, 0, -0.25]]} color={SC.pin} lineWidth={1.5} depthTest={false} />
            <group position={[0, 0, -0.25]}><Label text={g.texts.join(" / ")} color={sel ? "#ffffff" : "#7dd3fc"} /></group>
          </group>);
      })}
      {p.picked && (
        <group position={p.picked.point}>
          <mesh raycast={() => null} renderOrder={7}><sphereGeometry args={[0.025, 12, 8]} /><meshBasicMaterial color={SC.pick} depthTest={false} transparent /></mesh>
          <mesh raycast={() => null} renderOrder={7}><torusGeometry args={[0.07, 0.008, 6, 24]} /><meshBasicMaterial color={SC.pick} depthTest={false} transparent /></mesh>
        </group>
      )}
      <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
        <GizmoViewport labels={["x", "y", "z↓"]} axisColors={["#e5484d", "#2f9e6e", "#2f6fdd"]} labelColor="#fff" />
      </GizmoHelper>
    </Canvas>
  );
}
