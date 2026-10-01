// 3D view of a course over the capture, drawn directly in the COURSE frame (z down).
// three.js world coordinates = course coordinates; the camera's up vector is -z, so
// "up" on screen is altitude and nothing is converted twice.
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, Grid, Line, OrbitControls, TransformControls } from "@react-three/drei";
import type { OrbitControls as OrbitImpl } from "three-stdlib";
import type { Geometry, Preview } from "../api";
import { Box, Course, displayPos, inside, pos0, Vec3 } from "./model";
import { Drone, DroneModel, yawQuat } from "./Drone";

export type Tool = "move" | "yaw" | "add" | "goal";
export interface ViewOpts { points: boolean; colorBy: "rgb" | "altitude"; pointSize: number; cameraPath: boolean; boxes: boolean; drone: boolean }

const C = {
  kf: "#2f6fdd", kfSel: "#f59e0b", kfBad: "#e5484d", path: "#8a8f98", box: "#8a8f98", wbox: "#2f9e6e",
  slow: new THREE.Color("#3b82f6"), fast: new THREE.Color("#f59e0b"), danger: new THREE.Color("#e5484d"), goal: "#a855f7", cursor: "#ec4899",
};

const wrapPi = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

function boxEdges(b: Box): Vec3[] {
  const [x0, y0, z0] = [0, 1, 2].map((a) => Math.min(b.lo[a], b.hi[a]));
  const [x1, y1, z1] = [0, 1, 2].map((a) => Math.max(b.lo[a], b.hi[a]));
  const v: Vec3[] = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const e = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  return e.flatMap(([a, b]) => [v[a], v[b]]);
}

/**
 * Screen-sized text label drawn into a sprite. (drei's <Html> throws "removeChild" when
 * keyframes or the whole canvas unmount under React 19, so labels stay inside WebGL.)
 */
function Label({ text, color, offset = [0, 0, 0], px = 13, bold = false }:
  { text: string; color: string; offset?: Vec3; px?: number; bold?: boolean }) {
  const { tex, aspect } = useMemo(() => {
    const scale = 2, pad = 4;
    const cv = document.createElement("canvas");
    const ctx = cv.getContext("2d")!;
    const font = `${bold ? "600 " : ""}${px * scale}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(text).width) + pad * 2 * scale, h = Math.ceil(px * scale * 1.4);
    cv.width = w; cv.height = h;
    ctx.font = font; ctx.textBaseline = "middle";
    ctx.lineWidth = 4; ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.strokeText(text, pad * scale, h / 2);
    ctx.fillStyle = color; ctx.fillText(text, pad * scale, h / 2);
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    return { tex: t, aspect: w / h };
  }, [text, color, px, bold]);
  useEffect(() => () => tex.dispose(), [tex]);
  const hgt = 0.028 * (px / 13);
  return (
    <sprite position={offset} scale={[hgt * aspect, hgt, 1]} center={new THREE.Vector2(-0.15, -0.4)} raycast={() => null} renderOrder={10}>
      <spriteMaterial map={tex} sizeAttenuation={false} depthTest={false} transparent />
    </sprite>
  );
}

function Cloud({ geo, opts }: { geo: Geometry; opts: ViewOpts }) {
  const g = useMemo(() => {
    const bg = new THREE.BufferGeometry();
    const p = new Float32Array(geo.points ?? []);
    bg.setAttribute("position", new THREE.BufferAttribute(p, 3));
    const n = p.length / 3, col = new Float32Array(n * 3);
    if (opts.colorBy === "rgb" && geo.colors) {
      for (let i = 0; i < n * 3; i++) col[i] = geo.colors[i] / 255;
    } else {
      const lo = geo.points_box?.lo[2] ?? -3, hi = geo.points_box?.hi[2] ?? 0, c = new THREE.Color();
      for (let i = 0; i < n; i++) {
        const u = Math.min(1, Math.max(0, (hi - p[i * 3 + 2]) / (hi - lo || 1)));   // 0 floor … 1 ceiling
        c.setHSL(0.62 - 0.55 * u, 0.55, 0.52);
        col.set([c.r, c.g, c.b], i * 3);
      }
    }
    bg.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return bg;
  }, [geo, opts.colorBy]);
  useEffect(() => () => g.dispose(), [g]);
  return (
    <points geometry={g} raycast={() => null}>
      <pointsMaterial size={opts.pointSize} sizeAttenuation vertexColors transparent opacity={0.9} depthWrite={false} />
    </points>
  );
}

function Frame({ geo }: { geo: Geometry | null }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.PerspectiveCamera; controls: OrbitImpl | null };
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (!geo || !controls || done.current === geo.scene) return;
    const b = geo.points_box ?? geo.camera_box;
    const c = new THREE.Vector3(...b.lo).add(new THREE.Vector3(...b.hi)).multiplyScalar(0.5);
    const span = Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], 2);
    camera.position.set(c.x - 0.9 * span, c.y + 0.9 * span, c.z - 0.8 * span);   // above: z is down
    controls.target.copy(c);
    controls.update();
    done.current = geo.scene;
  }, [geo, controls, camera]);
  return null;
}

export default function Scene3D(p: {
  geo: Geometry | null; course: Course; sel: number | null; onSelect: (i: number | null) => void;
  preview: Preview | null; cursor: number | null; tool: Tool; opts: ViewOpts;
  goalSelected: boolean; onGoalSelect: (on: boolean) => void;
  onDragStart: () => void; onMove: (i: number, v: Vec3) => void; onYaw: (i: number, yaw: number) => void;
  onAdd: (v: Vec3) => void; onGoalMove: (v: Vec3) => void;
  drone?: Drone | null;
}) {
  const { geo, course, sel, preview, opts } = p;
  const [obj, setObj] = useState<THREE.Object3D | null>(null);
  const [goalObj, setGoalObj] = useState<THREE.Object3D | null>(null);
  const box = geo?.camera_box ?? null;
  const solved = useMemo(() => {
    if (!preview) return null;
    const m = new Map(preview.keyframes.map((k) => [k.name, k.pos as Vec3]));
    return course.kfs.map((k) => m.get(k.name) ?? null);
  }, [preview, course]);
  const floorZ = geo?.points_box?.hi[2] ?? (geo ? geo.camera_box.hi[2] + 1.2 : 0);

  const pathColors = useMemo(() => {
    if (!preview) return null;
    const vmax = Math.max(0.1, preview.stats.v_max);
    const bad = new Array(preview.t.length).fill(false);
    const mark = (iv: [number, number][]) => iv.forEach(([a, b]) => preview.t.forEach((t, i) => { if (t >= a && t <= b) bad[i] = true; }));
    if (preview.clearance) mark(preview.clearance.below);
    if (preview.inside) mark(preview.inside.outside_intervals);
    return preview.speed.map((s, i) => {
      const c = bad[i] ? C.danger.clone() : C.slow.clone().lerp(C.fast, Math.min(1, s / vmax));
      return [c.r, c.g, c.b] as [number, number, number];
    });
  }, [preview]);

  let ci: number | null = null;
  if (preview && p.cursor !== null) {
    ci = 0;
    for (let i = 1; i < preview.t.length; i++) if (Math.abs(preview.t[i] - p.cursor) < Math.abs(preview.t[ci] - p.cursor)) ci = i;
  }
  const selKf = sel !== null ? course.kfs[sel] : null;
  const addZ = selKf ? displayPos(course, sel!, solved?.[sel!])[2] : course.kfs[0] ? displayPos(course, 0)[2] : -1;
  const selYaw = selKf?.fo[3][0] ?? 0;

  // Where the drone model goes: on the preview at the cursor (position and FiGS's attitude),
  // else level on the selected keyframe, else on the first one.
  let dronePose: { pos: Vec3; q: THREE.Quaternion } | null = null;
  if (p.drone && opts.drone) {
    if (preview && ci !== null) {
      const q = preview.quat?.[ci];
      dronePose = { pos: preview.pos[ci], q: q ? new THREE.Quaternion(q[0], q[1], q[2], q[3]) : yawQuat(preview.yaw[ci]) };
    } else {
      const i = sel ?? (course.kfs.length ? 0 : null);
      if (i !== null) dronePose = { pos: displayPos(course, i, solved?.[i]), q: yawQuat(course.kfs[i].fo[3][0] ?? 0) };
    }
  }
  const bodyR = preview?.clearance?.body_radius ?? 0;

  return (
    <Canvas frameloop="demand" camera={{ up: [0, 0, -1], position: [-6, 6, -5], fov: 50, near: 0.02, far: 400 }}
      onPointerMissed={() => { p.onSelect(null); p.onGoalSelect(false); }}>
      <OrbitControls makeDefault enableDamping={false} />
      <ambientLight intensity={1.1} />
      <directionalLight position={[3, -2, -6]} intensity={2.2} />
      <directionalLight position={[-4, 3, 2]} intensity={0.6} />
      <Frame geo={geo} />
      <Grid position={[0, 0, floorZ]} rotation={[Math.PI / 2, 0, 0]} args={[40, 40]} cellSize={0.5} sectionSize={1}
        cellColor="#8a8f98" sectionColor="#6b7079" cellThickness={0.6} sectionThickness={1} fadeDistance={35}
        infiniteGrid side={THREE.DoubleSide} />
      {geo?.points && opts.points && <Cloud geo={geo} opts={opts} />}
      {geo && opts.cameraPath && geo.camera_path.length > 1 &&
        <Line points={geo.camera_path} color={C.path} lineWidth={1} transparent opacity={0.7} />}
      {geo && opts.boxes && (
        <>
          <Line points={boxEdges(geo.camera_box)} segments color={C.box} lineWidth={1} />
          <Line points={boxEdges(geo.waypoint_box)} segments dashed dashSize={0.15} gapSize={0.1} color={C.wbox} lineWidth={1.5} />
        </>
      )}

      {preview && pathColors && preview.pos.length > 1 &&
        <Line points={preview.pos} vertexColors={pathColors} lineWidth={3} />}
      {preview?.clearance && (
        <group position={preview.clearance.at_pos}>
          <mesh raycast={() => null}><sphereGeometry args={[0.05, 12, 8]} /><meshBasicMaterial color={C.kfBad} /></mesh>
          {bodyR > 0 && (
            <mesh raycast={() => null}><sphereGeometry args={[bodyR, 24, 16]} />
              <meshBasicMaterial color={C.kfBad} wireframe transparent opacity={0.35} /></mesh>
          )}
          <Label text={`${preview.clearance.min} m`} color="#ff6b6b" bold />
        </group>
      )}
      {preview && ci !== null && (
        <group position={preview.pos[ci]}>
          {!dronePose && <mesh raycast={() => null}><sphereGeometry args={[0.09, 16, 12]} /><meshBasicMaterial color={C.cursor} /></mesh>}
          <Line points={[[0, 0, 0], preview.vel[ci].map((v) => v * 0.3) as Vec3]} color={C.cursor} lineWidth={2.5} />
        </group>
      )}

      {course.kfs.map((k, i) => {
        const pos = displayPos(course, i, solved?.[i]);
        const free = pos0(k).some((v) => v === null);
        const out = box ? !inside(pos0(k), box) : false;
        const yaw = k.fo[3][0];
        const color = i === sel ? C.kfSel : out ? C.kfBad : C.kf;
        return (
          <group key={k.name + i} position={pos} rotation={[0, 0, yaw ?? 0]} ref={i === sel ? setObj : undefined}>
            <mesh onClick={(e) => { e.stopPropagation(); p.onSelect(i); p.onGoalSelect(false); }}>
              <sphereGeometry args={[i === sel ? 0.1 : 0.08, 20, 14]} />
              <meshBasicMaterial color={color} transparent opacity={free ? 0.45 : 1} />
            </mesh>
            {yaw !== null && <Line points={[[0, 0, 0], [0.4, 0, 0]]} color={color} lineWidth={2.5} />}
            <Label text={k.name} color={out ? "#ff6b6b" : "#ffffff"} />
          </group>
        );
      })}

      {p.drone && dronePose && <DroneModel drone={p.drone} position={dronePose.pos} quaternion={dronePose.q} />}

      {course.goal && (
        <group position={course.goal.position} ref={p.goalSelected ? setGoalObj : undefined}>
          <mesh onClick={(e) => { e.stopPropagation(); p.onGoalSelect(true); p.onSelect(null); }}>
            <octahedronGeometry args={[p.goalSelected ? 0.14 : 0.11]} />
            <meshBasicMaterial color={C.goal} wireframe={!p.goalSelected} />
          </mesh>
          <Label text={`goal: ${course.goal.label || "—"}`} color="#d8b4fe" bold />
        </group>
      )}

      {p.tool === "add" && (
        <mesh position={[0, 0, addZ]} onClick={(e) => { e.stopPropagation(); p.onAdd([e.point.x, e.point.y, addZ]); }}>
          <planeGeometry args={[200, 200]} />
          <meshBasicMaterial color={C.kf} transparent opacity={0.05} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}

      {selKf && obj && (p.tool === "move" || p.tool === "add") && (
        <TransformControls object={obj} mode="translate" space="world" size={0.8}
          showX={selKf.fo[0][0] !== null} showY={selKf.fo[1][0] !== null} showZ={selKf.fo[2][0] !== null}
          onMouseDown={p.onDragStart}
          onObjectChange={() => p.onMove(sel!, [obj.position.x, obj.position.y, obj.position.z])} />
      )}
      {selKf && obj && p.tool === "yaw" && selKf.fo[3][0] !== null && (
        <TransformControls object={obj} mode="rotate" space="world" size={0.8} showX={false} showY={false}
          onMouseDown={p.onDragStart}
          onObjectChange={() => p.onYaw(sel!, selYaw + wrapPi(obj.rotation.z - selYaw))} />
      )}
      {course.goal && p.goalSelected && goalObj && (
        <TransformControls object={goalObj} mode="translate" space="world" size={0.7}
          onMouseDown={p.onDragStart}
          onObjectChange={() => p.onGoalMove([goalObj.position.x, goalObj.position.y, goalObj.position.z])} />
      )}

      <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
        <GizmoViewport labels={["x", "y", "z↓"]} axisColors={["#e5484d", "#2f9e6e", "#2f6fdd"]} labelColor="#fff" />
      </GizmoHelper>
    </Canvas>
  );
}
