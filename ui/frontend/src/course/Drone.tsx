// The team's airframe (CAD → ui/tools/drone_model.py → public/models/drone.glb) drawn at true
// scale in the course editor. The GLB is in FiGS's body frame (FRD: x forward, y right,
// z down), so placing it is just the course-frame position plus FiGS's own attitude
// quaternion ([x, y, z, w], the same order as THREE.Quaternion). No file, no drone: the
// editor works as before.
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import type { Vec3 } from "./model";

export interface DroneMeta {
  name: string; file: string; frame: string;
  radius: number; radius_xy: number; half_extent: Vec3; size: Vec3;
  rotors: Vec3[]; prop_radius: number | null; guards: boolean;
  triangles: { source: number; glb: number | null; ratio: number }; bytes: number;
  source: { obj: string; sha256: string; unit: string; forward: string; up: string; origin_rule: string };
}
export interface Drone { meta: DroneMeta; scene: THREE.Object3D }

const BASE = "models/";   // public/models, served next to index.html
let pending: Promise<Drone | null> | null = null;

/** Loads models/drone.json and the GLB it names, once per page. Resolves null if absent. */
export function loadDrone(): Promise<Drone | null> {
  if (!pending) {
    pending = (async () => {
      const r = await fetch(`${BASE}drone.json`);
      if (!r.ok) return null;
      const meta = (await r.json()) as DroneMeta;
      const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(`${BASE}${meta.file}`);
      gltf.scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        const mats = Array.isArray(m.material) ? m.material : [m.material];
        for (const mt of mats) if (mt.transparent) mt.depthWrite = false;   // props: see through
      });
      return { meta, scene: gltf.scene };
    })().catch((e) => { console.warn("drone model not loaded:", e); return null; });
  }
  return pending;
}

export function useDrone(): Drone | null {
  const [d, setD] = useState<Drone | null>(null);
  useEffect(() => { let alive = true; loadDrone().then((x) => alive && setD(x)); return () => { alive = false; }; }, []);
  return d;
}

export const yawQuat = (yaw: number): THREE.Quaternion =>
  new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw);   // about z (down), FiGS's yaw

/** One placed copy (geometry and materials shared with the loaded model). */
export function DroneModel({ drone, position, quaternion }: { drone: Drone; position: Vec3; quaternion: THREE.Quaternion }) {
  const obj = useMemo(() => {
    const o = drone.scene.clone(true);
    o.traverse((c) => { if ((c as THREE.Mesh).isMesh) c.raycast = () => {}; });   // never steal clicks from keyframes/gizmos
    return o;
  }, [drone]);
  return <primitive object={obj} position={position} quaternion={quaternion} />;
}
