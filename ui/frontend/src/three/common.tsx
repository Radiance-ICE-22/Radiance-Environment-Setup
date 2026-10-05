// 3D helpers shared by the course editor (course/Scene3D.tsx) and the splat editor
// (splat/SplatScene.tsx). Everything is drawn in the COURSE frame (z down): the cameras' up
// vector is −z, so "up" on screen is altitude.
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import type { OrbitControls as OrbitImpl } from "three-stdlib";

export type V3 = [number, number, number];
export interface Box3 { lo: V3 | number[]; hi: V3 | number[] }

/** The 12 edges of an axis-aligned box as line-segment pairs. */
export function boxEdges(b: Box3): V3[] {
  const [x0, y0, z0] = [0, 1, 2].map((a) => Math.min(b.lo[a], b.hi[a]));
  const [x1, y1, z1] = [0, 1, 2].map((a) => Math.max(b.lo[a], b.hi[a]));
  const v: V3[] = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const e = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  return e.flatMap(([a, c]) => [v[a], v[c]]);
}

/**
 * Screen-sized text label drawn into a sprite. (drei's <Html> throws "removeChild" when
 * keyframes or the whole canvas unmount under React 19, so labels stay inside WebGL.)
 */
export function Label({ text, color, offset = [0, 0, 0], px = 13, bold = false }:
  { text: string; color: string; offset?: V3; px?: number; bold?: boolean }) {
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

/** The default view of a box: from above one corner, looking at its middle. */
export function frameBox(b: Box3, camera: THREE.Camera, controls: OrbitImpl) {
  const c = new THREE.Vector3(...(b.lo as V3)).add(new THREE.Vector3(...(b.hi as V3))).multiplyScalar(0.5);
  const span = Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], 2);
  camera.position.set(c.x - 0.9 * span, c.y + 0.9 * span, c.z - 0.8 * span);   // above: z is down
  controls.target.copy(c);
  controls.update();
}
/** Frames `box` once per `id` (a scene name), as soon as the controls exist. */
export function Frame({ box, id }: { box: Box3 | null; id: string | undefined }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.PerspectiveCamera; controls: OrbitImpl | null };
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (!box || !id || !controls || done.current === id) return;
    frameBox(box, camera, controls);
    done.current = id;
  }, [box, id, controls, camera]);
  return null;
}

// ── keyboard navigation ────────────────────────────────────────────────────────
// Arrows: fly forward/back and sideways (level, in the course's horizontal plane);
// PageUp/PageDown or E/Q: up/down; Ctrl+arrows: look around in place; + / −: move
// toward / away from the orbit centre; Shift: 3× faster; Home: default view; F: centre
// on the selection. Movement eases in and out, so holding a key glides.
const NAV_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "e", "q", "E", "Q", "+", "=", "-", "_"]);
export const UP = new THREE.Vector3(0, 0, -1);   // course frame: z is down
export function KeyNav({ enabled, speed, box, focus }: { enabled: boolean; speed: number; box: Box3 | null; focus: V3 | null }) {
  const { camera, controls, invalidate } = useThree() as unknown as { camera: THREE.PerspectiveCamera; controls: OrbitImpl | null; invalidate: () => void };
  const keys = useRef(new Set<string>());
  const mods = useRef({ shift: false, ctrl: false });
  const vel = useRef(new THREE.Vector3());      // m/s, world
  const rot = useRef({ yaw: 0, pitch: 0 });      // rad/s
  const live = useRef({ enabled, box, focus });
  live.current = { enabled, box, focus };
  useEffect(() => {
    const typing = (e: KeyboardEvent) => { const el = e.target as HTMLElement; return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable); };
    const down = (e: KeyboardEvent) => {
      mods.current = { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey };
      if (!live.current.enabled || typing(e) || e.altKey) return;
      if (e.key === "Home" && live.current.box && controls) { e.preventDefault(); frameBox(live.current.box, camera, controls); vel.current.set(0, 0, 0); invalidate(); return; }
      if ((e.key === "f" || e.key === "F") && !e.ctrlKey && live.current.focus && controls) {
        e.preventDefault();
        const d = new THREE.Vector3().subVectors(camera.position, controls.target);
        controls.target.set(...live.current.focus); camera.position.copy(controls.target).add(d.setLength(Math.min(d.length(), 3)));
        controls.update(); invalidate(); return;
      }
      if (!NAV_KEYS.has(e.key)) return;
      if ((e.key === "e" || e.key === "q" || e.key === "E" || e.key === "Q") && (e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      keys.current.add(e.key.length === 1 ? e.key.toLowerCase() : e.key);
      invalidate();
    };
    const up = (e: KeyboardEvent) => { mods.current = { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey }; keys.current.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key); };
    const clear = () => keys.current.clear();
    addEventListener("keydown", down); addEventListener("keyup", up); addEventListener("blur", clear);
    return () => { removeEventListener("keydown", down); removeEventListener("keyup", up); removeEventListener("blur", clear); };
  }, [camera, controls, invalidate]);
  useEffect(() => { if (!enabled) keys.current.clear(); }, [enabled]);
  useFrame((_, dtRaw) => {
    if (!controls) return;
    const dt = Math.min(dtRaw, 0.05);
    const k = keys.current, look = mods.current.ctrl;
    const has = (x: string) => k.has(x);
    const dist = camera.position.distanceTo(controls.target);
    const v = speed * (mods.current.shift ? 3 : 1) * THREE.MathUtils.clamp(dist * 0.5, 0.6, 4);   // m/s, scaled to the view's size
    // desired motion
    const fwd = new THREE.Vector3(); camera.getWorldDirection(fwd);
    const level = fwd.clone().addScaledVector(UP, -fwd.dot(UP));
    if (level.lengthSq() < 1e-6) level.set(1, 0, 0).applyQuaternion(camera.quaternion);
    level.normalize();
    const right = new THREE.Vector3().crossVectors(level, UP).normalize();
    const want = new THREE.Vector3();
    if (!look) {
      if (has("ArrowUp")) want.add(level); if (has("ArrowDown")) want.sub(level);
      if (has("ArrowRight")) want.add(right); if (has("ArrowLeft")) want.sub(right);
    }
    if (has("PageUp") || has("e")) want.add(UP); if (has("PageDown") || has("q")) want.sub(UP);
    if (want.lengthSq() > 0) want.setLength(v);
    const wantYaw = look ? ((has("ArrowLeft") ? 1 : 0) - (has("ArrowRight") ? 1 : 0)) * 1.2 * (mods.current.shift ? 2 : 1) : 0;
    const wantPitch = look ? ((has("ArrowUp") ? 1 : 0) - (has("ArrowDown") ? 1 : 0)) * 0.9 : 0;
    const dolly = (has("+") || has("=") ? 1 : 0) - (has("-") || has("_") ? 1 : 0);
    // ease toward it (≈ 0.1 s), so a tap nudges and a held key glides
    const a = 1 - Math.exp(-dt * 10);
    vel.current.lerp(want, a);
    rot.current.yaw += (wantYaw - rot.current.yaw) * a;
    rot.current.pitch += (wantPitch - rot.current.pitch) * a;
    let moved = false;
    if (vel.current.lengthSq() > 1e-6) {
      const d = vel.current.clone().multiplyScalar(dt);
      camera.position.add(d); controls.target.add(d); moved = true;
    } else vel.current.set(0, 0, 0);
    if (Math.abs(rot.current.yaw) > 1e-4 || Math.abs(rot.current.pitch) > 1e-4) {
      // look around: swing the orbit centre around the camera
      const off = new THREE.Vector3().subVectors(controls.target, camera.position);
      off.applyAxisAngle(UP, rot.current.yaw * dt);
      const r = new THREE.Vector3().crossVectors(off, UP).normalize();
      const p = off.clone().applyAxisAngle(r, rot.current.pitch * dt);
      if (Math.abs(p.clone().normalize().dot(UP)) < 0.97) off.copy(p);   // not straight up or down
      controls.target.copy(camera.position).add(off); moved = true;
    } else { rot.current.yaw = 0; rot.current.pitch = 0; }
    if (dolly) {
      const off = new THREE.Vector3().subVectors(camera.position, controls.target);
      const len = Math.max(0.3, off.length() * Math.exp(-dolly * dt * 1.5 * (mods.current.shift ? 2 : 1)));
      camera.position.copy(controls.target).add(off.setLength(len)); moved = true;
    }
    if (moved) { controls.update(); invalidate(); }
    else if (k.size) invalidate();
  });
  return null;
}
