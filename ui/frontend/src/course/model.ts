// Course file model for the editor. Mirrors SousVide/configs/courses/<name>.json exactly:
// keyframes in file order, each with t and a 4-row fo matrix (x, y, z, yaw) whose columns
// are derivative orders (position, velocity, acceleration, jerk, snap). null = free.
// Everything is in the COURSE frame: course = (x, -y, -z) of the splat, z points down.

export type Cell = number | null;
export interface Keyframe { t: number; fo: Cell[][] }
export interface CourseFile {
  waypoints: { Nco: number; keyframes: Record<string, Keyframe>; [k: string]: unknown };
  forces: Record<string, unknown> | null;
  semantic_goal?: SemanticGoal;
  [k: string]: unknown;
}
/** Hook for the thesis extension. FiGS and SousVide read only `waypoints` and `forces`. */
export interface SemanticGoal { label: string; position: [number, number, number] }

/** Editor state: keyframes as an ordered list (names are unique keys in the file). */
export interface KF { name: string; t: number; fo: Cell[][] }
export interface Course { Nco: number; kfs: KF[]; forces: Record<string, unknown> | null;
  goal: SemanticGoal | null; extra: Record<string, unknown>; wpExtra: Record<string, unknown> }

export const AXES = ["x", "y", "z", "yaw"] as const;
export const ORDERS = ["pos", "vel", "acc", "jerk", "snap"] as const;
export const MAX_ORDERS = 5;

export function fromFile(f: CourseFile): Course {
  const { waypoints, forces, semantic_goal, ...extra } = f;
  const { Nco, keyframes, ...wpExtra } = waypoints;
  return {
    Nco, forces: forces ?? null, goal: semantic_goal ?? null, extra, wpExtra,
    kfs: Object.entries(keyframes).map(([name, k]) => ({ name, t: k.t, fo: k.fo.map((r) => [...r]) })),
  };
}

export function toFile(c: Course): CourseFile {
  const keyframes: Record<string, Keyframe> = {};
  for (const k of c.kfs) keyframes[k.name] = { t: k.t, fo: k.fo.map((r) => [...r]) };
  const out: CourseFile = { waypoints: { Nco: c.Nco, keyframes, ...c.wpExtra }, forces: c.forces, ...c.extra };
  if (c.goal) out.semantic_goal = c.goal;
  return out;
}

export type Vec3 = [number, number, number];
export interface Box { lo: Vec3; hi: Vec3 }

/** The course step's test (figs_pipeline.course_inside): only constrained axes are checked. */
export function inside(p: (number | null)[], box: Box): boolean {
  return [0, 1, 2].every((i) => p[i] === null || (p[i]! >= box.lo[i] && p[i]! <= box.hi[i]));
}

export const pos0 = (k: KF): (number | null)[] => [0, 1, 2].map((i) => k.fo[i]?.[0] ?? null);

/**
 * Where to draw a keyframe. Fixed axes come from the file. A free axis takes the solver's
 * value when a preview exists, else a time-interpolation between neighbours that fix it.
 */
export function displayPos(c: Course, i: number, solved?: Vec3 | null): Vec3 {
  const k = c.kfs[i];
  return [0, 1, 2].map((ax) => {
    const v = k.fo[ax]?.[0];
    if (v !== null && v !== undefined) return v;
    if (solved) return solved[ax];
    let a = i - 1, b = i + 1;
    while (a >= 0 && (c.kfs[a].fo[ax]?.[0] ?? null) === null) a--;
    while (b < c.kfs.length && (c.kfs[b].fo[ax]?.[0] ?? null) === null) b++;
    if (a < 0 || b >= c.kfs.length) return 0;
    const va = c.kfs[a].fo[ax][0]!, vb = c.kfs[b].fo[ax][0]!;
    const u = (k.t - c.kfs[a].t) / (c.kfs[b].t - c.kfs[a].t || 1);
    return va + u * (vb - va);
  }) as Vec3;
}

export interface Problem { kf?: number; msg: string }

/** Same rules as the backend's Pydantic model, checked as you type. */
export function problems(c: Course): Problem[] {
  const out: Problem[] = [];
  if (c.kfs.length < 2) out.push({ msg: "a course needs at least 2 keyframes" });
  const names = new Set<string>();
  c.kfs.forEach((k, i) => {
    if (!/^[A-Za-z0-9_\-]+$/.test(k.name)) out.push({ kf: i, msg: `keyframe name “${k.name}”: letters, digits, _ and - only` });
    if (names.has(k.name)) out.push({ kf: i, msg: `duplicate keyframe name “${k.name}”` });
    names.add(k.name);
    if (!Number.isFinite(k.t) || k.t < 0) out.push({ kf: i, msg: `${k.name}: t must be a number ≥ 0` });
    if (i > 0 && !(k.t > c.kfs[i - 1].t)) out.push({ kf: i, msg: `${k.name}: t must be later than ${c.kfs[i - 1].name}` });
    if (k.fo.length !== 4) out.push({ kf: i, msg: `${k.name}: fo must have 4 rows` });
    k.fo.forEach((row, r) => {
      if (row.length < 1 || row.length > MAX_ORDERS) out.push({ kf: i, msg: `${k.name}: ${AXES[r]} needs 1–5 columns` });
      if (row.some((v) => v !== null && !Number.isFinite(v))) out.push({ kf: i, msg: `${k.name}: ${AXES[r]} has a bad number` });
    });
  });
  for (const i of [0, c.kfs.length - 1]) {
    const k = c.kfs[i];
    if (k && k.fo.some((r) => r[0] === null)) out.push({ kf: i, msg: `${k.name}: the ${i === 0 ? "first" : "last"} keyframe must fix x, y, z and yaw` });
  }
  return out;
}

export function nextName(c: Course, base = "k"): string {
  let n = c.kfs.length;
  while (c.kfs.some((k) => k.name === `${base}${n}`)) n++;
  return `${base}${n}`;
}

/** New pass-through keyframe between i and i+1 (position fixed, derivatives free). */
export function insertAfter(c: Course, i: number, at?: Vec3): Course {
  const j = Math.min(i, c.kfs.length - 2);          // never after the last keyframe
  const a = c.kfs[j], b = c.kfs[j + 1];
  const pa = displayPos(c, j), pb = displayPos(c, j + 1);
  const p = at ?? (pa.map((v, ax) => (v + pb[ax]) / 2) as Vec3);
  const yawA = a.fo[3][0], yawB = b.fo[3][0];
  const yaw = yawA !== null && yawB !== null ? (yawA + yawB) / 2 : yawA ?? yawB ?? 0;
  const kf: KF = { name: nextName(c), t: round((a.t + b.t) / 2, 3),
    fo: [...p.map((v) => [round(v, 3), null, null, null]), [round(yaw, 3), null, null, null]] };
  const kfs = [...c.kfs];
  kfs.splice(j + 1, 0, kf);
  return { ...c, kfs };
}

/**
 * A four-corner loop inside the recommended waypoint box: a starting point for a new room.
 * On an axis where the box is empty (camera span < 2 × margin, common for height: a phone
 * is carried at a steady height) the camera box's middle is used instead.
 */
export function blankLoop(wbox: Box | null, cbox: Box | null): Course {
  const lo = [0, 1, 2].map((a) => (wbox && wbox.lo[a] <= wbox.hi[a] ? wbox.lo[a] : cbox ? (cbox.lo[a] + cbox.hi[a]) / 2 : [-1, -1, -1.2][a]));
  const hi = [0, 1, 2].map((a) => (wbox && wbox.lo[a] <= wbox.hi[a] ? wbox.hi[a] : cbox ? (cbox.lo[a] + cbox.hi[a]) / 2 : [1, 1, -1.2][a]));
  const z = round(Math.min(Math.max(-1.2, lo[2]), hi[2]), 2);
  const x0 = round(lo[0] + 0.2 * (hi[0] - lo[0]), 2), x1 = round(hi[0] - 0.2 * (hi[0] - lo[0]), 2);
  const y0 = round(lo[1] + 0.2 * (hi[1] - lo[1]), 2), y1 = round(hi[1] - 0.2 * (hi[1] - lo[1]), 2);
  const pass = (x: number, y: number, yaw: number): Cell[][] => [[x, null, null, null], [y, null, null, null], [z, null, null, null], [yaw, null, null, null]];
  const rest = (x: number, y: number, yaw: number): Cell[][] => [[x, 0], [y, 0], [z, 0], [yaw, 0]];
  return {
    Nco: 6, forces: null, goal: null, extra: {}, wpExtra: {},
    kfs: [
      { name: "start", t: 0, fo: rest(x0, y0, 0) },
      { name: "k1", t: 3, fo: pass(x1, y0, 1.571) },
      { name: "k2", t: 6, fo: pass(x1, y1, 3.142) },
      { name: "k3", t: 9, fo: pass(x0, y1, 4.712) },
      { name: "end", t: 12, fo: rest(x0, y0, 6.283) },
    ],
  };
}

/** Axes on which the recommended box is empty (inset larger than half the camera span). */
export const emptyAxes = (b: Box | null) => (b ? [0, 1, 2].filter((a) => b.lo[a] > b.hi[a]).map((a) => "xyz"[a]) : []);

export const round = (v: number, d = 3) => Math.round(v * 10 ** d) / 10 ** d;
