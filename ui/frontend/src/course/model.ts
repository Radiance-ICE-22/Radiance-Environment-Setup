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
/**
 * The natural-language goal (thesis extension). FiGS and SousVide read only `waypoints` and
 * `forces`. Set by the splat editor's Send to course: the query it was resolved from, the
 * backend, the candidate's score and box, and the approach point (also the final keyframe when
 * that option was on). `score` is only meaningful while `position` is the resolved centroid, so
 * moving the goal by hand drops it (withGoalAt).
 */
export interface SemanticGoal {
  label: string; position: [number, number, number];
  query?: string; backend?: string; score?: number;
  extent?: { lo: [number, number, number]; hi: [number, number, number] };
  approach?: [number, number, number];
  [k: string]: unknown;
}
export function withGoalAt(g: SemanticGoal, position: [number, number, number]): SemanticGoal {
  const { score: _s, ...rest } = g;
  return { ...rest, position };
}

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

const dist3 = (p: Vec3, q: Vec3) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

/** The course's mean speed (m/s) over its keyframes' straight-line legs; null with no positive duration. */
export function meanSpeed(c: Course): number | null {
  let d = 0, t = 0;
  for (let i = 0; i + 1 < c.kfs.length; i++) {
    const dt = c.kfs[i + 1].t - c.kfs[i].t;
    if (dt > 0) { d += dist3(displayPos(c, i), displayPos(c, i + 1)); t += dt; }
  }
  return t > 0 && d > 0 ? d / t : null;
}

/**
 * New pass-through keyframe between i and i+1 (position fixed, derivatives free). It INSERTS time
 * rather than splitting the interval: each new leg gets distance / speed (the course's mean speed,
 * clamped to 0.5–2 m/s; 0.8 m/s when there is none), at least 0.5 s, and later keyframes shift
 * when the old interval is too short. Halving the interval instead (the old rule) squeezed a 3 s
 * course into segments of 0.375 s for 4 m — a starting guess the expert's time optimisation
 * cannot recover from (6 Oct).
 */
export function insertAfter(c: Course, i: number, at?: Vec3): Course {
  const j = Math.min(i, c.kfs.length - 2);          // never after the last keyframe
  const a = c.kfs[j], b = c.kfs[j + 1];
  const pa = displayPos(c, j), pb = displayPos(c, j + 1);
  const p = at ?? (pa.map((v, ax) => (v + pb[ax]) / 2) as Vec3);
  const yawA = a.fo[3][0], yawB = b.fo[3][0];
  const yaw = yawA !== null && yawB !== null ? (yawA + yawB) / 2 : yawA ?? yawB ?? 0;
  const v = Math.min(2, Math.max(0.5, meanSpeed(c) ?? 0.8));
  const dA = Math.max(0.5, dist3(pa, p) / v), dB = Math.max(0.5, dist3(p, pb) / v);
  const old = b.t - a.t;
  const shift = Math.max(0, dA + dB - old);
  const t = shift > 0 ? a.t + dA : a.t + (old * dA) / (dA + dB);
  const kf: KF = { name: nextName(c), t: round(t, 3),
    fo: [...p.map((v) => [round(v, 3), null, null, null]), [round(yaw, 3), null, null, null]] };
  const kfs = c.kfs.map((k, n) => (n > j && shift > 0 ? { ...k, t: round(k.t + shift, 3) } : k));
  kfs.splice(j + 1, 0, kf);
  return { ...c, kfs };
}

/**
 * Same rule as figs_pipeline.course_fast_segments: legs whose keyframe times are far too short for
 * their distance (> 5 m/s, or > 3 × the course's mean speed and > 2.5 m/s), or whose time does not
 * increase. Positions use the axes both keyframes fix, as the pipeline does.
 */
export function fastSegments(c: Course, maxSpeed = 5, maxRatio = 3): string[] {
  const segs: { a: string; b: string; d: number; dt: number }[] = [];
  for (let i = 0; i + 1 < c.kfs.length; i++) {
    const A = c.kfs[i], B = c.kfs[i + 1];
    let s = 0;
    for (let ax = 0; ax < 3; ax++) {
      const x = A.fo[ax]?.[0] ?? null, y = B.fo[ax]?.[0] ?? null;
      if (x !== null && y !== null) s += (x - y) ** 2;
    }
    segs.push({ a: A.name, b: B.name, d: Math.sqrt(s), dt: B.t - A.t });
  }
  const tt = segs.reduce((n, g) => n + (g.dt > 0 ? g.dt : 0), 0);
  const mean = tt > 0 ? segs.reduce((n, g) => n + (g.dt > 0 ? g.d : 0), 0) / tt : 0;
  const out: string[] = [];
  for (const g of segs) {
    if (g.dt <= 0) { out.push(`${g.a}→${g.b}: t does not increase (${g.dt >= 0 ? "+" : ""}${g.dt.toFixed(3)} s)`); continue; }
    const sp = g.d / g.dt;
    if (sp > maxSpeed || (sp > 2.5 && mean > 0 && sp > maxRatio * mean))
      out.push(`${g.a}→${g.b}: ${g.d.toFixed(2)} m in ${g.dt.toFixed(3)} s = ${sp.toFixed(1)} m/s (course mean ${mean.toFixed(1)} m/s)`);
  }
  return out;
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

// ── semantic goal → course (splat editor ▸ Send to course) ────────────────────
export const wrapPi = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
/** Yaw (course frame: heading (cos ψ, sin ψ) in x, y) from `from` looking at `to`. */
export const yawToward = (from: Vec3, to: Vec3) => Math.atan2(to[1] - from[1], to[0] - from[0]);
const rest = (p: Vec3, yaw: number): Cell[][] => [[round(p[0]), 0], [round(p[1]), 0], [round(p[2]), 0], [round(yaw), 0]];

/**
 * Append the approach point as the new final keyframe, at rest and facing the goal. The old
 * final keyframe becomes a pass-through (position and yaw kept, derivatives freed); the new
 * yaw is unwrapped to within π of it; the time grows with the distance at `speed` m/s.
 */
export function appendApproach(c: Course, approach: Vec3, goal: Vec3, speed = 0.8): Course {
  const last = c.kfs[c.kfs.length - 1];
  const prevPos = displayPos(c, c.kfs.length - 1);
  const prevYaw = last?.fo[3][0] ?? 0;
  const yaw = prevYaw + wrapPi(yawToward(approach, goal) - prevYaw);
  const dist = Math.hypot(approach[0] - prevPos[0], approach[1] - prevPos[1], approach[2] - prevPos[2]);
  const t = round((last?.t ?? 0) + Math.max(2.5, dist / speed), 2);
  const kfs = c.kfs.map((k, i) => (i === c.kfs.length - 1 ? { ...k, fo: k.fo.map((r) => [r[0]]) } : k));
  let name = "goal", n = 1;
  while (kfs.some((k) => k.name === name)) name = `goal${n++}`;
  return { ...c, kfs: [...kfs, { name, t, fo: rest(approach, yaw) }] };
}

/**
 * A new two-keyframe course to the approach point: from where the capture started (the first
 * camera position, moved into the waypoint box), at rest at both ends, facing the goal at the end.
 */
export function courseToGoal(start: Vec3, approach: Vec3, goal: Vec3, wbox: Box | null, speed = 0.8): Course {
  const s = start.map((v, a) => (wbox && wbox.lo[a] <= wbox.hi[a] ? Math.min(Math.max(v, wbox.lo[a]), wbox.hi[a]) : v)) as Vec3;
  const yawEnd = yawToward(approach, goal);
  const yawStart = yawToward(s, approach);
  const dist = Math.hypot(approach[0] - s[0], approach[1] - s[1], approach[2] - s[2]);
  return {
    Nco: 6, forces: null, goal: null, extra: {}, wpExtra: {},
    kfs: [
      { name: "start", t: 0, fo: rest(s, yawStart) },
      { name: "goal", t: round(Math.max(3, dist / speed), 2), fo: rest(approach, yawStart + wrapPi(yawEnd - yawStart)) },
    ],
  };
}
