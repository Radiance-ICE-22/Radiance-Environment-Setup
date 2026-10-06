// course/model.ts: the semantic goal hand-off (splat editor ▸ Send to course).
import { test } from "node:test";
import assert from "node:assert/strict";
import { appendApproach, blankLoop, courseToGoal, problems, toFile, fromFile, withGoalAt, yawToward } from "../src/course/model.ts";

test("yawToward: heading (cos ψ, sin ψ) in the course x, y plane", () => {
  assert.equal(yawToward([0, 0, -1], [1, 0, -1]), 0);
  assert.ok(Math.abs(yawToward([0, 0, -1], [0, 2, -1]) - Math.PI / 2) < 1e-12);
});

test("withGoalAt keeps the query but drops the score", () => {
  const g = { label: "chest", position: [1, 2, 3] as [number, number, number], query: "red tool chest", backend: "lift", score: 9 };
  const m = withGoalAt(g, [0, 0, 0]);
  assert.deepEqual(m, { label: "chest", position: [0, 0, 0], query: "red tool chest", backend: "lift" });
});

test("appendApproach: new final keyframe at rest facing the goal; old one a pass-through; still valid", () => {
  const c = blankLoop({ lo: [-2, -2, -1.5], hi: [2, 2, -0.5] }, null);
  const last = c.kfs[c.kfs.length - 1];
  const out = appendApproach(c, [1, 1, -1], [2, 1, -1]);
  assert.equal(out.kfs.length, c.kfs.length + 1);
  const g = out.kfs[out.kfs.length - 1];
  assert.equal(g.name, "goal");
  assert.deepEqual(g.fo.map((r) => r[0]).slice(0, 3), [1, 1, -1]);
  assert.deepEqual(g.fo.map((r) => r[1]), [0, 0, 0, 0]);            // at rest
  const yaw = g.fo[3][0]!;
  assert.ok(Math.abs(Math.atan2(Math.sin(yaw), Math.cos(yaw))) < 1e-3);   // facing +x (the goal)
  assert.ok(Math.abs(yaw - last.fo[3][0]!) <= Math.PI + 1e-9);      // unwrapped next to the old last yaw (6.283)
  assert.ok(g.t > last.t + 2.4);
  assert.deepEqual(out.kfs[out.kfs.length - 2].fo, last.fo.map((r) => [r[0]]));   // derivatives freed
  assert.deepEqual(problems(out), []);
  assert.equal(appendApproach(out, [0, 0, -1], [1, 0, -1]).kfs.at(-1)!.name, "goal1");
});

test("courseToGoal: two rest keyframes, start clamped into the waypoint box, valid, round-trips", () => {
  const c = courseToGoal([5, 0, -1], [0, 1, -1], [0, 2, -1], { lo: [-2, -2, -1.5], hi: [2, 2, -0.5] });
  assert.equal(c.kfs.length, 2);
  assert.deepEqual(c.kfs[0].fo.map((r) => r[0]).slice(0, 3), [2, 0, -1]);
  assert.ok(c.kfs[1].t >= 3);
  assert.ok(Math.abs(Math.atan2(Math.sin(c.kfs[1].fo[3][0]!), Math.cos(c.kfs[1].fo[3][0]!)) - Math.PI / 2) < 1e-3);
  assert.deepEqual(problems(c), []);
  const withGoal = { ...c, goal: { label: "x", position: [0, 2, -1] as [number, number, number], query: "x", score: 1, approach: [0, 1, -1] as [number, number, number] } };
  assert.deepEqual(fromFile(JSON.parse(JSON.stringify(toFile(withGoal)))), withGoal);
});

// ── keyframe timing (6 Oct: Add squeezed a 3 s course and the expert's re-time could not recover) ──
import { fastSegments, insertAfter, meanSpeed } from "../src/course/model.ts";
import type { Course } from "../src/course/model.ts";

const twoKf = (): Course => fromFile({ waypoints: { Nco: 6, keyframes: {
  start: { t: 0, fo: [[-0.025, 0], [-6.715, 0], [-1.211, 0], [0.163, 0]] },
  goal: { t: 3, fo: [[-0.63, 0], [1.111, 0], [-1.109, 0], [1.552, 0]] } } }, forces: null } as any);

test("insertAfter adds time for a far keyframe instead of halving the interval", () => {
  const c = twoKf();
  const v = Math.min(2, Math.max(0.5, meanSpeed(c)!));
  const out = insertAfter(insertAfter(c, 0, [0.836, -4.752, -1.519]), 1, [1.368, -0.67, -1.465]);
  assert.deepEqual(out.kfs.map((k) => k.name), ["start", "k2", "k3", "goal"]);
  const t = out.kfs.map((k) => k.t);
  assert.ok(t.every((x, i) => i === 0 || x > t[i - 1]));                    // still increasing
  assert.deepEqual(fastSegments(out), []);                                  // no squeezed leg
  assert.ok(out.kfs[3].t > 3);                                              // the course got longer
  const leg = Math.hypot(1.368 - 0.836, -0.67 + 4.752, -1.465 + 1.519);
  assert.ok(Math.abs((t[2] - t[1]) - leg / v) < 0.01);                      // distance / course speed
});

test("insertAfter splits proportionally when the interval already has room", () => {
  const c = twoKf();
  const slow = { ...c, kfs: c.kfs.map((k, i) => (i === 1 ? { ...k, t: 60 } : k)) };
  const out = insertAfter(slow, 0);                                          // midpoint, plenty of time
  assert.equal(out.kfs[2].t, 60);
  assert.ok(Math.abs(out.kfs[1].t - 30) < 0.01);
});

test("fastSegments flags the squeezed course and the old halving rule, not a sane one", () => {
  const sq = fromFile({ waypoints: { Nco: 6, keyframes: {
    start: { t: 0, fo: [[-0.025, 0], [-6.715, 0], [-1.211, 0], [0.163, 0]] },
    k3: { t: 2.25, fo: [[0.836], [-4.752], [-1.519], [1.631]] },
    k4: { t: 2.625, fo: [[1.368], [-0.67], [-1.465], [1.967]] },
    goal: { t: 3, fo: [[-0.63, 0], [1.111, 0], [-1.109, 0], [1.552, 0]] } } }, forces: null } as any);
  const f = fastSegments(sq);
  assert.equal(f.length, 2);
  assert.match(f[0], /^k3→k4: 4\.12 m in 0\.375 s = 11\.0 m\/s/);
  assert.deepEqual(fastSegments(twoKf()), []);
  const back = { ...sq, kfs: sq.kfs.map((k, i) => (i === 2 ? { ...k, t: 2.0 } : k)) };
  assert.match(fastSegments(back).join(" "), /t does not increase/);
});
