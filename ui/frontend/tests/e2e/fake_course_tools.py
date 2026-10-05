"""Stand-in for figs/course_tools.py (kitchen env): geometry, splat and a straight-line preview."""
import json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
sys.path.insert(0, os.environ["E2E_DIR"])
import synth

args = sys.argv[1:]
def out(d): print("GALLEY_JSON " + json.dumps(d)); sys.exit(0)

if args[0] == "geometry":
    sc = synth.scene()
    pts = [v for (p, s, c) in sc[::40] for v in p]
    cols = [v for (p, s, c) in sc[::40] for v in c[:3]]
    out({"ok": True, "scene": args[args.index("--scene") + 1], "camera_path": synth.camera_path(),
         "camera_box": {"lo": [-2.2, -2.2, -1.05], "hi": [2.2, 2.2, -0.95]},
         "waypoint_box": {"lo": [-1.7, -1.7, -0.55], "hi": [1.7, 1.7, -1.45], "margin": 0.5},
         "bounds_splat": {"lo": [-3.5, -3.5, 0], "hi": [3.5, 3.5, 2.5]},
         "points": pts, "colors": cols, "points_box": {"lo": [-3.5, -3.5, -2.5], "hi": [3.5, 3.5, 0.0]},
         "n_points": len(pts) // 3, "n_points_sent": len(pts) // 3})
elif args[0] == "splat":
    path = args[args.index("--out") + 1]
    sc = synth.scene(); synth.write_splat(path, sc)
    out({"ok": True, "n_total": len(sc), "n_written": len(sc), "bytes": os.path.getsize(path), "min_opacity": 0.05, "step": 29999, "seconds": 0.5})
else:   # preview: straight lines between the keyframes at the file's times
    course = json.loads(sys.stdin.read())
    kf = list(course["waypoints"]["keyframes"].items())
    P = [[k["fo"][a][0] if k["fo"][a][0] is not None else 0.0 for a in range(3)] for _, k in kf]
    T = [k["t"] for _, k in kf]
    t, pos = [], []
    for i in range(len(kf) - 1):
        for j in range(10):
            u = j / 10; t.append(T[i] + u * (T[i + 1] - T[i])); pos.append([P[i][a] + u * (P[i + 1][a] - P[i][a]) for a in range(3)])
    t.append(T[-1]); pos.append(P[-1])
    m = len(t); z = [[0.0, 0.0, 0.0]] * m
    out({"ok": True, "mode": "fixed", "pilot": "Viper", "frame": "carl", "hz": 10, "kT": None, "use_l2_time": False, "solve_s": 0.01,
         "keyframes": [{"name": n, "t_file": k["t"], "t_solved": k["t"], "pos": P[i], "yaw": k["fo"][3][0]} for i, (n, k) in enumerate(kf)],
         "duration_file": T[-1], "duration_solved": T[-1], "t": t, "pos": pos, "vel": z, "acc": z, "yaw": [0.0] * m,
         "speed": [0.5] * m, "acc_norm": [0.1] * m,
         "stats": {"v_max": 0.5, "v_mean": 0.5, "a_max": 0.1, "length_m": 1.0, "nonfinite_inputs": 0},
         "inputs": {"names": ["thrust", "wx", "wy", "wz"], "lower": [-1, -5, -5, -5], "upper": [0, 5, 5, 5],
                    "u": [[-0.5] * m, [0.0] * m, [0.0] * m, [0.0] * m], "max_use": [0.5, 0, 0, 0], "violations": {}},
         "clearance": None, "inside": None})
