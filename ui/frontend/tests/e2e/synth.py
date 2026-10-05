"""Synthetic room for the splat editor's end-to-end check (stdlib only, deterministic).

Splat frame = nerfstudio world; course = (x, -y, -z). Floor at splat z = 0, objects above it
(splat z > 0, course z < 0), cameras walked at 1 m height. Objects (course frame centres):
  red box     (1.5, -1.0, -0.4)   0.8 × 0.6 × 0.8 m
  green ball  (-1.5, 1.2, -0.3)   r 0.3
  blue chair  (-1.2, -1.6, -0.45) 0.5 × 0.5 × 0.9 m
"""
import math, os, random, struct

OBJECTS = {
    "red box": {"c": (1.5, -1.0, -0.4), "rgb": (205, 40, 35), "kind": "box", "size": (0.8, 0.6, 0.8)},
    "green ball": {"c": (-1.5, 1.2, -0.3), "rgb": (40, 170, 60), "kind": "ball", "size": (0.3,)},
    "blue chair": {"c": (-1.2, -1.6, -0.45), "rgb": (40, 70, 200), "kind": "box", "size": (0.5, 0.5, 0.9)},
}
COLOURS = {"red": (205, 40, 35), "green": (40, 170, 60), "blue": (40, 70, 200), "white": (235, 235, 235), "grey": (130, 130, 135),
           "floor": (130, 130, 135), "wall": (225, 220, 205), "box": (205, 40, 35), "ball": (40, 170, 60), "chair": (40, 70, 200)}


def scene(n_target=None):
    """List of (course_xyz, scale, rgba) in .splat order (most visible first is not needed here)."""
    n_target = n_target or int(os.environ.get("N_GAUSS", "120000"))
    rnd = random.Random(7)
    out = []
    n_obj = int(n_target * 0.12)
    per = n_obj // len(OBJECTS)
    for o in OBJECTS.values():
        cx, cy, cz = o["c"]
        for _ in range(per):
            if o["kind"] == "box":
                sx, sy, sz = o["size"]
                f = rnd.randrange(6); u, v = rnd.uniform(-0.5, 0.5), rnd.uniform(-0.5, 0.5)
                p = [[(1 if f == 0 else -1) * 0.5, u, v], [u, (1 if f == 2 else -1) * 0.5, v], [u, v, (1 if f == 4 else -1) * 0.5]][f // 2]
                xyz = (cx + p[0] * sx, cy + p[1] * sy, cz + p[2] * sz)
            else:
                r = o["size"][0]; th, ph = rnd.uniform(0, 2 * math.pi), math.acos(rnd.uniform(-1, 1))
                xyz = (cx + r * math.sin(ph) * math.cos(th), cy + r * math.sin(ph) * math.sin(th), cz + r * math.cos(ph))
            c = tuple(max(0, min(255, int(v + rnd.gauss(0, 12)))) for v in o["rgb"])
            out.append((xyz, 0.02, c + (240,)))
    rest = n_target - len(out)
    for i in range(rest):
        k = rnd.random()
        if k < 0.55:     # floor 7 × 7 m at z = 0
            xyz = (rnd.uniform(-3.5, 3.5), rnd.uniform(-3.5, 3.5), 0.0); base = COLOURS["floor"]
        else:            # walls at x = 3.5 and y = 3.5, 2.5 m high
            h = -rnd.uniform(0, 2.5)
            xyz = (3.5, rnd.uniform(-3.5, 3.5), h) if k < 0.78 else (rnd.uniform(-3.5, 3.5), 3.5, h)
            base = COLOURS["wall"]
        c = tuple(max(0, min(255, int(v + rnd.gauss(0, 10)))) for v in base)
        out.append((xyz, 0.035, c + (230,)))
    return out


def write_splat(path, sc):
    with open(path, "wb") as f:
        for (x, y, z), s, rgba in sc:
            f.write(struct.pack("<6f", x, -y, -z, s, s, s) + bytes(rgba) + bytes([255, 128, 128, 128]))


def camera_path():
    return [[2.2 * math.cos(t / 20 * 2 * math.pi), 2.2 * math.sin(t / 20 * 2 * math.pi), -1.0] for t in range(21)]
