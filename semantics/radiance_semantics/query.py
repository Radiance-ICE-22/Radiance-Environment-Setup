"""Text query → ranked 3D candidates → an approach point a drone can fly to.

1. Encode the query and LERF's canonical negatives with the CLIP text tower.
2. Relevancy per Gaussian (LERF): rel_i = min_j softmax(T·[s_q, s_neg_j])_0 with s = cosine
   similarity and temperature T = 10, i.e. min_j sigmoid(T (s_q − s_neg_j)). 0.5 = no
   preference; never-seen Gaussians (weight 0) get 0.
3. Select relative to the query's own peak: τ = threshold + rel_alpha · (peak − threshold), where
   peak = mean of the top `peak_k` relevancies (robust to a single odd Gaussian). Keep Gaussians with
   rel ≥ τ and opacity ≥ min_opacity; score = (rel − τ) · opacity. A fixed τ = 0.55 let every white
   flat surface into "whiteboard" (75 k Gaussians, room-long clusters; backroom, 5 Oct); relative to
   the peak, only what is close to the query's best match survives. rel_alpha = 0 restores the fixed
   threshold.
4. Cluster the selected centres by voxel connected components (26-neighbourhood, `voxel` m),
   so separate instances become separate candidates. Deterministic, scipy only.
5. Rank clusters by summed score; each gets a weighted centroid, a 5–95 % box, and the margin of
   the top score over the runner-up (small margin = ambiguous).
6. Approach point: `standoff` m from the centroid, horizontally, towards the training cameras that
   looked at it; clipped into the waypoint box (camera box inset by `margin`); its gap = distance
   to the k-th nearest sparse point minus the drone's body radius (course_tools' clearance).

Everything here is in the SPLAT frame (transforms.json, z up). Galley's course frame is
(x, −y, −z); `to_course` converts. Inputs are plain arrays so the CLI and the Phase 2 worker share it.
"""

import time
from dataclasses import dataclass, field

import numpy as np

NEGATIVES = ["object", "things", "stuff", "texture"]      # LERF's canonical phrases
TEMPERATURE = 10.0


def to_course(p):
    p = np.asarray(p, dtype=float)
    return p * np.array([1.0, -1.0, -1.0])


to_splat = to_course                                       # 180° about x, self-inverse


class TextEncoder:
    """CLIP text tower (CPU by default — a query must never compete with a GPU job)."""

    def __init__(self, device="cpu", loader=None):
        self.device = device
        self._loader = loader
        self._m = None

    def encode(self, texts):
        import torch
        if self._m is None:
            if self._loader is not None:
                self._m = self._loader()
            else:
                from .models import load_clip
                m, _, tok = load_clip(self.device)
                self._m = (m, tok)
        m, tok = self._m
        with torch.no_grad():
            e = m.encode_text(tok(list(texts)).to(self.device)).float()
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return e.cpu().numpy()


def relevancy(feats, weight, q, negs, temperature=TEMPERATURE, chunk=262_144):
    """feats [n, D] (unit rows, any float dtype, may be a memmap), q [D], negs [k, D] → rel [n]."""
    q = np.asarray(q, np.float32)
    negs = np.atleast_2d(np.asarray(negs, np.float32))
    P = np.concatenate([q[None], negs]).T                  # [D, 1 + k]
    n = len(feats)
    rel = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        s = np.asarray(feats[a:a + chunk], dtype=np.float32) @ P
        d = temperature * (s[:, :1] - s[:, 1:])            # [m, k]
        rel[a:a + chunk] = (1.0 / (1.0 + np.exp(-d))).min(axis=1)
    rel[np.asarray(weight) <= 0] = 0.0
    return rel


def voxel_labels(points, voxel=0.1, max_cells=20_000_000):
    """Connected-component label (1..k) per point on a voxel grid (26-connectivity)."""
    from scipy import ndimage
    if len(points) == 0:
        return np.zeros(0, int), 0, voxel
    lo = points.min(0)
    while True:
        idx = np.floor((points - lo) / voxel).astype(np.int64)
        dims = idx.max(0) + 1
        if int(np.prod(dims)) <= max_cells:
            break
        voxel *= 2.0
    occ = np.zeros(tuple(int(d) for d in dims), dtype=bool)
    occ[tuple(idx.T)] = True
    lab, k = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=int))
    return lab[tuple(idx.T)], int(k), voxel


@dataclass
class Settings:
    threshold: float = 0.55       # floor: below this a Gaussian never counts
    rel_alpha: float = 0.5        # τ = threshold + rel_alpha · (peak − threshold); 0 = fixed threshold
    peak_k: int = 100             # peak = mean of the top-k relevancies
    large_diag: float = 4.0       # m: candidates with a larger box diagonal are flagged "large" (walls, ceilings)
    min_opacity: float = 0.1
    voxel: float = 0.1
    min_gaussians: int = 15
    top: int = 5
    standoff: float = 1.0
    margin: float = 0.5           # waypoint box inset from the camera box (course editor default)
    body_radius: float = 0.19     # drone bounding sphere incl. prop guards
    min_gap: float = 0.15
    clearance_k: int = 5
    view_cos: float = 0.82        # camera counts as "looking at" the target within ~35°
    ambiguous_margin: float = 0.25
    negatives: list = field(default_factory=lambda: list(NEGATIVES))


@dataclass
class Scene:
    """What a query needs besides the table. All splat frame."""
    cam_c2w: np.ndarray           # [m, 3, 4] or [m, 4, 4] training camera-to-world (OpenGL)
    sparse_xyz: np.ndarray = None # [p, 3] SfM points for the gap check (None → no gap check)

    def camera_box(self, margin):
        """Waypoint box: the camera box inset by `margin`. On an axis the cameras spanned less than
        2·margin (typically height, for a capture filmed at one steady height) the inset box would be
        empty, so that axis falls back to the cameras' own range — the drone stays where the camera flew."""
        P = self.cam_c2w[:, :3, 3]
        lo, hi = P.min(0) + margin, P.max(0) - margin
        flat = lo > hi
        lo[flat], hi[flat] = P.min(0)[flat], P.max(0)[flat]
        return lo, hi

    def kdtree(self):
        if self.sparse_xyz is None or len(self.sparse_xyz) == 0:
            return None
        if not hasattr(self, "_tree"):
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.sparse_xyz)
        return self._tree


def cluster_candidates(points, scores, s):
    """Ranked clusters: list of dicts (splat frame), best first, at most s.top."""
    labels, k, vox = voxel_labels(points, s.voxel)
    out = []
    for lab in range(1, k + 1):
        m = labels == lab
        if m.sum() < s.min_gaussians:
            continue
        w = scores[m]
        c = (points[m] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
        lo, hi = np.percentile(points[m], [5, 95], axis=0)
        out.append({"score": float(w.sum()), "centroid": c, "box_lo": lo, "box_hi": hi, "n": int(m.sum())})
    out.sort(key=lambda d: -d["score"])
    return out[:s.top], vox


def approach_point(centroid, scene, s):
    """(point, n_cameras_used): standoff m from the centroid, horizontally towards the cameras
    that looked at it, clipped into the waypoint box. Splat frame (z up)."""
    C = scene.cam_c2w[:, :3, 3]
    fwd = -scene.cam_c2w[:, :3, 2]                        # OpenGL camera looks along −z
    to_t = centroid[None] - C
    dist = np.linalg.norm(to_t, axis=1)
    cos = (fwd * to_t).sum(1) / np.maximum(dist, 1e-9)
    sel = cos >= s.view_cos
    if not sel.any():
        sel = np.ones(len(C), bool)
    d = (C[sel] - centroid[None]) / np.maximum(dist[sel], 1e-9)[:, None] ** 2   # nearer cameras weigh more
    d = d.sum(0)
    d[2] = 0.0
    if np.linalg.norm(d) < 1e-9:
        d = C[sel][np.argmin(dist[sel])] - centroid
        d[2] = 0.0
    d = d / max(np.linalg.norm(d), 1e-9)
    p = centroid + s.standoff * d
    lo, hi = scene.camera_box(s.margin)
    return np.clip(p, lo, hi), int(sel.sum())


def gap(point, scene, s):
    t = scene.kdtree()
    if t is None:
        return None
    k = max(1, min(s.clearance_k, t.n))
    dk, _ = t.query(point, k=k)
    return float(np.atleast_1d(dk)[-1] - s.body_radius)


def run_query(table, scene, text, encoder, s=None):
    """Full query. Returns a JSON-ready dict; positions in the course frame unless noted."""
    s = s or Settings()
    t0 = time.time()
    E = encoder.encode([text] + list(s.negatives))
    t_enc = time.time() - t0
    geom = np.asarray(table.geom, dtype=np.float32)
    rel = relevancy(table.clip, table.weight, E[0], E[1:])
    t_rel = time.time() - t0 - t_enc
    opac = geom[:, 3]
    valid = (opac >= s.min_opacity) & (rel > 0)
    rv = rel[valid]
    k = min(s.peak_k, len(rv))
    peak = float(np.partition(rv, len(rv) - k)[len(rv) - k:].mean()) if k else 0.0
    tau = s.threshold + s.rel_alpha * max(0.0, peak - s.threshold)
    sel = valid & (rel >= tau)
    pts = geom[sel, :3].astype(np.float64)
    scores = ((rel[sel] - tau) * opac[sel]).astype(np.float64)
    cands, vox = cluster_candidates(pts, scores, s)
    res = []
    for r, c in enumerate(cands):
        a, ncam = approach_point(c["centroid"], scene, s)
        g = gap(a, scene, s)
        lo_c, hi_c = to_course(c["box_lo"]), to_course(c["box_hi"])
        diag = float(np.linalg.norm(c["box_hi"] - c["box_lo"]))
        res.append({"rank": r + 1, "score": round(c["score"], 4), "n": c["n"], "large": diag > s.large_diag,
                    "centroid": [round(float(v), 3) for v in to_course(c["centroid"])],
                    "box": {"lo": [round(float(v), 3) for v in np.minimum(lo_c, hi_c)],
                            "hi": [round(float(v), 3) for v in np.maximum(lo_c, hi_c)]},
                    "approach": [round(float(v), 3) for v in to_course(a)], "cameras": ncam,
                    "gap": None if g is None else round(g, 3), "gap_ok": None if g is None else g >= s.min_gap,
                    "centroid_splat": [round(float(v), 4) for v in c["centroid"]]})
    margin = None
    if len(res) >= 2:
        margin = round((res[0]["score"] - res[1]["score"]) / max(res[0]["score"], 1e-12), 3)
    elif len(res) == 1:
        margin = 1.0
    qs = np.percentile(rv, [50, 90, 99, 99.9]) if len(rv) else np.zeros(4)
    return {"text": text, "negatives": list(s.negatives), "threshold": s.threshold, "tau": round(tau, 4),
            "peak": round(peak, 4), "rel_alpha": s.rel_alpha,
            "rel_pct": {"p50": round(float(qs[0]), 4), "p90": round(float(qs[1]), 4), "p99": round(float(qs[2]), 4),
                        "p99.9": round(float(qs[3]), 4)},
            "n_selected": int(sel.sum()), "rel_max": round(float(rel.max()) if len(rel) else 0.0, 4),
            "rel_p99": round(float(np.percentile(rel[rel > 0], 99)) if (rel > 0).any() else 0.0, 4),
            "voxel": vox, "candidates": res, "margin": margin,
            "ambiguous": margin is not None and margin < s.ambiguous_margin,
            "frame": "course (x, -y, -z), z down",
            "ms": {"encode": round(t_enc * 1000), "relevancy": round(t_rel * 1000),
                   "total": round((time.time() - t0) * 1000)}}


def score_against(result, position_course, hit_radius=0.75, box_pad=0.3):
    """Gate metric for one annotated query: is the TOP candidate the annotated object?
    Hit = annotation inside the top box grown by box_pad, or within hit_radius of its centroid."""
    if not result["candidates"]:
        return {"hit": False, "error": None, "inside_box": False, "rank_of_hit": None}
    p = np.asarray(position_course, float)

    def check(c):
        e = float(np.linalg.norm(np.asarray(c["centroid"]) - p))
        lo, hi = np.asarray(c["box"]["lo"]) - box_pad, np.asarray(c["box"]["hi"]) + box_pad
        inside = bool(np.all(p >= lo) and np.all(p <= hi))
        return e, inside, inside or e <= hit_radius
    e, inside, hit = check(result["candidates"][0])
    rank = next((c["rank"] for c in result["candidates"] if check(c)[2]), None)
    return {"hit": hit, "error": round(e, 3), "inside_box": inside, "rank_of_hit": rank}
