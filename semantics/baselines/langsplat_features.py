"""LangSplat's 2D language features for Occam's LGS, with the ORIGINAL LangSplat preprocessing — `occamlgs` env.

    source semantics/baselines/env.sh occamlgs
    python semantics/baselines/langsplat_features.py --scene backroom

Occam's LGS lifts the feature maps LangSplat's `preprocess.py` produces (paper §4.2: SAM ViT-H masks at three
levels — subpart / part / whole — and one OpenCLIP ViT-B/16 embedding per mask). This runs that script's own
functions (minghanqin/LangSplat, cloned in ~/Radiance/third_party/LangSplat, with its SAM fork
segment-anything-langsplat whose mask generator returns default / s / m / l masks) on our training images:
same SamAutomaticMaskGenerator settings, same mask NMS (iou 0.8, score 0.7, inner 0.5), same crops (mask on black,
padded square, 224 px), same image rule (height > 1080 → 1080). Output per image, in LangSplat's format:
<stem>_s.npy (seg maps [4, H, W], -1 = no mask; levels default, s, m, l, ids offset into the feature list) and
<stem>_f.npy (features [n_masks, 512], fp16, unit norm).

Three execution changes, no change to the result:
  * `create()` is called one image at a time (it keeps every seg map in memory otherwise), and finished images are
    skipped on a re-run;
  * SAM decodes 16 prompt points per batch instead of 64 (each point is decoded independently; the batches are
    merged before any filtering across points), which keeps 1080p images within 8 GB;
  * `mask_nms` is computed from a pairwise-intersection matrix (mask_nms_matrix below) instead of a Python loop over
    all mask pairs, which is infeasible here (one image has ~2,500 masks; a GPU copy of them is 5 GB). Same
    arithmetic and rules; verified to select the same masks as upstream's function on real images.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path.home() / "Radiance/third_party"))


def mask_nms_matrix(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, chunk=1 << 17, **kwargs):
    """LangSplat's mask_nms with the same arithmetic, vectorised. Upstream compares every pair of full-resolution
    masks in a Python double loop (one image here has ~2,500 masks: ~3 M pairs); this computes all pairwise
    intersections as integer counts, summed over pixel chunks on the GPU, then applies upstream's rules
    elementwise in the same order: iou = I / U; for j ≥ i, inner[i, j] = 1 − (I/A_j)(I/A_i) when I/A_i < 0.5 and
    I/A_j ≥ 0.85, and inner[j, i] = the same value when I/A_i ≥ 0.85 and I/A_j < 0.5; then upstream's thresholds
    and top-3 fall-backs. Checked against upstream on real images (test_langsplat_nms in the run log)."""
    import torch
    scores, idx = scores.sort(0, descending=True)
    M = idx.shape[0]
    flat = masks.reshape(masks.shape[0], -1)[idx.view(-1)]                    # masks_ord, [M, P] bool (CPU)
    inter = torch.zeros((M, M), dtype=torch.float64, device="cuda")
    for c0 in range(0, flat.shape[1], chunk):
        x = flat[:, c0:c0 + chunk].to("cuda", non_blocking=True).float()
        inter += (x @ x.T).double()                                           # exact: counts ≤ chunk < 2^24
    area = torch.diagonal(inter).clone()
    inter, area = inter.float(), area.float()                                 # upstream sums are float32
    union = area[:, None] + area[None, :] - inter
    iou = inter / union
    ri = inter / area[:, None]                                                # I / A_i  (row i)
    rj = inter / area[None, :]                                                # I / A_j  (column j)
    val = 1 - rj * ri
    upper = torch.triu(torch.ones((M, M), dtype=torch.bool, device="cuda"))  # pairs j ≥ i, as upstream's loop
    inner = torch.zeros((M, M), dtype=torch.float, device="cuda")
    c1 = upper & (ri < 0.5) & (rj >= 0.85)
    inner[c1] = val[c1]
    c2 = upper & (ri >= 0.85) & (rj < 0.5)
    inner.T[c2] = val[c2]                                                     # inner[j, i]
    iou_matrix = iou * upper
    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_max_u, _ = torch.triu(inner, diagonal=1).max(dim=0)
    inner_iou_max_l, _ = torch.tril(inner, diagonal=1).max(dim=0)
    scores_c = scores.to("cuda")
    keep = iou_max <= iou_thr
    keep_conf = scores_c > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    if keep_conf.sum() == 0:
        keep_conf[scores_c.topk(3).indices] = True
    if keep_inner_u.sum() == 0:
        keep_inner_u[scores_c.topk(3).indices] = True
    if keep_inner_l.sum() == 0:
        keep_inner_l[scores_c.topk(3).indices] = True
    keep = keep & keep_conf & keep_inner_u & keep_inner_l
    return idx[keep.cpu()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="read the frames from exports/<scene>/cameras.json (baseline runs)")
    ap.add_argument("--frames", help="JSON list of {stem, image} instead of --scene (the semantic pipeline's teacher step)")
    ap.add_argument("--out-dir", help="write <stem>_s.npy / _f.npy here instead of <out>/<scene>")
    ap.add_argument("--exports", default=str(THIRD_PARTY / "exports"))
    ap.add_argument("--out", default=str(THIRD_PARTY / "langsplat_features"))
    ap.add_argument("--sam-ckpt", default=str(THIRD_PARTY / "checkpoints/sam_vit_h_4b8939.pth"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--points-per-batch", type=int, default=16,
                    help="SAM prompts decoded per batch (upstream default 64 runs out of memory on 8 GB at 1080p; "
                         "points are decoded independently, so the masks do not depend on it)")
    a = ap.parse_args()

    sys.path.insert(0, str(THIRD_PARTY / "LangSplat"))
    import cv2
    import torch
    import preprocess as P                      # LangSplat's preprocess.py (its __main__ block does not run)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    P.seed_everything(42)
    torch.set_default_dtype(torch.float32)
    if bool(a.scene) == bool(a.frames):
        ap.error("give exactly one of --scene and --frames")
    cams = json.loads(Path(a.frames).read_text()) if a.frames else \
        json.loads((Path(a.exports) / a.scene / "cameras.json").read_text())
    if a.limit:
        cams = cams[:a.limit]
    out = Path(a.out_dir) if a.out_dir else Path(a.out) / a.scene
    out.mkdir(parents=True, exist_ok=True)

    # LangSplat's __main__ set-up, verbatim settings
    P.model = P.OpenCLIPNetwork(P.OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=a.sam_ckpt).to("cuda")
    P.mask_generator = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.7, box_nms_thresh=0.7, stability_score_thresh=0.85,
        crop_n_layers=1, crop_n_points_downscale_factor=1, min_mask_region_area=100,
        points_per_batch=a.points_per_batch)

    P.mask_nms = mask_nms_matrix

    t_all = time.time()
    done = 0
    for k, c in enumerate(cams):
        stem = c["stem"]
        if (out / f"{stem}_f.npy").exists() and (out / f"{stem}_s.npy").exists():
            continue
        t0 = time.time()
        image = cv2.imread(c["image"])
        h, w = image.shape[:2]
        scale = h / 1080 if h > 1080 else 1.0            # LangSplat: --resolution -1 rule
        image = cv2.resize(image, (int(w / scale), int(h / scale)))
        img = torch.from_numpy(image).permute(2, 0, 1)[None]     # [1, 3, H, W], BGR uint8 as upstream
        P.create(img, [f"{stem}.png"], str(out))
        done += 1
        print(f"  {k + 1}/{len(cams)} {stem}: {time.time() - t0:.1f} s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    (out / "meta.json").write_text(json.dumps({
        "scene": a.scene or a.frames, "images": len(cams), "computed_now": done, "seconds": round(time.time() - t_all, 1),
        "sam": "vit_h (sam_vit_h_4b8939)", "clip": "ViT-B-16 laion2b_s34b_b88k",
        "source": "LangSplat preprocess.py (create, sam_encoder, masks_update; mask_nms vectorised, verified identical)"}, indent=1) + "\n")
    print(f"done: {done} images in {time.time() - t_all:.0f} s → {out}")


if __name__ == "__main__":
    main()
