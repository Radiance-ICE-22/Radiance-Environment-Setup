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
  * `mask_nms` runs on the GPU: upstream builds full-resolution boolean masks on the CPU and compares them pairwise
    in Python loops (minutes per image at 1080p); the same tensors on CUDA give the same selection.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path.home() / "Radiance/third_party"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
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
    cams = json.loads((Path(a.exports) / a.scene / "cameras.json").read_text())
    if a.limit:
        cams = cams[:a.limit]
    out = Path(a.out) / a.scene
    out.mkdir(parents=True, exist_ok=True)

    # LangSplat's __main__ set-up, verbatim settings
    P.model = P.OpenCLIPNetwork(P.OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=a.sam_ckpt).to("cuda")
    P.mask_generator = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.7, box_nms_thresh=0.7, stability_score_thresh=0.85,
        crop_n_layers=1, crop_n_points_downscale_factor=1, min_mask_region_area=100,
        points_per_batch=a.points_per_batch)

    _nms = P.mask_nms                           # same function, tensors on the GPU

    def mask_nms_cuda(masks, scores, **kw):
        return _nms(masks.cuda(), scores.cuda(), **kw).cpu()
    P.mask_nms = mask_nms_cuda

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
        "scene": a.scene, "images": len(cams), "computed_now": done, "seconds": round(time.time() - t_all, 1),
        "sam": "vit_h (sam_vit_h_4b8939)", "clip": "ViT-B-16 laion2b_s34b_b88k",
        "source": "LangSplat preprocess.py (create, sam_encoder, masks_update; mask_nms on CUDA)"}, indent=1) + "\n")
    print(f"done: {done} images in {time.time() - t_all:.0f} s → {out}")


if __name__ == "__main__":
    main()
