"""LUDVIG's full feature pipeline with the ORIGINAL code (naver/ludvig) — run in the `ludvig` env.

    source semantics/baselines/env.sh ludvig
    python semantics/baselines/ludvig_full.py --scene backroom            # → lift_ludvigfull_raw/

What LUDVIG does for open-vocabulary localisation on LERF (script/lerf_uplift.sh, configs/lerf_{dinov2,clip}.yaml),
on our Gaussians and refined cameras (exports/<scene>, from export_scene.py):

  DINOv2  their `dinov2.model.DINOv2` (ViT-g/14, checkpoint dinov2_vitg14_reg4_pretrain.pth, their config
          dinov2/configs/vitg14_pretrain.yaml), last-layer patch tokens of the whole image (no sliding window in
          lerf_dinov2.yaml), standardised per channel, PCA to 40 components fitted on a 500k-token subsample,
          bilinearly upsampled to the image (predictors.dino.DINOv2Dataset); uplifted (utils.solver.uplifting;
          its prune_gaussians 600000 keeps every one of our ≤ 532k Gaussians, so it is not applied — it would
          only re-sort them).
  CLIP    their `predictors.clip.CLIPDataset`: OpenCLIP ViT-B/16 (laion2b_s34b_b88k) on a sliding window at 7 tile
          ratios 0.05–0.5, per-scale maps upsampled to the image and averaged, unit norm; uplifted and l2-normalised
          (normalize: True).

Images are read by their `utils.image.image_from_path` (longest side ≤ 1600 px); cameras render at that size, so
feature maps and renders match as in their pipeline.

Execution changes, no change to the result: (1) DINOv2Dataset keeps every token of every image on the GPU for the
PCA (270 × 7,296 × 1,536 floats ≈ 12 GB on backroom; their GPU had 48 GB). Here the tokens are kept in host RAM and
the same steps run there — per-channel mean/std over all tokens, PCA fit on 500k random tokens, projection of all.
(2) The CLIP crop pyramid adds each scale's upsampled map in 64-channel chunks instead of one [512, H, W] temporary.
(3) Crops are embedded in batches of 128 instead of all crops of a scale at once.
(4) Each CLIP map's storage is released before the next is built (uplifting's loop variable kept two alive).
(5) Run DINOv2 and CLIP as separate processes (--only dino, --only clip) on an 8 GB card.

Note (a property of the released code, kept): their ViT has no register tokens, and the checkpoint is loaded with
strict=False, so the reg4 checkpoint's `register_tokens` are dropped. The run log lists the keys the load skips.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path.home() / "Radiance/third_party"))
LUDVIG = THIRD_PARTY / "ludvig"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--suffix", default="_ludvigfull")
    ap.add_argument("--exports", default=str(THIRD_PARTY / "exports"))
    ap.add_argument("--only", choices=["dino", "clip"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.chdir(LUDVIG)                                   # their configs use paths relative to the repo
    sys.path.insert(0, str(LUDVIG))
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from gaussiansplatting.scene import GaussianModel
    from gaussiansplatting.scene.cameras import Simple_Camera
    from utils.image import image_from_path
    from utils.solver import uplifting
    from predictors.clip import CLIPDataset
    from dinov2.model import DINOv2

    ex = Path(a.exports) / a.scene
    man = json.loads((ex / "manifest.json").read_text())
    cams_j = json.loads((ex / "cameras.json").read_text())
    if a.limit:
        cams_j = cams_j[:a.limit]
    raw = Path(man["semantics_dir"]) / man["run"] / f"lift{a.suffix}_raw"
    raw.mkdir(parents=True, exist_ok=True)
    img_dir = str(ex / "images")

    g = GaussianModel(sh_degree=0)
    g.load_ply(str(ex / "point_cloud.ply"))
    N = g.get_xyz.shape[0]
    probe = image_from_path(img_dir, cams_j[0]["stem"])
    H, W = probe.shape[1:]                             # their image size (longest side ≤ 1600)
    cams = [Simple_Camera(colmap_id=0, R=np.asarray(c["R"]), T=np.asarray(c["T"]), FoVx=c["fovx"], FoVy=c["fovy"],
                          h=H, w=W, image_name=c["stem"], uid=0) for c in cams_j]
    print(f"{a.scene}: {N:,} Gaussians, {len(cams)} views, images/renders {W}x{H} → {raw}", flush=True)
    stats = json.loads((raw / "lift.json").read_text()) if (raw / "lift.json").exists() else {}

    # total blend weight per Gaussian (1-channel pass; LUDVIG's count, divided out of its channel factor)
    if not (raw / "weight.npy").exists():
        ones3, w = torch.ones((N, 1), device="cuda"), torch.zeros((N, 1), device="cuda")
        img1 = torch.ones((1, H, W), device="cuda")
        acc = torch.zeros((N, 1), device="cuda")
        for cam in cams:
            g.apply_weights(cam, acc, w, img1)
        w = w[:, 0]
        w[w <= 1e-8] = 0
        np.save(raw / "weight.npy", w.cpu().numpy().astype(np.float32))
        del acc, ones3, img1

    if a.only in (None, "dino"):
        t0 = time.time()
        model = DINOv2("dinov2/configs/vitg14_pretrain.yaml", "checkpoints/dinov2_vitg14_reg4_pretrain.pth").cuda()
        sd = torch.load("checkpoints/dinov2_vitg14_reg4_pretrain.pth", map_location="cpu")
        mk = set(model.feature_model.state_dict())
        print("  DINOv2 load: checkpoint keys not in the model:", sorted(set(sd) - mk),
              "| model keys not in the checkpoint:", sorted(mk - set(sd)), flush=True)
        del sd
        feats, shape = [], None
        for k, cam in enumerate(cams):                 # DINOv2Dataset.extract, no sliding window
            img = image_from_path(img_dir, cam.image_name, normalize=True)
            Hr, Wr = (H // 14) * 14, (W // 14) * 14
            img = nn.functional.interpolate(img[None], size=(Hr, Wr), mode="bilinear").squeeze()
            with torch.no_grad():
                f = model.predict(img)                 # [1536, h, w]
            D, h, w_ = f.shape
            shape = (D, h, w_)
            feats.append(f.view(D, -1).T.float().cpu().numpy())
            if k == 0 or (k + 1) % 50 == 0 or k == len(cams) - 1:
                print(f"  dinov2: {k + 1}/{len(cams)} images, {time.time() - t0:.0f} s, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        del model
        torch.cuda.empty_cache()
        F_all = np.concatenate(feats)                  # DINOv2Dataset.apply_pca (their GPU branch), in host RAM
        del feats
        mean, std = F_all.mean(0), F_all.std(0)
        F_all -= mean
        F_all /= std
        rng = np.random.default_rng(0)
        sub = F_all[rng.choice(len(F_all), min(500_000, len(F_all)), replace=False)]
        pca = PCA(n_components=40).fit(sub)
        del sub
        npix = shape[1] * shape[2]
        proj = [((F_all[i * npix:(i + 1) * npix] - pca.mean_) @ pca.components_.T).astype(np.float32)
                for i in range(len(cams))]
        del F_all
        print(f"  dinov2: PCA 40 of {shape[0]} (explained {pca.explained_variance_ratio_.sum():.3f}), "
              f"{time.time() - t0:.0f} s", flush=True)

        def dino_loader():                             # DINOv2Dataset.__getitem__
            for i, cam in enumerate(cams):
                f = torch.from_numpy(proj[i]).cuda().T.view(40, shape[1], shape[2])
                yield nn.functional.interpolate(f[None], size=(H, W), mode="bilinear").squeeze(0), cam
        # lerf_dinov2.yaml's prune_gaussians: 600000 keeps all of our ≤ 532k Gaussians (it would only re-sort them by
        # weight, and flightroom is below its 400k minimum anyway), so it is left out to keep checkpoint order
        f3, _ = uplifting(dino_loader(), g)
        np.save(raw / "dino.npy", f3.cpu().numpy().astype(np.float32))
        stats["dino_seconds"] = round(time.time() - t0, 1)
        stats["dino_pca_explained"] = round(float(pca.explained_variance_ratio_.sum()), 4)
        stats["dino_tokens"] = [int(shape[1]), int(shape[2])]
        del proj, f3

    if a.only in (None, "clip"):
        t0 = time.time()
        import torch.nn.functional as Fn
        import predictors.clip as PC

        class Batched:                                 # tile_and_embed encodes all tiles of a scale in one batch
            def __init__(self, m, n=128):
                self.m, self.n = m, n

            def encode_image(self, x):
                return torch.cat([self.m.encode_image(x[i:i + self.n]) for i in range(0, len(x), self.n)])

        def sw_call(self, image, model, chunk=64):
            """predictors.clip.SlidingWindow.__call__ with each scale's upsampling added in channel chunks (the
            upstream line adds one full [512, H, W] temporary per scale: +2.75 GB at 1600 × 900)."""
            h_img, w_img = image.shape[1:]
            out = torch.zeros((self.clip_dim, h_img, w_img), device="cuda", dtype=torch.float32)
            for tile_ratio, stride_ratio in zip(self.tile_ratios, self.stride_ratios):
                kernel_size = int(tile_ratio * h_img)
                stride_h = stride_w = int(kernel_size * stride_ratio)
                e = self.tile_and_embed(image, Batched(model), kernel_size, stride_h, stride_w).permute(2, 0, 1)[None]
                for c0 in range(0, self.clip_dim, chunk):
                    out[c0:c0 + chunk] += Fn.interpolate(e[:, c0:c0 + chunk], (h_img, w_img), mode="bilinear").squeeze(0)
            out /= out.norm(dim=0, keepdim=True)
            return out
        PC.SlidingWindow.__call__ = sw_call
        ds = CLIPDataset(directory=img_dir, scene=a.scene, gaussian=g, cameras=cams, render_fn=None,
                         height=H, width=W)
        def clip_loader():                             # uplifting's loop variable keeps the previous map alive while
            prev = None                                # the next one is built (2 × 2.75 GB); release it first
            for i in range(len(ds)):
                if prev is not None:
                    prev.data = torch.empty(0, device="cuda")
                feat, cam = ds[i]
                prev = feat
                yield feat, cam
        f3, _ = uplifting(clip_loader(), g)
        f3 /= f3.norm(dim=1, keepdim=True) + 1e-6       # normalize: True (lerf_clip.yaml)
        np.save(raw / "clip.npy", f3.cpu().numpy().astype(np.float16))
        stats["clip_seconds"] = round(time.time() - t0, 1)

    w = np.load(raw / "weight.npy")
    stats.update({"views": len(cams), "seen": int((w > 0).sum()), "n": int(N), "seen_frac": round(float((w > 0).mean()), 4),
                  "render_width": int(W), "teacher_tag": "ludvig-full", "key": man["key"],
                  "impl": "LUDVIG original: DINOv2 ViT-g (released loader) PCA-40 + CLIPDataset 7-scale, uplifting()",
                  "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2**20)})
    stats["seconds"] = round(stats.get("dino_seconds", 0) + stats.get("clip_seconds", 0), 1)
    (raw / "lift.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"done → {raw}: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
