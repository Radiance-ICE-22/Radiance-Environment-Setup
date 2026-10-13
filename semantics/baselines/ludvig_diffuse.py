"""LUDVIG's DINOv2 graph diffusion of CLIP relevancy, with the ORIGINAL code — run in the `ludvig` env.

    source semantics/baselines/env.sh ludvig
    python semantics/baselines/ludvig_diffuse.py --scene backroom          # after ludvig_full.py

Input: lift_ludvigfull_raw/{clip,dino}.npy (ludvig_full.py, checkpoint order). For every query text of the scene
(queries.json: Phase 5 set + development set, positives and absent objects — LUDVIG diffuses all prompts of a scene
together, as ludvig_clip.py does for the LERF prompts) it computes LUDVIG's per-Gaussian relevancy
(clip_utils.openclip_encoder.OpenCLIPNetwork.get_max_across: LERF's relevancy, canonical phrases object / things /
stuff / texture) and runs `diffusion.clip.GraphDiffusionCLIP` with configs/lerf_eval.yaml's diffusion_cfg
(200 neighbours, 200 iterations, feature bandwidth 0.5, logistic-regression unary term C = 0.001 with
reg_bandwidth 2). Output: lift_ludvigfull_raw/diffused.npz — texts, diffused relevancy [P, N] and the raw
relevancy [P, N], checkpoint order. Our import step (import_diffused.py, kitchen) reorders it into the table.

Two execution changes, no change to the result (their GPU had 48 GB; ours has 8):
  * precompute_similarities: ‖f_i − f_j‖ over the 201 neighbours is computed in chunks of nodes instead of one
    [N, 201, 40] tensor (17 GB on backroom);
  * get_stationary: the symmetrised sparse matrix (2 · N · 201 ≈ 214 M entries) and the 200 power iterations run on
    the CPU instead of CUDA. Same code otherwise (normalisation by the unary term, per-iteration feature norm).
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
DIFFUSION_CFG = dict(num_neighbors=200, num_iterations=200, feature_bandwidth=0.5, reg_bandwidth=2, logreg=0.001)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--suffix", default="_ludvigfull")
    ap.add_argument("--exports", default=str(THIRD_PARTY / "exports"))
    a = ap.parse_args()

    os.chdir(LUDVIG)
    sys.path.insert(0, str(LUDVIG))
    import torch
    from gaussiansplatting.scene import GaussianModel
    from gaussiansplatting.scene.cameras import Simple_Camera
    from clip_utils.openclip_encoder import OpenCLIPNetwork
    from diffusion.clip import GraphDiffusionCLIP
    from utils.graph import normalize_torch

    class Diffusion8GB(GraphDiffusionCLIP):
        def precompute_similarities(self, features, chunk=50_000):
            idx = torch.from_numpy(self.knn_neighbor_indices).to(features.device)
            out = torch.empty(idx.shape, dtype=features.dtype, device=features.device)
            for i in range(0, len(idx), chunk):
                out[i:i + chunk] = torch.norm(features[i:i + chunk, None] - features[idx[i:i + chunk]], dim=-1)
            self.similarities = out

        def get_stationary(self, neighbors, similarities, normalize=False, normalize_f=True, f=None, binarize=None,
                           unary_term=None, symmetrize=True):
            dev = "cpu"                                  # upstream: "cuda"
            n = self.knn_neighbor_indices.shape[0]
            if f is None:
                f = self.initial_features
            f = f.to(dev)
            similarities = similarities.to(dev)
            if binarize:
                similarities = (similarities > binarize).type(torch.float32)
            row_indices, col_indices = torch.tensor(neighbors[0]), torch.tensor(neighbors[1])
            if symmetrize:
                A = torch.sparse_coo_tensor(
                    torch.stack([torch.cat([row_indices, col_indices]), torch.cat([col_indices, row_indices])]),
                    torch.cat([similarities, similarities]), (n, n), device=dev, dtype=f.dtype).coalesce().to_sparse_csr()
            else:
                A = torch.sparse_csr_tensor(torch._convert_indices_from_coo_to_csr(row_indices, n), col_indices,
                                            similarities, (n, n), dtype=f.dtype, device=dev)
            if unary_term is not None:
                unary_term = unary_term.to(dev)
            if normalize:
                if unary_term is not None:
                    unary_term /= torch.sqrt(unary_term) * (A @ torch.sqrt(unary_term)) + 1e-8
                else:
                    A = normalize_torch(A, n)
            for i in range(self.num_iterations):
                if normalize_f:
                    f /= self.eps + f.norm(dim=0, keepdim=True)
                if unary_term is not None:
                    f = torch.sqrt(unary_term) * (A @ (torch.sqrt(unary_term) * f))
                else:
                    f = A @ f
            return f.cuda()

    ex = Path(a.exports) / a.scene
    man = json.loads((ex / "manifest.json").read_text())
    raw = Path(man["semantics_dir"]) / man["run"] / f"lift{a.suffix}_raw"
    stats = json.loads((raw / "lift.json").read_text())
    texts = sorted({q["text"] for q in json.loads((Path(man["semantics_dir"]) / "queries.json").read_text())["queries"]})

    g = GaussianModel(sh_degree=0)
    g.load_ply(str(ex / "point_cloud.ply"))
    W = int(stats["render_width"])
    cams = []
    for c in json.loads((ex / "cameras.json").read_text()):
        H = int(round(c["height"] * W / c["width"]))
        cams.append(Simple_Camera(colmap_id=0, R=np.asarray(c["R"]), T=np.asarray(c["T"]), FoVx=c["fovx"],
                                  FoVy=c["fovy"], h=H, w=W, image_name=c["stem"], uid=0))

    t0 = time.time()
    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(texts)
    sem = torch.from_numpy(np.load(raw / "clip.npy").astype(np.float32)).cuda()
    sem /= torch.norm(sem, dim=-1, keepdim=True) + 1e-6              # LUDVIGCLIP.compute_relevancies
    relev = clip_model.get_max_across(sem).float()                   # [P, N]
    del sem
    print(f"{a.scene}: {len(texts)} prompts × {relev.shape[1]:,} Gaussians, relevancy in {time.time() - t0:.0f} s", flush=True)

    def render_shape(f, cam):                                        # compute_Z only needs the map's shape
        return torch.zeros((f.shape[1], cam.image_height, cam.image_width), device="cuda")

    gd = Diffusion8GB(gaussian=g, render_fn=render_shape, cameras=cams, load_dino=str(raw / "dino.npy"),
                      logdir=None, eps=1e-6, relev=relev, **DIFFUSION_CFG)
    diffused = gd().T                                                # LUDVIGCLIP.run_diffusion: [P, N]
    secs = round(time.time() - t0, 1)
    np.savez_compressed(raw / "diffused.npz", texts=np.array(texts), diffused=diffused.cpu().numpy().astype(np.float32),
                        relevancy=relev.cpu().numpy().astype(np.float16),
                        initial_positive=(gd.initial_features > 0).sum(0).cpu().numpy())
    stats.update(diffusion_seconds=secs, diffusion_prompts=len(texts), diffusion_cfg=DIFFUSION_CFG,
                 diffusion_peak_vram_mib=int(torch.cuda.max_memory_allocated() / 2**20))
    (raw / "lift.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"diffused {len(texts)} prompts in {secs:.0f} s → {raw / 'diffused.npz'}")


if __name__ == "__main__":
    main()
