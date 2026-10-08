"""OpenCLIP and DINOv2: one place that loads them, fetches their weights, and smoke-tests them.

    python -m radiance_semantics.models --fetch            # download into $HF_HOME / $TORCH_HOME
    python -m radiance_semantics.models --smoke [--device cuda]

Weights land in the caches figs_env.sh exports (HF_HOME, TORCH_HOME under the prefix), so a
host without internet works once they are fetched. DINOv2 is loaded through torch.hub; after
the first fetch it is loaded from the cached hub checkout with source="local", so later runs
never re-pull the repository's main branch (its code is frozen at fetch time).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from . import CLIP_DIM, CLIP_MODEL, CLIP_PRETRAINED, DINO_DIM, DINO_MODEL, DINO_REPO
from .log import emit, fail, info, ok, section


def dino_local_dir():
    import torch
    return Path(torch.hub.get_dir()) / (DINO_REPO.replace("/", "_") + "_main")


def provenance_file():
    import torch
    return Path(torch.hub.get_dir()).parent / "radiance_semantics_models.json"


def load_clip(device="cpu", model_name=None, pretrained=None):
    """(model, preprocess, tokenizer) for an OpenCLIP model (default: the pinned ViT-B/16), in eval mode."""
    import open_clip
    name, pre = model_name or CLIP_MODEL, pretrained or CLIP_PRETRAINED
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device=device)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(name)


def clip_dim_of(model):
    """Embedding width of a loaded OpenCLIP model (512 for ViT-B/16, 768 for ViT-L/14)."""
    return int(getattr(model.visual, "output_dim", 512))


def load_dino(device="cpu"):
    """The pinned DINOv2 backbone, from the frozen hub checkout when it exists."""
    import torch
    d = dino_local_dir()
    if (d / "hubconf.py").exists():
        model = torch.hub.load(str(d), DINO_MODEL, source="local")
    else:
        # skip_validation: no GitHub API call (rate limits on shared campus IPs)
        model = torch.hub.load(DINO_REPO, DINO_MODEL, trust_repo=True, skip_validation=True)
    return model.to(device).eval()


def fetch():
    section("Fetching OpenCLIP and DINOv2 weights")
    from importlib.metadata import version
    import torch
    t0 = time.time()
    load_clip("cpu")
    oc = version("open_clip_torch")              # open_clip 2.24 has no __version__ attribute
    ok(f"OpenCLIP {CLIP_MODEL} / {CLIP_PRETRAINED} (open_clip {oc})")
    load_dino("cpu")
    d = dino_local_dir()
    ok(f"DINOv2 {DINO_MODEL} (hub checkout {d})")
    prov = {"clip_model": CLIP_MODEL, "clip_pretrained": CLIP_PRETRAINED, "open_clip": oc,
            "dino_repo": DINO_REPO, "dino_model": DINO_MODEL, "dino_hub_dir": str(d),
            "torch": torch.__version__, "fetched": datetime.now().isoformat(timespec="seconds")}
    if not provenance_file().exists():            # first fetch only: records when the hub code was frozen
        provenance_file().write_text(json.dumps(prov, indent=2) + "\n")
    info(f"provenance: {provenance_file()}")
    info(f"{time.time() - t0:.0f} s")
    return prov


def smoke(device):
    """A functional check, not just an import: CLIP must match a red image to "red" and a blue
    one to "blue", and DINOv2 must return 384-d patch tokens on a 16x16 grid for 224 px."""
    import torch
    section(f"OpenCLIP + DINOv2 smoke test on {device}")
    res = {"device": device}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model, _, tok = load_clip(device)
    size = model.visual.image_size
    size = size[0] if isinstance(size, (tuple, list)) else size
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    imgs = torch.zeros(2, 3, size, size, device=device)
    imgs[0, 0] = 1.0                                   # pure red
    imgs[1, 2] = 1.0                                   # pure blue
    texts = tok(["a photo of the color red", "a photo of the color blue"]).to(device)
    with torch.no_grad():
        fi = model.encode_image((imgs - mean) / std)
        ft = model.encode_text(texts)
    fi = fi / fi.norm(dim=-1, keepdim=True)
    ft = ft / ft.norm(dim=-1, keepdim=True)
    sim = (fi @ ft.T).float().cpu()
    res["clip_dim"] = int(fi.shape[-1])
    res["clip_sim"] = [[round(float(x), 4) for x in row] for row in sim]
    res["clip_seconds"] = round(time.time() - t0, 1)
    clip_ok = res["clip_dim"] == CLIP_DIM and int(sim[0].argmax()) == 0 and int(sim[1].argmax()) == 1
    (ok if clip_ok else fail)(f"CLIP {CLIP_MODEL}: dim {res['clip_dim']}, red→{sim[0].tolist()}, blue→{sim[1].tolist()}")
    del model

    t0 = time.time()
    dino = load_dino(device)
    with torch.no_grad():
        out = dino.forward_features(torch.randn(1, 3, 224, 224, device=device))
    pt = out["x_norm_patchtokens"]
    res["dino_tokens"] = list(pt.shape)
    res["dino_seconds"] = round(time.time() - t0, 1)
    dino_ok = list(pt.shape) == [1, 256, DINO_DIM] and bool(torch.isfinite(pt).all())
    (ok if dino_ok else fail)(f"DINOv2 {DINO_MODEL}: patch tokens {list(pt.shape)} (expected [1, 256, {DINO_DIM}])")

    if device == "cuda":
        res["peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        info(f"peak allocated {res['peak_mib']} MiB")
    res["ok"] = clip_ok and dino_ok
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download weights into the caches")
    ap.add_argument("--smoke", action="store_true", help="functional check of both models")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    a = ap.parse_args(argv)
    if not (a.fetch or a.smoke):
        ap.error("nothing to do: pass --fetch and/or --smoke")
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    try:
        if a.fetch:
            out["fetch"] = fetch()
        if a.smoke:
            out["smoke"] = smoke(device)
    except Exception as e:                       # network, cache or import problems: say which
        fail(f"{type(e).__name__}: {e}")
        emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    good = out.get("smoke", {}).get("ok", True)
    emit({"ok": good, **out})
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
