"""tiny-cuda-nn probes for the FMGS field: which parts run on this GPU?

    python -m radiance_semantics.fmgs.diag                  # the matrix below, each case in its own process
    python -m radiance_semantics.fmgs.diag --probe encoding --cfg '{"levels": 24, ...}' --n 200000

Each probe builds the part with tiny-cuda-nn, runs forward and backward on random inputs and
synchronises, in a CHILD process with CUDA_LAUNCH_BLOCKING=1 (a failed kernel launch is then
reported where it happens, and cannot disturb the caller's CUDA context). The trainer's
`impl="auto"` uses `resolve()` to pick tiny-cuda-nn or PyTorch for the encoding and the heads.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

PROBE_N = 262_144     # ≥ the trainer's batch: the visible trainable Gaussians (faithful) or the 480×270 pixels (blite)
FMGS = {"levels": 24, "features": 8, "log2_table": 20, "base": 16, "finest": 512, "hidden": 256, "layers": 2}
MATRIX = [
    ("encoding", "nerfacto-like 16 lvl × 2 feat, 2^19", {"levels": 16, "features": 2, "log2_table": 19, "finest": 2048}, 65536),
    ("encoding", "LERF-like 12 lvl × 8 feat, 2^19", {"levels": 12, "features": 8, "log2_table": 19}, 65536),
    ("encoding", "24 lvl × 8 feat, 2^19", {"levels": 24, "features": 8, "log2_table": 19}, 65536),
    ("encoding", "FMGS 24 lvl × 8 feat, 2^20", FMGS, 4096),
    ("encoding", "FMGS 24 lvl × 8 feat, 2^20, 200k points", FMGS, 200_000),
    ("encoding", "FMGS as 2 grids × 12 lvl (split 2), 200k points", {**FMGS, "split": 2}, 200_000),
    ("encoding", "24 lvl × 4 feat, 2^20", {"levels": 24, "features": 4, "log2_table": 20}, 65536),
    ("encoding", "24 lvl × 2 feat, 2^20", {"levels": 24, "features": 2, "log2_table": 20}, 65536),
    ("heads", "FMGS heads: CutlassMLP 2 × 256 → 512 / 384", FMGS, 65536),
    ("heads-ff", "FullyFusedMLP 2 × 128 → 512 / 384", {**FMGS, "hidden": 128}, 65536),
]


def probe(part, cfg, n, device="cuda"):
    """Build and run one part in THIS process. Returns {"ok", "msg", "ms", "peak_mib"}."""
    import torch
    from .field import FieldConfig, tcnn_encoding, tcnn_head
    cfg = dict(cfg)
    impl = cfg.pop("impl", "tcnn")
    c = FieldConfig(**{**asdict(FieldConfig()), **cfg})
    t0 = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        if part == "field":                                         # the whole field, as the trainer builds it
            from .field import FeatureField
            m = FeatureField(torch.zeros(3), torch.ones(3), c, impl).to(device)
            out = m(torch.rand(n, 3, device=device))
            ys = [out["clip"], out["dino"]]
            loss = sum(y.float().square().mean() for y in ys)
            loss.backward()
            torch.cuda.synchronize()
            bad = [p for p in m.parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            ok = bool(torch.isfinite(loss).item()) and not bad and list(ys[0].shape) == [n, c.clip_dim]
            return {"ok": ok, "msg": "ok" if ok else "non-finite output or gradient", "ms": round((time.time() - t0) * 1000),
                    "peak_mib": int(torch.cuda.max_memory_allocated() / 2 ** 20), "out": [list(y.shape) for y in ys],
                    "dtype": str(ys[0].dtype), "impl": m.impl}
        if part == "encoding":
            m = tcnn_encoding(c).to(device)
            x = torch.rand(n, 3, device=device)
        else:
            otype = "FullyFusedMLP" if part == "heads-ff" else "CutlassMLP"
            m = torch.nn.ModuleList([tcnn_head(c, c.clip_dim, otype), tcnn_head(c, c.dino_dim, otype)]).to(device)
            x = torch.rand(n, c.enc_dim, device=device)
        ys = [m(x)] if part == "encoding" else [h(x) for h in m]
        loss = sum(y.float().square().mean() for y in ys)
        loss.backward()
        torch.cuda.synchronize()
        bad = [p for p in m.parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        ok = bool(torch.isfinite(loss).item()) and not bad
        return {"ok": ok, "msg": "ok" if ok else "non-finite output or gradient", "ms": round((time.time() - t0) * 1000),
                "peak_mib": int(torch.cuda.max_memory_allocated() / 2 ** 20), "out": [list(y.shape) for y in ys],
                "dtype": str(ys[0].dtype)}
    except Exception as e:                                          # noqa: BLE001 — the message is the result
        return {"ok": False, "msg": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", "ms": round((time.time() - t0) * 1000)}


def probe_child(part, cfg, n, timeout=300):
    """probe() in a child process with CUDA_LAUNCH_BLOCKING=1."""
    env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
    try:
        p = subprocess.run([sys.executable, "-m", "radiance_semantics.fmgs.diag", "--probe", part, "--cfg", json.dumps(cfg),
                            "--n", str(n)], env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "msg": f"timed out after {timeout} s"}
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        tail = " | ".join((p.stderr or p.stdout).strip().splitlines()[-3:])
        return {"ok": False, "msg": f"child exited {p.returncode}: {tail[:300]}"}
    return json.loads(line)


def resolve(cfg, device, log=print):
    """impl 'auto' → ("<enc>/<heads>", probe notes, FieldConfig overrides).

    Encoding: tiny-cuda-nn's grid as configured; else the same levels as consecutive tiny-cuda-nn
    grids (split 2, 3, …: on intellisense08 one 24 × 8 grid fails to launch, 12 × 8 runs); else
    PyTorch with the grid unchanged. Heads: tiny-cuda-nn, else PyTorch."""
    import torch
    if str(device).startswith("cpu") or not torch.cuda.is_available():
        return "torch/torch", {}, {}
    try:
        import tinycudann  # noqa: F401
    except Exception as e:                                          # noqa: BLE001
        log(f"  tiny-cuda-nn not importable ({e}): PyTorch field")
        return "torch/torch", {"import": str(e)}, {}
    cfg = dict(cfg)
    levels, notes, over = cfg.get("levels", 24), {}, {}
    enc = "torch"
    for split in [cfg.get("split", 1)] + [s for s in (2, 3, 4, 6) if s > cfg.get("split", 1) and levels % s == 0 and levels // s >= 2]:
        r = probe_child("encoding", {**cfg, "split": split}, PROBE_N)
        notes[f"encoding split {split}"] = r
        log(f"  tiny-cuda-nn encoding, {levels} levels as {split} grid{'s' if split > 1 else ''}: "
            f"{'ok' if r['ok'] else 'FAILED'} ({r['msg']}, {r.get('ms', '?')} ms)")
        if r["ok"]:
            enc = "tcnn"
            if split != cfg.get("split", 1):
                over["split"] = split
            break
    if enc == "torch":
        log("  → PyTorch encoding (same grid, slower)")
    r = probe_child("heads", cfg, PROBE_N)
    notes["heads"] = r
    head = "tcnn" if r["ok"] else "torch"
    log(f"  tiny-cuda-nn heads: {'ok' if r['ok'] else 'FAILED — using PyTorch'} ({r['msg']}, {r.get('ms', '?')} ms)")
    return f"{enc}/{head}", notes, over


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", choices=["encoding", "heads", "heads-ff", "field"])
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--n", type=int, default=65536)
    a = ap.parse_args(argv)
    if a.probe:
        print(json.dumps(probe(a.probe, json.loads(a.cfg), a.n)))
        return 0
    import torch
    from importlib.metadata import version
    try:
        import tinycudann  # noqa: F401
        tv_ok = True
    except Exception as e:                                          # noqa: BLE001
        tv_ok, why = False, f"{type(e).__name__}: {e}"
    try:
        tv = version("tinycudann")
    except Exception:                                               # noqa: BLE001
        tv = "installed" if tv_ok else "not importable"
    if not torch.cuda.is_available():
        print(f"torch {torch.__version__} · tinycudann {tv} · no GPU: nothing to probe (the trainer uses torch/torch)")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) · tinycudann {tv} · "
          f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")

    def show(part, name, n, r):
        extra = f"{r.get('ms', '?')} ms, peak {r.get('peak_mib', '?')} MiB, {r.get('dtype', '')}" if r["ok"] else r["msg"]
        print(f"  {'OK  ' if r['ok'] else 'FAIL'}  {part:<9} {name:<46} n={n:<7} {extra}", flush=True)

    if tv_ok:
        for part, name, cfg, n in MATRIX:
            show(part, name, n, probe_child(part, cfg, n))
    else:
        print(f"  tinycudann cannot be imported ({why[:200]}): the PyTorch field only")
    impl, _, over = resolve(FMGS, "cuda", log=lambda m: None) if tv_ok else ("torch/torch", {}, {})
    # The verdict: the field the trainer will build (impl auto) runs forward + backward on this GPU.
    r = probe_child("field", {**FMGS, **over, "impl": impl}, PROBE_N)
    what = f"{impl}" + (f", split {over['split']}" if over.get("split") else "")
    show("field", f"the trainer's field: {what}", PROBE_N, r)
    print(f"  → the trainer (impl auto) will use: {what} (encoding/heads)  — {'usable' if r['ok'] else 'NOT usable'}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
