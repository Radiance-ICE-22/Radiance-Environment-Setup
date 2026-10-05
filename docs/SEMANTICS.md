# Semantic embeddings on the splat: status and handoff

Per-Gaussian CLIP + DINOv2 features attached to a trained (frozen) FiGS splat, so a text query
resolves offline to a 3D goal (`semantic_goal`) that becomes waypoints. The plan, phase by phase,
is `docs/SEMANTICS_PLAN.md` (live copy with tick-boxes:
https://claude.ai/code/artifact/d12ee1d2-3737-424f-bca9-6827f3e195aa). This file records what is
built, how to run it, and the numbers measured.

## 1. Status

| Phase | What | State |
| --- | --- | --- |
| 0 | Environment and pose check | Built and cloud-tested 5 Oct 2026. **Gate not yet run on intellisense08.** |
| 1 | Teacher features, lift backend, CLI query | Not started (waits on the Phase 0 gate) |
| 2 | Galley backend | Not started |
| 3 | Splat editor UI | Not started |
| 4 | FMGS backend (`splatfacto-sem`) | Not started |
| 5 | Evaluation and comparison | Not started |
| 6 | Language → waypoints → SV-Net | Not started |

## 2. Phase 0: what was added

| Path | |
| --- | --- |
| `semantics/` | The `radiance_semantics` package (pip-installed into kitchen, editable). Phase 0 modules: `paths.py` (scene/run layout, the active model, Galley's cache key), `env_check.py` (pinned-stack guard, constraints file), `models.py` (OpenCLIP/DINOv2 load, fetch, smoke test), `render.py` (feature rendering: gsplat, plus a CPU reference for tests), `probe.py`, `cameras.py`, `log.py`. Tests in `semantics/tests/`. |
| `setup_scripts/install_semantics.sh` | Installs the above into kitchen without moving anything already there, and caches the weights. |
| `setup_scripts/install_figs.sh` | New soft step `semantics` (13 steps now) after `envfile`; `--relocated` redoes it; verification lists `radiance_semantics`. |
| `setup_scripts/verify_figs.sh` | `--semantics` adds a section: package, open_clip 2.24.0, pinned stack, CLIP + DINOv2 smoke, quick gsplat gradient probe. |
| `ui/deploy/sem0_gate.sh` | The Phase 0 gate (below). |

## 3. Running Phase 0 on a host

```bash
cd ~/Radiance/Radiance-Environment-Setup && git pull
tmux new -d -s sem0 'bash ui/deploy/sem0_gate.sh > ~/sem0_gate.log 2>&1'     # Galley queue idle
```

Or by hand:

```bash
setup_scripts/install_semantics.sh --prefix ~/Radiance/figs          # once; idempotent
setup_scripts/verify_figs.sh --prefix ~/Radiance/figs --quick --semantics
source ~/Radiance/figs/figs_env.sh
python -m radiance_semantics.probe --record                          # synthetic, 1.5 M Gaussians
python -m radiance_semantics.probe --scene backroom --record         # the real splat
python -m radiance_semantics.cameras --scene backroom --save-images --record
```

What each check proves (all must pass for the gate):

- **Pinned stack** — torch 2.1.2, nerfstudio 1.1.4, gsplat 1.0.0, numpy 1.x, unchanged by the install (versions compared before and after).
- **CLIP + DINOv2 smoke** — OpenCLIP ViT-B-16/laion2b_s34b_b88k matches a red image to "red" and a blue one to "blue"; DINOv2 vits14 returns [1, 256, 384] patch tokens.
- **Probe** — four exact identities of gsplat's feature rendering, which the lift backend relies on: linearity, finite differences, blend weights reproducing the alpha image (the lift denominators), and channel chunking (64 at a time). Plus ms per 64-channel forward + backward, peak VRAM, and a lift-time estimate (CLIP 512 + DINO 384 channels + one weight pass per training image).
- **Cameras** — refined-pose PSNR on training views ≥ raw-pose PSNR (−0.05 dB tolerance); also reports the size of the SO3xR3 corrections and held-out PSNR. Image strips (refined | raw | training image) are written for a visual check.

## 4. Decisions and findings (Phase 0)

- **OpenCLIP is pinned at 2.24.0.** 3.x requires `timm>=1.0.17`; nerfstudio 1.1.4 pins `timm==0.6.7`. 2.24 imports timm only optionally and needs just ftfy, regex, tqdm and huggingface_hub. Verified: under the constraints file, asking for 3.3.0 is refused and nothing moves.
- **huggingface_hub < 1.0.** 1.x replaced requests with httpx; open_clip 2.24 was written against 0.x (0.36.2 installs cleanly).
- **The constraints file is built from package metadata, not `pip freeze`.** `pip freeze` writes conda-installed packages as `name @ file://…`, which cannot constrain anything — numpy among them in kitchen. While setting up the cloud test env, one unconstrained dependency pulled numpy 2.2 and torch 2.1.2 broke at import ("_ARRAY_API not found"): the exact failure this prevents.
- **splatfacto renders the raw pose in eval mode.** The SO3xR3 correction is applied only while training (`get_outputs: if self.training`). `cameras.optimized_c2w` applies `camera_optimizer.apply_to_camera` explicitly; the lift must use it.
- **gsplat 1.0.0** renders N-D features with `sh_degree=None`, up to 512 channels; other counts are padded to the next power of two (DINO's 384 → 512 internally), so the lift renders in 64-channel chunks.
- **DINOv2 code is frozen at fetch time.** The first fetch caches the `facebookresearch/dinov2` hub checkout; later loads use it with `source="local"`, never re-pulling main.
- **pipeline loading needs cwd = `gsplats/workspace`** (ns-train ran there; config.yml paths are relative). `cameras.in_workspace` handles it, and `load_pipeline` forces `cache_images=cpu` to keep VRAM for rendering.

Verified in the cloud (no GPU): 26 tests, against torch 2.1.2 (CPU), gsplat 1.0.0's own projection code, nerfstudio 1.1.4 and open_clip 2.24.0 — our view matrix equals splatfacto's `get_viewmat`, `optimized_c2w` equals nerfstudio's `CameraOptimizer` correction, the probe's identities hold on a dense reference renderer and fail on a deliberately non-linear one. `load_pipeline` was run end to end on a tiny CPU-built splatfacto run in the FiGS layout (checkpoint pose corrections recovered exactly, cwd restored). `install_semantics.sh` was dry-run against a Python 3.10 env. Not verifiable in the cloud: anything on CUDA (gsplat's rasterizer, timings) and the weight downloads.

## 5. Measured on the hosts

To fill from `~/sem0_gate.log` and `SousVide/runs/semantics_p0_*.json`.

| Host | Probe (synthetic 1.5 M) 480×270 / 960×540 | Probe (backroom) | Peak VRAM | Lift estimate (backroom) | Refined vs raw PSNR | Corrections |
| --- | --- | --- | --- | --- | --- | --- |
| intellisense08 (RTX 2080 8 GB) | — | — | — | — | — | — |
