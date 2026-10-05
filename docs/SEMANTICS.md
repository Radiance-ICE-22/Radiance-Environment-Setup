# Semantic embeddings on the splat: status and handoff

Per-Gaussian CLIP + DINOv2 features attached to a trained (frozen) FiGS splat, so a text query
resolves offline to a 3D goal (`semantic_goal`) that becomes waypoints. The plan, phase by phase,
is `docs/SEMANTICS_PLAN.md` (live copy with tick-boxes:
https://claude.ai/code/artifact/d12ee1d2-3737-424f-bca9-6827f3e195aa). This file records what is
built, how to run it, and the numbers measured.

## 1. Status

| Phase | What | State |
| --- | --- | --- |
| 0 | Environment and pose check | **DONE.** Gate PASSED on intellisense08, 5 Oct 09:48 (second run; the first found gsplat's 32-channel backward limit). |
| 1 | Teacher features, lift backend, CLI query | First gate 5 Oct (fc7aed2): pipeline PASSED (35 min), queries 3 of 5 (need 4). Fixed: relative threshold (whiteboard), per-chunk upsample (VRAM); red tool chest annotation under review. **Gate to re-run (queries only).** |
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
- **Probe** — four exact identities of gsplat's feature rendering, which the lift backend relies on: linearity, finite differences, blend weights reproducing the alpha image (the lift denominators), channel chunking, and 2C channels WITH gradients through render_features' automatic 32-channel chunking. Plus ms per 32-channel forward + backward, peak VRAM, and a lift-time estimate (CLIP 512 + DINO 384 channels + one weight pass per training image).
- **Cameras** — refined-pose PSNR on training views ≥ raw-pose PSNR (−0.05 dB tolerance); also reports the size of the SO3xR3 corrections and held-out PSNR. Image strips (refined | raw | training image) are written for a visual check.

## 3b. Phase 1: what was added

| Path | |
| --- | --- |
| `figs/semantic_pipeline.py` | Resumable steps `preflight → cameras → teachers → lift → export` (svnet_pipeline's pattern: chained fingerprints incl. the checkpoint key, one child process per step, `--status`, `--from/--only/--redo`). |
| `figs/semantic_query.py` | Phrase → ranked candidates (course frame) with approach point and gap; `--eval` scores queries.json (gate: 4 of 5). |
| `semantics/radiance_semantics/teachers.py` | CLIP pyramid (7 scales 0.05–0.5 of the short side, half-crop stride, averaged on one grid) and DINOv2 (896 px wide) per frame, fp16, cached per tag, resumable. |
| `semantics/radiance_semantics/lift.py` | Blend-weighted average of the teacher maps onto every Gaussian over the training views with refined poses; 32-channel chunks; accumulators on the GPU when ≤ 2.5 GB. |
| `semantics/radiance_semantics/store.py` | The per-Gaussian table (clip.f16, dino.f16, weight.f32, geom.f32, pca_rgb.u8, index.json), atomic swap, order hash. |
| `semantics/radiance_semantics/query.py` | LERF relevancy (T = 10, canonical negatives), voxel connected-component clusters, approach point, gap, hit scoring. |
| `semantics/radiance_semantics/annotations.py` | queries.json (course frame) with the five gate objects. |
| `figs/course_tools.py` | `splat_order()` factored out of `splat` (identical result, tested against the old code) — the one definition of the .splat record order. |
| `ui/deploy/sem1_gate.sh` | The Phase 1 gate. |

```bash
source ~/Radiance/figs/figs_env.sh
python figs/semantic_pipeline.py --scene backroom              # teachers dominate (tens of minutes)
python figs/semantic_query.py --scene backroom "red tool chest"
python figs/semantic_query.py --scene backroom --eval          # after §6
```

## 4. Decisions and findings (Phase 0)

- **OpenCLIP is pinned at 2.24.0.** 3.x requires `timm>=1.0.17`; nerfstudio 1.1.4 pins `timm==0.6.7`. 2.24 imports timm only optionally and needs just ftfy, regex, tqdm and huggingface_hub. Verified: under the constraints file, asking for 3.3.0 is refused and nothing moves.
- **huggingface_hub < 1.0.** 1.x replaced requests with httpx; open_clip 2.24 was written against 0.x (0.36.2 installs cleanly).
- **The constraints file is built from package metadata, not `pip freeze`.** `pip freeze` writes conda-installed packages as `name @ file://…`, which cannot constrain anything — numpy among them in kitchen. While setting up the cloud test env, one unconstrained dependency pulled numpy 2.2 and torch 2.1.2 broke at import ("_ARRAY_API not found"): the exact failure this prevents.
- **splatfacto renders the raw pose in eval mode.** The SO3xR3 correction is applied only while training (`get_outputs: if self.training`). `cameras.optimized_c2w` applies `camera_optimizer.apply_to_camera` explicitly; the lift must use it.
- **gsplat 1.0.0: forward ≤ 512 channels, backward ≤ 32.** The forward kernel takes 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512 channels (others padded up); the backward kernel only 1–4, 8, 16, 32. The first gate died with `RuntimeError: Unsupported number of channels: 64` in `rasterize_to_pixels_bwd`. `render_features` now splits features that require grad into 32-channel chunks; rendering without grad (baking, visualisation) still goes in one call. Consequence: a lift is 16 + 12 + 1 = 29 passes per image; FMGS training renders 28 chunks per step (Phase 4 note).
- **`open-clip-torch 2.24.0 requires sentencepiece` (pip warning) is expected.** sentencepiece is only for Hugging Face-tokenizer models; ViT-B-16 uses open_clip's own tokenizer.
- **DINOv2 code is frozen at fetch time.** The first fetch caches the `facebookresearch/dinov2` hub checkout; later loads use it with `source="local"`, never re-pulling main.
- **pipeline loading needs cwd = `gsplats/workspace`** (ns-train ran there; config.yml paths are relative). `cameras.in_workspace` handles it, and `load_pipeline` forces `cache_images=cpu` to keep VRAM for rendering.

Verified in the cloud (no GPU): 26 tests, against torch 2.1.2 (CPU), gsplat 1.0.0's own projection code, nerfstudio 1.1.4 and open_clip 2.24.0 — our view matrix equals splatfacto's `get_viewmat`, `optimized_c2w` equals nerfstudio's `CameraOptimizer` correction, the probe's identities hold on a dense reference renderer and fail on a deliberately non-linear one. `load_pipeline` was run end to end on a tiny CPU-built splatfacto run in the FiGS layout (checkpoint pose corrections recovered exactly, cwd restored). `install_semantics.sh` was dry-run against a Python 3.10 env. Not verifiable in the cloud: anything on CUDA (gsplat's rasterizer, timings) and the weight downloads.

## 4b. Decisions (Phase 1)

- **Teachers for all 300 frames, lift over the 270 training views.** Frames are keyed by file stem; the 30 held-out frames' features are there for later evaluation.
- **"Unseen" means weight exactly 0** everywhere (lift, table, query). The first end-to-end run counted 10,741 seen Gaussians in the lift and 11,592 in the table: weights in (0, 1e-8] had zeroed features but positive weight. Fixed and tested.
- **Selection score = (relevancy − threshold) × opacity**, not × volume: weighting by volume would favour walls and floor.
- **Flat captures:** when the cameras span less than 2 × margin on an axis (usually height — the phone held at one height), the waypoint box falls back to the cameras' own range on that axis instead of becoming empty, so the approach altitude stays where the camera flew.
- **Hit rule for the gate:** the top candidate's box grown by 0.3 m contains the annotation, or its centroid is within 0.75 m.
- **Probe finite-difference tolerance 5e-3** (others stay 2e-3): single-entry differences carry fp32 atomic noise (1.7e-3 measured).

## 5. Measured on the hosts

To fill from `~/sem0_gate.log` and `SousVide/runs/semantics_p0_*.json`.

First gate, intellisense08 (RTX 2080 8 GB, driver 535), 5 Oct 09:22–09:24, repo d0faa98:

| Check | Result |
| --- | --- |
| install_semantics.sh | PASS. 280 packages frozen; added open_clip_torch 2.24.0 and ftfy 6.3.1 (regex, huggingface_hub 0.36.2, tqdm already present); torch 2.1.2, torchvision 0.16.2, numpy 1.26.4, timm 0.6.7, gsplat 1.0.0, nerfstudio 1.1.4, tinycudann 2.0 unchanged. Weights fetched in 75 s; DINOv2 hub code frozen at that fetch. |
| CLIP + DINOv2 smoke | PASS, peak allocated 602 MiB |
| gsplat N-channel probe | FAIL: backward kernel rejects 64 channels (fixed since; re-run) |
| Camera check, backroom 2026-09-26_190524 | PASS. 532,361 Gaussians, 270 training cameras. Refined-pose PSNR 31.68 dB vs raw 27.46 dB (+4.22 dB; every one of 10 views better, +3.3 to +7.0 dB). Corrections: translation mean 4.1 mm (max 39.0 mm), rotation mean 0.108° (max 1.99°). Held-out PSNR 27.06 dB — rendered with raw poses (eval cameras get no correction), so it understates the model like the raw train figure does. |

The camera result settles the design point: features lifted with transforms.json poses would be projected through cameras that render ~4 dB worse — up to 39 mm and 2° off. The lift uses the refined poses.

Second gate, intellisense08, 5 Oct 09:47–09:48 — **all five checks PASS** (records:
`SousVide/runs/semantics_p0_probe_{synthetic,backroom}_2026-10-05_0947.json`,
`semantics_p0_cameras_backroom_2026-10-05_0948.json`). Install re-run changed nothing (weights cached, 4 s);
camera check identical to the first run.

gsplat probe, 32 channels, one forward + backward (lift estimate = 29 passes × 300 transforms.json frames):

| Scene | Resolution | Contributing Gaussians | fwd + bwd | Peak allocated / device | Lift estimate | Worst identity error |
| --- | --- | --- | --- | --- | --- | --- |
| backroom (532,361) | 480×270 | 50,412 (9%) | 8 ms | 242 / 1893 MiB | 1.1 min | finite diff 1.9e-4 |
| backroom (532,361) | 960×540 | 54,086 (10%) | 14 ms | 323 / 2153 MiB | 2.0 min | finite diff 7.5e-4 |
| synthetic (1.5 M) | 480×270 | 3,898 (0.3%) | 22 ms | 633 / 4197 MiB | 3.3 min | finite diff 1.7e-3 |
| synthetic (1.5 M) | 960×540 | 3,917 (0.3%) | 51 ms | 764 / 4565 MiB | 7.5 min | finite diff 1.5e-3 |

Reading it:
- The lift itself is cheap: ~2 min for backroom at 960×540, ~2.2 GB on the device. Teacher extraction (the CLIP pyramid) will dominate Phase 1's time, not the lift.
- The synthetic scene is dense and opaque (mean alpha 1.00), so only the front ~0.3% of Gaussians get any weight; "visible" in the probe means *contributing*, and the warning there is about occlusion, not field of view. Its numbers bound cost for a 1.5 M splat; backroom's are the ones to plan with.
- Chunked gradients (64 channels with grad through two 32-channel calls) match unchunked ones to ≤ 1.1e-6.
- The synthetic finite-difference error (1.7e-3) is close to the shared 2e-3 tolerance — fp32 atomic accumulation on single small gradients, not a bug (linearity over all Gaussians is 4e-5–4e-4). Phase 1 gives that check its own tolerance (5e-3) so it cannot flake.

## 6. Phase 1 gate: annotations

The five objects, chosen 5 Oct from the Phase 0 image strips (`D:\Projects\FYP\semantics_p0\p0_cameras`):
red tool chest, shop vacuum, green foam mats, garden cart, whiteboard. For each: Galley → course editor on
backroom → Show ▸ Splat → place the Goal marker on the object → read its position (course frame), then on
intellisense08:

```bash
source ~/Radiance/figs/figs_env.sh
python -m radiance_semantics.annotations --scene backroom init
python -m radiance_semantics.annotations --scene backroom set "red tool chest" X Y Z    # once per object
python -m radiance_semantics.annotations --scene backroom list
tmux new -d -s sem1 'bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem1_gate.sh > ~/sem1_gate.log 2>&1'
```

The gate can run before the annotations: the pipeline completes and the gate reports PENDING; re-run with
`SKIP_PULL=1` once they are set (finished steps are skipped).

## 7. Phase 1, first gate (intellisense08, 5 Oct 10:57–11:32, fc7aed2)

| Step | Time | Peak VRAM (device) | Result |
| --- | --- | --- | --- |
| cameras | 12 s | 1.3 GB | refined 30.64 dB vs raw 25.34 dB (+5.30, 6 views) |
| teachers | 31 min 29 s | 5.6 GB | 300 frames × 3,529 CLIP crops (7 scales), 6.3 s/frame, 1,338 MB |
| lift | 3 min 11 s | **7.6 GB** | 531,951 of 532,361 Gaussians seen, 7,830 render passes |
| export | 10 s | — | 532,361 rows, .splat order 22cd5cc9509bc6e5, 924 MB |

Queries (fixed threshold 0.55; CLIP text tower cold start 5.7 s, then ~0.75 s per query):

| Object | Result | Top candidate | Reading |
| --- | --- | --- | --- |
| shop vacuum | HIT 0.035 m | 0.34 × 0.56 × 0.53 m | clean |
| garden cart | HIT 0.22 m | only 344 Gaussians ≥ 0.55 (max 0.63) | weak but right |
| green foam mats | HIT 0.34 m, ambiguous | box 2.7 m long | runner-up = black rubber floor mats (flat, z ≈ 0) |
| red tool chest | MISS 1.30 m | score 1047 vs 8.8; 1.7 × 0.9 × 1.6 m at 1.0 m height, y = 2.82 | annotation at y = 1.53 is at the edge of where the camera walked — likely a depth error when placing it; **to verify** |
| whiteboard | MISS, right object at rank 2 | 3.2 × 10 m at 2.9 m height; 75,596 Gaussians ≥ 0.55 | every white flat surface (ceiling, walls) passes a fixed 0.55, and connected walls merge |

Changes after this gate:
- **Relative threshold** (query.py): τ = 0.55 + 0.5 · (peak − 0.55), peak = mean of the top 100 relevancies; candidates with a box diagonal > 4 m are flagged LARGE; the CLI prints the relevancy distribution and τ. `--rel-alpha 0` restores the fixed threshold. Tested on a synthetic wall (rel 0.6) vs object (rel 0.8).
- **Per-chunk upsampling in the lift**: the full 512-channel CLIP map at 960×540 (~1 GB float32) was upsampled at once; now only the 32 channels being rendered. Same numbers, lower peak — no need to re-run the lift.
- **Method note:** the five gate queries were used to choose these changes, so they are now a development set. Phase 5's comparison uses a separate, frozen set of ≥ 15 queries per scene annotated before any results are seen.
