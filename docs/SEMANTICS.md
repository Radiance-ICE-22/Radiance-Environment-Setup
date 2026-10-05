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
| 1 | Teacher features, lift backend, CLI query | **DONE.** Gate PASSED 5 Oct 16:12: 5 of 5 queries hit (second run, after the relative threshold and a corrected red tool chest annotation). |
| 2 | Galley backend | **DONE.** Gate PASSED on intellisense08, 5 Oct 17:24 (second run; the first failed only on a host-dependent test, §9): cold query 5.8 s, warm ≤ 789 ms, 5 of 5 hits. |
| 3 | Splat editor UI | **DONE.** Gate PASSED on intellisense08, 5 Oct: automated half 19:01 (query → course → flight, tracking max 8 mm) and browser half 19:21 (whiteboard sent from the editor, 3-keyframe course flown, tracking max 73 mm); 532 k Gaussians recoloured in 20 ms + 71 ms frame (§10). |
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

## 3c. Phase 2: what was added (Galley backend)

| Path | |
| --- | --- |
| `figs/semantic_worker.py` | Long-lived query server in kitchen, JSON lines on stdin/stdout (`ping`, `query`, `labels`, `unload`). CPU only (`CUDA_VISIBLE_DEVICES=""`). Keeps the CLIP text tower and memory-mapped tables loaded; reloads a table when its `index.json` changes; exits after `--idle` s. Library prints go to stderr, so stdout carries only replies. |
| `ui/backend/galley/semworker.py` | Galley's client: starts the worker through `figs_env.sh` on first use, one request at a time, restarts it after an idle exit or a crash (a crash costs only the request in flight), kills it on a timeout. Log: `<data_dir>/semantic_worker.log`. |
| `ui/backend/galley/semantics.py` | `SemanticRun` → `semantic_pipeline.py` argv (`feat_width` defaults to the profile's `semantic_feat_width`), status (steps for the active run, every table with stale/fresh by checkpoint key), `queries.json` read/write, relevancy cache. |
| `ui/backend/galley/app.py` | `POST /api/jobs/semantics`; `/api/scenes/{scene}/semantics` (status), `…/query`, `…/relevancy/{id}`, `…/{backend}/pca`, `…/labels`, `…/queries` (GET/PUT); `/api/semantics/worker` (+ `/stop`). Listed in `ui/README.md`. |
| `semantics/radiance_semantics/query.py` | `run_query(..., return_relevancy=True)` so the worker sends the heatmap without computing relevancy twice. |
| `ui/machines/*.toml` | `semantic_feat_width` (intellisense08 960, dummy 480), `semantic_worker_idle_s` 600. |
| `ui/deploy/sem2_gate.sh` | The Phase 2 gate. |

Design points:
- **Stale means "built for another checkpoint"**: the table key (run + checkpoint stem + mtime) is the same string as Galley's `.splat` cache key, so a retrain or a promote makes both stale together, and the relevancy/pca bytes of a stale table are never painted on a splat they do not match (the API says `stale` and sends `X-Table-Stale`).
- **No new course endpoint**: `semantic_goal` with the extra fields (`query`, `backend`, `score`, `extent`, `approach`) already round-trips through `PUT /api/configs/courses/{name}` (tested).
- **Archive/Promote wait for semantic jobs** like figs and SV-Net jobs (the job carries the scene).
- In the cloud end-to-end run (50 k rows, stand-in text encoder): cold query 1.4 s, warm 0.19 s. backroom has 532 k rows; Phase 1's CLI measured ~0.72 s warm per query there.

## 3d. Phase 3: what was added (splat editor)

`#/splat/<scene>` (Explorer ▸ Scenes ▸ a scene ▸ Semantics) with the contextual **Semantics** ribbon tab:
Build (features job, Continue, stop the worker) · Query (backend, candidates, standoff, threshold, relative,
negatives) · View (colour / relevancy / PCA, floor, candidates only, pins, camera path, key speed) · Goal
(Send to course, approach keyframe) · Annotate (label, save, Query all).

| Path | |
| --- | --- |
| `ui/frontend/src/splat/format.ts` | Parses the browser `.splat` (32-byte records; course frame (x, −y, −z)); positions, largest scale, opacity, colours. |
| `ui/frontend/src/splat/SplatMesh.tsx` | The splat renderer, vendored from drei 10.7.9's `<Splat>` (same shaders, covariance packing and worker sort) but fed a parsed buffer, with a `colors` prop that rewrites only the colour words and re-uploads that texture. Sorts only when the view changed. |
| `ui/frontend/src/splat/recolor.ts` | Colour modes: rgb, relevancy (heat from the floor up, the rest a dimmed grey), PCA; *candidates only* greys everything outside a box. |
| `ui/frontend/src/splat/pick.ts` | Click → the Gaussian with the largest blend weight T·α along the ray (what the pixel mostly shows). A plain pass over all centres. |
| `ui/frontend/src/splat/load.tsx` | `useSplat(scene)`: export (cached on the host) → download with progress → parse; one parsed copy per URL shared by the course editor and the splat editor. `SplatLayer` = renderer + error boundary. |
| `ui/frontend/src/splat/SplatScene.tsx` | The editor's 3D view: splat, candidate boxes, goal, approach point with the drone's sphere, annotation pins, picked Gaussian. |
| `ui/frontend/src/three/common.tsx` | `Label`, `boxEdges`, `Frame`, `KeyNav` moved out of `course/Scene3D.tsx`, shared by both views. |
| `ui/frontend/src/pages/SplatEditor.tsx` | The page: query bar, Candidates / Features / Annotations tiles, Properties (picked Gaussian + best labels, candidate, query and table), Send to course and Build dialogs. |
| `ui/frontend/src/course/model.ts` | `SemanticGoal` gains `query`, `backend`, `score`, `extent`, `approach`; `withGoalAt` (moving the goal drops `score`); `appendApproach`, `courseToGoal`. |
| `ui/frontend/src/pages/Course.tsx`, `course/Scene3D.tsx` | Splat through the shared loader; goal tile shows the query with *open in splat editor*; the approach point is drawn; the course reloads when the splat editor saved it. |
| `ui/frontend/src/pages/Scene.tsx` | Semantics tile: lift / fmgs state, rows, size, lift time, peak VRAM. |
| `ui/frontend/src/api.ts`, `shell/*`, `main.tsx` | `semApi`; the Semantics tab; route; Explorer entry; `semantics` jobs re-run and filter; 4 icons. Ribbon fix: a contextual tab asked for by a lazily loaded document is no longer reset to Home before the document registers. |
| `ui/frontend/tests/` | `*.test.ts` (`npm test`, Node ≥ 22.6); `compare/` (drei vs SplatMesh pixel diff); `e2e/` (Galley + stand-in tools over a synthetic room, Playwright). Cloud checks; nothing here ships. |
| `ui/deploy/sem3_gate.sh` | The Phase 3 gate. |

**Send to course.** New course: two keyframes at rest — the first camera position (kept inside the
waypoint box) and the approach point facing the object — plus `semantic_goal`. Existing course: the
approach point is appended as the final keyframe (at rest, yaw facing the object, unwrapped to within π
of the previous one; the old last keyframe becomes a pass-through); only the goal changes if the option is
off. The course editor then opens it for Fly.

**Cloud measurements (5 Oct).** Renderer vs drei on 3,000 random anisotropic Gaussians: mean |Δ| 0.18/255,
max 8 (normalised vs raw quantised quaternions), identical coverage. Recolour CPU time (colour build +
colour-word rewrite): 1 M Gaussians 27 ms in node; 60 k in Chromium 3–12 ms; the texture upload happens in
the next frame (software GL in the cloud, so its time there means nothing — read it on the host). Pick:
1 M Gaussians within the 200 ms test budget in node.

**Not in Phase 3:** *Compare* (lift | FMGS side by side) waits for an FMGS table (Phase 4); the command is
there, disabled.

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

## 8. Phase 1, second gate (intellisense08, 5 Oct 16:12) — PASSED, 5 of 5

Pipeline steps were cached; only the queries re-ran (8.4 s, CLIP text cold start 5.5 s, then ~0.72 s per
query). Annotations corrected by Suhan after checking in the splat: red tool chest (-0.366, 2.6, -0.931),
whiteboard (-1.8, -8.37, -1.233). The chest correction was checked against the object itself.

| Object | Error | τ (peak) | Selected | Top box (m) | Next best score | Approach gap |
| --- | --- | --- | --- | --- | --- | --- |
| red tool chest | 0.346 m | 0.669 (0.789) | 7,326 | 1.23 × 0.61 × 0.86 | only candidate | +0.40 ok |
| shop vacuum | 0.015 m | 0.589 (0.628) | 1,112 | 0.23 × 0.29 × 0.35 | 0.12 vs 17.4 | +0.20 ok |
| green foam mats | 0.209 m | 0.634 (0.719) | 1,976 | 0.68 × 0.81 × 0.52 | 2.4 vs 41.6 | −0.12 TOO CLOSE |
| garden cart | 0.218 m | 0.570 (0.590) | 131 | 0.07 × 0.24 × 0.11 | 0.33 vs 1.55 | +0.20 ok |
| whiteboard | 0.379 m | 0.623 (0.696) | 1,224 | 0.88 × 1.19 × 0.89 | 0.74 vs 28.9 | +0.26 ok |

What changed between the runs: the relative threshold cut the whiteboard query from 75,596 selected
Gaussians (10 m wall/ceiling clusters) to 1,224 and put the board first; every query is now unambiguous
(runner-up ≤ 6% of the top score; mats were ambiguous before). Mean error 0.23 m, max 0.38 m.

Notes for later phases:
- **Garden cart is weak**: peak relevancy 0.59, only 131 Gaussians, a 7 × 24 × 11 cm cluster — a small part
  of the cart, not the whole object. Right place, fragile. Candidate for the DINO-assisted cluster growth
  (L-CD variant, Phase 5) and a natural first test for the FMGS backend.
- **Approach points** come out at the camera height (z ≈ −0.97, the flat-capture fallback) and the
  mats' one is too close to clutter (gap −0.12 m). Phase 6's feasibility loop (try other directions
  around the object) is the fix; nothing to change in Phase 1.
- The five queries are a development set now (they shaped the threshold). Phase 5 uses a separate frozen
  set of ≥ 15 queries per scene, annotated before results are seen.

## 9. Phase 2 gate (intellisense08, 5 Oct) — PASSED

First run (6a1cb86): the API checks ran against the restarted Galley and called it the same way the splat editor will. All passed:

| Check | Result |
| --- | --- |
| Status | Fresh lift table for the active run; the semantics job had every step cached and succeeded in seconds. |
| Cold query (worker stopped first) | 5.9 s, mostly CLIP text tower load plus mapping the 924 MB table. |
| Warm queries (5 annotated) | 716–754 ms each, mean 737 ms (need < 1 s); **5 of 5 hit**, same as the Phase 1 gate. |
| Relevancy / pca bytes | 532,361 = rows; pca 3 × rows. |
| Labels, worker | Labels for row 0 returned; worker running CPU-only, idle exit after 600 s (seen in `semantic_worker.log`). |
| `.splat` records | `n_written` = table rows, so the heatmap bytes line up with the splat. |

Backend tests: 106 passed and 1 failed, `test_worker_errors_map_to_http`. The test sent
`/api/scenes/../semantics/query`, and httpx normalises that to `/api/semantics/query`. With a built
frontend (`ui/frontend/dist`, present on the host but not in the cloud copy), that path falls through to
the static mount, which answers POST with 405 instead of 404. The fix is in the test, not the API:
it now sends a scene name the route rejects (`bad.name` → 400). The suite was re-checked with and
without a `dist` directory.

Second run (5 Oct 17:24, with the test fix): **PASSED**, backend tests and API checks.

| Query | Warm | Error | Approach (course frame) | Gap |
| --- | --- | --- | --- | --- |
| red tool chest | 737 ms | 0.346 m | (−0.630, 1.111, −1.109) | +0.40 |
| shop vacuum | 730 ms | 0.015 m | (−1.081, −5.628, −0.973) | +0.20 |
| green foam mats | 717 ms | 0.209 m | (−0.171, −5.060, −0.973) | −0.12 |
| garden cart | 721 ms | 0.218 m | (−1.108, −4.910, −0.973) | +0.20 |
| whiteboard | 789 ms | 0.379 m | (−1.108, −7.014, −1.401) | +0.26 |

Cold query 5.8 s (worker 5.47 s); warm max 789 ms, mean 739 ms. Errors are identical to the Phase 1 CLI
gate (§8), so going through Galley and the worker changes nothing in the results. Relevancy max byte 204
(0.80); labels for row 0: floor 0.255, wall 0.218, ceiling 0.217. Worker restarts 0. `.splat` records =
table rows = 532,361.

## 10. Phase 3 gate (intellisense08, 5 Oct, 4e4a5a5) — PASSED

| Check | Result |
| --- | --- |
| Backend tests | 107 passed, 1 skipped (the built-frontend test now runs) |
| `.splat` vs table | 532,361 records, 16.2 MB = 32 × rows: record i is table row i |
| Query “red tool chest” | 1 candidate, 6.3 s (first query after the restart: CLIP load), τ 0.669, 7,384 Gaussians lit; relevancy bytes = rows |
| Top candidate | centroid (−0.599, 2.783, −1.109), 0.35 m from the annotation (same as Phase 1) · approach (−0.63, 1.111, −1.109), gap 0.40 m |
| Course `sem_gate_red_tool_chest` | built exactly as Send to course does for a new course; final keyframe = approach point, yaw 1.552 facing the chest |
| Flight (job 18) | succeeded in 20 s · course: 2 keyframes, 0 outside · tracking mean 0.003 / max 0.008 m · 33 frames · pixel std 47.9 · no dark frames |

In the browser (Suhan's screenshots): “tool box” resolves to the red tool chest (5,297 Gaussians, score
114.9, gap 0.45 m) — a paraphrase, not the annotated phrase; worker 729 ms, 823 ms end to end; **recoloured
in 20 ms + 71 ms for the next frame at 532,361 Gaussians** (gate: about 2 s at 1 M), so the vendored-renderer
route meets the budget with room to spare.

Notes:
- The new-course path is short here: the first camera position, clamped into the waypoint box, is only
  0.2 m from the chest's approach point, so this flight proves the hand-off (goal, final keyframe, yaw,
  through course/simulate/validate/record) more than it exercises flying. Sending to an existing course (the
  approach appended to a loop) gives a real flight; Phase 6's feasibility loop chooses better starts.
- Fixed after the gate (UI text only): the Candidates tile read `margin` as the runner-up's share of the top
  score. `query.py` defines it as (top − runner-up) / top, 1.0 with a single cluster, ambiguous below 0.25, so a
  lone candidate showed "runner-up at 100%". It now says "only one cluster" and shows the runner-up as
  (1 − margin). The cloud stand-in worker had the same misreading, which is why the E2E run missed it.
  The redundant "goal" label (it overlapped the annotation pin) is gone.

**Browser half (5 Oct 19:21) — PASSED.** Whiteboard queried and sent from the splat editor, then flown from
the course editor (job 20), checked with `sem3_gate.sh --check sem_whiteboard`:

| Check | Result |
| --- | --- |
| `semantic_goal` | “whiteboard” · lift · score 28.87 · approach (−1.108, −7.014, −1.401) |
| Final keyframe | the approach point, yaw −2.203 facing the board |
| Course | 3 keyframes (approach appended), 0 outside the captured volume |
| Flight | succeeded · 70 frames at 20 Hz (3.5 s) · tracking mean 0.015 / max 0.073 m · pixel std 47.3 · no dark frames |

With the automated half and the browser recolour figure above, every Phase 3 gate item is met:
query → pick → Send to course → Save and fly passes course, simulate and validate and ends at the approach
point, and recolouring is far under the 2 s budget (no need for further loader work).
