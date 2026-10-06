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
| 4 | FMGS backend | **DONE.** Gate PASSED on intellisense08, 6 Oct 13:31 (run 3, 7344e7a, §11): 4,200 steps in 31.4 min, default config (2^20, faithful, 480×270, tcnn split 2), no fallback; peak 5.5 GB PyTorch / 7.95 GB device; Gaussians and checkpoint unchanged; the dev queries hit 4 of 5 on FMGS vs 5 of 5 on the lift. Runs 1–2 failed on tcnn launch limits and then memory, and a NaN came out with the OOM fix (§3e). Standalone trainer (§3e), editor Compare. |
| 5 | Evaluation and comparison | **Results in (§13)**, 6 Oct: four variants × backroom + flightroom on frozen sets; lift best (0.62 / 0.63 top-1), F-CD's CLIP channel fragmented. GTN_lab_v1 excluded (folded, §12). Open: sweeps, Galley metrics tile, Compare check. |
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

## 3e. Phase 4: what was added (FMGS backend)

Decisions (Suhan, 5 Oct): a **standalone trainer** instead of the planned `ns-train splatfacto-sem`
plugin — the splat is frozen, so training needs only the cameras and the Phase 1 teacher maps, not the
RGB images a nerfstudio datamanager would load, and it shares the lift's refined cameras and gsplat
rendering path exactly (a fair Phase 5 comparison); **faithful FMGS at 480×270** by default (the teacher
grids are only ~71 × 40 CLIP and 64 × 36 DINO cells); the editor's **Compare** view in this phase.

| Path | |
| --- | --- |
| `semantics/radiance_semantics/fmgs/field.py` | `FeatureField`: Instant-NGP hash grid (24 levels, 16 → 512, 2^20, 8 features = 192-d) on centres normalised by the scene's 1st–99th percentile box (+5 %), CLIP 512 and DINO 384 heads (2 × 256 ReLU). Implementation per part, `"<encoding>/<heads>"`: `tcnn` (HashGrid, CutlassMLP) or a PyTorch twin `torch` (same hashing and level resolutions; each level checkpointed under autograd, so 262k points do not keep 2 GB of gathers). `field.pt` records which. |
| `.../fmgs/losses.py` | 0.2 · CLIP Huber (δ 1.25) + 0.8 · DINO L2 + 0.01 · pixel alignment, means over pixels with rendered alpha ≥ 0.5. Pixel alignment (our reading): mean \|cos(clip_p, clip_q) − cos(dino_p, dino_q)\| over sampled pixels and 3 × 3 neighbours at dilation 2, DINO = the fixed teacher map. |
| `.../fmgs/train.py` | The trainer. Per step: one training view (refined pose), the trainable Gaussians in its frustum (the most opaque 40 %, picked once), field → features, gsplat render in 32-channel chunks, **divided by the rendered alpha** (so dropping the untrained 60 % does not darken the maps), losses against the upsampled teachers. Adam 1e-2 → 1e-3 exponential, eps 1e-15 (LERF's hash-field settings). Checkpoints every 1,000 steps (resume on the same settings), TensorBoard in `tb/`, `train.json`. Out-of-memory ladder (variant auto): hash table 2^19 → half the visible Gaussians per step → B-lite (render the 192-d encoding, heads per pixel; labelled as a variant); restarts from step 0, recorded. Gaussian checksum before/after. `python -m radiance_semantics.fmgs.train … --steps 200` is the smoke run (exit 0 = loss fell and Gaussians unchanged). |
| `.../fmgs/diag.py` | tiny-cuda-nn probes (incl. the split grid), each in its own process with `CUDA_LAUNCH_BLOCKING=1`: a matrix of hash-grid and MLP configurations, then the field the trainer will build. `resolve()` is the trainer's `impl auto` (the default): tcnn for each part that passes its probe at 262,144 points, else PyTorch; CPU → torch/torch. `train.json` records the choice and the probe results. |
| `.../fmgs/bake.py` | Field at every Gaussian → unit rows. |
| `.../lift.py` | `blend_weights()`: the lift's denominator pass on its own, so both tables agree on unseen rows. |
| `figs/semantic_pipeline.py` | `--backend fmgs`: steps `fmgs` (train into `semantics/<run>/fmgs_train/`, checkpoint-file SHA before/after) and `bake` (table in `semantics/<run>/fmgs/`, `field.pt` copied in; unseen rows from the lift table when it is for the same checkpoint, else a blend-weight pass). `--fmgs-steps/-width/-variant/-impl/-table`. Backend is no longer sticky in `config.json` (a bare run is the lift). |
| `ui/backend/galley/semantics.py` | `SemanticRun.backend` lift \| fmgs with per-backend step validation and `fmgs_*` options; profile key `semantic_fmgs_width` (480 on both hosts); status lists all seven steps and an fmgs summary per table. |
| `ui/frontend/src/pages/SplatEditor.tsx`, `SplatCompare.tsx`, `splat/SplatScene.tsx` | Build dialog: backend choice and FMGS options. **View ▸ Compare**: the right view runs every query on the other backend with its own colours, candidates and picked labels, can send its own candidate to a course, and follows the left view's camera (`CamLink`). Features tile and scene tile: fmgs variant, fallback, loss, VRAM. |
| `ui/deploy/sem4_gate.sh` | The Phase 4 gate. |

Cloud checks (5 Oct): 60 semantics tests (12 new: losses by hand, hash grid, subset/frustum, faithful and
B-lite training on a synthetic scene whose teacher maps are rendered from known features — loss falls,
Gaussians unchanged, resume, the OOM ladder, bake); the real pipeline `--backend fmgs` on the CPU
splatfacto run (torch field, reference renderer): trained, checkpoint unchanged, baked 50,000 rows, and
both tables answer through the query worker; 113 Galley tests; browser runs 26/26 (Phase 3 regression)
and 12/12 (Phase 4: build fmgs from the editor, Compare with a linked camera, send the FMGS candidate).
Not checkable in the cloud: tiny-cuda-nn, GPU memory and speed — the gate's first two steps.

**Gate run 1 (73e7968) failed** at the tiny-cuda-nn smoke on intellisense08: the HashGrid forward
(24 × 8, 2^20, 200k points) raised `CUDA error: invalid configuration argument` in the kitchen build.
Response: `fmgs/diag.py` (which configurations fail), `impl auto` with a per-part PyTorch fallback, and the
gate's step 2 now runs the diag and fails only if the chosen field does not run.

**Diag on intellisense08** (tinycudann 2.0, CUDA 11.8, sm_75): every hash grid up to 96 output dims runs
(16 × 2, 12 × 8, 24 × 4, 24 × 2, at 2^19 and 2^20); every 24 × 8 = 192-dim grid fails to launch, at any
table size and point count. Both head types run. So the limit is the grid's width, not memory. The PyTorch
encoding works (262k points forward + backward 2.8 s, 4.4 GB) but is ~20× slower than tcnn.
Fix: `FieldConfig.split` — the 24 levels as consecutive tcnn grids of 12 × 8 (96 dims each), concatenated
in level order. Same capacity (24 levels × 8 features, 2^20 entries per level) and the same ladder except
that the second grid's base resolution is an integer (tcnn's `base_resolution` is): levels 16 → 84 and
98 → 512 instead of one ladder 16 → 512 (every level within 1 %). The PyTorch twin computes the same split
grid. `resolve()` tries the configured grid, then split 2, 3, 4, 6, then PyTorch; the choice (split in
`field.pt`'s cfg and `train.json`) is recorded and is a reported deviation from FMGS.

**Gate run 2 (70d2178, 5 Oct 21:55) failed** in the smoke run and in the queue (job 21):
`tiny-cuda-nn/gpu_memory.h:563 cuMemCreate(...) failed: CUDA_ERROR_OUT_OF_MEMORY` in tcnn's backward.
Diagnosed on intellisense08 on 6 Oct (Claude Code on the host) with a one-step profile at 480 × 270, split 2,
2^20, over the views with the most trainable Gaussians in frustum (of 212,944 trainable: median 66,563 per
view, p90 120,550, max 140,374):

| Stage (worst view, 140k Gaussians) | PyTorch live | Device in use |
| --- | --- | --- |
| field + Adam state, persistent (2^20) | 2.2 GB | |
| + teachers upsampled, field forward, 28 render chunks, losses | 5.9 GB peak | 7.3–7.9 GB |
| tcnn backward needs its own `cuMemCreate` | | **OOM** |
| after `torch.cuda.empty_cache()` | 2.6 GB | 4.0 GB |

So the live tensors fit; what failed was the split between two allocators. PyTorch's caching allocator kept
1–3 GB of free but fragmented blocks reserved, which tiny-cuda-nn's own arena cannot use. Without
expandable segments even a 2^19 table failed on its first step. On top of that, the OOM ladder in `train()`
caught only `torch.cuda.OutOfMemoryError`, and tcnn raises a plain `RuntimeError`, so the ladder never ran.

Fixes (7344e7a):
- `train()` switches PyTorch's allocator to **expandable segments**.
- tcnn's `RuntimeError` counts as out of memory (`is_oom`). A step that runs out is **retried once** with
  the cache emptied and the same random state. A second failure goes down the ladder (which now works for
  tcnn's error too; seen in a debug run).
- Shorter tensor lifetimes: the fp32 head outputs and the un-normalised render are dropped once used
  (−1 GB at 140k). The losses no longer keep full-map products for backward: `masked_mean` sums the
  channels first, and pixel alignment normalises only the sampled pixels (same values and gradients to
  1e-14, checked on CPU and GPU).

**Then the loss went NaN at step 11.** The default 2^20 config fits once the OOM is fixed, and that exposed a
second, older bug. tiny-cuda-nn gets the gradient of its fp16 outputs already cast to fp16; its own ×128
loss scale comes after the cast. The CLIP Huber term is ~4 orders below the DINO term (0.0008 vs 4–5), so
its gradients (~1e-8) underflowed: only 1–2 % of the CLIP-output gradient was nonzero, and the CLIP head
hardly learned (CLIP loss flat at 0.0008). Pixel alignment then normalised those ~0 CLIP vectors. The
gradient of x/‖x‖ is ~1/‖x‖, which made per-Gaussian gradients of 2.25e4, and ×128 overflowed fp16 in the
CLIP head's backward (`torch.autograd.detect_anomaly` points at tcnn's `_module_functionBackward`).
Fixes (same commit):
- **Dynamic loss scaling** (`torch.cuda.amp.GradScaler`, initial 2^10, growth every 200 steps) whenever a
  field part is tcnn. Adam's update does not depend on the scale. Steps whose gradients overflow are
  skipped and counted (`overflow_skipped_steps` and `loss_scale` in `train.json`).
- Pixel alignment normalises the CLIP vectors with a **floor of 1e-3 on the norm**. The teacher vectors
  have norm 1, so this only matters while the render is still ~0. A numerical guard, not a change to the
  term; recorded here as part of our reading of it.

Smoke after the fixes (200 steps, backroom, default config, no fallback): loss 3.43 → 2.70, CLIP 0.0020 →
0.0001, 2.2 it/s, PyTorch peak 5.5 GB, device 7.95 GB including the desktop's 0.4 GB (with expandable
segments PyTorch keeps its high-water mark reserved), loss scale 2048, 0 skipped steps, 0 retries, Gaussian
checksum unchanged. The semantics tests ran on the host for the first time: 68 passed on CPU (pytest from
a scratch `--target` directory on `PYTHONPATH`, nothing added to kitchen).

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

## 11. Phase 4 gate, run 3 (intellisense08, 6 Oct 12:55–13:31, 7344e7a) — PASSED

Run by Claude Code on the host (`SKIP_PULL=1 ui/deploy/sem4_gate.sh`, tmux). Log: `~/sem4_gate.log`; run 2's
log is kept as `~/Radiance/sem4_gate_run2.log`. The fixes are in §3e.

| Check | Result |
| --- | --- |
| Galley backend tests | 113 passed, 1 skipped |
| Semantics tests | *skipped by the gate* (pytest was not in kitchen yet). Run by hand before the gate: 68 passed on CPU. pytest was added to kitchen after the gate (6 Oct), so later gates run them |
| Field probes | 24 × 8 grid still fails to launch on sm_75; impl auto → tcnn/tcnn, split 2 (285 ms, 4.7 GB at 262k points) |
| Smoke, 200 steps | loss 3.42 → 2.70, Gaussians unchanged, 91 s |
| Full run (Galley job 22) | succeeded in 32.0 min; **4,200 steps in 31.4 min (2.23 it/s)**, loss 3.51 → 1.60 |
| Config | **default, no fallback**: 2^20 table, faithful (28 chunks), 480 × 270, 113.5 M parameters, 212,944 trainable Gaussians |
| Peak VRAM | **5,480 MiB PyTorch / 7,954 MiB device** (incl. ~0.4 GB desktop and PyTorch's reserved high-water mark) < 8,192 |
| fp16 | final loss scale 2048; 15 of 4,200 steps skipped on overflow (GradScaler growth probes every 200 steps); 0 out-of-memory retries |
| Gaussians | checksum d03b7b5bac0bd434 → d03b7b5bac0bd434; checkpoint file SHA 581ff4a1e53bb8e7 → 581ff4a1e53bb8e7 |
| Table | fresh, 532,361 rows = `.splat` records, 1,356.5 MB |

Development queries (the five Phase 1 annotations; they shaped the lift's threshold, so this is a first look,
not the Phase 5 result):

| Query | Lift | FMGS |
| --- | --- | --- |
| red tool chest | HIT 0.35 m, 802 ms | HIT 0.48 m, 791 ms |
| shop vacuum | HIT 0.01 m, 742 ms | HIT 0.10 m, 752 ms |
| green foam mats | HIT 0.21 m, 723 ms | HIT 0.18 m, 775 ms |
| garden cart | HIT 0.22 m, 738 ms | **MISS**, 730 ms |
| whiteboard | HIT 0.38 m, 731 ms | HIT 0.32 m, 813 ms |

Notes:
- The CLIP loss term flattens at ~1e-4 by step ~250 while DINO keeps falling (4.39 → ~2.0). The CLIP Huber
  term is tiny at this weighting (unit-length 512-d teacher vectors), so DINO drives most of the training.
  That fits FMGS's 0.2 / 0.8 weights, but it is worth checking in Phase 5 whether the CLIP channel of the
  FMGS table is as sharp as the lift's (the garden cart miss is a place to start). Not tuned here: the dev
  queries must not shape FMGS's settings.
- Not checked yet: View ▸ Compare in the browser (lift | FMGS) on the new table.


## 12. Phase 5: decisions and progress (6 Oct, in progress)

**Decisions (Suhan, 6 Oct):**
- Second scene: GTN_lab_v1, diagnosed first rather than recaptured (below).
- **L-CD** = lift table + DINO at query time: relevancy diffused over a spatial 16-NN graph whose edges are the
  clipped DINO cosine (α 0.5, 2 iterations), and a candidate split in two where its DINO rows form two groups
  (spherical 2-means, mean directions with cosine < 0.5, each ≥ 15 Gaussians). No retraining.
- **F-C** = FMGS with the DINO loss **and** pixel alignment off (w_dino = w_pa = 0), so F-C vs F-CD isolates
  everything DINO contributes. `semantic_pipeline.py --backend fmgs_c` (steps `fmgs_c`, `bake_c`; table
  `semantics/<run>/fmgs_c/`, no DINO rows; only the 16 CLIP chunks are rendered).
- **Query sets**: drafted and placed by Claude from the training photos (Suhan reviews). The five Phase 1 queries
  stay out (development set).

**GTN_lab_v1 is metric.** Its run record looked wrong (camera path 1,309 m, splat bounds ±168 m). Checked on the
41 frames where marker 0 is the only marker: splat distance / marker (PnP) distance over 733 camera pairs has
median **1.003** (IQR 0.988–1.015). The scale is right; **3 of 600 cameras** (frames 292, 297, 538) were registered
100–219 m away by SfM. Without them the camera volume is 11.3 × 8.4 × 1.9 m. `query.pose_inliers` (a camera
> 4 × the p90 distance from the median camera) now drops such cameras from the query's camera box and approach
directions and from the training views of lift / FMGS / blend weights (dea2e41). backroom and flightroom lose
none. The GTN splat itself is weaker: training views render at 18.1 dB PSNR (backroom 31.7 dB) — HDR/HLG footage
without tone mapping, many washed-out frames.

**Annotating from photos (`annotations.py locate`).** Pick the object's pixel in a training photo; the SfM sparse
points projecting within 40 px give the nearest depth cluster (20th-percentile depth ± 0.25 m), which is
unprojected through that frame's pose (CPU only, no semantic features involved). Validated against Suhan's dev
annotations: shop vacuum 0.09 m, garden cart 0.37 m, red tool chest 0.42 m (one pick lands on the visible face;
picks from 2+ sides are averaged). Each GTN point is also reprojected into other photos to check it lands on
the object: thin panels (banners) and far walls leak to the background there, so those objects wait for
splat-depth picks (dense) once the GPU is free.

Draft sets (`semantics/queries.json`, set `phase5`; frozen copies go into the repo when complete):
- backroom, 16: recycling bin + "blue trash can" (synonym, one instance seen from 4 sides); office chair +
  "swivel chair" (3 instances); grandfather clock + "clock"; children's road play mat; camera tripod; purple foam
  mat; white bucket; glass door cabinet; hose reel (ceiling); moving box; negatives bicycle, sofa, ceiling fan.
  Dropped as unreliable ground truth: window (no SfM points on glass), stools and a red panel (depth leaks).
- GTN_lab_v1, 9 so far: television + "wall-mounted screen", GTN logo, green bean bag, round table, flower
  planter (2 instances so far); negatives red tool chest, shop vacuum (backroom objects), bicycle. To place:
  ceiling fans, air conditioner, lectern, backpack, the Success / Visualize / MindSET banners, wall mural.

**Evaluator (`radiance_semantics/evaluate.py`, 8bb8b87).** Per variant and query: top candidate vs every annotated
instance (hit by the gate rule on any instance; error to the nearest), failures sorted into no candidate / bad
position (missed, top centroid ≤ 1.5 m from an instance) / wrong object. Negatives: false-positive rate at the
fixed floor (any candidate) and, because the relative threshold always keeps the best match, a threshold-free
AUROC of peak relevancy, positives vs negatives. Build time, VRAM and table size from the table's index;
`--report DIR` writes `variants.csv` and `variants.tex`. 77 semantics tests pass.

Check on the dev set (not the evaluation): L-C reproduces the Phase 1 gate exactly (5/5, same errors); L-CD 5/5,
+90 ms per query, 10 s once per table for the DINO graph. The lifted DINO rows of neighbouring Gaussians are
very alike (mean edge weight 0.945), so diffusion has little contrast to work with — a finding to report, not
a setting to tune.

GPU queue (one job at a time): GTN_lab_v1 lift (job 23, running) → splat-depth picks for GTN → backroom F-C →
GTN F-CD → GTN F-C → evaluate both scenes → report.

**GTN_lab_v1 is folded; second scene is flightroom (Suhan, 6 Oct 16:20).** GTN_lab_v1's lift was built (job 23:
537 of 540 views after the 3 outliers, 714,736 of 714,979 Gaussians seen, peak 3.8 GB, 70 min), but its
geometry is not one consistent room: 395 of 600 cameras face within ±45° of +x (backroom and flightroom spread
over all directions), photos of the slatted TV wall and of the mural wall (each with a GTN logo) register onto
the same wall, the banner row in front of the mural projects onto the TV wall in photos where nothing stands
there, and the splat's own depth is full of floaters (picks at 0.4–0.5 m depth; the same logo 1 m apart from two
views). Most likely SfM fused the two walls (a "doppelganger" failure on repeated structure). Annotating and
scoring a folded room would be weak evidence, so Phase 5 uses **flightroom** (Stanford's sample: mocap poses,
metric without SfM, 499 views in every direction; jobs 25–27 build lift, F-CD, F-C). GTN_lab_v1's partial
query draft and lift table stay on disk. Our own second capture remains open: the intellisense lab footage
(Suhan's Drive, 29 Jul) could be added through Galley's New capture as a third scene.

Galley accepts `backend: fmgs_c` (steps fmgs_c / bake_c) since this change; backend tests 114 passed.

flightroom annotation: sparse-point picks are consistent for objects on solid supports (round table + red cup
+ water bottle + floor lamp + water jug within 0.7 m; MSL poster and garden cart from two views within 0.15 m)
but leak to the background on thin ones (monitor, tripods, armchair, ladders), so those wait for splat-depth
picks after the GPU queue (flightroom's mocap-posed splat should not have GTN's floaters).

**Power cuts, 6 Oct 17:47 and ~18:14** (lightning, per Suhan). The host lost power twice: during flightroom
F-CD (job 26, step ~3950 of 4200) and during flightroom F-C (job 29, after step 3000). Galley marks the running
and queued jobs interrupted on restart; no training was lost, because FMGS checkpoints every 1,000 steps and a
re-queued job resumes from the latest one on the same settings (job 28 resumed from step 3000 and finished;
F-C was re-queued the same way).

## 13. Phase 5 results (intellisense08, 6 Oct 18:56–19:20; query sets frozen in f38225b before any run)

Four variants × two scenes, the frozen `phase5` sets (backroom 13 positive + 3 negative; flightroom 16 + 3),
query settings unchanged from Phase 1 (never tuned on these sets). Table: `docs/phase5_results/variants.csv` and
`variants.tex` (for `fyp_report.tex`); per-query records in `semantics/<run>/eval/<variant>.json` on the host.

| Scene | Variant | Top-1 hit | Err. median / p90 (m) | Ambig. | Feasible | Neg. FP | Neg. AUROC | Build (min) | VRAM (GB) | Table (MB) | Query (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backroom | L-C | **8/13 (0.62)** | 0.19 / 3.70 | 0.08 | 0.88 | 1/3 | 0.85 | 3.0 | 7.4 | 924 | 720–890 |
| backroom | L-CD | **8/13 (0.62)** | 0.19 / 3.91 | 0.00 | 0.75 | 1/3 | 0.85 | 3.0 | 7.4 | 924 | 790 |
| backroom | F-C | 6/13 (0.46) | 0.32 / 2.06 | 0.00 | 0.83 | 1/3 | 0.80 | 13.9 | 5.8 | 967 | 737 |
| backroom | F-CD | 7/13 (0.54) | 0.20 / 1.90 | 0.00 | 0.71 | 1/3 | 0.80 | 31.4 | 7.8 | 1357 | 742 |
| flightroom | L-C | **10/16 (0.63)** | 0.22 / 0.75 | 0.06 | 0.80 | 1/3 | 0.90 | 4.4 | 2.4 | 659 | 537 |
| flightroom | L-CD | **10/16 (0.63)** | 0.21 / 0.76 | 0.06 | 0.90 | 0/3 | 0.90 | 4.4 | 2.4 | 659 | 587 |
| flightroom | F-C | 8/16 (0.50) | **0.11** / 3.58 | 0.00 | 0.75 | 0/3 | 0.81 | 11.5 | 5.1 | 814 | 538 |
| flightroom | F-CD | 4/16 (0.25) | 0.18 / 0.56 | 0.00 | 1.00 | 0/3 | **0.96** | 25.2 | 7.5 | 1092 | 546 |

Build = the backend's own step; the CLIP + DINOv2 teachers are shared by all four variants and cost more than
any of them (backroom 31.5 min, 300 frames; flightroom ≈ 50 min, 499 frames). Error = top candidate to the
nearest annotated instance, over queries that returned a candidate. Feasible = hits whose approach point has
≥ 0.15 m clearance. Flightroom's two FMGS runs resumed from step 3000 after the power cuts; build time is
steps ÷ it/s (eval fix in this commit).

Failures (positives; no candidate / wrong object / bad position):
- backroom: lift 2 / 3 / 0 (no candidate: swivel chair, camera tripod; wrong: purple foam mat, white bucket,
  glass door cabinet); F-C 2 / 2 / 3; F-CD 5 / 1 / 0.
- flightroom: lift 4 / 1 / 1 (no candidate: armchair, upholstered chair, keyboard, floor lamp — peak relevancy
  below the 0.55 floor in every variant; wrong: drone gate; bad position: water jug 0.76 m); F-C 4 / 4 / 0;
  F-CD **12** / 0 / 0.

**Findings (research question 2):**
- **Accuracy.** The training-free lift localises best on both scenes (0.62 and 0.63 top-1), with ~0.2 m median
  error. FMGS on CLIP alone (F-C) finds fewer objects (0.46, 0.50) but localises the ones it finds most
  tightly (0.11 m median on flightroom). FMGS with its paper weights (F-CD) is worst on flightroom (0.25).
- **Why F-CD fails: its CLIP channel is spatially fragmented.** For "red cup" on flightroom, F-CD's Gaussians
  above τ form 61 clusters of ≤ 3 (none reaches the 15-Gaussian minimum), where F-C forms one of 288 and the
  lift one of 379. With unit-norm 512-d CLIP teachers the 0.2-weighted CLIP Huber term is ~4 orders below the
  0.8-weighted DINO L2 (Phase 4: 1e-4 vs ~2), so the shared hash grid is shaped almost only by DINO and the
  CLIP head's output is noisy at Gaussian scale. FMGS's weights presumably assume differently scaled CLIP
  targets; rebalancing is the obvious next experiment, but it was not tuned here (it would need its own
  development set).
- **DINO at query time (L-CD)** changes no hit; it removes the one flightroom false positive ("recycling bin")
  and backroom's one ambiguous result, for +50–70 ms per query. The lifted DINO rows are smooth (mean neighbour
  similarity 0.945), which limits what diffusion and splitting can do.
- **Negatives.** At the fixed floor 1 of 3 negatives returns a candidate in most variants ("ceiling fan" in
  backroom). Separation by peak relevancy is 0.80–0.96 AUROC: the score separates absent objects reasonably,
  but a single global floor does not.
- **Memory.** All variants fit 8 GB: the lift peaked at 7.4 GB on backroom (960 px renders, 532k Gaussians) and
  2.4 GB on flightroom; FMGS 5.1–7.8 GB on the device. Tables 0.66–1.36 GB.
- **Time.** Teachers dominate (31–50 min). Then lift 3–4 min ≪ F-C 12–14 min < F-CD 25–31 min. Queries take
  0.54–0.9 s on the CPU for every variant.
- **Hard queries for all variants**: thin or low-texture objects (tripod, keyboard, floor lamp, armchair) and
  colour + material phrases where a similar object exists (purple foam mat → another mat).

Caveats: 29 positive and 6 negative queries over two scenes, one training seed per FMGS variant; ground truth
placed by Claude from photos (dev-set check 0.09–0.42 m from Suhan's placements; flightroom positions
reprojection-checked); GTN_lab_v1 excluded (folded reconstruction, §12).

**Sensitivity of the lift (L-C) to the query settings** (`evaluate --sweep L-C`, one setting at a time around
the defaults; `docs/phase5_results/sweep_L-C.csv`; reported only, the defaults stay as fixed on the dev set):

| Setting | backroom hits · neg FP | flightroom hits · neg FP |
| --- | --- | --- |
| floor 0.50 / 0.525 / **0.55** / 0.575 / 0.60 | 9·2/3, 9·1/3, **8·1/3**, 7·1/3, 7·1/3 | 12·2/3, 11·1/3, **10·1/3**, 10·0/3, 9·0/3 |
| rel_alpha 0 / 0.25 / **0.5** / 0.75 | 8 each; ambiguous 0.23 → 0.15 → **0.08** → 0 | 10 each; median error 0.18 → 0.20 → **0.22** → 0.25 m |
| voxel 0.05 / **0.1** / 0.2 / 0.3 m | 8 each | 10, **10**, 10, 11 |

- The **relevancy floor is the one setting that matters**: lowering it to 0.50 finds 1–2 more objects per scene
  but doubles the false positives on absent objects (2 of 3); raising it to 0.575 removes flightroom's false
  positive at no cost there but loses one backroom hit. 0.55 sits at the balance point on both scenes.
- rel_alpha trades ambiguity (higher = fewer runner-up clusters) against localisation (higher = tighter
  selection, slightly larger error); hits do not change. The cluster voxel hardly matters (0.05–0.3 m).
- Not swept: the lift's feature width (480 vs 960 px) — it needs re-lifts on the GPU written to a separate
  table (`--feat-width` overwrites the lift table today).

Galley: `GET /api/scenes/<scene>/semantics` now returns `eval` (each variant's metrics and its failed queries,
stale flag by checkpoint key). The Semantics-tile table on the Scene page needs a frontend rebuild, and Node is
not installed on intellisense08 (ask Suhan, or build on dummy).

Still open in Phase 5: the feature-width sweep, the Semantics-tile table (frontend), and the browser check of
View ▸ Compare.
