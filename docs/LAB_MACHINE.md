# intellisense05 — FiGS lab machine reference

*Last verified: 2026-08-06, with a full capture-to-flight run. Written for whoever picks this
machine up next, including future me.*

This machine is the **primary FiGS box** and the only hardware on which the *complete* pipeline
has been run. Every other document in this folder was written against an RTX 3050 Ti laptop
(4 GB) where splat training was never attempted. Those documents' VRAM pessimism no longer
applies — but their filming guidance still does.

---

## 1. The machine

| | |
|---|---|
| Hostname | `intellisense05-EWISPro9900G` |
| User | `intellisense05` |
| LAN address | `10.8.100.30` — must be on the lab network |
| Physical | Monitor + keyboard in the lab; SSH and console are equivalent |
| GPU | **NVIDIA GeForce RTX 2080, 8192 MiB**, compute capability **7.5** |
| NVIDIA driver | 580.159.04 |
| System ffmpeg | 8.1.2 |

```bash
ssh intellisense05@10.8.100.30
```

Single user in practice — no GPU contention. Still worth `nvidia-smi` before a long run;
~106 MiB sits resident from the desktop session, which is normal.

---

## 2. Layout

```
~/FYP-Radiance/
├── docs/                          ← this file and the runbooks
├── figs/                          ← PROJECT_ROOT
│   ├── figs_env.sh                ← the only entry point. source this, always.
│   ├── figs_pipeline.py           ← the capture-to-flight script
│   ├── .figs_pipeline_state/      per-scene step markers (resume state)
│   ├── .figs_pipeline_logs/       per-scene stage logs
│   ├── miniconda3/                ← self-contained conda (env: kitchen)
│   └── SousVide/                  ← the repo
│       ├── configs/{captures,courses,frames,pilots,methods}/
│       ├── gsplats/
│       │   ├── capture/           ← source videos MUST live here
│       │   └── workspace/<scene>/ images/, sfm/, transforms.json, sparse_pc.ply
│       │       └── outputs/<scene>/splatfacto/<timestamp>/nerfstudio_models/*.ckpt
│       ├── runs/                  ← JSON run records written by figs_pipeline.py
│       └── FiGS/src/figs/         core package
├── setup_scripts/{install_figs.sh, verify_figs.sh, install.log}
└── video_captures/                ← raw footage staging area (NOT visible to FiGS)
```

**`PROJECT_ROOT` is `~/FYP-Radiance/figs`, not `~/projects/figs_validation`.** That second path
is the default baked into both `install_figs.sh` and `verify_figs.sh`, and it is wrong here.
Every invocation needs `--prefix ~/FYP-Radiance/figs` or the script reports a healthy install as
missing. This is the single most likely thing to waste an afternoon.

**`video_captures/` is a staging area only.** FiGS hardcodes its search to
`SousVide/gsplats/capture/` and globs by substring. Footage must be copied there, under an
unambiguous name — two files matching the scene string raises `ValueError`.

---

## 3. Daily use

```bash
source ~/FYP-Radiance/figs/figs_env.sh
```

Activates `kitchen`, exports `ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH`, sets
`PYTHONNOUSERSITE=1` (guards against a stray `~/.local` torch shadowing the env), and drops you
in `SousVide`. Confirm with `echo $CONDA_DEFAULT_ENV` → `kitchen`.

**For a new capture, use `figs_pipeline.py`** — one resumable script covering the whole pipeline,
with preflight checks, on-disk step state, VRAM monitoring and error diagnosis. Usage:
`docs/FiGS_pipeline_script_guide.md`.

```bash
source ~/FYP-Radiance/figs/figs_env.sh
~/FYP-Radiance/figs/figs_pipeline.py --scene <name> --video <path> --marker-length <m> --course <name>
```

`docs/figs_capture_pipeline.ipynb` is the earlier notebook version. It works, but a runtime
error mid-run leaves you hand-picking cells to re-run and a kernel restart loses in-memory
state — the script keeps its state on disk instead. Prefer the script.

---

## 4. Install provenance

Built **in place on this machine** by `setup_scripts/install_figs.sh` — not copied from the
portable SSD, not cloned from the 3050 Ti box.

**tiny-cuda-nn is compiled for exactly one GPU architecture**, here `TCNN_CUDA_ARCHITECTURES=75`
for the RTX 2080. Moving this install to a different compute capability breaks it; the fix is
`./install_figs.sh --prefix ~/FYP-Radiance/figs --redo tcnn` (15–40 min, run under `tmux`).
Full portability rules: `README_PORTABLE.md`.

Logs: `setup_scripts/install.log`, plus per-step logs in `figs/.figs_install_logs/` and
completion markers in `.figs_install_state/`. Re-running skips completed steps.

---

## 5. Verified state — 2026-08-06

`./verify_figs.sh --prefix ~/FYP-Radiance/figs --scene backroom --course circuit`
→ **22 passed, 1 warning.**

| Component | Version |
|---|---|
| torch | 2.1.2, CUDA runtime 11.8 |
| nerfstudio | 1.1.4 (`splatfacto` registered) |
| COLMAP (binary) | 3.11.1, CUDA-enabled |
| **pycolmap (python)** | **4.1.1 — see §8, this needs a pin** |
| CasADi | 3.7.2 — newer than acados officially supports; works, see §8 |
| tinycudann | ok, forward pass returns `[64, 16]` |
| gsplat | `rasterization()` ok — `(1, 128, 128, 3)` |
| hloc | commit `e334220`, 2025-03-24 (**patched** — see §8) |

### Measured performance

| Workload | Result |
|---|---|
| Flight sim, `backroom`/`circuit` (371 MB ckpt) | 76 s, **peak 635 MiB** |
| Flight sim, `intellisense` (751 MB ckpt) | 51 s for 16 s of flight |
| **Splat training, 30k steps** | **34 min, peak 5203 MiB** |

**635 MiB for simulation, out of 8192.** Simulation is cheap — roughly 7.5 GB of headroom, which
means **CLIPSeg or a comparable semantic-embedding model can run in the same process as the
simulator**, not as a separate pass. That settles the question `FiGS_test_runbook.md` left open.

**5203 MiB for training.** Fits with ~3 GB spare. *Correction (2026-09-27):* this does not mean
a 4 GB card cannot train. The figure includes the training images cached on the GPU (nerfstudio's
default); with `--cache-images cpu` the same 960×540 training of `backroom` peaked at 2360 MiB on the
RTX 3050 Ti. See `FiGS_pipeline_script_guide.md` §9.

**4K training is probably out of reach.** Forcing `--downscale-factor 1` quadruples the
rasterization buffers on top of the same Gaussian count, against ~3 GB of headroom. Recorded as
a reasoned expectation, not a tested result.

### The one warning, and why it's fine

```
sousvide    ! ModuleNotFoundError: No module named 'sousvide'
```

Expected. `sousvide` is the SOUS VIDE policy-distillation layer (SV-Net behaviour cloning),
downstream of everything FiGS does. `verify_figs.sh` treats it as a soft check. Ignore it until
the project reaches the policy-learning stage.

### Shipped example data

`backroom`, `flightroom`, `mid_gate`, `src_open` all have trained checkpoints.
**`backroom` + `circuit` is the known-good smoke test — always fly it first when something seems
wrong**, to separate "the install broke" from "my capture is bad."

---

## 6. What this machine can and cannot do

| Stage | Status |
|---|---|
| Load an existing splat and fly it | ✅ 635 MiB, 76 s |
| Custom waypoints through an existing splat | ✅ |
| Frame extraction + SfM (COLMAP/hloc) | ✅ CUDA-enabled, not VRAM-bound |
| ArUco metric alignment | ✅ |
| **Splat training (`ns-train splatfacto`)** | ✅ **34 min, 5203 MiB peak** |
| 4K-resolution training | ⚠️ untested, expected to OOM |
| SOUS VIDE policy distillation (SV-Net) | ❌ `sousvide` not installed |

Disk: budget **~8 GB per scene** at 1080p — 300 PNGs (~540 MB), SfM working directory (~1.5 GB
plus 140 MB features + 249 MB matches h5), and a 300–750 MB checkpoint.

---

## 7. Reference run — `intellisense`, 2026-08-06

The first complete capture-to-flight on this hardware. All numbers measured.

### Capture

| | |
|---|---|
| Device | Samsung S23, handheld |
| Source | 3840×2160 **HEVC** 8-bit, 60 fps, 191.7 s, 11,432 frames, **3.0 GB** |
| Transcoded to | 1920×1080 H.264, 30 fps, 5,754 frames, 273 MB (8m14s, `speed=0.388x`) |
| ArUco marker | `DICT_4X4_50`, **ID 0**, **0.18 m** side |
| Marker visibility | ~682 frames (11.9%), spread over five 10-s windows; 71% in the first 20 s |
| Marker apparent size | 49–192 px median by window (→ ~1.3–5.0 m range) |
| Config | `num_images: 300`, **`num_marked: 40`** (raised from 20 — see below) |

**Why transcode.** hloc's SuperPoint runs at `resize_max: 1024`, and nerfstudio auto-downscales
to ≤1600 px, so 4K pixels are discarded twice over. 1080p30 matches the validated config, cuts
SfM and decode cost, and avoids the training-VRAM risk. Keep the 4K original as the master.

**Why `num_marked: 40`.** Detections clustered heavily in the first 50 s, with one isolated
window at 110–120 s. That late window is what gives the RANSAC similarity fit a long baseline;
at `num_marked: 20` it would contribute only ~2 frames. Raising to 40 gives it ~4–5. The cost is
a handful of the 300 total frames.

### Pipeline timings

| Stage | Time | Notes |
|---|---|---|
| Transcode 4K→1080p | 8 min | CPU-bound HEVC decode |
| ArUco scan | 32 s | 1080p; ~10 min at 4K |
| Frame extraction | ~2 min | CPU only, VRAM flat at ~109 MiB |
| SuperPoint features | 20 s | 300 images, 14.8 it/s |
| **SuperGlue matching** | **38 min** | 44,850 exhaustive pairs, 19.6 it/s |
| COLMAP incremental mapping | 5.7 min | mostly CPU, low VRAM |
| ArUco alignment | seconds | |
| **splatfacto, 30k steps** | **34 min** | 67 ms/iter, 31 M rays/s, **peak 5203 MiB** |
| Flight sim, 16 s course | 51 s | |
| **Total** | **~1 h 40 m** | |

### Reconstruction quality

| Metric | Value | Read |
|---|---|---|
| Registered | **300 / 300 (100%)** | vs the ~90% floor |
| `num_points3D` | 40,542 | healthy |
| `mean_track_length` | 4.80 | each point in ~5 images |
| `mean_observations_per_image` | 648 | |
| `mean_reprojection_error` | 1.434 px | good (under 1.0 excellent, over 2 concerning) |
| Checkpoint | **751 MB** | ~2× `backroom` — texture-rich room, denser splat |

`Reconstructed 2 model(s)` is benign when the largest holds all images; hloc picks the largest.

### Scene geometry

Camera bounds from `transforms.json` (z-up, metres):

```
x  -3.05 .. +1.75   (span 4.79)
y  -1.54 .. +3.88   (span 5.42)
z  +0.54 .. +2.04   (span 1.51)
path length 352.3 m over 191.7 s → 1.84 m/s
```

A 4.8 × 5.4 m room with camera heights 0.54–2.04 m — plausible, which **implicitly validates the
0.18 m marker measurement**. That check is the only way to catch a `marker_length` error, since
nothing in the pipeline complains.

1.84 m/s is a brisk walk, faster than the guide's advice. It reconstructed perfectly anyway, but
it's the first parameter to relax if a future capture goes badly.

---

## 8. Known traps

### 8.1 — pycolmap is unpinned (**will recur on every rebuild**)

`install_figs.sh` does not pin `pycolmap`, so it resolves to whatever is newest on install day.
It landed on **4.1.1**, which renamed a keyword the vendored hloc still uses:

```
TypeError: import_images(): incompatible function arguments
  accepts: ..., image_names=[], options=...
  sent:    ..., image_list=[],  options=...
```

**Fix applied** — one line in
`FiGS/Hierarchical-Localization/hloc/reconstruction.py` (backup at `reconstruction.py.bak`):

```diff
-            image_list=image_list or [],
+            image_names=image_list or [],
```

The other three pycolmap call sites (`incremental_mapping`, `verify_matches`,
`triangulate_points`) pass **dict** options, which pybind converts to whatever options object
the installed version expects — so they are version-agnostic and need no patch. This was checked
explicitly; do not assume broader breakage.

**This patch lives in a git submodule.** `install_figs.sh --redo clone` re-syncs it and silently
reverts the fix. Notebook cell 7 re-applies it idempotently. The durable fix is a pin —
`pycolmap==3.11.1`, matching the COLMAP binary — in the installer's `pips` step.

*Diagnosing a future variant:* read the accepted signature straight out of the pybind error, and
check `pycolmap.<fn>.__doc__` against the hloc call site. Loose kwargs → break; dict → fine.

### 8.2 — CasADi 3.7.2 is newer than acados supports

```
Warning: officially supported CasADi versions are ... 3.6.7. Version 3.7.2 currently in use.
```

Generated working solver code on this run — no action needed. Same unpinned-dependency pattern
as pycolmap; if acados codegen ever produces broken C, this is the first suspect.

### 8.3 — z-sign convention between splat and course frames

`transforms.json` is **z-up** (camera heights positive). Course files are **z-down** —
`circuit.json` writes 0.7 m altitude as `z = -0.70`. Verified against the known-good
`backroom` + `circuit` pair:

| Axis | `backroom` cameras | `circuit` waypoints | Relation |
|---|---|---|---|
| x | −1.61 .. +1.98 | −1.40 .. +1.50 | same sign |
| y | −1.61 .. +7.48 | −7.00 .. 0.00 | **negated** |
| z | +0.48 .. +1.90 | −0.70 | **negated** |

```
course = (x, −y, −z)   relative to transforms.json
```

A 180° rotation about x — a proper rotation, so yaw carries over unchanged. Get it wrong and the
drone flies below the floor: **the sim runs without error and renders mush**, which reads as
"the splat is bad." Notebook cell 10 re-derives and cell 11 validates every waypoint.

### 8.4 — `ns-viewer` holds ~4.6 GB

Leaving the viewer running contaminates VRAM measurements and risks an OOM during training.
`Ctrl-C` releases it. Relaunch with:

```bash
cd ~/FYP-Radiance/figs/SousVide/gsplats/workspace
ns-viewer --load-config outputs/<scene>/splatfacto/<timestamp>/config.yml
# browse http://localhost:7007  (listens on 0.0.0.0; or tunnel: ssh -L 7007:localhost:7007)
```

**`cd` into `gsplats/workspace` matters** — `config.yml` holds paths relative to the directory
`ns-train` ran from.

### 8.5 — `--prefix` on every script invocation

Covered in §2, repeated because it is the most common time-waster.

### 8.6 — OpenCV 5.0.0 segfaults in forked DataLoader workers

hloc's feature extraction died with
`DataLoader worker (pid N) is killed by signal: Segmentation fault` at `0/300` — before touching
a single image, so not a data problem. RAM (58 GB free), `/dev/shm` (32 GB), disk (1.1 TB) and
every extracted PNG were all clean.

The environment is a major version ahead of what hloc and nerfstudio target: **cv2 5.0.0** against
their 4.x assumptions. `cv2.imread` inside a forked worker with `pin_memory=True` is a classic
place for that to surface as a segfault rather than an exception, and it was intermittent — the
first capture processed 300 images at 14.8 it/s without complaint.

**Fix:** `num_workers=0` in `FiGS/Hierarchical-Localization/hloc/extract_features.py` (~line 263).
The loader then runs in-process and cannot fork-crash. Costs a few seconds on a 20-second stage.
`figs_pipeline.py` applies this automatically in its `patch` step.

Also observed: the failure was more likely when launched from a Jupyter kernel than from a plain
shell, which is consistent with a fork-context problem. Prefer the shell.

Pinning `opencv-python<5` in the installer would be the durable fix.

### 8.7 — Do not casually install into `kitchen`

Every compiled extension (tiny-cuda-nn, gsplat) is built against a specific libtorch ABI. If a
package pulls a different torch they become silently invalid while still importing. The
installer guards this and will say `FATAL: torch changed X -> Y`; the response is exactly
`--redo conda_env --redo tcnn --redo pips`. Use `pip install --no-deps` when you must, and
verify `torch.__version__` is still `2.1.2` afterwards.

---

## 9. Running a new capture

**Use `docs/figs_capture_pipeline.ipynb`.** Edit cell 0 (paths, marker ID and length, scene
name), then run top to bottom. It preflights the environment, transcodes, scans the marker with
a time-distribution histogram, writes the capture config, re-applies the hloc patch, generates
the splat under VRAM monitoring, verifies registration, derives the frame conversion, validates
the course, flies it, and writes a JSON run record to `runs/`.

Every stage is idempotent — a failure part-way is resumed by re-running from the top. hloc
caches features and matches to h5, so a crash after matching costs seconds, not 38 minutes.

The four quiet failures it exists to catch, none of which raise an error on their own:

1. **Wrong `marker_length`** → metrically wrong splat. Nothing errors. Every waypoint inherits
   the scale error. Cell 10's bounds check is the only detector: do the spans look like the room?
2. **Wrong `marker_id`**, or passing the wrong `capture_cfg_name` → detector finds nothing, FiGS
   pads with unmarked frames, alignment dies an hour later with
   `Mismatched number of aruco and sfm transforms`. Cell 5 catches it in 30 seconds.
3. **Partial SfM registration** → trains a worse splat silently. Cell 9 hard-fails below 90%.
4. **Waypoints outside the captured volume** → runs fine, renders floaters exactly there.
   Cell 11 checks each keyframe.

Manual equivalents are in `docs/FiGS_intellisense_capture_runbook.md`.

---

## 10. Related documents

| File | What it's for |
|---|---|
| `figs_pipeline.py` | **The pipeline script — start here for a new capture** (lives in `figs/`, but location-independent) |
| `FiGS_pipeline_script_guide.md` | How to use it, what each step does, what to expect |
| `figs_capture_pipeline.ipynb` | Earlier notebook version; superseded by the script |
| `FiGS_intellisense_capture_runbook.md` | Manual step-by-step for this machine |
| `FiGS_custom_video_guide.md` | **Filming guidelines (Part 1)** — read before shooting; those decisions are unrecoverable |
| `FiGS_pipeline.md` | How FiGS works internally — architecture, data flow, stage I/O |
| `FiGS_setup_guide.md` | Original from-scratch install narrative |
| `FiGS_validation_notes.md` | Build errors on the 3050 Ti and their resolutions |
| `README_PORTABLE.md` | Moving the install to other hardware |
| `FiGS_rebuild_runbook.md` | Clean rebuild via `install_figs.sh` |

Everything except this file, the notebook, and the capture runbook targets the old 4 GB machine
at `~/projects/figs_validation`. **Translate paths, ignore the VRAM pessimism.**

---

## 11. Open items

- [x] ~~Splat training on this GPU~~ — **34 min, peak 5203 MiB, 2026-08-06**
- [x] ~~SfM registration rate for a self-filmed capture~~ — **300/300 (100%)**
- [x] ~~Wall-clock for `generate_gsplat`~~ — **~1 h 20 m** (38 min of it SuperGlue matching)
- [ ] Pin `pycolmap==3.11.1` and `casadi<3.7` in `install_figs.sh` so a rebuild is clean
- [ ] Carry the hloc patch as a repo-local diff applied post-clone, rather than a live edit
- [ ] Test whether `--downscale-factor 1` (1080p training) fits in 8 GB, now that 960×540
      is known to peak at 5203 MiB
- [ ] Run CLIPSeg alongside the simulator — the 635 MiB sim footprint says it should fit
- [ ] Decide whether `sousvide` is worth installing for the policy-learning stage
