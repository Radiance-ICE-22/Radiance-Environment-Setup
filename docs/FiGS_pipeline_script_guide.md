# `figs_pipeline.py` — usage guide

One resumable script that takes a phone video of a room and returns a rendered drone flight
through a Gaussian-Splat reconstruction of it.

Replaces `figs_capture_pipeline.ipynb`. Same phases, but a notebook keeps its state in a live
kernel — one runtime error and you are hand-picking which cells to re-run, and a kernel restart
loses everything not written to disk. This script keeps state **on disk**, so an interrupt costs
you only the step that was running.

*Written against the validated run on `intellisense05` (RTX 2080, 8 GB), 2026-08-06/07.*

---

## 1. Setup

```bash
scp figs_pipeline.py intellisense05@10.8.100.30:~/FYP-Radiance/figs/
ssh intellisense05@10.8.100.30
chmod +x ~/FYP-Radiance/figs/figs_pipeline.py
```

**The script's location does not matter.** It resolves `PROJECT_ROOT` at runtime rather than
assuming a path — in order: `--project-root`, then `$FIGS_PROJECT_ROOT`, then derived from
`$ACADOS_SOURCE_DIR` (which `figs_env.sh` always exports as `<root>/SousVide/FiGS/acados`), then
by probing next to itself, and only then the conventional path. A valid root is one containing
both `figs_env.sh` and `SousVide/`.

That matters because a hardcoded default is exactly what makes `install_figs.sh` and
`verify_figs.sh` report a healthy install as missing on this machine — they assume
`~/projects/figs_validation` and need `--prefix` every time. This script does not.

The resolved root is printed on every run; check that line first if paths look wrong.

**Every invocation must be preceded by:**

```bash
source ~/FYP-Radiance/figs/figs_env.sh
```

`conda activate kitchen` is *not* sufficient. It gives you the right Python but none of
`ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`, or `PYTHONNOUSERSITE`. Without them the run proceeds
happily until `Simulator()` dies on `libqpOASES_e.so: cannot open shared object file` — about
90 minutes in. `preflight` now catches this in the first few seconds by loading the library
**by soname**, which exercises the same loader search that acados will.

`LD_LIBRARY_PATH` is read once by glibc at process start, so this genuinely cannot be repaired
from inside a running process. Re-source and re-run.

**Use `tmux`.** The `gsplat` step is ~80 minutes and a dropped SSH session kills it.

```bash
tmux new -s figs
# detach: Ctrl-b then d      reattach: tmux attach -t figs
```

A tmux session belongs to the user, not the connection — you can detach over SSH and reattach at
the machine's physical console (`tmux attach -d -t figs` to steal it from the other client).

---

## 2. Quickstart

```bash
source ~/FYP-Radiance/figs/figs_env.sh

~/FYP-Radiance/figs/figs_pipeline.py \
  --scene lab3 \
  --video ~/FYP-Radiance/video_captures/lab3_raw.mp4 \
  --marker-id 0 \
  --marker-length 0.18 \
  --course lab3_loop
```

That runs all 13 steps. On a first run for a new room it will **stop at `course`**, because the
course file does not exist yet and the waypoints depend on bounds you cannot know until the
splat is aligned. That is expected — see §6.

### Choosing `--scene`

FiGS globs `gsplats/capture/` **by substring**, so a scene name that is a substring of another
scene's staged file makes *that* scene un-runnable with an ambiguous-match `ValueError`.

`intellisense` matches `intellisense_02.mp4`. `lab3` matches neither. Pick disjoint names.

The script checks for collisions in the `transcode` step and refuses rather than letting FiGS
fail later.

---

## 3. Files — where input goes, what each step writes

### Input

| What | Where | Notes |
|---|---|---|
| Raw video from the phone | `~/FYP-Radiance/video_captures/<anything>.mp4` | **Staging only — FiGS cannot see this directory.** Pass the path with `--video`; the `transcode` step copies it into the repo. |
| Printed ArUco marker | physical world | `DICT_4X4_50`, ID passed via `--marker-id`, side length in metres via `--marker-length` |
| Course file *(authored by you)* | `SousVide/configs/courses/<course>.json` | Written after the `bounds` step — see §6 |

Everything else the script creates itself. `PROJECT_ROOT` is `~/FYP-Radiance/figs` and
`REPO` is `$PROJECT_ROOT/SousVide`; all paths below are relative to `REPO` unless stated.

### Outputs by step

| Step | Writes | Path | Format | Size |
|---|---|---|---|---|
| `preflight` | — | | | |
| `probe` | facts only, into the run record | | | |
| `transcode` | staged capture | `gsplats/capture/<scene>.mp4` | H.264 MP4 | ~270 MB |
| `aruco` | histogram only, into the run record | | | |
| `config` | capture config | `configs/captures/<scene>.json` | JSON | <1 KB |
| `patch` | edited submodule + backup | `FiGS/Hierarchical-Localization/hloc/{reconstruction,extract_features}.py{,.bak}` | Python | — |
| **`gsplat`** | *see breakdown below* | `gsplats/workspace/<scene>/` and `gsplats/workspace/outputs/<scene>/` | | **~7 GB** |
| `verify` | — (reads only) | | | |
| `bounds` | — (reads only) | | | |
| `course` | — (reads `configs/courses/<course>.json`) | | | |
| `simulate` | flight video | `<scene>_flight.mp4` *(repo root)* | H.264 MP4 | ~1 MB |
| `validate` | — (reads only) | | | |
| `record` | run record | `runs/<scene>_<timestamp>.json` | JSON | ~2 KB |

### What `gsplat` produces

This is where essentially all the disk goes. Sizes are from the reference 300-image 1080p run.

| Artefact | Path | Format | Size | Purpose |
|---|---|---|---|---|
| Extracted frames | `gsplats/workspace/<scene>/images/frame_*.png` | PNG ×300 | ~540 MB | training images |
| Downscaled copies | `gsplats/workspace/<scene>/sfm/images_{2,4,8}/` | PNG | ~400 MB | nerfstudio auto-downscale; training actually uses 960×540 |
| SuperPoint features | `gsplats/workspace/<scene>/sfm/colmap/features.h5` | HDF5 | **140 MB** | cached — makes a resume cheap |
| SuperGlue matches | `gsplats/workspace/<scene>/sfm/colmap/matches.h5` | HDF5 | **249 MB** | cached — this is the 38-minute stage |
| COLMAP database + model | `gsplats/workspace/<scene>/sfm/colmap/` | `.db` + binary | ~1.5 GB | SfM working directory |
| **Camera poses** | `gsplats/workspace/<scene>/transforms.json` | JSON | 260 KB | per-frame 4×4 camera-to-world + intrinsics, **metrically aligned**. This is what `bounds` reads. |
| **Sparse point cloud** | `gsplats/workspace/<scene>/sparse_pc.ply` | binary PLY | 1.1–1.4 MB | splatfacto initialisation |
| Training config | `gsplats/workspace/outputs/<scene>/splatfacto/<ts>/config.yml` | YAML | small | frozen at train time; **`ns-viewer --load-config` points here** |
| Dataparser transform | `…/<ts>/dataparser_transforms.json` | JSON | small | nerfstudio's own normalisation |
| **Trained splat** | `…/<ts>/nerfstudio_models/step-000029999.ckpt` | PyTorch `.ckpt` | **371–751 MB** | the Gaussians: positions, scales, rotations, opacities, spherical-harmonic colour coefficients |
| Stage log | `$PROJECT_ROOT/.figs_pipeline_logs/<scene>/gsplat_<ts>.log` | text | ~1 MB | full subprocess output |

`<ts>` is a timestamp like `2026-08-07_103309`. **Only one such directory may exist per scene** —
`verify` hard-fails otherwise, because FiGS globs for `config.yml` and will not guess.

### Script bookkeeping

| What | Path | Format |
|---|---|---|
| Step completion markers | `$PROJECT_ROOT/.figs_pipeline_state/<scene>/<step>.done` | text: parameter fingerprint + timestamp |
| Carried-over results | `$PROJECT_ROOT/.figs_pipeline_state/<scene>/results.json` | JSON |
| Per-step logs | `$PROJECT_ROOT/.figs_pipeline_logs/<scene>/` | text |

Deleting a `.done` marker is equivalent to `--redo <step>`.

Note these live under **`PROJECT_ROOT`**, not next to the script — so moving
`figs_pipeline.py` never orphans your progress.

### Disk budget and what is safe to delete

Roughly **7–8 GB per scene** at 1080p. If you need to reclaim space after a successful run:

| Safe to delete | Reclaims | Cost |
|---|---|---|
| `sfm/colmap/` (database + model) | ~1.5 GB | cannot resume SfM; a re-run repeats COLMAP mapping (~6 min) |
| `sfm/*.h5` | ~390 MB | a re-run repeats SuperGlue matching (**38 min**) |
| `sfm/images_{2,4,8}/` | ~400 MB | nerfstudio regenerates them if you retrain |

**Do not delete** `images/`, `transforms.json`, `sparse_pc.ply`, or anything under
`outputs/<scene>/` — the first three are needed to retrain without re-extracting, and the last is
the splat itself. `ns-viewer` and `Simulator` both resolve dataset paths recorded in `config.yml`,
so moving `images/` breaks them.

---

## 4. What each step does, and what to expect

| # | Step | Typical time | GPU | What it produces |
|---|---|---|---|---|
| 1 | `preflight` | 5 s | — | nothing; fails fast on a bad environment |
| 2 | `probe` | 1 s | — | source video facts, transcode decision |
| 3 | `transcode` | 8 min | — | `gsplats/capture/<scene>.mp4` |
| 4 | `aruco` | 30 s | — | marker ID confirmation + time histogram |
| 5 | `config` | instant | — | `configs/captures/<scene>.json` |
| 6 | `patch` | 1 s | — | re-applies two hloc submodule fixes |
| 7 | **`gsplat`** | **~80 min** | ✔ | images, SfM, `transforms.json`, `.ckpt` |
| 8 | `verify` | 5 s | — | registration rate, point count |
| 9 | `bounds` | 1 s | — | capture extent, waypoint box |
| 10 | `course` | 1 s | — | per-keyframe inside/outside check |
| 11 | `simulate` | 1 min | ✔ | `<scene>_flight.mp4` |
| 12 | `validate` | 5 s | — | confirms the MP4 is a real render |
| 13 | `record` | instant | — | `runs/<scene>_<timestamp>.json` |

`--list-steps` prints this list with the parameters each step depends on.

### `preflight`

Checks conda env, acados environment variables, shared-library resolution, nine imports with a
torch pin, five binaries, and the GPU.

**Expect:** all ✔, and a warning if VRAM is already in use. `ns-viewer` holds ~4.6 GB — close it
before `gsplat` or the peak measurement is meaningless and you risk an OOM.

**`sousvide` is not checked.** It is the SOUS VIDE policy-distillation layer, downstream of
everything FiGS does, and is not installed here. Its absence is expected.

### `probe`

Reports codec, pixel format, resolution, frame rate, duration, size. Warns on non-8-bit pixel
formats (HDR/10-bit decodes slowly and sometimes with shifted colour) and detects variable frame
rate by comparing nominal fps against `nb_frames / duration`.

### `transcode`

Normalises to 1080p30 H.264 in `gsplats/capture/<scene>.mp4`.

**Why downscale 4K.** hloc's SuperPoint runs at `resize_max: 1024` and nerfstudio auto-downscales
to ≤1600 px, so 4K pixels are discarded twice over. You would pay 4× the SfM cost and ~3× the
disk to train at the same 960×540. Keep the 4K original as your master.

**Expect:** `speed≈0.4x` against 4K HEVC — about 8 minutes for 3 minutes of footage. ffmpeg
reports `drop=N` when converting 60→30 fps; that is the decimation working, not an error. A
completed encode ends with `Lsize=` and a libx264 statistics block.

Writes to `.partial` and renames on success, so an interrupt cannot leave a truncated file that
a later existence check would accept.

### `aruco`

Scans every second frame for `DICT_4X4_50` markers and prints a **10-second histogram** with
median apparent size.

**Expect:** your marker ID dominant; one or two other IDs with single-digit counts are ArUco
false positives and are ignored.

**Read three things:**

- **Total detections** — needs to comfortably exceed `--num-marked`. A pool 3× the requirement is
  the floor; the reference run had ~680 against 40.
- **Spread** — how many 10-second windows are non-empty. Three or more is fine. All detections in
  one window means every ground-truth correspondence comes from one vantage, and the RANSAC
  similarity fit becomes noise-sensitive: scale error then propagates into every waypoint.
- **Median apparent size** — under ~40 px means corner localisation is imprecise. The reference
  run spanned 49–192 px, roughly 1.3–5.0 m from an 18 cm marker.

**Hard-fails** if the marker ID never appears. No configuration change fixes that; it is a
re-shoot.

### `config`

Writes the capture config. Two numbers matter:

- **`--marker-length`** — metres, the **black square's** side, not the paper or the white quiet
  zone. This is the only value in the entire pipeline that nothing downstream can check. A wrong
  value reconstructs perfectly at the wrong scale, and every waypoint coordinate inherits the
  error. Use a tape measure.
- **`--marker-id`** — wrong ID means the detector silently finds nothing, FiGS pads the quota with
  unmarked frames, and alignment dies an hour later with
  `Mismatched number of aruco and sfm transforms`.

`--num-marked` defaults to **40**, raised from FiGS's default of 20. If detections cluster in
time, a distant cluster contributes only ~2 frames at 20 — and that cluster is exactly what gives
the alignment fit a long baseline. Raising it costs a handful of the 300 total frames.

### `patch`

Re-applies two fixes to git submodules, every run, because `install_figs.sh --redo clone` reverts
them. `.bak` files are kept.

1. **`reconstruction.py`** — newer pycolmap renamed `import_images(image_list=)` to
   `image_names=`. Detected by inspecting the installed signature, so it is correct on either
   version.
2. **`extract_features.py`** — sets the DataLoader to `num_workers=0`. See §7.

### `gsplat` — the long one

Frame extraction → hloc SuperPoint → SuperGlue matching → COLMAP incremental mapping → ArUco
metric alignment → `ns-train splatfacto` to 30,000 steps. Runs as a subprocess so a crash cannot
poison the parent; VRAM is sampled in the background.

**Reference timings (300 images, 1080p):**

| Sub-stage | Time | Signal |
|---|---|---|
| Frame extraction | ~2 min | CPU only, VRAM flat at ~109 MiB |
| SuperPoint features | 20 s | 14.8 it/s progress bar |
| **SuperGlue matching** | **38 min** | 44,850 pairs at 19.6 it/s |
| COLMAP mapping | 5.7 min | mostly CPU, sparse output |
| ArUco alignment | seconds | |
| **splatfacto** | **34 min** | step counter, 67 ms/iter, **peak 5203 MiB** |

**Long silences are normal**, especially during COLMAP mapping. Low VRAM there is not a hang —
check CPU with `htop` if you want reassurance. hloc caches features and matches to h5, so an
interrupt after matching resumes in seconds rather than repeating 38 minutes.

### `verify`

**Hard-fails below 90% registration.** That threshold matters: `generate_gsplat()` does not
surface COLMAP's statistics and does not fail on a partial reconstruction — it silently trains a
worse splat. Below ~90% means insufficient overlap or motion blur while filming, and everything
downstream inherits the holes. Re-shoot rather than proceeding.

Also **hard-fails if more than one trained model exists** under `outputs/<scene>/`. FiGS requires
exactly one `config.yml` per scene and will not guess. Retraining a scene leaves the previous
timestamped directory behind; the error prints the `mv` command to archive it.

**Reference:** 300/300 registered, 40,542 sparse points, mean track length 4.80, mean reprojection
error 1.434 px, 751 MB checkpoint.

### `bounds`

Prints where the camera actually went, in metres, and converts to the flight frame.

**The sign convention is the trap in this whole pipeline.** `transforms.json` is **z-up** — camera
heights are positive. Course files are **z-down** — `circuit.json` writes 0.7 m altitude as
`z = -0.70`. Derived from the known-good `backroom` + `circuit` pair:

```
course = (x, −y, −z)      # 180° rotation about x; yaw carries over unchanged
```

Get it wrong and the drone flies *below the floor*: the simulation runs without error and renders
mush, which reads as "the splat is bad."

**This is also your only check on `--marker-length`.** If the printed spans look like the room you
filmed, the measurement was right. If they come back at 0.3 m or 40 m, it is wrong by that ratio.

**Reference:** 4.79 × 5.42 m footprint, camera heights 0.54–2.04 m, 352 m of path over 191 s
(1.84 m/s — brisk; slower walking is the first thing to relax if a capture reconstructs poorly).

### `course`

Maps every keyframe back into splat coordinates and checks it against the camera bounding box.

**Refuses to fly if any waypoint is outside**, unless `--allow-outside`. This was the last silent
failure: waypoints beyond the captured volume produce a clean run and a video of blurry floaters
at exactly those points.

Note the bounds are where the **camera travelled**, not a map of free space. Furniture does not
appear in a bounding box — check the course against the splat in `ns-viewer` too.

### `simulate`

Loads the splat, builds the ACADOS MPC around a minimum-snap trajectory, runs the closed loop:
MPC solves body rates → 9-DOF dynamics integrates → the splat renders the onboard camera at every
control step.

**Expect** acados C codegen on the first run for a new course — slow once, cached after. Reference:
51 s for a 16 s flight against a 751 MB checkpoint. Simulation VRAM is small: 635 MiB for a 371 MB
checkpoint, leaving ~7.5 GB of headroom on this card.

Reports MPC tracking error; over 0.5 m max suggests the trajectory is too aggressive — stretch the
course `t` values.

### `validate`

Confirms the MP4 is a real render, not a valid file full of nothing.

- File exists and is non-empty
- Frame count in the file matches what the simulator produced (catches truncated writes)
- Duration matches the course
- **Pixel standard deviation** — below 5 means near-uniform frames, i.e. the camera is outside the
  splat entirely. Usually a sign-convention error in the course.
- **Near-black frame count** — a cluster localises which waypoints left the captured volume

### `record`

Writes `runs/<scene>_<timestamp>.json` with every measurement, and prints the `ns-viewer` command
for the new splat.

---

## 5. Resuming, and what is preserved

State lives in `<PROJECT_ROOT>/.figs_pipeline_state/<scene>/`. A step writes its `.done` marker
**only after succeeding**, so:

- **Ctrl-C** → marker not written → that step re-runs, everything before it is skipped
- **Crash** → same
- **Re-running the identical command is always the correct resume action** — the script prints it
  on exit

Three further protections:

**Atomic writes.** The transcode and the output MP4 are written to `.partial` and renamed on
success. A killed process cannot leave a truncated file that a later existence check accepts.

**Parameter fingerprints.** Each step hashes the arguments it depends on. Change
`--marker-length` and the `config` and `gsplat` steps invalidate themselves rather than reusing a
splat built at the old scale. You will see
`'gsplat' was completed with different parameters — redoing`.

**hloc's h5 cache.** Interrupting during SuperGlue matching resumes from where it stopped.

```bash
figs_pipeline.py --scene lab3 --status     # what's done, and when
```

Deleting a `.done` marker by hand is equivalent to `--redo`.

---

## 6. First run for a new room

The course cannot be authored until the splat exists, because waypoints live in a metric frame you
do not know in advance.

```bash
# 1. everything up to and including the bounds report
figs_pipeline.py --scene lab3 --video ~/.../lab3_raw.mp4 \
    --marker-length 0.18 --stop-after bounds

# 2. write configs/courses/lab3_loop.json using the printed box.
#    z is NEGATIVE: 1.1 m altitude → z = -1.1

# 3. resume
figs_pipeline.py --scene lab3 --video ~/.../lab3_raw.mp4 \
    --marker-length 0.18 --course lab3_loop --from course
```

If the new capture is the **same room with the same marker**, the coordinate frames match and an
existing course works as-is. Two independent runs of the reference room agreed to within 2–5 cm on
every axis — that is also your noise floor when comparing captures.

---

## 7. Flags

| Flag | Default | Notes |
|---|---|---|
| `--project-root` | auto | resolved from `$ACADOS_SOURCE_DIR` / script location; override only if detection fails |
| `--scene` | *required* | must not be a substring of another scene's staged filename |
| `--video` | — | required until `transcode` has run once |
| `--marker-id` | `0` | `DICT_4X4_50` |
| `--marker-length` | `0.18` | **metres, tape-measured, black square only** |
| `--num-images` | `300` | total training frames |
| `--num-marked` | `40` | of those, marker-visible; raised from FiGS's 20 |
| `--width/--height/--fps` | `1920/1080/30` | transcode target |
| `--course` | `intellisense_loop` | under `configs/courses/` |
| `--frame` / `--pilot` / `--method` | `carl` / `Viper` / `eval_single` | generic; fine to reuse |
| `--margin` | `0.5` | waypoint-box inset from the capture extent, metres |
| `--dataloader-workers` | `0` | `0` avoids the OpenCV fork segfault |
| `--allow-outside` | off | fly even if waypoints leave the capture |
| `--redo STEP` | — | repeatable |
| `--from` / `--only` / `--stop-after` | — | step selection |
| `--status` / `--list-steps` | — | inspection only |

---

## 8. Troubleshooting

Errors are matched against a table of known failure modes and printed with a remedy rather than a
bare traceback. All of these were hit for real on this machine.

| Symptom | Cause and fix |
|---|---|
| `cannot open shared object file: libqpOASES_e.so` | Started without `source figs_env.sh`. Re-source and re-run — it cannot be fixed in-process. |
| `DataLoader worker ... segmentation fault` | OpenCV 5.0.0 under fork. `--dataloader-workers 0` (the default) prevents it. Also more common when launched from a Jupyter kernel than a plain shell. |
| `import_images(): incompatible function arguments` | pycolmap renamed a kwarg. The `patch` step fixes it; if it recurs, `--redo patch`. |
| `returned multiple configurations` | Two trained models for one scene. Archive one out of `outputs/`; the error prints the command. |
| `Mismatched number of aruco and sfm transforms` | Wrong `--marker-id`, or too few marker-visible frames. Re-read the `aruco` histogram. |
| `Could not reconstruct any model` | Insufficient overlap or motion blur. Re-shoot. |
| Registration below 90% | Same cause, milder. Re-shoot rather than train on holes. |
| `ValueError` ambiguous match | Two files in `gsplats/capture/` contain the scene string. Rename to disjoint names. |
| `CUDA out of memory` | Close `ns-viewer` (~4.6 GB). If it persists during training, run `ns-train` manually with `--downscale-factor 2`, reusing the existing SfM output. |
| `FATAL: torch changed` | **Stop.** Something moved torch off 2.1.2, invalidating tiny-cuda-nn and gsplat. `install_figs.sh --redo conda_env --redo tcnn --redo pips`. |
| Video plays but is uniform grey | `validate` catches this. Course sign convention — z must be negative. |
| Video has blurry floaters at specific points | Those waypoints left the captured volume. Shrink the course. |
| `captum requires torch>=2.3` from `pip check` | Red herring. Unused transitive nerfstudio dependency. "Fixing" it would break the install. |

Full logs: `<PROJECT_ROOT>/.figs_pipeline_logs/<scene>/`. Note tqdm's carriage returns make these
look like binary files to `grep` — use `tr -d '\000' < log | sed 's/\r/\n/g'`.

---

## 9. Reference numbers

From the validated `intellisense` run, RTX 2080 8 GB, 2026-08-06.

| | |
|---|---|
| Source | Samsung S23, 4K60 HEVC, 3.0 GB, 191.7 s |
| Transcoded | 1080p30 H.264, 273 MB |
| Marker | `DICT_4X4_50` ID 0, 0.18 m |
| Registration | 300 / 300 (100%) |
| Sparse points | 40,542 |
| Reprojection error | 1.434 px |
| Checkpoint | 751 MB |
| **Training peak VRAM** | **5203 MiB / 8192** |
| Training time | 34 min |
| Simulation VRAM | 635 MiB (371 MB ckpt) |
| Total pipeline | ~1 h 40 m |

**Implications.** Training fits on 8 GB with ~3 GB spare — the 4 GB card would have OOMed.
Simulation is cheap enough that CLIPSeg or a comparable model can run in the same process.
4K training is expected to exceed the budget and has not been attempted.

---

## 10. See also

| File | |
|---|---|
| `LAB_MACHINE.md` | Machine reference, install provenance, known traps |
| `FiGS_custom_video_guide.md` | **Filming guidelines — read before shooting** |
| `FiGS_pipeline.md` | How FiGS works internally |
| `README_PORTABLE.md` | Moving the install to other hardware |

Everything not listed above targets the old 4 GB machine at `~/projects/figs_validation` —
translate paths and ignore the VRAM pessimism.
