# Runbook — `intellisense_capture_1.mp4` → trained splat → simulated flight

Target server: `intellisense05@10.8.100.30` (`intellisense05-EWISPro9900G`)
Project root: `~/FYP-Radiance/figs`   ·   Repo: `~/FYP-Radiance/figs/SousVide`   ·   Env: `kitchen`
GPU: **RTX 2080, 8 GB** — compute capability **7.5**

This is `FiGS_custom_video_guide.md` Part 2 / `FiGS_test_runbook.md` Phase 3 rewritten for
*this* server's layout and *this* capture. Two things differ from every other doc in this folder
and drive most of what follows:

1. **Paths are different.** Every existing doc targets `hanzo@dummy:~/projects/figs_validation`.
   Here `PROJECT_ROOT` is `~/FYP-Radiance/figs`, and both `install_figs.sh` and `verify_figs.sh`
   default `PROJECT_ROOT` to `$HOME/projects/figs_validation` — so **you must pass `--prefix`**
   to either script or they will look in the wrong place.
2. **8 GB, not 4 GB.** Splat training (`ns-train splatfacto`) needs ~6 GB and has never been run
   on any of your hardware. On this card it should fit. The "training will OOM" warnings
   throughout the other docs do not apply here — but Phase 3.3 keeps the fallback in case.

**Also note:** your video lives in `~/FYP-Radiance/video_captures/`, which FiGS cannot see.
It must be moved into the repo's `gsplats/capture/` (Step 2). This is a hard precondition.

---

## Phase 0 — Confirm the install works on this machine

### 0.1 — Does the env activate?

```bash
ssh intellisense05@10.8.100.30
ls -la ~/FYP-Radiance/figs/figs_env.sh
source ~/FYP-Radiance/figs/figs_env.sh
echo "$CONDA_DEFAULT_ENV"; which python; echo "$ACADOS_SOURCE_DIR"
```

**Expect:** `kitchen`, a python under `~/FYP-Radiance/figs/miniconda3/envs/kitchen`, acados
pointing at `SousVide/FiGS/acados`, and cwd already inside `SousVide`.

### 0.2 — Verify, with the right prefix

```bash
cd ~/FYP-Radiance/setup_scripts
./verify_figs.sh --prefix ~/FYP-Radiance/figs --quick
```

**Expect:** torch 2.1.2 / cuda 11.8, all imports ✔, COLMAP 3.11.1, `ns-train` → `registered`,
tinycudann forward pass ✔, and `NVIDIA GeForce RTX 2080 ... 8192 MiB`.

**The check that matters most on this box: the tinycudann forward pass.** Per
`README_PORTABLE.md` §3, tiny-cuda-nn is compiled for exactly one GPU architecture. If this
install was built elsewhere (e.g. on the 3050 Ti, compute 8.6) and moved here (2080, compute
7.5), it will fail to load or misbehave. If it fails:

```bash
cd ~/FYP-Radiance/setup_scripts
./install_figs.sh --prefix ~/FYP-Radiance/figs --redo tcnn
```

15–40 min of near-silent compilation. Run it under `tmux`.

### 0.3 — Fly a shipped scene first

Do not debug a new capture and a possibly-broken install at the same time.

```bash
./verify_figs.sh --prefix ~/FYP-Radiance/figs --scene backroom --course circuit \
  2>&1 | tee ~/verify_$(date +%F_%H%M).log
```

**Expect:** `✔ render produced: circuit.mp4`. If `no trained checkpoint for scene "backroom"`,
re-run with a scene it lists (`flightroom`, `mid_gate`, `src_open`).

---

## Phase 1 — Sanity-check the capture *before* spending an hour on it

Your marker is **not** ID 0 at 0.341 m, so the defaults are wrong for this video and the
failure mode is quiet: the detector looks for the wrong pattern, finds nothing, and FiGS takes
its "no aruco markers found" fallback — which fills the quota with unmarked frames and then
fails at alignment with `Mismatched number of aruco and sfm transforms`. Two minutes here saves
that.

Fill in your two measured values first:

```bash
MARKER_ID=<your id>              # integer, DICT_4X4_50
MARKER_LEN=<your side length>    # metres, tape-measured — NOT estimated
```

Then scan the video for the marker:

```bash
source ~/FYP-Radiance/figs/figs_env.sh
python - <<PY
import cv2
cap = cv2.VideoCapture("$HOME/FYP-Radiance/video_captures/intellisense_capture_1.mp4")
d   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
n = hits = 0
ids_seen = {}
while True:
    ok, f = cap.read()
    if not ok: break
    n += 1
    if n % 5: continue                      # every 5th frame is plenty
    corners, ids, _ = det.detectMarkers(f)
    if ids is not None:
        hits += 1
        for i in ids.flatten():
            ids_seen[int(i)] = ids_seen.get(int(i), 0) + 1
print(f"total frames      : {n}")
print(f"fps / resolution  : {cap.get(cv2.CAP_PROP_FPS):.2f} / "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"sampled frames w/ any marker: {hits}")
print(f"marker IDs seen (id: count)  : {ids_seen}")
PY
```

**Expect:** your `MARKER_ID` present with a healthy count — comfortably more than `num_marked`
(20) once you account for the 5× sampling and FiGS's own time-distributed subsampling. A few
hundred hits is good; under ~50 is a warning sign.

**If your ID doesn't appear at all:** stop. Either the marker is from a different dictionary,
or it was never in shot clearly enough. No amount of downstream config fixes this — it's a
re-shoot. See `FiGS_custom_video_guide.md` Part 1 §1–2.

**If the count is low but non-zero:** the marker was only visible briefly. You can lower
`num_marked` below 20 in the capture config, but alignment quality degrades — RANSAC has fewer
correspondences to fit scale/rotation/translation against, and a bad fit means a metrically
wrong splat. Prefer a re-shoot if it's borderline.

---

## Phase 2 — Stage the video and write the capture config

### 2.1 — Move the video into the repo

FiGS hardcodes `gsplats/capture/` relative to the repo root and globs it by substring.

```bash
cd ~/FYP-Radiance/figs/SousVide
mkdir -p gsplats/capture
cp ~/FYP-Radiance/video_captures/intellisense_capture_1.mp4 gsplats/capture/intellisense.mp4
ls -lh gsplats/capture/
```

**Why rename to `intellisense.mp4`:** the scene name you pass to `generate_gsplat()` is matched
against the filename by substring, and *multiple matches raise a `ValueError`*. If you later
add `intellisense_capture_2.mp4` next to `intellisense_capture_1.mp4`, a scene name of
`"intellisense"` would match both. One clean unambiguous name per scene avoids that entirely.
Scene name from here on: **`intellisense`**.

### 2.2 — Write the capture config

```bash
cd ~/FYP-Radiance/figs/SousVide
cat configs/captures/default.json          # see what you're starting from

cat > configs/captures/intellisense.json <<EOF
{
    "camera": null,
    "extractor": {
        "num_images": 300,
        "num_marked": 20,
        "marker_length": ${MARKER_LEN},
        "marker_id": ${MARKER_ID}
    }
}
EOF
cat configs/captures/intellisense.json
```

| Field | Meaning | Getting it wrong |
|---|---|---|
| `marker_length` | printed side length **in metres** | Silently produces a metrically wrong splat. Nothing errors. Every waypoint coordinate you author later inherits the scale error, and so does any sim-to-real transfer. |
| `marker_id` | the `DICT_4X4_50` ID | Detector finds nothing → fallback path → alignment fails with `Mismatched number of aruco and sfm transforms`. |
| `num_images` | total frames extracted for training | Fewer = faster SfM, thinner reconstruction. 300 is the validated number. |
| `num_marked` | how many of those must show the marker | Used for the metric alignment fit. Below 20, RANSAC gets shaky. |
| `camera` | `null` = trust COLMAP's estimated intrinsics | Fine for a first run. Supplying real calibration (fx, fy, cx, cy, distortion) makes alignment more accurate. |

`marker_length` is the one number in this entire pipeline that nothing can sanity-check for
you. Measure it with a tape measure, not from the PDF you printed.

---

## Phase 3 — Generate the splat

### 3.1 — Run it

This chains frame extraction → SfM (hloc + COLMAP) → ArUco metric alignment → `ns-train
splatfacto`, in one call. Expect **1–2 hours** end to end; the training tail is the long part.

Run it under `tmux` — a dropped SSH session kills it.

```bash
tmux new -s gsplat
source ~/FYP-Radiance/figs/figs_env.sh
cd ~/FYP-Radiance/figs/SousVide

python -c "
import figs.render.capture_generation as pg
pg.generate_gsplat('intellisense', capture_cfg_name='intellisense')
" 2>&1 | tee ~/gsplat_intellisense_$(date +%F_%H%M).log
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t gsplat`.

In a second SSH session, watch VRAM the whole time:

```bash
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Record the peak.** This is the first time splat training has been run on any of your
hardware — the number belongs in `FiGS_validation_notes.md` under "Still untested", which is
currently a plan rather than a finding. Expect roughly 6 GB of the 8 GB during splatfacto;
long silences during SfM and training are normal (neither prints progress continuously).

Note the second argument is `capture_cfg_name='intellisense'` — the config from 2.2, **not**
`'default'`. Passing `'default'` here is the single easiest way to silently get a wrong-scale
splat.

### 3.2 — Verify each stage actually worked

`generate_gsplat()` does not surface COLMAP's registration statistics and does not hard-fail on
a partial reconstruction. A 60%-registered SfM run trains a worse splat and tells you nothing.

```bash
cd ~/FYP-Radiance/figs/SousVide

# frames extracted
ls gsplats/workspace/intellisense/images | wc -l

# frames SfM actually registered a pose for — compare to the number above
python -c "
import json
t = json.load(open('gsplats/workspace/intellisense/transforms.json'))
print(f'{len(t[\"frames\"])} frames registered')
"

# sparse point cloud — should be tens of thousands, not near-zero
python -c "
import open3d as o3d
p = o3d.io.read_point_cloud('gsplats/workspace/intellisense/sparse_pc.ply')
print(f'{len(p.points)} points')
"

# training checkpoint
find gsplats/workspace/outputs/intellisense -name '*.ckpt'
```

**Expect:** ~300 images, **≥90% registered**, a healthy point cloud, and a
`nerfstudio_models/step-000029999.ckpt` around 300–400 MB.

**Registration below ~90% means insufficient overlap during filming.** Re-shoot rather than
proceeding — everything downstream inherits the holes, and they surface later as unexplained
render artifacts that read as "the splat is bad."

### 3.3 — If training OOMs anyway

Unlikely at 8 GB, but if it happens, SfM output is already on disk so you don't repeat it:

1. Run `ns-train splatfacto` manually with `--downscale-factor 2`, reusing the existing
   `transforms.json` / `sparse_pc.ply`. Copy the exact arguments from
   `FiGS/src/figs/render/capture_generation.py` and add the flag.
2. Cap Gaussian count via `--pipeline.model.cull-alpha-thresh` and related pruning flags.
3. Train on a bigger GPU: move `gsplats/workspace/intellisense/` off, run `ns-train` there,
   move the checkpoint back. Simulation is far lighter and stays here.

---

## Phase 4 — Author a course inside the captured volume

### 4.1 — Find out where the splat actually is

Waypoints are in the splat's metric frame — the frame your ArUco marker established, origin at
the marker. You have no intuition for those coordinates yet, so read them off the camera poses
rather than guessing:

```bash
source ~/FYP-Radiance/figs/figs_env.sh
cd ~/FYP-Radiance/figs/SousVide
python - <<'PY'
import json, numpy as np
tf = json.load(open('gsplats/workspace/intellisense/transforms.json'))
P = np.array([f['transform_matrix'] for f in tf['frames']])[:, :3, 3]
print(f'{len(P)} camera poses')
for i, ax in enumerate('xyz'):
    print(f'  {ax}: {P[:,i].min():+.2f} .. {P[:,i].max():+.2f}   (median {np.median(P[:,i]):+.2f})')
PY
```

These bounds are where the camera *went*, which is the region the splat renders well from.
Keep every waypoint comfortably inside them — ideally well inside, not hugging the edges.
Remember **z is negative-up**: 1.2 m altitude is `z = -1.2`.

### 4.2 — Write the course

Start from your existing `square_loop.json` and shrink it to fit the bounds you just printed:

```bash
scp "D:\Projects\FYP\figs_validation\square_loop.json" \
    intellisense05@10.8.100.30:~/FYP-Radiance/figs/SousVide/configs/courses/intellisense_loop.json
```

`square_loop.json` spans x,y ∈ [-1.5, 1.5], z ∈ [-1.2, -1.0]. If the printed bounds don't
comfortably contain that box, edit the `fo` values before flying. Each keyframe row is
`[x, y, z, yaw]`; a scalar means "position only, derivatives free," a list is
`[pos, vel, accel, jerk]` with `null` meaning "let the minimum-snap solver choose." `t` is the
time in seconds to reach that keyframe.

Flying outside the captured volume runs without error and renders blurry floaters exactly at
the offending waypoints — which looks like a bad splat but isn't.

---

## Phase 5 — Simulate

```bash
source ~/FYP-Radiance/figs/figs_env.sh
cd ~/FYP-Radiance/figs/SousVide
nvidia-smi --query-gpu=memory.used --format=csv    # baseline, expect ~0 MiB

cat > script_intellisense.py <<'PY'
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.visualize.generate_videos as gv

sim = Simulator('intellisense', 'eval_single', 'carl')
ctl = VehicleRateMPC('Viper', 'intellisense_loop', 'carl')

t0, tf, x0 = ctl.tXUd[0,0], ctl.tXUd[-1,0], ctl.tXUd[0,1:11]
Tro, Xro, Uro, Fro, Rgb, Dpt, Aux = sim.simulate(ctl, t0, tf, x0)

gv.images_to_mp4(Rgb, 'intellisense_flight.mp4', ctl.hz)
PY

time python script_intellisense.py
```

`carl` (frame) and `Viper` (pilot) are generic quadcopter configs, not tied to `backroom` —
reuse them for the first run. Only author your own when modelling a specific physical drone.

**Expect:** acados codegen chatter on the first run for a new course (it compiles C for the
solver; cached afterwards), then the sim, then `intellisense_flight.mp4` in the repo root.

```bash
ls -lh intellisense_flight.mp4
nvidia-smi --query-gpu=memory.used --format=csv    # expect back to ~0 MiB
```

Pull it back to look at it — from PowerShell on your Windows box:

```powershell
scp intellisense05@10.8.100.30:~/FYP-Radiance/figs/SousVide/intellisense_flight.mp4 "D:\Projects\FYP\figs_validation\"
```

### Failure modes

| Symptom | Cause |
|---|---|
| `FileNotFoundError` on the video | not in `gsplats/capture/`, or scene name isn't a substring of the filename — Step 2.1 |
| `ValueError` ambiguous match | two files in `gsplats/capture/` match the scene name — Step 2.1 |
| `Mismatched number of aruco and sfm transforms` | wrong `marker_id`/`marker_length`, or too few marker-visible frames — Phase 1 |
| SfM registers well under 90% | insufficient overlap / motion blur during filming — re-shoot |
| `FileNotFoundError` on the course | course JSON not in `configs/courses/` — Step 4.2 |
| acados solver infeasible | keyframes too aggressive for the time budget — stretch `t` values and retry |
| Runs fine, video blurry / full of floaters | waypoints outside the captured volume — Step 4.1 |
| tinycudann import or forward-pass failure | built for a different GPU architecture — `--redo tcnn`, Step 0.2 |

---

## Numbers worth recording

Three of these don't exist anywhere in your notes yet:

- **Whether splat training completes on the RTX 2080, and peak VRAM during it** (Phase 3.1).
  This is the open question the entire "Still untested" section of `FiGS_validation_notes.md`
  is waiting on.
- **SfM registration rate** for a capture you filmed yourself (Phase 3.2) — the empirical
  quality bar for your filming technique.
- **Wall-clock for the full `generate_gsplat()` run** and for a single simulated flight
  (Phase 5) — prices out behaviour-cloning data collection later.
