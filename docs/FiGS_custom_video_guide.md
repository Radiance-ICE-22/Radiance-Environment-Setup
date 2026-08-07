# Using FiGS With Your Own Video, From Scratch

This is a practical, from-scratch guide to taking your own video of a real space and turning it into a FiGS
Gaussian Splat you can simulate a drone flight in. Every step explains what to do, how to do it, why it's needed,
and what happens downstream if you get it wrong. It assumes the environment is already set up — see
`FiGS_setup_guide.md` if not — and builds on what `FiGS_pipeline.md` documents about how the pipeline works
internally.

**Important caveat up front:** everything we've validated on this server so far (RTX 3050 Ti, 4GB VRAM) used an
*already-trained* splat. Splat *training* — the step this guide walks you into — is untested on this hardware and
is the step most likely to hit the VRAM ceiling. Section 6 covers what to do if it does.

---

## Part 1 — Filming Guidelines

Everything below feeds directly into two automated steps later in the pipeline: Structure-from-Motion (SfM, via
COLMAP/hloc) and ArUco-based metric alignment. Both are unforgiving of a bad capture in ways that are hard to fix
after the fact, so it's worth getting this part right.

### 1. The ArUco marker

**What:** A printed marker from the `DICT_4X4_50` ArUco dictionary, with a specific numeric ID (ID `0` is FiGS's
default, matching `configs/captures/default.json`).

**How:** Generate and print one — e.g. with OpenCV:
```python
import cv2
marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 800)
cv2.imwrite("marker_id0.png", marker)
```
Print it flat (no lamination glare), mount it on something rigid (foam board, clipboard) so it doesn't warp, and
**measure its printed side length precisely in meters** — this number goes directly into the capture config as
`marker_length` and is not something the pipeline can sanity-check for you.

**Why it's needed:** COLMAP's Structure-from-Motion reconstructs the scene in an arbitrary, unscaled coordinate
frame — it has no concept of real-world units or which way is "up." The ArUco marker is the *only* source of
ground truth: FiGS uses `cv2.solvePnP` on the marker's known real-world size to compute a small set of true-scale
3D positions, then RANSAC-fits a similarity transform (scale + rotation + translation) between those and SfM's
unscaled points.

**Impact on the pipeline if done wrong:** A mis-measured `marker_length` silently produces a *metrically wrong*
splat — everything reconstructs, nothing errors, but the drone's physics simulation (which assumes real meters)
will be scaled incorrectly, and if you ever deploy on real hardware, the sim-to-real gap will be exactly your
measurement error. Get this number right with an actual ruler/tape measure, not an estimate.

### 2. Placing the marker

Place it somewhere static, flat against a wall or the floor, fully visible (not at a grazing angle) from a
meaningful chunk of your camera path — ideally somewhere central enough that you pass near it multiple times
during the walkthrough, not just once at the start.

**Why:** The alignment step needs the marker correctly detected in at least `num_marked` frames (20 by default).
If it's only visible for a few seconds at the very start of your walk, you may not get enough tagged frames after
FiGS's frame-sampling step, which specifically tries to bin frames into "marker visible" vs. "not visible" and
draw the configured count from each.

### 3. Camera motion

**What/how:** Move slowly and smoothly. Walk (don't fly/spin the camera fast) around the space, covering it from
multiple heights and angles, with generous overlap between consecutive views — think "orbit the space at several
different radii and heights," not "one fast lap." If this is meant to represent a volume a drone will actually fly
through, walk the camera through that volume too, not just around its perimeter.

**Why:** COLMAP/hloc reconstructs 3D structure from *parallax* — the same point seen from multiple, sufficiently
different viewpoints. Fast motion causes motion blur (kills feature matching), and viewpoints that are too similar
to their neighbors (e.g. pure in-place rotation) don't provide the triangulation baseline SfM needs. Insufficient
overlap between frames is the single most common reason SfM fails to register a large fraction of images.

**Impact if done wrong:** COLMAP will either fail outright, register only a subset of your frames (leaving holes
in reconstructed geometry), or produce a fragmented/noisy point cloud. Because FiGS's `generate_gsplat()` doesn't
surface COLMAP's registration statistics back to you, a partial failure can be silent — the pipeline will still
"succeed" and train a splat, just a worse one. Section 6 below covers how to actually check reconstruction
quality.

### 4. Scene content

Prefer a mostly static scene with real texture — avoid moving objects (people walking through, opening doors,
curtains blowing) during the capture, since both SfM and Gaussian Splatting assume a static world. Also avoid
large textureless surfaces (blank white walls, glass, mirrors, glossy floors) where possible — they give SfM
nothing to match on and Gaussian Splatting nothing to represent well.

**Impact:** Moving objects during capture typically show up as "ghost" blurry artifacts baked into the splat.
Textureless regions either fail to reconstruct at all (holes) or get filled with low-confidence, blurry Gaussians.

### 5. Coverage of the intended flight volume

If you're building this splat to simulate a drone flying through it, make sure your capture actually covers the
airspace the drone will occupy — including higher up if it will climb, and close to any obstacles it will pass
near.

**Why:** Gaussian Splatting only renders well from viewpoints reasonably close to where training images were
taken; render quality degrades (blur, floaters, missing geometry) when you query a camera pose far outside the
training distribution. If your course's waypoints send the simulated drone through airspace the video never
covered, expect visibly bad renders — or a `Simulator` that runs without error but produces garbage frames — at
exactly those points in the flight.

### 6. Duration, resolution, and format

Any resolution/frame rate OpenCV can decode is technically fine (our validated example was 1920×1080, 30fps
H.264). One to three minutes is a reasonable target — FiGS subsamples down to a configurable frame count anyway
(300 by default), so a much longer video mostly just gives it a larger pool to sample from, with diminishing
returns and a longer frame-extraction pass. Keep the file as `.mp4` or `.mov` — both are what's been tested; OpenCV
generally handles standard H.264 containers without issue.

---

## Part 2 — Step-by-Step: From Video to Simulated Flight

### Step 1 — Get the video onto the server

**What:** Transfer your video file into `gsplats/capture/`, named so that it contains whatever you'll use as your
`scene_name` (FiGS matches by substring, e.g. a file containing `"myroom"` for `capture_name = "myroom"`).

**How:**
```powershell
scp "C:\path\to\myroom.mp4" hanzo@dummy.stargazer-haddock.ts.net:/home/hanzo/projects/figs_validation/SousVide/gsplats/capture/myroom.mp4
```

**Why here specifically:** `generate_gsplat()` hardcodes the search location as `gsplats/capture/` relative to the
repo root and does a glob match on the filename — it will error immediately (`FileNotFoundError`) if the file
isn't there or the name doesn't match what you pass as `scene_file_name`.

**Impact if skipped/wrong:** Nothing runs — this is a hard precondition, not a quality issue. If you name two
files similarly (e.g. `myroom.mp4` and `myroom_backup.mp4`), the glob match will find both and raise a `ValueError`
for ambiguity — keep capture filenames unambiguous.

### Step 2 — Decide whether you need a custom capture config

**What:** `configs/captures/default.json` controls frame extraction:
```json
{
    "camera": null,
    "extractor": {
        "num_images": 300,
        "num_marked": 20,
        "marker_length": 0.341,
        "marker_id": 0
    }
}
```

**How:** If your marker size, ID, or desired frame counts differ from the defaults, copy this file to
`configs/captures/<yourname>.json` and edit `marker_length` (meters, measured — see Part 1) and `marker_id` to
match your printed marker. `num_images` is the total frames extracted for training; `num_marked` is how many of
those must have the marker visible (used for alignment). `"camera": null` tells FiGS to trust COLMAP's own
estimated intrinsics rather than a supplied camera calibration — fine for a first attempt, but supplying real
intrinsics (fx, fy, cx, cy, distortion coefficients) here if you have them from a calibration will make alignment
more accurate.

**Why:** These are the only two per-video-shoot parameters that materially affect what gets extracted. Wrong
`marker_length` = wrong scale (see Part 1); wrong `marker_id` = the detector silently ignores your real marker
because it's looking for a different pattern, and you'll get the "no aruco markers found" fallback path instead
(FiGS will fall back to using extra unmarked frames instead — meaning the alignment step will fail outright since
it needs `num_marked` valid marker detections).

**Impact:** Get this wrong and either alignment fails with an explicit error (`Mismatched number of aruco and sfm
transforms`), or — worse — it silently succeeds with the wrong marker size and produces a metrically incorrect
splat.

### Step 3 — Run the generation pipeline

**What:** Kick off frame extraction → SfM → alignment → splat training in one call.

**How:**
```bash
conda activate kitchen
cd ~/projects/figs_validation/SousVide
python -c "
import figs.render.capture_generation as pg
pg.generate_gsplat('myroom', capture_cfg_name='default')  # or your custom config name from Step 2
"
```

**Why this is one call and not several:** The four sub-steps (extraction, SfM, alignment, training) are chained
because each depends on the previous one's output files (`transforms.json`, `sparse_pc.ply`) — there's no
supported way to run them independently through the public API, though you can inspect intermediate outputs (see
Step 4) while it's running or after it finishes each stage, since they're written to disk incrementally.

**Impact / what to expect:** This is by far the longest and most resource-intensive step in the whole workflow.
SfM (COLMAP+hloc) on ~300 images can take anywhere from several minutes to tens of minutes depending on scene
complexity and GPU. Splat training (`ns-train splatfacto`, hardcoded to run to nerfstudio's default step count)
is the VRAM-heavy part — this is the step we have **not** yet validated on the 4GB RTX 3050 Ti. Watch VRAM in a
second session:
```bash
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
If it OOMs, see Section 6 below before re-running.

### Step 4 — Verify the outputs before moving on

**What:** Confirm each stage actually produced something reasonable, rather than assuming success just because no
exception was raised (`generate_gsplat` prints COLMAP/training subprocess output but doesn't hard-fail on a
partial reconstruction).

**How:**
```bash
# How many images actually got extracted?
ls gsplats/workspace/myroom/images | wc -l

# Did SfM register most of your images? (compare to the count above)
python3 -c "
import json
tfm = json.load(open('gsplats/workspace/myroom/transforms.json'))
print(f'{len(tfm[\"frames\"])} frames registered with poses')
"

# Sparse point cloud sanity check — should be a non-trivial number of points, not near-zero
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('gsplats/workspace/myroom/sparse_pc.ply')
print(f'{len(pcd.points)} points')
"

# Did training actually complete and leave a final checkpoint?
find gsplats/workspace/outputs/myroom/splatfacto -name "*.ckpt"
```

**Why this matters:** If SfM only registered, say, 60% of your images, that's a strong sign of insufficient
overlap during filming (Part 1, Step 3) — better to catch it here and consider re-shooting than to discover it
later as unexplained render artifacts.

### Step 5 — Author a course (flight path) for the new scene

**What:** A JSON file under `configs/courses/` describing waypoints as a sequence of timed keyframes in flat-output
space (position `x, y, z` + yaw, each optionally with velocity/acceleration/jerk constraints).

**How:** Based on the structure of the existing `circuit.json` course, each keyframe looks like:
```json
{
    "waypoints": {
        "Nco": 3,
        "keyframes": {
            "start": {
                "t": 0.0,
                "fo": [
                    [ 0.0,  0.0],
                    [ 0.0,  0.0],
                    [-1.0,  0.0],
                    [ 0.0,  0.0]
                ]
            },
            "mid": {
                "t": 3.0,
                "fo": [
                    [ 1.0, null, null, null],
                    [ 1.0, null, null, null],
                    [-1.0, null, null, null],
                    [ null, null, null, null]
                ]
            },
            "end": {
                "t": 6.0,
                "fo": [
                    [ 0.0, 0.0],
                    [ 0.0, 0.0],
                    [-1.0, 0.0],
                    [ 0.0, 0.0]
                ]
            }
        }
    },
    "forces": null
}
```
Each row of `"fo"` is `[x, y, z, yaw]`; each entry can be a single value (position only, derivatives free) or a
list `[pos, vel, accel, jerk]` where `null` means "unconstrained, let the minimum-snap solver choose." `t` is the
time (seconds) the drone should reach that keyframe. `Nco` is the polynomial order used internally by the
minimum-snap trajectory generator — 6, matching the existing examples, is a safe default. **Coordinates are in the
splat's metric frame that Step 3's ArUco alignment established** — so they need to correspond to real positions in
your physical space, in meters, relative to wherever your marker put the origin.

**Why:** This is what `MinTimeSnap` (used inside `VehicleRateMPC`) turns into an actual smooth, dynamically
feasible trajectory for the MPC to track. Skipping intermediate keyframes or spacing them unrealistically close in
time relative to distance will produce a trajectory the MPC can't physically track well (aggressive accelerations,
tracking error, or infeasible solves).

**Impact on the rest of the pipeline:** This directly determines what the drone flies through and therefore what
you'll see rendered — and per Part 1 Step 5, if your waypoints go somewhere your original video capture didn't
cover, expect degraded rendering right at those points.

### Step 6 — Frame and pilot configs

**What:** `configs/frames/*.json` (drone physical specs: mass, inertia, arm lengths, motor coefficients, camera
intrinsics/mount pose) and `configs/pilots/*.json` (MPC tuning: cost weights, horizon, control-rate, input
bounds).

**How:** For a first test, just reuse the existing `carl` frame and `Viper` pilot — they're generic quadcopter
parameters, not tied to the `backroom` example specifically. Only author your own if you're modeling a specific
real drone you intend to eventually fly (matching its actual mass/geometry/camera), or want a different flight
style (e.g. gentler tracking — lower `Rk` values relax control effort penalties, higher `Qk` values tighten
position tracking).

**Why default configs are fine to start with:** The frame config's camera intrinsics/mount transform determine
what the *rendered onboard view* looks like (field of view, position on the airframe) — reasonable generic values
are enough to validate that the pipeline works end-to-end before you invest in matching a specific physical drone.

### Step 7 — Simulate

**What:** Load the trained splat and run the same simulate → render → export flow used with `backroom`.

**How:**
```python
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import figs.visualize.generate_videos as gv

sim = Simulator("myroom", "eval_single", "carl")
ctl = VehicleRateMPC("Viper", "my_course_name", "carl")

t0, tf, x0 = ctl.tXUd[0,0], ctl.tXUd[-1,0], ctl.tXUd[0,1:11]
Tro, Xro, Uro, Fro, Rgb, Dpt, Aux = sim.simulate(ctl, t0, tf, x0)

gv.images_to_mp4(Rgb, "myroom_flight.mp4", ctl.hz)
```

**Why this mirrors the `backroom` walkthrough exactly:** Once a scene has a trained splat and a course, it's
interchangeable with any other scene from `Simulator`'s perspective — this is the same code path we validated in
`figs_examples.ipynb`.

**Impact:** This is the payoff step — if Steps 1-6 were all done correctly, this produces a photorealistic flight
video of a real space you filmed yourself, with a physically simulated drone flying through it.

---

## Part 3 — VRAM Risk and What to Do If Training OOMs

Splat training (Step 3) is untested on this server's 4GB card. `generate_gsplat()`'s `ns-train splatfacto`
invocation is hardcoded without VRAM-reduction flags, so if it OOMs, you have three options, roughly in order of
effort:

1. **Reduce image resolution before training.** Lower `num_images` in your capture config won't help VRAM (it
   affects dataset size, not per-image memory) — instead, downscale the extracted images themselves, or pass
   `--downscale-factor 2` (or higher) by running `ns-train splatfacto` manually with the same arguments
   `generate_gsplat()` uses (visible in `FiGS/src/figs/render/capture_generation.py`) instead of calling the
   wrapper function, once `transforms.json`/`sparse_pc.ply` already exist from a successful SfM run.
2. **Cap the Gaussian count / use lighter splatfacto settings**, e.g.
   `--pipeline.model.cull-alpha-thresh` and related pruning flags, again by invoking `ns-train` directly.
3. **Train elsewhere.** Run Steps 1-4 (frame extraction through SfM/alignment — none of which are GPU-VRAM-heavy
   in the same way) on this server, then move the `workspace/<scene>/` folder to a machine with more VRAM just to
   run `ns-train splatfacto`, and move the resulting `outputs/<scene>/splatfacto/.../` checkpoint back. Steps 6-7
   (simulation) are the part we've confirmed fits in 4GB, so this server can still do all *simulation* work even
   if it can't do *training* for a particular scene.

If/when we actually run a custom capture through Step 3, this section should get updated with what actually
happened — right now it's a plan, not a confirmed result.
