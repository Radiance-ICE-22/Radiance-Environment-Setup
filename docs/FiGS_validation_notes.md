# FiGS Validation Notes

Source paper: "SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum" (arXiv 2412.16346), Stanford MSL.
FiGS ("Flying in Gaussian Splats") is the simulator component: couples a lightweight 9-DOF drone dynamics model with a
Gaussian Splatting scene reconstruction, rendering up to 130 fps. Trained from a short video + single ArUco tag.

## Repos
- Core lib: https://github.com/StanfordMSL/FiGS (submodule, includes `acados` for MPC)
- Examples/entry point: https://github.com/StanfordMSL/FiGS-Examples (GPL-3.0)
- Full pipeline (sim + distillation + SV-Net policy): https://github.com/StanfordMSL/SousVide

## Target server (home lab, per server_documentation.md + live check)
| Component | Value |
|---|---|
| CPU | i7-12700H, 20 threads |
| RAM | 16GB |
| GPU | RTX 3050 Ti Mobile — **4GB VRAM** |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | **6.17.0-35-generic — pinned, do not let this drift** (see Host constraints below) |
| NVIDIA driver | 570.211.01, CUDA 12.8 (system) |
| Disk free | ~422GB free (46G/468G used) |
| SSH | hanzo@dummy.stargazer-haddock.ts.net (Tailscale) |

## Dependency requirements (from FiGS-Examples/environment_x86.yml)
- Python 3.10, PyTorch 2.1.2, torchvision 0.16.2
- **pytorch-cuda=11.8** + `nvidia/label/cuda-11.8.0::cuda-toolkit` (conda installs its own CUDA 11.8 toolkit — does NOT
  need to match system CUDA 12.8; NVIDIA drivers are backward compatible, 570.x supports CUDA 11.8 workloads fine)
- `tiny-cuda-nn` (built from source via `pip install git+...`) — needs `ninja`, working nvcc from the conda CUDA 11.8 toolkit
- `nerfstudio==1.1.4` (pulls in `gsplat`/splatfacto)
- `colmap` (via conda-forge) — for structure-from-motion / camera pose estimation from video
- `acados` — built locally via cmake, not pip-installed
- `Hierarchical-Localization` (hloc) — installed as editable submodule

## Key risk: GPU VRAM
- Standard nerfstudio `splatfacto` (Gaussian Splatting) training: ~6GB VRAM minimum for typical scenes, 8-12GB recommended
  for larger ones. The RTX 3050 Ti's 4GB is **below the practical minimum**.
- Mitigations to test:
  - `--downscale-factor` to reduce image resolution during splat training (reduces VRAM at cost of splat fidelity)
  - Reduce max Gaussians / use `--pipeline.model.cull-alpha-thresh` and similar splatfacto flags to cap splat count
  - Consider training splats on a separate, more powerful machine (e.g. cloud GPU) and only using this server for
    FiGS *flight simulation / inference* (rendering from an existing splat), which is much lighter than training
  - tiny-cuda-nn compilation and colmap SfM are CPU/VRAM-light and should be fine on this hardware
- acados (MPC solver) has no GPU dependency — CPU only, should run fine.

## Live server check results (2026-07-25)
| Check | Result |
|---|---|
| GPU | RTX 3050 Ti Laptop, 4096 MiB total, **0 MiB used at idle** — full 4GB available to start |
| Driver | 570.211.01, compute capability 8.6 |
| System nvcc | CUDA 12.8 (matches driver; irrelevant to conda's own 11.8 toolkit) |
| git | present (`/usr/bin/git`) |
| cmake | present (`/usr/bin/cmake`) — needed for acados build |
| conda/mamba | **not installed** — needs Miniconda/Miniforge before `environment_x86.yml` can be used |
| gdown | not installed (comes via conda env's pip section, so not a blocker) |
| Python (system) | 3.12.3 — irrelevant, conda env pins its own 3.10 |
| Disk | 399G free of 468G — plenty of headroom |
| RAM | 15Gi total, 14Gi available — should be fine for colmap/training at reduced settings |

**Conclusion:** No hard blockers. Full 4GB VRAM is free at idle, matching the borderline-viable case for splatfacto
if resolution/Gaussian-count is reduced. Compute capability 8.6 is well within tiny-cuda-nn/gsplat support. Missing
piece is conda — needs to be installed first.

## Setup progress (2026-07-25)
- [x] Miniconda installed (`conda 26.5.3`)
- [x] Repo cloned to `~/projects/figs_validation/SousVide` (submodules: FiGS, acados, Hierarchical-Localization) — note:
  actual path differs slightly from original plan (`~/projects/figs_validation/` vs `~/figs_validation/`)
- [x] acados built clean, no errors
- [x] conda env `kitchen` created from `environment_x86.yml`
- [x] `tiny-cuda-nn`, nerfstudio, FiGS (editable) installed — hit two build issues along the way, both resolved:
  1. Build-isolation pulled a fresh unpinned `setuptools` missing `pkg_resources` → fixed with `pip install
     "setuptools<81"` + `--no-build-isolation` so the env's own setuptools is used
  2. System `nvcc` (12.8) was ahead of conda's CUDA 11.8 toolkit on PATH, and once fixed, CUDA 11.8's nvcc rejected
     Ubuntu 24.04's default GCC 13 → fixed by `export PATH=$CONDA_PREFIX/bin:$PATH`, `CUDA_HOME=$CONDA_PREFIX`, and
     installing `gcc-11`/`g++-11` via apt, set as `CC`/`CXX` for the build
- [x] Verify installed packages import correctly and detect the GPU — **all pass**:
  - `torch 2.1.2`, `torch.cuda.is_available() == True`, device = RTX 3050 Ti Laptop GPU
  - `tinycudann`, `nerfstudio`, `gsplat`, `hloc`, `figs` all import cleanly
  - `ns-train --help` lists `splatfacto` / `splatfacto-big` among registered methods
  - `colmap -h` → COLMAP 3.11.1, built with CUDA
- [x] Example GSplats downloaded — gdown hit Google Drive's per-file quota on both the folder walk and the single
  zip (shared academic link, throttled independent of us); worked around by downloading the zip manually via browser
  and `scp`-ing it to the server. Landed at `~/projects/figs_validation/SousVide/gsplats/{capture,workspace}`
  (5.0GB total: `backroom.MOV` capture + `backroom`/`flightroom`/`mid_gate`/`src_open` workspaces, all with COLMAP
  output already computed and `backroom`/`flightroom`/`mid_gate`/`src_open` each having a trained splatfacto
  checkpoint under `workspace/outputs/`)
- [x] `figs_examples` notebook run to confirm inference within 4GB VRAM — **success**
  - Notebook's default example (`capture_name = "button"`) isn't in our downloaded set — patched to use `"backroom"`
    / `scene_name, course_name = "backroom", "circuit"` (the paper's actual "cluttered trajectory" example, and
    explicitly labeled as such in the notebook's own comments)
  - `backroom` already had a complete trained splatfacto checkpoint (`step-000029999.ckpt`), so skipped
    `pg.generate_gsplat()` (training) and ran straight to loading the splat + MPC-controlled flight simulation —
    the realistic "use FiGS for sim/inference" path rather than the VRAM-heavy training path
  - Ran headless via `jupyter nbconvert --execute` (no GUI on this server): 0 errored cells, produced
    `notebooks/circuit.mp4` (onboard-camera render of the simulated flight)
  - VRAM returned to 0 MiB / 4096 MiB after completion — confirms FiGS flight simulation (splat rendering + MPC
    control) fits comfortably within the RTX 3050 Ti's 4GB, consistent with the plan to treat this server as an
    inference/simulation box rather than a splat-training box

## Clean rebuild via `install_figs.sh` (2026-08-02)

Rebuilt from near-scratch with the installer rather than by hand. Conda now lives at
`~/projects/figs_validation/miniconda3` (not `~/miniconda3`), with pip/conda/torch/HF caches
redirected off `$HOME`. The July manual install remains at `~/miniconda3` as a fallback.

Reused rather than redownloaded: the `SousVide` clone, the acados build, and the 5 GB
`gsplats/` dataset — so no repeat of the Google Drive quota problem.

**Final verification: 22 passed, 2 warnings, 0 failed.** Both warnings are expected
(`sousvide` module absent — policy-distillation layer, not needed for FiGS sim; and the
4 GB VRAM training note).

### Measured performance — the numbers that were missing

| Metric | Value |
|---|---|
| **Peak VRAM, `backroom`/`circuit` flight sim** | **608 MiB of 4096** (~15%) |
| Idle baseline VRAM | 114 MiB |
| Wall clock, 1 simulated flight | **40 s** (headless notebook, end to end) |
| Output | `notebooks/circuit.mp4`, 1.1 MB |
| tiny-cuda-nn build (single arch, `TCNN_CUDA_ARCHITECTURES=86`) | 2m29s |

**Implication for the semantic-goal work:** simulation leaves ~3.4 GB of VRAM free. That is
enough to co-host a CLIPSeg model in the same process as the running sim rather than
requiring a separate pass — which was the open architectural question. Rollout cost of 40 s
also sets the price of behaviour-cloning data collection.

### Pinned dependency versions — do not let these drift

| Package | Version | Why |
|---|---|---|
| nerfstudio | **1.1.4** | pinned by `environment_x86.yml`; 1.1.5 pulls an incompatible gsplat |
| gsplat | **1.0.0** | hard requirement of nerfstudio 1.1.4 |
| torch / torchvision | 2.1.2 / 0.16.2 | every compiled extension is built against this libtorch ABI |
| numpy | 1.26.4 | numpy 2.x broke the C ABI; extensions built against 1.x segfault |

Drift observed and corrected this session: nerfstudio 1.1.5 and gsplat 1.4.0 had been
layered on top of 1.1.4/1.0.0. Also noted: ffmpeg is now 8.1.2.

---

## Host constraints (learned the hard way, 2026-08-02)

Four independent host-level faults, none of them FiGS-related, cost most of a day. All are
fixed properties of this machine and will resurface if the mitigations are undone.

### 1. Kernel is pinned — the NVIDIA 570 driver will not build against kernel 7.0.x

An unattended upgrade installed `7.0.0-28-generic`. The `nvidia/570.211.01` DKMS module
builds only for 6.17.x; on 7.0 it fails outright, so `nvidia-smi` reports *"couldn't
communicate with the NVIDIA driver"* and nothing GPU-shaped works.

The kernel *packages* were correctly held (`hi` state). The failure was the bootloader:
`GRUB_DEFAULT=saved` with `GRUB_SAVEDEFAULT=true` is **not a pin** — it means "boot whatever
booted last," so a single boot into 7.0.0-28 made it permanently sticky.

Fixed by pinning to an explicit menuentry ID (never an index — indices shift when kernels
are added or removed):

```
GRUB_SAVEDEFAULT=false
GRUB_DEFAULT="gnulinux-advanced-8eddb211-…>gnulinux-6.17.0-35-generic-advanced-8eddb211-…"
```

plus `sudo grub-editenv - unset saved_entry`.

**Debugging gotcha:** `dpkg -l | grep '^ii'` hides held packages — their state code is `hi`,
not `ii`. Use `grep '^.i'`. This wasted a diagnostic cycle chasing a "missing" kernel that
was installed the whole time.

### 2. Secure Boot — the DKMS signing key must be enrolled

Secure Boot is enabled and the kernel runs in lockdown mode. DKMS signs modules with the
machine-owner key at `/var/lib/shim-signed/mok/MOK.der`, but that key was **not enrolled** in
firmware, so the module loaded correctly on disk and was rejected at insert:

```
modprobe: ERROR: could not insert 'nvidia': Key was rejected by service
Kernel is locked down from EFI Secure Boot mode
```

Note this presents identically to a missing driver while `dkms status` cheerfully reports
`installed` — it is installed, just untrusted. Fixed with `sudo mokutil --import
/var/lib/shim-signed/mok/MOK.der`, then enrolling at the blue MOK Manager screen on reboot.
**Requires physical access** — the pre-boot UI is not reachable over SSH and times out.

Expect this to recur after any firmware update, which can clear enrolled MOKs without
touching anything on the Linux side.

### 3. Idle suspend must stay masked

`lightdm` suspended the machine mid-install, truncating the `pips` step. Masked at the
systemd level so desktop settings can't reintroduce it:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
# /etc/systemd/logind.conf: HandleLidSwitch=ignore, IdleAction=ignore
```

### 4. `MAX_JOBS=4` — 20 threads against 16 GB RAM will hard-lock this machine

Torch's `cpp_extension` ninja builds default to one job per core. With 20 threads and ~1–2 GB
per `nvcc`/`cicc` process, the gsplat JIT compile exhausted 15.3 GB RAM *and* 4 GB swap,
thrashing the box so hard that new SSH connections could not be established and SIGINT was
never delivered. Recovery required Magic SysRq (`Alt+SysRq` then `S`, `U`, `B` — Ubuntu's
default `kernel.sysrq=176` allows sync/remount-ro/reboot but **not** process signalling, so
`Alt+SysRq+F` does nothing).

`export MAX_JOBS=4` is now baked into `figs_env.sh`. Anything that JIT-compiles in this env
hits the same wall — this is a permanent property of 20 threads on 16 GB, not a one-off.

The July note that RAM "should be fine for colmap/training at reduced settings" is the case
where it wasn't.

### Consequence: interrupted pip installs leave doubled packages

The suspend during `pips` left **two** copies each of nerfstudio (1.1.4 + 1.1.5) and gsplat
(1.0.0 + 1.4.0): pip writes the new `dist-info` before removing the old, so an interruption
leaves both metadata directories over one merged package tree.

The symptom was baffling — gsplat's JIT builder globs every `.cu` in `cuda/csrc/`, so it
compiled the orphaned `sh.cu` (from 1.0.0) against 1.4.0's headers:

```
sh.cu(412): error: identifier "DEVICE_GUARD" is undefined
```

Note also that the two tools disagreed: `pip show gsplat` said 1.0.0 while `conda list` said
1.4.0, and `importlib.metadata.requires('nerfstudio')` returned the *wrong* pin because it
read the duplicate's metadata. Sweep for this after any interrupted install:

```bash
python -c "
import importlib.metadata as m, collections
c = collections.Counter(d.metadata['Name'] for d in m.distributions())
print([(k,v) for k,v in c.items() if v > 1])
"
```

Repair is a hard purge — `pip uninstall` twice, then `rm -rf` both the package directory and
every matching `*.dist-info` — followed by `./scripts/install_figs.sh --redo pips`.

---

## Script fixes applied (2026-08-02)

- **`install_figs.sh` preflight**: guarded the GPU probe. It tested `command -v nvidia-smi`
  (binary exists) rather than whether nvidia-smi *runs*; when the driver was down the error
  string was fed into `(( vram < 6000 ))`, so bash parsed `NVIDIA` as an identifier and
  `set -u` aborted with `line 343: NVIDIA: unbound variable` — an error pointing nowhere near
  the cause. Now detects the installed-but-not-communicating case and prints the actual
  remediation (DKMS rebuild / reboot / modprobe).
- **Known remaining issue in `do_pips`**: it guards with `python -c "import nerfstudio"` and
  skips its own `==1.1.4` pin whenever the conda yml already installed *something*. That is
  how a 1.1.5 could persist unnoticed. Should check the installed *version*, not mere
  presence.

## Still untested
- Training a *new* splat from scratch (`pg.generate_gsplat()`) on this hardware — this is the step most likely to
  hit the 4GB VRAM ceiling, per the original risk assessment. Not yet attempted since `backroom` already had a
  pre-trained checkpoint. If/when a new capture needs training, test with `--downscale-factor` and reduced
  Gaussian-count flags first, and be ready to fall back to training on a separate machine and only running FiGS
  simulation here.
- `sous_vide_examples.ipynb` (policy distillation / SV-Net) — not run yet.

**Note for reproducibility:** the `PATH`/`CUDA_HOME`/`CC`/`CXX` exports above were only set for the install session.
Anyone rebuilding this env from scratch needs to set them again before the `tiny-cuda-nn` pip install step specifically.

## Plan
1. Verify server env (commands issued to user for live server check).
2. Install acados (cmake build) + conda env `figs-env` from FiGS-Examples/environment_x86.yml.
3. Download example GSplats, run `figs_examples` notebook to confirm inference path works within 4GB VRAM.
4. If training a new splat from scratch is needed later, attempt at reduced resolution first; escalate to external
   GPU if VRAM is insufficient.
