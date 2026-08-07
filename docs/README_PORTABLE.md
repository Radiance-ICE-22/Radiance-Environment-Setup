# Running this FiGS install on another Ubuntu machine

The whole environment lives on the SSD — conda, the `kitchen` env (including its own
CUDA 11.8 toolkit and COLMAP), the SousVide repo, the acados build, and the example
GSplats. It is *mostly* portable. Four things are not.

## 1. The mount path must match — this is the one that breaks things

Conda bakes absolute paths into script shebangs, `.pth` files, and activation hooks.
Envs are **not relocatable**. This install was built at:

```
/media/yutharsan/105747cb-865f-46fb-b615-40315bb47374/FIGS
```

Note that path contains a **username**. Ubuntu auto-mounts removable drives at
`/media/<whoever-is-logged-in>/<UUID>`, so on another machine it lands somewhere
else and the env stops working.

Mount it at the original path explicitly. The filesystem UUID never changes, and
this works no matter who is logged in on the new machine:

```bash
sudo mkdir -p /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374
sudo mount UUID=105747cb-865f-46fb-b615-40315bb47374 \
           /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374
```

If the desktop already auto-mounted it elsewhere, unmount that first
(`udisksctl unmount -b /dev/sdX1`).

Then:

```bash
export FIGS=/media/yutharsan/105747cb-865f-46fb-b615-40315bb47374/FIGS
source $FIGS/figs_env.sh
```

`install_figs.sh` detects a path mismatch during preflight and tells you rather
than letting you debug it blind.

## 2. The NVIDIA driver belongs to the host

Never on the SSD. The host needs a driver supporting CUDA 11.8 — **>= 520**
recommended (>= 450 is the hard floor). Check with `nvidia-smi`. If absent:

```bash
sudo ubuntu-drivers install
```

## 3. tiny-cuda-nn is compiled for ONE GPU architecture

It was built for the machine it was installed on. On a GPU with a different
compute capability it will fail to load or misbehave. The installer records the
architecture it built for and warns you on a mismatch.

| GPU | Compute capability |
|---|---|
| RTX 2080 / 2080 Ti | 7.5 |
| RTX 3050–3090 | 8.6 |
| RTX 4070–4090 | 8.9 |
| RTX 5080 / 5090 | 12.0 |
| A100 | 8.0 |

Rebuild takes 15–40 min.

## 4. apt packages are host-side

`cmake`, `build-essential`, `ffmpeg`, `libgl1`, `libglib2.0-0`, `pkg-config`.
Not on the drive.

## Ubuntu version floor

The env links against system glibc in places. **Same or newer Ubuntu is fine**
(22.04 → 24.04 works). **Older will not** (22.04 → 20.04 fails) — glibc is
backward but not forward compatible.

Built on: **Ubuntu 22.04.5 LTS**, glibc 2.35.

---

## The actual procedure on a new machine

```bash
# 1. mount at the original path (section 1)
sudo mkdir -p /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374
sudo mount UUID=105747cb-865f-46fb-b615-40315bb47374 \
           /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374
export FIGS=/media/yutharsan/105747cb-865f-46fb-b615-40315bb47374/FIGS

# 2. confirm the mount allows execution — auto-mounts sometimes set noexec,
#    which breaks conda and acados in confusing ways
findmnt --target $FIGS
sudo mount -o remount,exec "$(findmnt -no TARGET --target $FIGS)"   # only if needed

# 3. redo just the host-specific parts (apt deps, host compiler, tiny-cuda-nn,
#    env file). Everything else on the drive is reused; gsplats are skipped.
cd $FIGS/../figs_setup     # wherever the scripts are
./install_figs.sh --prefix $FIGS --new-host

# 4. verify
./verify_figs.sh --prefix $FIGS --scene src_open
```

If the new machine has the **same GPU architecture**, you can skip the rebuild:

```bash
source $FIGS/figs_env.sh
./install_figs.sh --prefix $FIGS --verify-only
```

## Option B: copy the install onto the machine's internal disk

Faster than running off USB, and you keep the SSD as a pristine master copy.
Conda is **not relocatable**, so conda itself and the `kitchen` env get rebuilt —
but the large downloads are all path-independent and reused.

| Component | Size | Copied path change |
|---|---|---|
| `gsplats/` | ~5 GB | reused as-is |
| `SousVide/` git clone + submodules | ~1 GB | reused as-is |
| `miniconda3/pkgs/` package cache | ~8 GB | **reused** — env re-links locally, no re-download |
| `.cache/pip` wheels | ~3 GB | **reused** |
| `miniconda3/bin`, `envs/kitchen` | ~15 GB | rebuilt (absolute paths baked in) |
| tiny-cuda-nn | — | recompiled (15–40 min, GPU-architecture specific) |

```bash
# 1. copy (preserve permissions and symlinks — -a matters)
sudo cp -a /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374/FIGS /opt/FIGS
sudo chown -R "$USER:$USER" /opt/FIGS
export FIGS=/opt/FIGS

# 2. rebuild only what the path change invalidated
cd /path/to/scripts
./install_figs.sh --prefix $FIGS --relocated

# 3. verify
./verify_figs.sh --prefix $FIGS --scene src_open
```

`--relocated` reuses `clone` and `gsplats`, and rebuilds `miniconda`, `conda_env`,
`acados`, `tcnn`, `pips`, `envfile` plus host-side `apt_deps`. It removes the stale
env explicitly while keeping `pkgs/`, so env creation is local linking (a few
minutes) rather than a fresh download (~25 min).

Without `--relocated` a plain re-run would see `conda_env.done` and skip it, then
fail confusingly on activation. Preflight also detects the path mismatch
independently and tells you.

Use `rsync -aH --info=progress2` instead of `cp -a` if you want resumable copying —
it's ~30 GB.

## Making the mount permanent on a machine you use often

```bash
echo 'UUID=105747cb-865f-46fb-b615-40315bb47374 /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374 ext4 defaults,nofail,x-systemd.device-timeout=10 0 2' \
  | sudo tee -a /etc/fstab
sudo mkdir -p /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374
sudo mount -a
```

`nofail` matters — without it the machine refuses to boot when the SSD is unplugged.

## Things that will still differ between machines

- **VRAM.** Splatfacto training needs ~6 GB minimum. The 8 GB RTX 2080 clears it;
  a 4 GB laptop GPU does not. FiGS *simulation* (rendering an existing splat) is
  far lighter and runs on 4 GB.
- **CPU count.** acados was built with `make -j$(nproc)`; harmless, but COLMAP SfM
  speed varies a lot.
- **`~/.bashrc` is untouched by design.** `conda init` was deliberately skipped so
  that unplugging the drive doesn't break every new shell. `figs_env.sh` is the
  only entry point — there is no global `conda` command.

## Safe ejection

Never yank it mid-write. A corrupted conda env means a full rebuild.

```bash
conda deactivate
cd ~
sudo umount /media/yutharsan/105747cb-865f-46fb-b615-40315bb47374
```
