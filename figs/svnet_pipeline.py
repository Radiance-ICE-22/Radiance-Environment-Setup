#!/usr/bin/env python3
"""
svnet_pipeline.py — SOUS-VIDE's learning half (SV-Net), in one resumable script.

The sibling of figs_pipeline.py. That script takes one scene from phone video to an expert
flight; this one takes a *cohort* — a set of rollouts, observations and trained pilots — from
expert rollouts to an evaluated SV-Net policy. It replaces the upstream notebook
(notebooks/sous_vide_examples.ipynb) step for step and calls the upstream functions
unchanged:

    rollout     sousvide.synthesize.rollout_generator.generate_rollout_data
    observe     sousvide.synthesize.observation_generator.generate_observation_data
    train_hist  sousvide.instruct.train_policy.train_roster(..., "histNet", ...)
    train_comm  sousvide.instruct.train_policy.train_roster(..., "commNet", ..., regen=True)
    deploy      sousvide.flight.deploy_figs.deploy_roster(..., mode="visualize")

    source ~/projects/figs_validation/figs_env.sh        # REQUIRED, as for figs_pipeline.py
    ./svnet_pipeline.py --cohort first --scene backroom --courses circuit --method data_alpha
    ./svnet_pipeline.py --cohort first --status
    ./svnet_pipeline.py --cohort first --from train_comm --comm-epochs 100
    ./svnet_pipeline.py --cohort first --only deploy --deploy-method eval_nominal

The cohort's settings are remembered in <PROJECT_ROOT>/.svnet_pipeline_state/<cohort>/config.json,
so a resume needs only --cohort. Each step's marker fingerprint includes the marker of the
step before it: re-running an earlier step invalidates everything after it.

Only three things are added around upstream, none of which changes what it computes:
  * progress output: sousvide draws rich progress bars, which print nothing when stdout is not
    a terminal (Galley jobs, `tee`, tmux logs). They are replaced by plain lines, and training
    losses are also written per epoch to live_<pilot>_<network>.jsonl for the UI.
  * data hygiene: re-running `rollout` or `observe` moves the old files to
    cohorts/<cohort>/_archive/<timestamp>/ first. Upstream overwrites files by index, so a
    smaller re-run would silently mix old and new data. Nothing is deleted.
  * a second tracking-error figure in `deploy`: upstream's compute_flight_metrics takes the
    norm over the wrong axis (axis=0, across the whole reference path, instead of axis=1, per
    point), so its "TTE" is not a distance to the path. Its numbers are reported as they are,
    labelled upstream_*, next to a per-point figure computed the same way rollout_generator's
    tol_select check does. Note that upstream also uses its TTE to pick commNet's best
    checkpoint when in-loop evaluation is on.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_pipeline import (  # noqa: E402  (shared helpers; figs_pipeline imports only stdlib)
    StepFailed, VramMonitor, _C, course_digest, course_int_cells, diagnose, elapsed, fail, info,
    ok, resolve_project_root, section, vram_now, warn)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")

# Defaults follow the upstream notebook, except in-loop commNet evaluation (see --comm-eval).
DEFAULTS = {
    "scene": None, "courses": None, "method": "data_alpha", "roster": ["Maverick"],
    "expert": "Viper", "frame": "carl", "nro_ds": 50, "use_compress": False, "subsample": 1.0,
    "hist_epochs": 200, "comm_epochs": 300, "lr": 1e-4, "batch_size": 64, "lim_sv": 50,
    "comm_eval": "eval_nominal", "deploy_course": None, "deploy_method": "eval_nominal",
}
LISTS = ("courses", "roster")


# ════════════════════════════════════════════════════════════════════════════════
#  Progress: plain lines in place of rich's live bars
# ════════════════════════════════════════════════════════════════════════════════

def _plain(s):
    try:
        from rich.text import Text
        s = Text.from_markup(str(s)).plain
    except Exception:
        s = re.sub(r"\[/?[^\]]*\]", "", str(s))
    return re.sub(r"\s+", " ", s.replace("\\[", "[")).strip()


class PlainProgress:
    """Duck-types the parts of rich.progress.Progress that sousvide uses.

    Sample-level bars become carriage-return redraws (Galley shows them live but does not
    store them); dataset- and epoch-level bars become ordinary log lines. Training epochs are
    also appended to <state>/live_<pilot>_<network>.jsonl.
    """
    live_dir: Path | None = None
    redraw_pending = False                      # a \r-terminated line is on screen

    def __init__(self):
        self.tasks = {}
        self._n = 0
        self._last = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        PlainProgress._end_redraw()

    @staticmethod
    def _end_redraw():
        if PlainProgress.redraw_pending:        # move a terminal off the redraw line
            sys.stdout.write(" " * 100 + "\r")
            sys.stdout.flush()
            PlainProgress.redraw_pending = False

    def add_task(self, description, total=None, **fields):
        self._n += 1
        self.tasks[self._n] = {"description": description, "total": total, "completed": 0,
                               "fields": dict(fields), "t0": time.time()}
        return self._n

    def reset(self, task_id, description=None, total=None, completed=0, **fields):
        t = self.tasks[task_id]
        if description is not None:
            t["description"] = description
        t["total"], t["completed"], t["t0"] = total, completed, time.time()
        t["fields"].update(fields)

    def update(self, task_id, description=None, advance=None, total=None, completed=None,
               refresh=False, **fields):
        t = self.tasks[task_id]
        if description is not None:
            t["description"] = description
        if total is not None:
            t["total"] = total
        if completed is not None:
            t["completed"] = completed
        if advance:
            t["completed"] += advance
        t["fields"].update(fields)
        if advance or completed is not None:
            self._emit(t)

    def refresh(self):
        pass

    def _emit(self, t):
        units = t["fields"].get("units", "")
        desc = _plain(t["description"])
        tot = t["total"]
        count = f"{t['completed']}/{tot if tot is not None else '?'} {units}".strip()
        if units != "samples":
            PlainProgress._end_redraw()
        if units == "epochs":
            loss = t["fields"].get("loss")
            print(f"  {desc}  epoch {count}  train loss {loss:.5f}" if loss is not None else f"  {desc}  {count}",
                  flush=True)
            m = re.match(r"(\S+) > (\S+)", desc)
            if m and self.live_dir and loss is not None:
                with open(self.live_dir / f"live_{m.group(1)}_{m.group(2)}.jsonl", "a") as fh:
                    fh.write(json.dumps({"epoch": t["completed"], "loss": float(loss),
                                         "t": round(time.time(), 1)}) + "\n")
        elif units == "datasets":
            print(f"  {desc}  {count}  ({elapsed(t['t0'])})", flush=True)
        else:                                   # samples: a live redraw, throttled
            now = time.time()
            if now - self._last > 0.5 or (tot and t["completed"] >= tot):
                self._last = now
                sys.stdout.write(f"  {desc}  {count}".ljust(100) + "\r")   # trailing \r: a redraw
                sys.stdout.flush()
                PlainProgress.redraw_pending = True


def install_progress(live_dir):
    import sousvide.visualize.rich_utilities as ru
    PlainProgress.live_dir = live_dir
    ru.get_generation_progress = PlainProgress
    ru.get_training_progress = PlainProgress


# ════════════════════════════════════════════════════════════════════════════════
#  Context
# ════════════════════════════════════════════════════════════════════════════════

class Ctx:
    def __init__(self, a):
        self.a = a
        self.project_root = resolve_project_root(a.project_root).resolve()   # steps chdir into the repo
        self.repo = self.project_root / "SousVide"
        self.cohort = a.cohort
        self.cohort_dir = self.repo / "cohorts" / a.cohort
        self.state = self.project_root / ".svnet_pipeline_state" / a.cohort
        self.state.mkdir(parents=True, exist_ok=True)
        self.results = {"cohort": a.cohort}
        rp = self.state / "results.json"
        if rp.exists():
            try:
                self.results.update(json.loads(rp.read_text()))
            except ValueError:
                pass

    def save_results(self):
        _atomic(self.state / "results.json", json.dumps(self.results, indent=2, default=str) + "\n")

    def marker(self, name):
        m = self.state / f"{name}.done"
        return m.read_text().strip() if m.exists() else ""

    def fingerprint(self, keys, prev):
        vals = {k: getattr(self.a, k, None) for k in keys}
        vals["_prev"] = self.marker(prev) if prev else None
        return hashlib.sha256(json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def is_done(self, name, fp):
        m = self.state / f"{name}.done"
        if not m.exists():
            return False
        if m.read_text().strip().split("\n")[0] != fp:
            warn(f"'{name}' was completed with different settings or older inputs — redoing")
            m.unlink()
            return False
        return True

    def mark_done(self, name, fp):
        _atomic(self.state / f"{name}.done", f"{fp}\n{datetime.now().isoformat(timespec='seconds')}\n")

    def clear(self, name):
        (self.state / f"{name}.done").unlink(missing_ok=True)

    def archive(self, rel_paths, why):
        """Move existing cohort data aside (never delete) before a step regenerates it."""
        moved = []
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for rel in rel_paths:
            src = self.cohort_dir / rel
            if src.exists():
                dst = self.cohort_dir / "_archive" / stamp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved.append(rel)
        if moved:
            warn(f"{why}: moved {', '.join(moved)} to cohorts/{self.cohort}/_archive/{stamp}/ "
                 "(delete it yourself when you no longer need it)")
        return moved


def _atomic(p, text):
    p = Path(p)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def _du(p):
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _gb(n):
    return round(n / 1e9, 2)


def active_model(repo, scene):
    cfgs = sorted((repo / "gsplats" / "workspace" / "outputs" / scene).rglob("config.yml"))
    return cfgs


def _torch_load(p):
    import torch
    return torch.load(p, map_location="cpu")


# ════════════════════════════════════════════════════════════════════════════════
#  Steps
# ════════════════════════════════════════════════════════════════════════════════

def step_preflight(c):
    a, problems = c.a, []
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    (ok if env == "kitchen" else problems.append)(
        f"conda env: {env}" if env == "kitchen" else f"conda env is '{env or 'none'}', expected 'kitchen'")
    if not os.environ.get("ACADOS_SOURCE_DIR"):
        problems.append(f"ACADOS_SOURCE_DIR unset — `source {c.project_root}/figs_env.sh` first")

    import importlib
    try:
        import torch
        (ok if torch.__version__.startswith("2.1.2") else problems.append)(
            f"torch {torch.__version__}" if torch.__version__.startswith("2.1.2")
            else f"FATAL: torch changed — {torch.__version__}, expected 2.1.2")
    except Exception as e:
        problems.append(f"import torch failed: {e}")
    for mod in ("sousvide.synthesize.rollout_generator", "sousvide.synthesize.observation_generator",
                "sousvide.instruct.train_policy", "sousvide.flight.deploy_figs", "figs.simulator"):
        try:
            importlib.import_module(mod)
            ok(f"import {mod}")
        except Exception as e:
            problems.append(f"import {mod} failed: {e}. sousvide is installed with "
                            "`pip install --no-deps -e .` in the SousVide root (GALLEY_UI.md §5)")

    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
            used = vram_now()
            ok(f"GPU: {torch.cuda.get_device_name(0)} {total} MiB, {used} MiB in use")
            c.results["gpu"] = torch.cuda.get_device_name(0)
            if used > 1000:
                warn(f"{used} MiB already in use — ns-viewer holds ~4.6 GB; close it")
        else:
            problems.append("torch cannot see the GPU")

    # configs
    cfg = c.repo / "configs"
    need = [("courses", x) for x in a.courses] + [("methods", a.method), ("pilots", a.expert),
                                                 ("frames", a.frame), ("methods", a.deploy_method)]
    need += [("pilots", p) for p in a.roster]
    if a.comm_eval != "none":
        need.append(("methods", a.comm_eval))
    for fam, name in need:
        p = cfg / fam / f"{name}.json"
        if not p.exists():
            problems.append(f"configs/{fam}/{name}.json not found")
    for course in a.courses:
        p = cfg / "courses" / f"{course}.json"
        if p.exists():
            bad = course_int_cells(json.loads(p.read_text())["waypoints"]["keyframes"])
            if bad:
                problems.append(f"course {course}: integer fo cells {bad[:4]} — FiGS misreads them; "
                                "re-save it from Galley's course editor")
    for p in a.roster:
        pp = cfg / "pilots" / f"{p}.json"
        if pp.exists() and "networks" not in json.loads(pp.read_text()):
            problems.append(f"pilot {p} is not a student (no 'networks'); experts go in --expert")
    if a.deploy_course not in a.courses:
        warn(f"deploy course {a.deploy_course} is not one of the training courses {a.courses}")

    # scene: FiGS loads exactly one trained model
    models = active_model(c.repo, a.scene)
    if len(models) != 1:
        problems.append(f"scene {a.scene}: {len(models)} trained models under gsplats/workspace/outputs/"
                        f"{a.scene}/ — FiGS needs exactly one (archive/promote in Galley's Scene page)")
    else:
        ok(f"scene {a.scene}: model {models[0].parent.name}")

    # size estimate from the course files' times (the expert re-times them, so ±30 %)
    try:
        method = json.loads((cfg / "methods" / f"{a.method}.json").read_text())
        frame = json.loads((cfg / "frames" / f"{a.frame}.json").read_text())
        hz = json.loads((cfg / "pilots" / f"{a.expert}.json").read_text())["track"]["hz"]
        h, w = frame["camera"]["height"], frame["camera"]["width"]
        per_sample = h * w * 3 * 2                 # rgb + depth, uint8, as upstream stores them
        est_samples = est_rollouts = 0
        for course in a.courses:
            kf = json.loads((cfg / "courses" / f"{course}.json").read_text())["waypoints"]["keyframes"]
            ts = [k["t"] for k in kf.values()]
            dur = ts[-1] - ts[0]
            dt_ro = method["duration"] or dur
            rate = method["rate"] or 1 / dt_ro
            nro = int(round((method["reps"] or 1) * int(round(rate * dur))))
            est_rollouts += nro
            est_samples += int(nro * dt_ro * hz)
        est = est_samples * per_sample * (0.5 if a.use_compress else 1.0)
        free = shutil.disk_usage(c.repo).free
        c.results["estimate"] = {"rollouts": est_rollouts, "samples": est_samples,
                                 "rollout_gb": _gb(est), "free_gb": _gb(free)}
        info(f"estimate for {a.method}: ~{est_rollouts} rollouts, ~{est_samples:,} samples, "
             f"~{_gb(est)} GB of rollout data (+ observations); {_gb(free)} GB free")
        if est > 0.9 * free:
            problems.append(f"not enough disk: ~{_gb(est)} GB needed, {_gb(free)} GB free")
        elif est > 0.5 * free:
            warn("this cohort would use over half the free disk")
    except Exception as e:
        warn(f"could not estimate the data size: {e}")

    # keep cohorts/ out of the upstream clone's git status (local exclude, not .gitignore)
    exc = c.repo / ".git" / "info" / "exclude"
    if exc.parent.is_dir():
        text = exc.read_text() if exc.exists() else ""
        if not re.search(r"^/?cohorts/?$", text, re.M):
            exc.write_text(text + ("" if text.endswith("\n") or not text else "\n") + "cohorts/\n")
            info("added cohorts/ to SousVide/.git/info/exclude")

    if problems:
        print()
        for p in problems:
            fail(p)
        raise StepFailed(f"{len(problems)} preflight problem(s)")
    ok("preflight clean")


def step_rollout(c):
    a = c.a
    c.archive(["rollout_data", "observation_data"], "re-running rollouts")
    os.chdir(c.repo)
    import sousvide.synthesize.rollout_generator as rg
    info(f"expert {a.expert} flies {', '.join(a.courses)} in {a.scene} with {a.method}; "
         "first run for a course compiles acados (slow once)")
    t0 = time.time()
    with VramMonitor(interval=5) as vm:
        rg.generate_rollout_data(a.cohort, a.courses, a.scene, a.method,
                                 expert_name=a.expert, bframe_name=a.frame,
                                 Nro_ds=a.nro_ds, use_compress=a.use_compress)
    summary = {}
    for course in a.courses:
        d = c.cohort_dir / "rollout_data" / course
        trajs = sorted((d / "trajectories").glob("*.pt"))
        n_ro = n_dp = 0
        for f in trajs:
            data = _torch_load(f)
            n_ro += len(data)
            n_dp += sum(int(t["Ndata"]) for t in data)
        summary[course] = {"files": len(trajs), "rollouts": n_ro, "samples": n_dp, "gb": _gb(_du(d))}
        ok(f"{course}: {n_ro} rollouts kept (tol_select), {n_dp:,} samples, {summary[course]['gb']} GB")
    if not any(s["rollouts"] for s in summary.values()):
        raise StepFailed("no rollout met the method's tol_select: nothing to train on. Check the flight "
                         "with figs_pipeline.py first (tracking error), or loosen tol_select in a copy "
                         "of the method")
    c.results["rollout"] = {"courses": summary, "wallclock": elapsed(t0), "peak_vram_mib": vm.peak,
                            "scene_model": getattr(a, "scene_model", None)}


def step_observe(c):
    a = c.a
    c.archive([f"observation_data/{p}" for p in a.roster], "re-running observations")
    os.chdir(c.repo)
    import sousvide.synthesize.observation_generator as og
    t0 = time.time()
    with VramMonitor(interval=5) as vm:
        og.generate_observation_data(a.cohort, a.roster, subsample=a.subsample)
    summary = {}
    for p in a.roster:
        base = c.cohort_dir / "observation_data" / p
        summary[p] = {n.name: _gb(_du(n)) for n in sorted(base.iterdir()) if n.is_dir()} if base.is_dir() else {}
        ok(f"{p}: " + ", ".join(f"{k} {v} GB" for k, v in summary[p].items()))
    c.results["observe"] = {"pilots": summary, "wallclock": elapsed(t0), "peak_vram_mib": vm.peak}


def _fresh(c, net):
    if net not in c.a.fresh:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in c.a.roster:
        rdir = c.cohort_dir / "roster" / p
        for f in (f"{net}.pt", f"losses_{net}.pt"):
            if (rdir / f).exists():
                dst = rdir / "_archive" / stamp / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rdir / f), str(dst))
                info(f"{p}: moved {f} to roster/{p}/_archive/{stamp}/ (--fresh {net})")


def _losses(c, net):
    """Latest loss log per pilot, from sousvide's losses_<net>.pt, as plain lists for the UI."""
    out = {}
    for p in c.a.roster:
        lp = c.cohort_dir / "roster" / p / f"losses_{net}.pt"
        if not lp.exists():
            continue
        logs = _torch_load(lp)
        if not logs:
            continue
        key = sorted(logs)[-1]
        e = logs[key]

        def pairs(x):
            try:
                import numpy as np
                arr = np.asarray(x, dtype=float)
                return [] if arr.size == 0 or arr.ndim != 2 else [[float(s), float(v)] for s, v in arr.T]
            except Exception:
                return []
        out[p] = {"log": key, "epochs": e.get("N_eps"), "train_s": round(float(e.get("t_tn") or 0), 1),
                  "n_train": e.get("Nd_tn"), "n_test": e.get("Nd_tt"),
                  "loss_train": pairs(e.get("Loss_tn")), "loss_test": pairs(e.get("Loss_tt")),
                  "eval_tte_upstream": pairs(e.get("Eval_tte")), "n_logs": len(logs)}
        lt, lv = out[p]["loss_train"], out[p]["loss_test"]
        ok(f"{p} {net}: {out[p]['epochs']} epochs, train {lt[-1][1] if lt else '?':.5}, "
           f"test {lv[-1][1] if lv else '?':.5}, {out[p]['train_s']} s")
    return out


def _train(c, net, epochs, regen, deployment):
    a = c.a
    _fresh(c, net)
    for p in a.roster:
        (c.state / f"live_{p}_{net}.jsonl").unlink(missing_ok=True)
        rdir = c.cohort_dir / "roster" / p
        if (rdir / f"{net}.pt").exists():
            info(f"{p}: {net}.pt exists — upstream continues training it (use --fresh {net} to start over)")
    os.chdir(c.repo)
    import sousvide.instruct.train_policy as tp
    t0 = time.time()
    with VramMonitor(interval=5) as vm:
        tp.train_roster(a.cohort, a.roster, net, epochs, regen=regen, deployment=deployment,
                        lim_sv=a.lim_sv, lr=a.lr, batch_size=a.batch_size)
    c.results[f"train_{net}"] = {"epochs": epochs, "regen": regen, "deployment": deployment,
                                 "wallclock": elapsed(t0), "peak_vram_mib": vm.peak,
                                 "pilots": _losses(c, net)}


def step_train_hist(c):
    _train(c, "histNet", c.a.hist_epochs, False, None)


def step_train_comm(c):
    a = c.a
    dep = None if a.comm_eval == "none" else (a.deploy_course, a.scene, a.comm_eval)
    if dep:
        info(f"in-loop evaluation every {a.lim_sv} epochs on {a.deploy_course} with {a.comm_eval}; "
             "upstream keeps the checkpoint with the lowest (upstream) TTE")
    _train(c, "commNet", a.comm_epochs, True, dep)       # regen: histNet's features are inputs


def _tte_per_point(trajectories):
    import numpy as np
    errs, finals = [], []
    for tr in trajectories:
        P, Pd = tr["Xro"][:, 0:3], tr["tXUd"][:, 1:4]
        d = np.linalg.norm(P[:, None, :] - Pd[None, :, :], axis=2).min(axis=1)
        errs.append(d)
        finals.append(d[-1])
    allerr = np.hstack(errs)
    return {"mean_m": round(float(allerr.mean()), 4), "max_m": round(float(allerr.max()), 4),
            "within_0.3m": round(float((allerr < 0.3).mean()), 3),
            "final_mean_m": round(float(np.mean(finals)), 4), "rollouts": len(trajectories)}


def step_deploy(c):
    a = c.a
    os.chdir(c.repo)
    import sousvide.flight.deploy_figs as df
    captured = []
    upstream = df.fh.compute_flight_metrics

    def recording(trajectories, *args, **kw):          # observe, do not alter
        m = upstream(trajectories, *args, **kw)
        captured.append((m, _tte_per_point(trajectories)))
        return m
    df.fh.compute_flight_metrics = recording
    t0 = time.time()
    try:
        with VramMonitor(interval=5) as vm:
            df.deploy_roster(a.cohort, a.deploy_course, a.scene, a.deploy_method, a.roster,
                             expert_name=a.expert, bframe_name=a.frame, mode="visualize", show_table=True)
    finally:
        df.fh.compute_flight_metrics = upstream
    crew = ["expert"] + list(a.roster)
    rows = {}
    for name, (m, ours) in zip(crew, captured):
        pilot = a.expert if name == "expert" else name
        vid = c.cohort_dir / "deployment_data" / f"sim_{a.deploy_course}_{name}_rgb.mp4"
        rows[pilot] = {
            "role": "expert" if name == "expert" else "student",
            "upstream_tte_mean": round(float(m["TTE"]["mean"]), 4),
            "upstream_tte_best": round(float(m["TTE"]["best"]), 4),
            "upstream_pp": round(float(m["PP"]), 3),
            "hz_mean": round(float(m["hz"]["mean"]), 1), "hz_worst": round(float(m["hz"]["worse"]), 1),
            "tte": ours, "video": str(vid.relative_to(c.repo)) if vid.exists() else None,
        }
        ok(f"{pilot:<10} per-point tracking error mean {ours['mean_m']} m, max {ours['max_m']} m, "
           f"{ours['within_0.3m'] * 100:.0f}% within 0.3 m  |  upstream TTE {rows[pilot]['upstream_tte_mean']}, "
           f"PP {rows[pilot]['upstream_pp']}, {rows[pilot]['hz_mean']} Hz")
    c.results["deploy"] = {"course": a.deploy_course, "method": a.deploy_method, "scene": a.scene,
                           "pilots": rows, "wallclock": elapsed(t0), "peak_vram_mib": vm.peak,
                           "finished": datetime.now().isoformat(timespec="seconds")}


# (name, function, fingerprint keys, description). Each fingerprint also covers the marker
# of the step before it, so re-running a step invalidates everything downstream.
STEPS = [
    ("preflight", step_preflight, [], "environment, sousvide, configs, scene, disk estimate"),
    ("rollout", step_rollout, ["scene", "scene_model", "courses", "course_digests", "method", "expert",
                               "frame", "nro_ds", "use_compress"], "expert rollouts through the splat (LONG, GPU)"),
    ("observe", step_observe, ["roster", "subsample"], "per-pilot observations"),
    ("train_hist", step_train_hist, ["roster", "hist_epochs", "lr", "batch_size", "lim_sv"],
     "train histNet (dynamics history → [m, kt])"),
    ("train_comm", step_train_comm, ["roster", "comm_epochs", "lr", "batch_size", "lim_sv", "comm_eval",
                                     "deploy_course"], "regenerate observations, train commNet (GPU)"),
    ("deploy", step_deploy, ["roster", "deploy_course", "deploy_method", "expert", "frame"],
     "fly expert and students in FiGS, metrics and videos"),
]
_ALWAYS = {"preflight"}


def main():
    ap = argparse.ArgumentParser(description="SOUS-VIDE SV-Net pipeline (resumable)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cohort", required=True, help="experiment name: SousVide/cohorts/<cohort>/")
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--scene", help="trained splat to fly in (exactly one model)")
    ap.add_argument("--courses", help="comma-separated course names")
    ap.add_argument("--method", help="rollout method: data_alpha | data_beta | data_gamma | ...")
    ap.add_argument("--roster", help="comma-separated student pilots (default Maverick)")
    ap.add_argument("--expert", help="expert pilot (default Viper)")
    ap.add_argument("--frame", help="drone frame (default carl)")
    ap.add_argument("--nro-ds", type=int, help="rollouts per saved file (default 50)")
    ap.add_argument("--use-compress", choices=["yes", "no"], help="upstream image compression (default no)")
    ap.add_argument("--subsample", type=float, help="observation subsample ratio (default 1.0)")
    ap.add_argument("--hist-epochs", type=int, help="histNet epochs (default 200)")
    ap.add_argument("--comm-epochs", type=int, help="commNet epochs (default 300)")
    ap.add_argument("--lr", type=float, help="learning rate (default 1e-4)")
    ap.add_argument("--batch-size", type=int, help="batch size (default 64)")
    ap.add_argument("--lim-sv", type=int, help="save (and evaluate) every N epochs (default 50)")
    ap.add_argument("--comm-eval", help="method for commNet's in-loop evaluation, or 'none' "
                                        "(notebook: eval_nominal; eval_single is ~10x cheaper)")
    ap.add_argument("--deploy-course", help="course for evaluation (default: first course)")
    ap.add_argument("--deploy-method", help="evaluation method (default eval_nominal)")
    ap.add_argument("--fresh", action="append", default=[], choices=["histNet", "commNet"],
                    help="archive the pilot's existing network and train it from scratch")
    ap.add_argument("--redo", action="append", default=[], metavar="STEP")
    ap.add_argument("--from", dest="from_step", metavar="STEP")
    ap.add_argument("--only", metavar="STEP")
    ap.add_argument("--stop-after", metavar="STEP")
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    names = [s[0] for s in STEPS]
    if a.list_steps:
        for n, _, keys, d in STEPS:
            print(f"  {n:<11} {d}" + (f"   {_C['d']}[{','.join(keys)}]{_C['x']}" if keys else ""))
        return 0
    if not NAME_RE.match(a.cohort):
        ap.error("--cohort: letters, digits, '_' and '-'")
    for opt in (a.from_step, a.only, a.stop_after, *a.redo):
        if opt and opt not in names:
            ap.error(f"unknown step '{opt}'. Options: {', '.join(names)}")
    if a.only and (a.from_step or a.stop_after):
        ap.error("--only cannot be combined with --from/--stop-after")

    c = Ctx(a)

    if a.status:
        section(f"Status — cohort {a.cohort}")
        for n, _, _, d in STEPS:
            m = c.state / f"{n}.done"
            when = m.read_text().strip().split("\n")[-1] if m.exists() else ""
            print(f"  {'✔' if m.exists() else '·'} {n:<11} {when:<20} {d}")
        info(f"state: {c.state}   data: {c.cohort_dir}")
        return 0

    # settings: command line > the cohort's saved config > defaults
    cfg_path = c.state / "config.json"
    saved = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    for k, dflt in DEFAULTS.items():
        v = getattr(a, k)
        if v is None:
            v = saved.get(k, dflt)
        elif k in LISTS:
            v = [x.strip() for x in v.split(",") if x.strip()]
        elif k == "use_compress":
            v = v == "yes"
        setattr(a, k, v)
    if not a.scene or not a.courses:
        ap.error("--scene and --courses are required for a new cohort")
    a.deploy_course = a.deploy_course or a.courses[0]
    names_used = [getattr(a, k) for k in ("scene", "method", "expert", "frame", "comm_eval",
                                          "deploy_course", "deploy_method")] + a.courses + a.roster
    for v in names_used:
        if not NAME_RE.match(str(v)):
            ap.error(f"invalid name: {v!r}")
    _atomic(cfg_path, json.dumps({k: getattr(a, k) for k in DEFAULTS}, indent=2) + "\n")
    c.results["config"] = {k: getattr(a, k) for k in DEFAULTS}
    models = active_model(c.repo, a.scene)
    a.scene_model = models[0].parent.name if len(models) == 1 else None
    a.course_digests = [course_digest(c.repo, x) for x in a.courses]

    for n in a.redo:
        c.clear(n)
        info(f"cleared marker for '{n}'")
    selected = names
    if a.only:
        selected = [a.only]
    else:
        if a.from_step:
            selected = names[names.index(a.from_step):]
        if a.stop_after:
            selected = [n for n in selected if names.index(n) <= names.index(a.stop_after)]
    if "preflight" not in selected:
        selected = ["preflight"] + selected          # cheap, and catches a broken environment

    print(f"{_C['b']}SV-Net pipeline{_C['x']}  cohort={a.cohort}  scene={a.scene}  "
          f"courses={','.join(a.courses)}  roster={','.join(a.roster)}  steps={','.join(selected)}")
    info(f"data:  {c.cohort_dir}")
    info(f"state: {c.state}")

    t_all = time.time()
    prev = None
    for n, fn, keys, desc in STEPS:
        this_prev = prev if n != "preflight" else None
        if n != "preflight":
            prev = n
        if n not in selected:
            continue
        fp = c.fingerprint(keys, this_prev)
        if n not in _ALWAYS and c.is_done(n, fp) and n not in a.redo:
            print(f"  {_C['d']}· {n:<11} already done — skipping{_C['x']}")
            continue
        section(f"{n}  —  {desc}")
        if n != "preflight":
            install_progress(c.state)
        t0 = time.time()
        try:
            fn(c)
        except KeyboardInterrupt:
            print()
            warn(f"interrupted during '{n}' — marker NOT written, this step will re-run")
            c.save_results()
            return 130
        except StepFailed as e:
            print()
            fail(f"step '{n}' failed:\n\n{e}\n")
            c.save_results()
            return 1
        except Exception:
            print()
            traceback.print_exc()
            d = diagnose(traceback.format_exc())
            if d:
                print()
                fail(d)
            info(f"after fixing, resume with:  {Path(sys.argv[0]).name} --cohort {a.cohort} --from {n}")
            c.save_results()
            return 1
        if n not in _ALWAYS:
            c.mark_done(n, c.fingerprint(keys, this_prev))
            # downstream markers now mismatch through the chain; nothing else to clear
        c.save_results()
        info(f"'{n}' completed in {elapsed(t0)}")

    section(f"Done — total {elapsed(t_all)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
