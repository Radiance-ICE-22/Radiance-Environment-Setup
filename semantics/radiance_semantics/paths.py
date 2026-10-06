"""Where things live for one scene's trained model, and where semantic artefacts go.

Layout (PROJECT_ROOT is the FiGS install prefix: figs_env.sh + SousVide/):

    SousVide/gsplats/workspace/
        <scene>/                                   transforms.json, images, sparse_pc.ply
        <scene>/semantics/teachers/<tag>/          2D teacher features (Phase 1)
        <scene>/semantics/<run>/<backend>/         per-Gaussian tables (Phase 1/4)
        outputs/<scene>/splatfacto/<run>/          config.yml, dataparser_transforms.json,
                                                   nerfstudio_models/step-XXXXXXXXX.ckpt
    .semantic_pipeline_state/<scene>/<run>/        step markers and settings
    SousVide/runs/                                 JSON run records (as figs_pipeline.py)

ns-train runs with cwd = gsplats/workspace and --data <scene>, so config.yml holds paths
relative to gsplats/workspace; anything that loads the pipeline must run from there.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
_STEP_RE = re.compile(r"step-(\d+)\.ckpt$")


class SemanticsError(ValueError):
    """A problem the user can fix (wrong scene, missing model, ...). Printed without a traceback."""


def _valid_root(d):
    d = Path(d).expanduser()
    return d if (d / "figs_env.sh").exists() and (d / "SousVide").is_dir() else None


def resolve_project_root(explicit=None):
    """PROJECT_ROOT, in the same order of preference as figs_pipeline.resolve_project_root:
    --project-root, $FIGS_PROJECT_ROOT, then <root>/SousVide/FiGS/acados from $ACADOS_SOURCE_DIR
    (figs_env.sh always exports it)."""
    if explicit:
        d = Path(explicit).expanduser()
        if not _valid_root(d):
            raise SemanticsError(f"--project-root {d} is not a FiGS root (expected figs_env.sh and SousVide/ inside)")
        return d.resolve()
    env = os.environ.get("FIGS_PROJECT_ROOT")
    if env and _valid_root(env):
        return _valid_root(env).resolve()
    acados = os.environ.get("ACADOS_SOURCE_DIR")
    if acados:
        cand = Path(acados).resolve().parents[2]
        if _valid_root(cand):
            return cand
    raise SemanticsError("cannot find the FiGS project root: pass --project-root, or `source <root>/figs_env.sh` first")


def latest_checkpoint(models_dir):
    """The highest step-*.ckpt in a nerfstudio_models/ directory (eval_setup's choice too)."""
    best = None
    for p in Path(models_dir).glob("step-*.ckpt"):
        m = _STEP_RE.search(p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    if best is None:
        raise SemanticsError(f"no step-*.ckpt in {models_dir}: training has not finished")
    return best[1]


@dataclass(frozen=True)
class SceneRun:
    project_root: Path
    scene: str
    config_yml: Path

    # ── the trained model ──────────────────────────────────────────────────────
    @property
    def repo(self):
        return self.project_root / "SousVide"

    @property
    def workspace_root(self):
        return self.repo / "gsplats" / "workspace"

    @property
    def scene_dir(self):
        return self.workspace_root / self.scene

    @property
    def run_dir(self):
        return self.config_yml.parent

    @property
    def run(self):
        return self.run_dir.name

    @property
    def checkpoint(self):
        return latest_checkpoint(self.run_dir / "nerfstudio_models")

    @property
    def dataparser_transforms(self):
        return self.run_dir / "dataparser_transforms.json"

    @property
    def transforms_json(self):
        return self.scene_dir / "transforms.json"

    @property
    def key(self):
        """run + checkpoint name + mtime: the key Galley's .splat cache uses, so a retrain or a
        promote makes semantic artefacts stale exactly when it makes the splat stale."""
        ck = self.checkpoint
        return f"{self.run}-{ck.stem}-{int(ck.stat().st_mtime)}"

    # ── semantic artefacts ─────────────────────────────────────────────────────
    @property
    def semantics_dir(self):
        return self.scene_dir / "semantics"

    def teachers_dir(self, tag):
        return self.semantics_dir / "teachers" / tag

    def backend_dir(self, backend, suffix=""):
        """semantics/<run>/<backend><suffix>; a suffix (e.g. _w480) names a variant table for a sweep."""
        if backend not in ("lift", "fmgs", "fmgs_c"):
            raise SemanticsError(f"unknown backend {backend!r} (lift, fmgs or fmgs_c)")
        if suffix and not re.fullmatch(r"_[a-z0-9]{1,16}", suffix):
            raise SemanticsError(f"invalid table suffix {suffix!r} (_ then up to 16 lower-case letters/digits)")
        return self.semantics_dir / self.run / f"{backend}{suffix}"

    @property
    def state_dir(self):
        return self.project_root / ".semantic_pipeline_state" / self.scene / self.run

    @property
    def runs_dir(self):
        return self.repo / "runs"


def find_scene_run(project_root, scene):
    """The scene's single active splatfacto model. FiGS itself requires exactly one under
    gsplats/workspace/outputs/<scene>/ (older ones go to gsplats/workspace/_archive/)."""
    if not NAME_RE.match(scene or ""):
        raise SemanticsError(f"invalid scene name {scene!r}")
    root = Path(project_root)
    out = root / "SousVide" / "gsplats" / "workspace" / "outputs" / scene
    cfgs = sorted(out.rglob("config.yml")) if out.is_dir() else []
    if not cfgs:
        raise SemanticsError(f"{scene} has no trained model under {out}: run figs_pipeline.py through `train` first")
    if len(cfgs) > 1:
        runs = ", ".join(c.parent.name for c in cfgs)
        raise SemanticsError(f"{scene} has {len(cfgs)} trained models ({runs}); FiGS needs exactly one — "
                             "archive the others (Galley's Scene page, or move them to gsplats/workspace/_archive/)")
    return SceneRun(root, scene, cfgs[0])
