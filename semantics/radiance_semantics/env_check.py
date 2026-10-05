"""Versions of the ABI-sensitive stack in kitchen, and a guard that nothing moved.

tiny-cuda-nn and gsplat are C++/CUDA extensions compiled against torch 2.1.2 and numpy 1.x.
If an install drags torch (or numpy 2) in, they break with errors that look unrelated
("undefined symbol: at::_ops::...", "_ARRAY_API not found"). Reading versions from package
metadata needs no import, so this runs in a second and never initialises CUDA.

    python -m radiance_semantics.env_check                     # print, exit 1 on a broken pin
    python -m radiance_semantics.env_check --snapshot before.json
    python -m radiance_semantics.env_check --compare before.json   # exit 1 if a watched one changed
"""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Hard pins: the FiGS stack is validated with exactly these.
PINS = {"torch": "2.1.2", "nerfstudio": "1.1.4", "gsplat": "1.0.0"}
# Must not change during any semantic install (compiled extensions and their deps).
WATCH = ("torch", "torchvision", "numpy", "timm", "gsplat", "nerfstudio", "tinycudann")
# Informational: what the semantic step adds.
EXTRA = ("open_clip_torch", "ftfy", "regex", "huggingface_hub", "radiance-semantics")


def _v(dist):
    try:
        return version(dist).split("+")[0]
    except PackageNotFoundError:
        return None


def versions():
    return {d: _v(d) for d in WATCH + EXTRA}


def problems(vers):
    out = []
    for d, pin in PINS.items():
        if vers.get(d) is None:
            out.append(f"{d} is not installed (expected {pin})")
        elif vers[d] != pin:
            out.append(f"{d} is {vers[d]}, expected {pin} — a moved torch stack invalidates tiny-cuda-nn and gsplat")
    nv = vers.get("numpy")
    if nv and int(nv.split(".")[0]) >= 2:
        out.append(f"numpy {nv}: extensions in kitchen are built for the 1.x ABI")
    return out


def changed(before, after):
    """Watched packages whose version differs from the snapshot (installed before only)."""
    return [f"{d}: {before[d]} -> {after.get(d)}" for d in WATCH
            if before.get(d) is not None and before[d] != after.get(d)]


def constraints(exclude=("radiance-semantics", "open-clip-torch")):
    """`name==version` for EVERY distribution in the env, for `pip install -c`. Built from package
    metadata rather than `pip freeze`, which writes conda-installed packages as `name @ file://…`
    lines that cannot be used as constraints — numpy among them in kitchen."""
    import re
    from importlib.metadata import distributions
    seen, lines = set(), []
    for d in distributions():
        name = d.metadata.get("Name")
        if not name:
            continue
        norm = re.sub(r"[-_.]+", "-", name).lower()
        if norm in seen or norm in exclude:
            continue
        seen.add(norm)
        lines.append(f"{name}=={d.version}")
    return sorted(lines, key=str.lower)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", metavar="FILE", help="write the current versions as JSON")
    ap.add_argument("--compare", metavar="FILE", help="fail if a watched package differs from FILE")
    ap.add_argument("--constraints", metavar="FILE", help="write a pip constraints file freezing every installed package")
    a = ap.parse_args(argv)

    if a.constraints:
        lines = constraints()
        Path(a.constraints).write_text("\n".join(lines) + "\n")
        print(f"  {len(lines)} installed packages frozen in {a.constraints}")

    now = versions()
    width = max(len(d) for d in now)
    for d, v in now.items():
        print(f"  {d:<{width}}  {v or '-'}")

    if a.snapshot:
        Path(a.snapshot).write_text(json.dumps(now, indent=2) + "\n")
        print(f"  snapshot written to {a.snapshot}")

    bad = problems(now)
    if a.compare:
        diff = changed(json.loads(Path(a.compare).read_text()), now)
        if diff:
            bad += [f"changed during install: {x}" for x in diff]
        else:
            print("  no watched package changed")
    for b in bad:
        print(f"  FATAL: {b}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
