"""Put LUDVIG's diffused relevancy (ludvig_diffuse.py) into a table, in .splat order — kitchen env.

    source ~/Radiance/figs/figs_env.sh
    python semantics/baselines/import_diffused.py --scene backroom          # → lift_ludvigfull/ludvig_diffused.npz

Read by query.run_query when Settings.ludvig_diffusion is on (evaluator variant L-CL).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "semantics"), str(REPO / "figs")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--suffix", default="_ludvigfull")
    ap.add_argument("--project-root", default=str(Path.home() / "Radiance/figs"))
    a = ap.parse_args()

    import torch
    from course_tools import splat_order
    from radiance_semantics.paths import find_scene_run
    from radiance_semantics.store import check_order, read_table

    run = find_scene_run(Path(a.project_root), a.scene)
    table_dir = run.backend_dir("lift", a.suffix)
    raw = table_dir.with_name(table_dir.name + "_raw")
    ck = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
    st = ck.get("pipeline", ck)

    def p(name):
        return next(v for k, v in st.items() if k.endswith(f"gauss_params.{name}") or k.endswith(f"_model.{name}")
                    ).float().numpy()
    order = splat_order(p("means"), p("scales"), p("opacities").reshape(-1))
    table = read_table(table_dir)
    check_order(table, order)
    d = np.load(raw / "diffused.npz")
    texts = [str(t) for t in d["texts"]]
    np.savez_compressed(table_dir / "ludvig_diffused.npz", texts=np.array(texts),
                        diffused=d["diffused"][:, order].astype(np.float32),
                        key=np.array(json.loads((raw / "lift.json").read_text())["key"]))
    print(f"{len(texts)} diffused prompts → {table_dir / 'ludvig_diffused.npz'}")


if __name__ == "__main__":
    main()
