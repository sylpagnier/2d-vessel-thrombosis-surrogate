"""Write `configs/mirror_branch.json`: which vessels sit on the mirror flow branch.

For each anchor pack: is the mesh mirror-symmetric, and is the solved FEM t=0 flow COMSOL's flow
or COMSOL's flow reflected?  The rule and why it is decided from flow is in
`src/clot_ml/mirror_branch.py`; this only applies it.  Re-run when the pack corpus or the FEM
solver changes, and review the diff -- every published score reads this file.

    python scripts/build_mirror_branch_registry.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.clot_ml.mirror_branch import BRANCH_MARGIN, REGISTRY, flow_branch, mirror_permutation
from src.clot_ml.v0 import solve_fem_into_pack
from src.config import BiochemConfig
from src.core_physics.physics_wall_model import t0_flow_fields
from src.utils.paths import anchor_packs_dir
from src.utils.safe_load import load_untrusted_graph


def audit(stem: str) -> dict | None:
    d = load_untrusted_graph(anchor_packs_dir() / f"{stem}.pt")   # tensors + graph classes only
    d.graph_stem = stem
    m = mirror_permutation(d.x[:, :2].numpy())
    if not m.symmetric:
        return None
    solve_fem_into_pack(d)
    bio = BiochemConfig(phase="biochem")
    g = t0_flow_fields(d, bio, hops=3, flow_source="gt")
    f = t0_flow_fields(d, bio, hops=3, flow_source="fem")
    return flow_branch(np.stack([f.u, f.v], 1), np.stack([g.u, g.v], 1), m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    stems = sorted(p.stem for p in anchor_packs_dir().glob("*.pt"))
    rows = {}
    for s in stems:
        r = audit(s)
        if r is not None:
            rows[s] = r
            print(f"  {s:<20} {json.dumps(r)}", flush=True)
    REGISTRY.write_text(json.dumps(dict(
        rule=("score against mirrored GT iff the mesh is mirror-symmetric and the solved FEM t=0 "
              f"flow is >{BRANCH_MARGIN:g}x closer (rel-L2) to COMSOL's reflected flow than to "
              "COMSOL's flow; decided from flow only, applied to evaluation labels only "
              "(src/clot_ml/mirror_branch.py)"),
        n_packs_scanned=len(stems), vessels=rows), indent=2) + "\n", encoding="utf-8")
    flipped = [s for s, r in rows.items() if r["mirror_branch"]]
    print(f"[i] {len(rows)} symmetric of {len(stems)}; mirror branch: {flipped} -> {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
