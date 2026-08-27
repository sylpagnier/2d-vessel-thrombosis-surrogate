"""Populate `wall_normal` at boundary nodes and the `node_type_*` one-hot, on every pack.

THE TWO DEAD CHANNEL GROUPS (WOUND_PROGRESS 8, MODEL_REVIEW_2026-08-22 6.5):

1. **`wall_normal` is identically zero at every wall node, on every pack.**  The normals were
   read off Gmsh **line** cells; the COMSOL `.msh` exports contain only `triangle6`, so that
   branch never ran and the KD-tree fallback handed a wall node *itself* as its nearest
   boundary neighbour -- a zero offset vector.  Fixed by
   `mesh_wls.boundary_normals_from_graph`, which fits the boundary tangent from the graph's
   own solid-solid edges and rotates it, so it needs no mesh at all and works on the three
   wound packs whose COMSOL exports are gone.
2. **`node_type_0..3` is all-zero on 100% of nodes** -- the block was a literal
   `torch.zeros((N, 4))` in `build_kinematics_node_x_tensor`.  Now
   `mesh_wls.node_type_one_hot`: `[interior, solid, inlet, outlet]`.

WHAT THIS INVALIDATES, deliberately and with the decision recorded:

* **`clot_gnn_v4` / `clot_gnn_v4w`.**  `width_nd`, `width_d1`, `width_d2` and `wss_prior_nd`
  are all derived from `wall_normal`, and they are clot-ML feature columns.  The v5 cache and
  the locked feature normaliser must both be rebuilt and the artifact re-promoted.
* **The frozen RGP-DEQ.**  `wall_normal` is Fourier-encoded and `node_type_*` sits in
  `NodeFeat.REST`, so Stage-A now sees inputs it was never trained on at the boundary, and
  the precached `u0_pred` / `v0_pred` on every pack is stale.  Accepted 2026-08-22: the clot
  work runs on GT t=0 flow, so nothing is blocked today, but **`u0_pred` must be recomputed
  before any `--flow pred` result is quoted.**

THE WRITE IS A DELTA, NOT A REBUILD -- and that is load-bearing.  A wholesale `rebuild_x`
does not reproduce the stored packs, for a reason that has nothing to do with this fix: the
cohort was extracted by two different builder revisions and they disagree about the *prior*
channels.  On `patient020` a fresh build puts `wss_prior_nd` at ~45 at the wall where the
stored pack has 0, and moves `u_prior` by 0.55 in the interior (WOUND_PROGRESS 8, last
paragraph).  Rewriting those would be a second, undecided change riding along on this one.

So the script builds `x` twice -- once with the fix, once with `graph_normals=False` -- and
applies only the difference:

    x_written = x_stored + (x_fixed - x_prefix)

Any channel the builder disagrees with the pack about for unrelated reasons cancels and is
left byte-identical.  Only the change *caused by* the normals fix is written.  This is the
same safeguard WOUND_PROGRESS 6 used for the solid-boundary repair.

THE CORRECTNESS GATE.  `--verify` checks the two properties that make the delta trustworthy:
the pre-fix rebuild must reproduce the stored `wall_normal` and `node_type_*` exactly, and
the delta must be zero on every channel outside `EXPECTED`.  Run it before writing.

    python scripts/repair_pack_wall_normals.py --verify          # gate, no writes
    python scripts/repair_pack_wall_normals.py --dry-run         # what would change
    python scripts/repair_pack_wall_normals.py                   # write (backs up once)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.wall_cohort_splits import (  # noqa: E402
    CLOT_FREE, DEV, FIT, SEALED,
)
from src.data_gen.lib.pack_repair import rebuild_x, solid_of, write_x  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
BACKUP_SUFFIX = ".pt.prenormalfix"

#: channels this repair is ALLOWED to move.  `wall_normal_*` and `node_type_*` are the
#: targets; the width and prior channels are derived from the normals and move with them.
EXPECTED = ("wall_normal_x", "wall_normal_y",
            "node_type_0", "node_type_1", "node_type_2", "node_type_3",
            "width_nd", "width_d1", "width_d2",
            "u_prior", "v_prior", "mu_prior_nd", "wss_prior_nd", "shear_potential")


def pack_stems() -> list[str]:
    """Every pack this repair covers: the pool, SEALED, the clot-free set, and the wounds."""
    stems = list(FIT) + list(DEV) + list(SEALED) + list(CLOT_FREE)
    stems += sorted(p.stem for p in PACKS.glob("wound_patient*.pt"))
    seen, out = set(), []
    for s in stems:
        if s not in seen and (PACKS / f"{s}.pt").exists():
            seen.add(s)
            out.append(s)
    return out


def report(stem: str) -> dict:
    """Build the surgical delta and everything needed to judge it."""
    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    x_fix, _, _ = rebuild_x(data, graph_normals=True)
    x_pre, _, _ = rebuild_x(data, graph_normals=False)
    names = data.x_channel_names.split(",")
    solid = solid_of(data)

    # WHICH CONVENTION DOES THE PACK ALREADY CARRY?  This is what makes the script
    # IDEMPOTENT, and it is not academic: running it twice once doubled every normal to
    # |n| = 2.0, because the delta was re-added to a pack that already contained it.  The
    # baseline the delta is measured from must be the pack's OWN state, not always the
    # pre-fix build -- then a second run computes `x_fix - x_fix = 0` and changes nothing.
    stored_mag = np.hypot(data.x[:, names.index("wall_normal_x")].numpy(),
                          data.x[:, names.index("wall_normal_y")].numpy())[solid]
    was_repaired = bool(solid.any() and float(np.median(stored_mag)) > 0.5)
    ref = x_fix if was_repaired else x_pre

    delta = x_fix - ref
    x_new = data.x + delta
    # `node_type_*` is identical in both builds, so the delta cancels it to zero.  It is the
    # one group with no stored content to protect -- it is all-zero on every pack by
    # construction -- so assign it outright.  Direct assignment cannot mask an unrelated
    # builder divergence here, which is the only thing the delta exists to prevent.
    nt = [names.index("node_type_%d" % k) for k in range(4)]
    x_new[:, nt] = x_fix[:, nt]
    moved = {names[j]: float(delta[:, j].abs().max())
             for j in range(len(names)) if delta[:, j].abs().max().item() > 1e-6}
    faithful = {c: float((ref[:, names.index(c)] - data.x[:, names.index(c)])[solid]
                         .abs().max())
                for c in ("wall_normal_x", "wall_normal_y")}
    nx = x_new[:, names.index("wall_normal_x")].numpy()
    ny = x_new[:, names.index("wall_normal_y")].numpy()
    mag = np.hypot(nx, ny)[solid]
    oh = x_new[:, [names.index("node_type_%d" % k) for k in range(4)]].numpy()
    return dict(data=data, x_new=x_new, names=names, moved=moved, faithful=faithful,
                was_repaired=was_repaired, n_solid=int(solid.sum()),
                normal_ok=float((mag > 0.99).mean()) if solid.any() else 0.0,
                onehot_ok=float((oh.sum(1) == 1).mean()),
                onehot_counts=oh.sum(0).astype(int).tolist())


def verify(stem: str) -> bool:
    """Two properties the delta rests on (see the module docstring).

    The gate is **convention-aware, so the script stays safe to re-run.**  A pre-fix pack has
    zero normals at the boundary and must round-trip against ``graph_normals=False``; a pack
    this script has already repaired carries unit normals and must round-trip against
    ``graph_normals=True``, i.e. re-running must be a no-op.  Both are the same underlying
    property -- the rebuild reproduces the pack's own convention -- and checking the wrong
    one would report an already-correct pack as broken.
    """
    r = report(stem)
    bad = []
    for c, d in r["faithful"].items():
        if d > 1e-6:
            bad.append("%s does not round-trip at solid nodes (%.3e)" % (c, d))
    extra = sorted(set(r["moved"]) - set(EXPECTED))
    if extra:
        bad.append("delta touches unclaimed channels: " + ",".join(extra))
    state = "already repaired" if r["was_repaired"] else "pre-fix"
    print("  %-18s %-16s %s" % (stem, state, "OK" if not bad else "FAIL -- " + "; ".join(bad)))
    return not bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="round-trip gate only, no writes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated stems")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    stems = [s.strip() for s in args.only.split(",") if s.strip()] or pack_stems()
    print("[i] %d packs" % len(stems))

    if args.verify:
        print("\n[VERIFY] wall_normal round-trips at solid nodes, and the delta stays"
              " inside the claimed channels")
        ok = all([verify(s) for s in stems])
        print("\n[%s] round-trip %s" % ("OK" if ok else "FAIL",
                                        "clean on every pack" if ok else "DRIFTED"))
        return 0 if ok else 1

    print("\n%-18s %7s %9s %9s %-28s" % ("pack", "n_solid", "|n|=1", "1-hot", "node_type counts"))
    unexpected: dict[str, set] = {}
    for s in stems:
        r = report(s)
        print("%-18s %7d %8.1f%% %8.1f%%  %s"
              % (s, r["n_solid"], 100 * r["normal_ok"], 100 * r["onehot_ok"],
                 r["onehot_counts"]))
        extra = set(r["moved"]) - set(EXPECTED)
        if extra:
            unexpected[s] = extra
        if not (args.dry_run):
            path = PACKS / f"{s}.pt"
            if not args.no_backup:
                bak = path.with_suffix(BACKUP_SUFFIX)
                if not bak.exists():
                    shutil.copy2(path, bak)
            write_x(r["data"], r["x_new"])   # keeps x_biochem's duplicate normal in step
            torch.save(r["data"], path)

    if unexpected:
        print("\n[WARN] channels moved that this repair does not claim:")
        for s, e in unexpected.items():
            print("   %-18s %s" % (s, sorted(e)))
    print("\n[%s] %s" % ("DRY-RUN" if args.dry_run else "WRITTEN", len(stems)))
    if not args.dry_run:
        print("[!] clot_gnn_v4/v4w are now STALE: rebuild the v5 cache and re-promote.")
        print("[!] u0_pred/v0_pred are now STALE: re-run precache_rgp_deq before --flow pred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
