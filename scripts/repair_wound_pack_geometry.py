"""Repair wall-derived geometry channels on wound packs built before the solid-boundary fix.

The extract carves ``mask_wall`` disjoint from ``mask_wound`` (COMSOL ``dif1`` vs ``sel1``) so
the gated and ungated deposition laws stay separable.  Builders then measured every
wall-derived feature against ``mask_wall`` alone, so an injured node -- which is still a
no-slip boundary node -- was encoded as open lumen: its SDF became the distance to the
nearest *un-wounded* wall node (0.11-0.32 diameters), and ``wall_normal`` / ``width_nd`` /
``wss_prior`` followed it off the wall.  Interior nodes above the wound inherited the same
error, since their nearest "wall" was also forced around the gap.

The builders now use ``solid_boundary_mask`` (``src/data_gen/lib/mesh_wls.py``), so freshly
extracted packs are correct.  This script fixes packs already on disk, for which the
multi-GB COMSOL exports are no longer available.  It rebuilds ``data.x`` through the same
``build_kinematics_node_x_tensor`` call the extractor uses, with the same
``resolve_anchor_kine_phys_cfg`` Carreau config -- only the boundary mask differs.  ``y``,
the masks, edges and WLS operators are untouched; they never depended on the wall set.

``--verify STEM`` is the correctness gate: it runs the identical rebuild on a **no-wound**
pack, where the union is a no-op, and reports the per-channel drift against what is stored.
If that round-trips, the same code applied to a wound pack is trustworthy.

Usage:
    python scripts/repair_wound_pack_geometry.py --verify patient012
    python scripts/repair_wound_pack_geometry.py --dry-run
    python scripts/repair_wound_pack_geometry.py
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

from src.data_gen.lib.pack_repair import rebuild_x, write_x  # noqa: E402

DEFAULT_DIR = Path("data/processed/graphs_biochem_anchors")
SIDECAR_DIR = Path("data/raw/biochem_anchors")


def _channel_report(old: torch.Tensor, new: torch.Tensor, names: list[str], where: dict) -> None:
    print(f"    {'channel':16s} {'max|diff|':>11s} " + " ".join(f"{k:>22s}" for k in where))
    for j, nm in enumerate(names):
        a, b = old[:, j].numpy(), new[:, j].numpy()
        cells = " ".join(
            f"{np.median(a[m]):10.4f}->{np.median(b[m]):10.4f}" for m in where.values()
        )
        print(f"    {nm:16s} {np.abs(a - b).max():11.4f} {cells}")


def verify(path: Path) -> None:
    """Round-trip the rebuild on a pack and report drift (no writes)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    wound = getattr(data, "mask_wound", None)
    n_wound = int(wound.sum()) if torch.is_tensor(wound) else 0
    # Pre-fix convention on BOTH axes, so this reports only the solid-boundary drift this
    # script is about -- the 2026-08-22 wall-normal / node_type repair is a separate change
    # with its own gate in scripts/repair_pack_wall_normals.py.
    x_new, _, _ = rebuild_x(data, graph_normals=False)
    wl = data.mask_wall.numpy()
    inter = ~(wl | data.mask_inlet.numpy() | data.mask_outlet.numpy())
    if n_wound:
        inter = inter & ~wound.numpy()
    print(f"\n{path.name}  n={data.x.shape[0]}  wound_nodes={n_wound}")
    _channel_report(
        data.x, x_new, data.x_channel_names.split(","), {"wall": wl, "interior": inter}
    )


def repair(path: Path, *, dry_run: bool, backup: bool) -> bool:
    data = torch.load(path, map_location="cpu", weights_only=False)
    wound = getattr(data, "mask_wound", None)
    if wound is None or not torch.is_tensor(wound) or not bool(wound.any()):
        return False

    names = data.x_channel_names.split(",")
    w = wound.numpy()
    wl = data.mask_wall.numpy()
    x_new, _, _ = rebuild_x(data)

    # Surgical write.  The rebuild reproduces every geometry channel on an untouched pack
    # exactly, but not the prior channels (packs in this cohort were written by more than one
    # extractor revision and their stored priors disagree).  So only rows whose wall reference
    # was actually wrong get rewritten -- the wound itself, plus the interior nodes whose
    # nearest boundary was forced around the gap.  Every other row stays bit-identical to the
    # pack v4 was validated against.
    sdf_j = names.index("sdf_nd")
    moved = (data.x[:, sdf_j] - x_new[:, sdf_j]).abs().numpy() > 1e-6
    rows = torch.from_numpy(moved | w)

    print(f"\n{path.name}  n={data.x.shape[0]}  wall={int(wl.sum())} wound={int(w.sum())}")
    print(f"  rows with a corrected wall reference: {int(rows.sum())}"
          f"  ({int(rows.sum()) / len(w) * 100:.1f}% of mesh; {int((rows.numpy() & ~w).sum())} of them interior)")
    for nm in ("sdf_nd", "wall_normal_y", "width_nd"):
        j = names.index(nm)
        a, b = data.x[:, j].numpy(), x_new[:, j].numpy()
        print(f"  {nm:14s} wound {np.median(a[w]):9.4f} -> {np.median(b[w]):9.4f}"
              f"   | healthy wall (reference) {np.median(a[wl]):9.4f} -> {np.median(b[wl]):9.4f}")
    if dry_run:
        return True

    if backup:
        bak = path.with_suffix(".pt.prewoundfix")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"  [i] backup -> {bak.name}")
    x_out = data.x.clone()
    x_out[rows] = x_new[rows]
    write_x(data, x_out)
    torch.save(data, path)
    print(f"  [OK] rewrote {path.name}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--verify", metavar="STEM", help="round-trip the rebuild on one pack, no writes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    if args.verify:
        verify(root / f"{args.verify}.pt")
        return

    paths = sorted(root.glob("*.pt"))
    if not paths:
        raise SystemExit(f"[ERR] no packs under {root}")

    touched = 0
    for p in paths:
        try:
            if repair(p, dry_run=args.dry_run, backup=not args.no_backup):
                touched += 1
        except Exception as exc:  # noqa: BLE001 - one bad pack must not abort the batch
            print(f"[WARN] {p.name}: {exc}")
    verb = "would repair" if args.dry_run else "repaired"
    print(f"\n[i] {verb} {touched} wound pack(s); {len(paths) - touched} carry no wound and were skipped")


if __name__ == "__main__":
    main()
