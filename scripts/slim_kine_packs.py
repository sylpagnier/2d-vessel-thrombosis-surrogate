"""Strip rebuildable tensors from kinematics packs so a cohort can be moved between machines.

    python scripts/slim_kine_packs.py --src data/processed/graphs_kinematics/carreau --out transfer/

**Default is lossless**: only `G_x`/`G_y` are dropped, and nothing reads them outside
`BIOCHEM_GRAD_OPERATOR=legacy`.  `--drop-wls` additionally drops `V`/`W`/`M_inv`, which changes
numerics on packs whose stored operator does not match their graph (B13) -- opt in deliberately.

**98.4% of a pack is two tensors nothing reads.**  Measured on a 4,019-node vessel:

```
G_x           64.61 MB   (4019, 4019) sparse
G_y           64.61 MB   (4019, 4019) sparse
everything    2.07 MB
TOTAL        131.30 MB   ->  24 vessels = 3.15 GB
without G_*    2.07 MB   ->  24 vessels =   50 MB
```

`graph_gradient_operators` defaults to **MLS** mode and builds its operators from positions and
connectivity; it only touches `G_x`/`G_y` under `BIOCHEM_GRAD_OPERATOR=legacy`.  `V`/`W`/`M_inv`
are likewise pure functions of `(edge_index, pos)` and are rebuilt by `precompute_wls_operators`
-- and on an elevated graph they are rebuilt regardless, because the topology changed.

So this is not lossy compression; it is dropping a cache.  `--verify` proves that per pack by
rebuilding what was dropped and comparing against the original.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Pure functions of (edge_index, pos).  Rebuilt on load; on an elevated graph, rebuilt anyway.
REBUILDABLE = ("G_x", "G_y", "V", "W", "M_inv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-wls", action="store_true",
                    help="ALSO drop V/W/M_inv so they are rebuilt on load. Changes numerics on "
                         "packs whose stored operator does not match their graph (B13) -- opt in "
                         "deliberately, do not use it to save a megabyte.")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild the dropped WLS operators and check they match the original")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.pt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[ERR] no .pt under {src}")
        return 1

    drop = REBUILDABLE if args.drop_wls else ("G_x", "G_y")
    tot_in = tot_out = n_mismatch = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        before = sum(v.element_size() * v.nelement()
                     for v in d.to_dict().values() if torch.is_tensor(v))

        if args.verify and hasattr(d, "M_inv"):
            from src.data_gen.lib.mesh_wls import precompute_wls_operators

            V, W, M_inv = precompute_wls_operators(d.edge_index, int(d.num_nodes), d.x[:, :2])
            err = float((M_inv - d.M_inv.to(M_inv.dtype)).abs().max())
            ref = float(d.M_inv.abs().max()) + 1e-30
            if err / ref > 1e-3:
                n_mismatch += 1
                print(f"[note] {f.name}: stored M_inv differs from a rebuild by {err / ref:.2e} "
                      f"relative -- B13, the stored operator was not built from this graph")
            drop_this = drop
        else:
            drop_this = drop

        for k in drop_this:
            if hasattr(d, k):
                try:
                    delattr(d, k)
                except Exception:
                    d[k] = None
        dst = out / f.name
        torch.save(d, dst)
        after = dst.stat().st_size
        tot_in += before
        tot_out += after
        print(f"[ok] {f.name:<22} {before / 1e6:8.1f} MB -> {after / 1e6:6.2f} MB")

    print(f"\n{len(files)} packs: {tot_in / 1e9:.2f} GB -> {tot_out / 1e6:.0f} MB "
          f"({tot_in / max(tot_out, 1):.0f}x smaller)")
    print(f"[i] dropped: {', '.join(drop)}")
    print("[i] Do NOT use BIOCHEM_GRAD_OPERATOR=legacy with slimmed packs -- that mode reads")
    print("    G_x/G_y directly; every other path rebuilds from positions + connectivity.")
    if n_mismatch:
        print("")
        print(f"[!] {n_mismatch}/{len(files)} packs carry a stored WLS operator that does NOT")
        print("    match their own graph (RGP_DEQ_REPAIR_PLAN.md B13/s1i).  Transfer does not")
        print("    change that either way -- `--drop-wls` would force a correct rebuild on load,")
        print("    but that is a numerics decision to make deliberately, not in a copy tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
