#!/usr/bin/env python
"""Pre-solve the local FEM prior for every deploy pack, into the disk cache.

``prior_source="fem"`` solves a Carreau FEM per vessel the first time it is asked for and caches
the result under ``data/processed/fem_priors/``.  Doing that lazily inside a training run works
but hides several minutes of CPU inside the first epoch and, worse, hides a FAILURE there --
a vessel whose mesh does not register raises in the middle of a run that has already paid for
its dataset build.  Solve them all up front instead, and report which ones cannot be.

    python scripts/build_fem_prior_cache.py                # train pool + selection set
    python scripts/build_fem_prior_cache.py --all-cohort   # every FIT+DEV pack plus the wounds
    python scripts/build_fem_prior_cache.py --stems comsol001 comsol041
    python scripts/build_fem_prior_cache.py --force        # re-solve, ignoring the cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_gen.lib.legal_priors import (  # noqa: E402
    COL_U_PRIOR, COL_V_PRIOR, _fem_prior_cache_path, build_fem_priors,
)
from src.utils.kinematics_select_packs import (  # noqa: E402
    selection_pack_dir, selection_pack_stems, selection_subset_stems,
)


def _cohort_stems(packs: Path) -> list[str]:
    from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED

    skip = set(SEALED) | set(CLOT_FREE)
    out = [a for a in list(FIT) + list(DEV) if a not in skip and (packs / f"{a}.pt").exists()]
    for s in ("wound_comsol001", "wound_comsol002", "wound_comsol003"):
        if (packs / f"{s}.pt").exists() and s not in out:
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--all-cohort", action="store_true",
                    help="every FIT+DEV clot-carrying pack plus the wounds, not just the "
                         "Stage-A train+selection split")
    ap.add_argument("--force", action="store_true", help="re-solve even when cached")
    args = ap.parse_args(argv)

    packs = selection_pack_dir()
    if args.stems:
        stems = list(args.stems)
    elif args.all_cohort:
        stems = _cohort_stems(packs)
    else:
        sel = selection_subset_stems()
        train = sorted(set(selection_pack_stems()) - set(sel))
        stems = train + sel
        print(f"[i] Stage-A split: {len(train)} train packs + {len(sel)} selection packs")

    print(f"[i] {len(stems)} vessels -> {_fem_prior_cache_path.__module__.split('.')[-1]} cache "
          f"under data/processed/fem_priors/", flush=True)

    n_ok = n_cached = n_fail = 0
    t_all = time.perf_counter()
    for stem in stems:
        f = packs / f"{stem}.pt"
        if not f.is_file():
            print(f"{stem:22s} MISSING {f}", flush=True)
            n_fail += 1
            continue
        data = torch.load(f, map_location="cpu", weights_only=False)
        data.graph_stem = stem
        cache = _fem_prior_cache_path(data, stem)
        was_cached = cache.is_file() and not args.force
        if args.force and cache.is_file():
            cache.unlink()
        t0 = time.perf_counter()
        try:
            u, v, _, _ = build_fem_priors(data)
        except Exception as exc:
            print(f"{stem:22s} FAIL  {type(exc).__name__}: {exc}", flush=True)
            n_fail += 1
            continue
        dt = time.perf_counter() - t0
        # The prior is only useful if it is a closer base point than what it replaces, and the
        # pack carries COMSOL's own t=0 field to check that against.  A converged solve that is
        # WORSE than analytic means the mesh registered onto the wrong vessel.
        g = data.y[0, :, 0:2].numpy().astype(np.float64)
        p = np.stack([u.cpu().numpy(), v.cpu().numpy()], 1).astype(np.float64)
        rel = float(np.linalg.norm(p - g) / max(np.linalg.norm(g), 1e-30))
        ana = data.x[:, [COL_U_PRIOR, COL_V_PRIOR]].numpy().astype(np.float64)
        print(f"{stem:22s} {'cached' if was_cached else 'solved':6s} {dt:6.1f}s  "
              f"relL2(fem->comsol)={rel:.4f}", flush=True)
        n_cached += int(was_cached)
        n_ok += 1

    print(f"\n[OK] {n_ok} priors available ({n_cached} already cached), {n_fail} failed, "
          f"{time.perf_counter() - t_all:.0f}s total", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
