"""Lock `clot_gnn_v4w` -- v4 plus the wound complement -- as a deployable artifact.

WHAT THIS IS AND IS NOT.  The GNN ensemble, the temporal head, the readout and every
threshold are **byte-identical to `clot_gnn_v4`**: this artifact adds a boundary-condition
branch, not a retrained model.  COMSOL's wound law is the wall law with the two shear gates
deleted (`docs/WOUND_PROGRESS.md` 1), and v4 has no channel that can represent that -- on the
three wound vessels **100% of wound nodes clot and the t=0 gate fires on 0% of them**, so v4
scores that domain at exactly 0.

Why it may supersede v4 rather than sit beside it: **on a pack with no wound it returns v4's
own output unchanged**, and that is asserted per vessel here at promotion time, not argued.

Deploy numbers it reproduces (`scripts/eval_wound_complement.py`, LOVO constants, GT t=0
flow, cohort mean over the three wound vessels):

              FIN wall  FIN wound  FIN off  FIN full | MOT wall  MOT wound  MOT off  MOT full
    v4          0.7879     0.0000   0.0600   0.4989  |   0.6230     0.0000   0.0063   0.3662
    v4w         0.7879     1.0000   0.5883   0.8058  |   0.6230     0.9445   0.5871   0.7230

`FIN wall` / `MOT wall` are identical by construction -- the healthy wall is untouched.

The fitted content is TWO SCALARS, `(G_pre, G_post)`, refit on all three wound vessels here;
the leave-one-vessel-out evidence that they generalise is in
`outputs/clot_ml/wound_rate/lovo.json` and is reported next to them.  The per-node
`WoundRateNet` LOSES leave-one-vessel-out at n=3 and is deliberately not part of this
artifact.

    outputs/clot_ml/locked/clot_gnn_v4w/manifest.json    (kind -> temporal_v4_wound)
    data/reference/clot_gnn_locked.json                  (repointed with --repoint)

    python scripts/promote_clot_gnn_v4_wound.py
    python scripts/promote_clot_gnn_v4_wound.py --repoint
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.locked import (  # noqa: E402
    load_temporal_v4, predict_temporal_v4, predict_temporal_v4_wound,
)
from src.clot_ml.wound import (  # noqa: E402
    G_POST0, G_PRE0, OFFWALL_LAG_FRAC, TRIGGER_HOPS, has_wound,
)

#: DEFAULTS ONLY -- `--base` / `--name` rebind these in `main`.  They are module globals
#: because nine call sites read them; hardcoding the generation meant a new base ensemble
#: could not be given its wound complement without editing this file.
BASE = "clot_gnn_v4"
NAME = "clot_gnn_v4w"
LOCKED = REPO / "outputs/clot_ml/locked"
POINTER = REPO / "data/reference/clot_gnn_locked.json"
RATE = REPO / "outputs/clot_ml/wound_rate"
GRAPHS = REPO / "data/processed/graphs_biochem_anchors"
WOUND_STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
#: No-wound packs the artifact must leave bit-identical. Spot-check, not the whole cohort --
#: the property is structural (`has_wound` short-circuits), these catch a wiring mistake.
NOWOUND_CHECK = ("patient012", "patient020", "patient044")


def _rate_constants() -> tuple[dict, dict]:
    """(G_pre, G_post) refit on all three vessels, plus the LOVO evidence for them."""
    p = RATE / "lovo.json"
    if not p.exists():
        raise SystemExit(f"[ERR] {p} missing -- run scripts/train_wound_rate.py first")
    blob = json.loads(p.read_text())
    fa = blob.get("fitted_all") or {}
    if "g_pre" not in fa:
        raise SystemExit(f"[ERR] {p} has no fitted_all block")
    return (dict(g_pre=float(fa["g_pre"]), g_post=float(fa["g_post"])),
            dict(summary=blob.get("summary"), folds=blob.get("folds")))


def _assert_noop_without_wound(bundle: dict) -> list[str]:
    """The whole licence for superseding v4: no wound, no change. Asserted, not argued."""
    checked = []
    for stem in NOWOUND_CHECK:
        p = GRAPHS / f"{stem}.pt"
        if not p.exists():
            print(f"    [skip] {stem} not on disk")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        if has_wound(data):
            raise SystemExit(f"[ERR] {stem} unexpectedly carries a wound mask")
        T = int(data.y.shape[0])
        times = sorted({0, T // 3, 2 * T // 3, T - 1})
        a = predict_temporal_v4(bundle["base"], data, times, flow="gt")
        b = predict_temporal_v4_wound(bundle, data, times, flow="gt")
        if not np.array_equal(a["mask"], b["mask"]):
            raise SystemExit(f"[ERR] {stem}: v4w changed the no-wound MASK")
        if not np.array_equal(np.asarray(a["onset"]), np.asarray(b["onset"])):
            raise SystemExit(f"[ERR] {stem}: v4w changed the no-wound ONSET")
        for ti in times:
            if not np.array_equal(a["series"][int(ti)], b["series"][int(ti)]):
                raise SystemExit(f"[ERR] {stem}: v4w changed the no-wound SERIES at t={ti}")
        print(f"    [OK] {stem}: bit-identical to {BASE}")
        checked.append(stem)
    if not checked:
        raise SystemExit("[ERR] no no-wound pack available to verify the no-op property")
    return checked


def _assert_wound_is_covered(bundle: dict) -> dict:
    """And the other half: on a wound pack it must actually commit the injured segment."""
    from src.clot_ml.wound import wound_mask

    out = {}
    for stem in WOUND_STEMS:
        p = GRAPHS / f"{stem}.pt"
        if not p.exists():
            print(f"    [skip] {stem} not on disk")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        w = wound_mask(data)
        base = predict_temporal_v4(bundle["base"], data, [T - 1], flow="gt")
        new = predict_temporal_v4_wound(bundle, data, [T - 1], flow="gt")
        cov_b, cov_n = float(base["mask"][w].mean()), float(new["mask"][w].mean())
        if cov_n < 0.99:
            raise SystemExit(f"[ERR] {stem}: v4w commits only {cov_n:.1%} of the wound")
        untouched = ~w & ~np.asarray(new["owned"], bool)
        if not np.array_equal(base["mask"][untouched], new["mask"][untouched]):
            raise SystemExit(f"[ERR] {stem}: v4w changed nodes it does not own")
        print(f"    [OK] {stem}: wound coverage {cov_b:.1%} -> {cov_n:.1%}, "
              f"{int(np.asarray(new['owned']).sum())} nodes owned, rest untouched")
        out[stem] = dict(wound_coverage_v4=cov_b, wound_coverage_v4w=cov_n)
    if not out:
        raise SystemExit("[ERR] no wound pack available to verify coverage")
    return out


def main() -> int:
    global BASE, NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE,
                    help="base ensemble this complements (default: %(default)s)")
    ap.add_argument("--name", default="",
                    help="artifact name; default is <base> + 'w'")
    ap.add_argument("--repoint", action="store_true",
                    help="point data/reference/clot_gnn_locked.json at this artifact")
    args = ap.parse_args()
    BASE = args.base
    NAME = args.name or (BASE + "w")

    if not (LOCKED / BASE / "manifest.json").exists():
        raise SystemExit(f"[ERR] base artifact {BASE} not found under {LOCKED}")
    rate, evidence = _rate_constants()
    print(f"[i] wound rate constants (refit on all 3): "
          f"G_pre={rate['g_pre']:.3f} G_post={rate['g_post']:.3f}")
    if evidence.get("summary"):
        c = evidence["summary"].get("const", {})
        print(f"    LOVO evidence: onset MAE {c.get('onset_mae', float('nan')):.1f} steps "
              f"({c.get('onset_mae_frac', float('nan'))*100:.1f}% of horizon), "
              f"recall {c.get('recall', float('nan')):.3f}")

    root = LOCKED / NAME
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        name=NAME,
        kind="temporal_v4_wound",
        base_model=BASE,
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "clot_gnn_v4 plus the wound complement. The GNN ensemble, temporal head and "
            "readout are byte-identical to clot_gnn_v4; this adds a boundary-condition "
            "branch for injured wall, where COMSOL deletes the two shear gates from the "
            "deposition law. Returns v4's output unchanged on any pack without a wound mask."
        ),
        docs="docs/WOUND_PROGRESS.md",
        supersedes=BASE,
        wound=dict(
            g_pre=rate["g_pre"], g_post=rate["g_post"],
            off_att=0.16, lag_frac=OFFWALL_LAG_FRAC,
            trigger="self", k_hops=TRIGGER_HOPS,
            defaults=dict(g_pre0=G_PRE0, g_post0=G_POST0),
            fitted_on=list(WOUND_STEMS),
            note=("G_pre recovers ungated(1) + low-shear(1) = 2 independently in all three "
                  "LOVO folds (2.04 / 1.96 / 2.01). G_post is the genuinely fitted one and "
                  "is not stable across folds (22.1 / 20.9 / 11.9). The per-node "
                  "WoundRateNet loses LOVO at n=3 and is NOT in this artifact."),
        ),
        wound_rate_evidence=evidence,
        caveats=[
            "n=3 wound vessels; treat the magnitude as indicative and the sign as solid.",
            "GT flow at t=0 only (flow='gt'); not yet measured with predicted t=0 flow.",
            "wound_patient003 is externally triggered and remains an outlier "
            "(WOUND_PROGRESS 11); its coupling is built but inert on the deploy-legal path.",
            "mask_wall stays the HEALTHY wall label BY DECISION (MODEL_REVIEW 8e): "
            "mask_wound is 100% GT clot, so folding it into the wall domain would award "
            "free true positives. As of 2026-08-22 the off-wall domain is ~solid, true "
            "lumen, so the wound is in neither global domain -- it is scored by "
            "wound_region_masks. See src/clot_ml/data.eval_domains.",
        ],
    )

    print("[i] verifying the no-op property on no-wound packs")
    bundle = dict(base=load_temporal_v4(name=BASE), wound=manifest["wound"],
                  manifest=manifest)
    manifest["verified_noop_on"] = _assert_noop_without_wound(bundle)
    print("[i] verifying wound coverage")
    manifest["verified_wound"] = _assert_wound_is_covered(bundle)

    if (RATE / "wound_rate.pt").exists():
        shutil.copy2(RATE / "wound_rate.pt", root / "wound_rate.pt")
        manifest["wound_rate_file"] = "wound_rate.pt"
        print(f"    [i] copied wound_rate.pt (reference only; the deploy path uses the "
              f"two scalars above)")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[save] {root / 'manifest.json'}")

    if args.repoint:
        ptr = json.loads(POINTER.read_text())
        ptr.update(name=NAME, kind="temporal_v4_wound",
                   path=f"outputs/clot_ml/locked/{NAME}",
                   manifest=f"outputs/clot_ml/locked/{NAME}/manifest.json",
                   promoted_at=manifest["promoted_at"], docs="docs/WOUND_PROGRESS.md",
                   supersedes=BASE)
        # REFRESH THE EMBEDDED SCORES FROM THE ARTIFACT.  `dict.update` leaves untouched keys
        # alone, so `scores_strict_cv` and `readout` survived every repoint and went on
        # describing an artifact two generations old -- the pointer read wall 0.9176 /
        # off 0.7366 while resolving to weights that score 0.9008 / 0.5812.  The pointer must
        # never carry a number the artifact does not.
        base_manifest = LOCKED / BASE / "manifest.json"
        if base_manifest.exists():
            bm = json.loads(base_manifest.read_text())
            for k in ("scores_strict_cv", "readout", "fingerprint"):
                if k in bm:
                    ptr[k] = bm[k]
                elif k in ptr:
                    del ptr[k]
            ptr["scores_from"] = f"{BASE}/manifest.json"
        POINTER.write_text(json.dumps(ptr, indent=2))
        print(f"[save] {POINTER} -> {NAME}")
    else:
        print(f"[i] pointer NOT changed; re-run with --repoint to ship {NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
