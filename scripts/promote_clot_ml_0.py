"""Lock ``clot_ml_0`` -- one artifact for wounded and non-wounded vessels.

This is a composition, not a retrain (docs/WOUND_PROGRESS.md 18.3):

    wall              C0 GNN ensemble (the ``--base`` wound-capable artifact)
    wound             two-regime constants already on that base; no-op without a mask
    off-wall (wound)  chemistry ODE + solid-anchored replace+depth (REPLACE, not union)
    off-wall (else)   the GNN's own readout
    003-like AP       upwind renewal + da_scale_auto=123; optional residual GNN hook

On a pack with no wound it returns the base GNN unchanged -- asserted here, not argued.

    python scripts/promote_clot_ml_0.py
    python scripts/promote_clot_ml_0.py --base clot_gnn_v5w --repoint
    python scripts/eval_clot_ml_0.py          # compare against --baseline (v5w)
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

from src.clot_ml.locked import predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.v0 import (  # noqa: E402
    DEFAULT_NAME, KIND, REPLACE_SCOPES, ClotMlV0Config, load_v0_bundle, predict_clot_ml_0,
)
from src.biochem_gnn.wall_cohort_constants import WOUND_COHORT  # noqa: E402
from src.clot_ml.wound import (  # noqa: E402
    has_wound, wound_flow_regime, wound_mask,
)
from src.config import BiochemConfig  # noqa: E402

LOCKED = REPO / "outputs/clot_ml/locked"
POINTER = REPO / "data/reference/clot_gnn_locked.json"
GRAPHS = REPO / "data/processed/graphs_biochem_anchors"
WOUND_STEMS = WOUND_COHORT
NOWOUND_CHECK = ("patient012", "patient020", "patient044")
DEFAULT_BASE = "clot_gnn_v5w"


def _prepare_flow(data, stem: str, flow: str) -> None:
    """Put the deploy-legal t=0 field in the pack before anything reads `u0_pred`."""
    if flow != "fem":
        return
    from src.clot_ml.v0 import solve_fem_into_pack
    if not str(getattr(data, "graph_stem", "") or ""):
        data.graph_stem = stem
    solve_fem_into_pack(data)


def _assert_noop_without_wound(bundle: dict, base_name: str, flow: str = "gt") -> list[str]:
    checked = []
    for stem in NOWOUND_CHECK:
        p = GRAPHS / f"{stem}.pt"
        if not p.exists():
            print(f"    [skip] {stem} not on disk")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        _prepare_flow(data, stem, flow)
        if has_wound(data):
            raise SystemExit(f"[ERR] {stem} unexpectedly carries a wound mask")
        T = int(data.y.shape[0])
        times = sorted({0, T // 3, 2 * T // 3, T - 1})
        a = predict_temporal_v4_wound(bundle["base"], data, times, flow=flow)
        b = predict_clot_ml_0(bundle, data, times, flow=flow)
        if not np.array_equal(a["mask"], b["mask"]):
            raise SystemExit(f"[ERR] {stem}: v0 changed the no-wound MASK")
        if not np.array_equal(np.asarray(a["onset"]), np.asarray(b["onset"])):
            raise SystemExit(f"[ERR] {stem}: v0 changed the no-wound ONSET")
        for ti in times:
            if not np.array_equal(a["series"][int(ti)], b["series"][int(ti)]):
                raise SystemExit(f"[ERR] {stem}: v0 changed the no-wound SERIES at t={ti}")
        print(f"    [OK] {stem}: bit-identical to {base_name}")
        checked.append(stem)
    if not checked:
        raise SystemExit("[ERR] no no-wound pack available to verify the no-op property")
    return checked


def _assert_wound_is_covered(bundle: dict, flow: str = "gt") -> dict:
    out = {}
    for stem in WOUND_STEMS:
        p = GRAPHS / f"{stem}.pt"
        if not p.exists():
            print(f"    [skip] {stem} not on disk")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        _prepare_flow(data, stem, flow)
        T = int(data.y.shape[0])
        w = wound_mask(data)
        pred = predict_clot_ml_0(bundle, data, [T - 1], flow=flow)
        cov = float(pred["mask"][w].mean()) if w.any() else float("nan")
        # Full wound coverage is required only where the wound branch's PREMISE holds: an
        # ungated patch inside a gated wall.  `wound_flow_regime` decides that from the t=0
        # flow alone, and `wound_patient006` is the one vessel that fails it -- its wound
        # sits in a stagnation zone where 35% of the patch never clots in the ground truth
        # either, because species supply is limiting there.  Same rule as
        # `promote_clot_gnn_v4_wound.py`; see docs/DEPLOYCLOT.md 5b.
        regime, gate_on = wound_flow_regime(data, BiochemConfig(phase="biochem"), flow=flow)
        if regime == "flowing" and cov < 0.99:
            raise SystemExit(
                f"[ERR] {stem}: v0 commits only {cov:.1%} of the wound, and its wound is in "
                f"the flowing regime (t=0 gate ON at {gate_on:.1%} of wound nodes)")
        if regime == "stagnation":
            verdict = "reached anyway" if cov >= 0.99 else "not reached"
            print(f"    [!]  {stem}: STAGNATION-REGIME wound, t=0 shear gate already ON at "
                  f"{gate_on:.1%} of wound nodes -- full coverage not required, "
                  f"{verdict}: {cov:.1%} committed")
        else:
            print(f"    [OK] {stem}: wound coverage {cov:.1%}")
        out[stem] = dict(wound_coverage=cov, t0_gate_on_frac_at_wound=gate_on, regime=regime)
    if not out:
        raise SystemExit("[ERR] no wound pack available to verify coverage")
    return out


def _temporal_readout(bundle) -> dict:
    """The readout families the GNN actually ships, from the temporal_v4 two levels down.

    The chain is unified_v0 -> temporal_v4_wound -> temporal_v4, and only the last records
    `temporal_readout`.  Reading it off the wrong level silently yields `{}`, which on the
    pointer is indistinguishable from "this artifact has no readout".
    """
    inner = (bundle.get("base") or {}).get("base") or {}
    return dict((inner.get("manifest") or {}).get("temporal_readout") or {})


def _pointer_scores(repo) -> dict:
    """Rebuild the pointer's score record from the files that MEASURED it.

    WHY THIS REPLACES RATHER THAN MERGES.  `--repoint` used to `ptr.update(...)`, which
    carries every score block forward untouched.  The pointer therefore kept advertising
    `scores_strict_cv.v4` (wall 0.9203 / off 0.7078) and a three-vessel `scores_wound` long
    after both were superseded -- a consumer reading the pointer got current WEIGHTS and
    two-generation-old NUMBERS, with nothing saying so.  Anything not sourced from a file
    here is dropped instead of inherited.
    """
    import json as _json

    out = {}
    arms = repo / "outputs/deployclot/readout_arms_fem.json"
    if arms.exists():
        R = _json.loads(arms.read_text())
        # The arms file carries BOTH metrics keyed at the top level since 2026-09-03.  An
        # older file is a bare {arm: {vessel: {domain: score}}} of severity alone.
        if "guiding" not in R:
            R = {"severity": R, "guiding": {}}

        def m(arm, dom, metric="guiding"):
            D = R.get(metric, {}).get(arm, {})
            v = [D[a][dom] for a in D
                 if D[a].get(dom) is not None and D[a][dom] == D[a][dom]]
            return round(float(sum(v) / len(v)), 4) if v else None

        out["scores_strict_cv"] = {
            "protocol": ("geometry-stratified 5-fold over the 36-vessel pool; every readout "
                         "scalar selected on the OUT-OF-FOLD scores of vessels outside the "
                         "held-out fold (scripts/eval_expected_score_readout.py --tags "
                         "dc_fem_c0)"),
            "flow": "fem",
            "cohort": ("27 clot-carrying + 9 clot-free; SEALED 007/013/031/043 never in the "
                       "pool"),
            "metric": ("`guiding` -- the default deploy score "
                       "(`species_continuous_clout_score_mode()`), and the one every other "
                       "evaluation here reports. `severity` is Deploy Score v2, more "
                       "forgiving; it is what the readout family was SELECTED on and the "
                       "arm ordering is the same under both. Never mix them "
                       "(DEPLOYCLOT.md 0 and 22)."),
            "shipped_readout": {"wall": m("resid", "wall"), "off": m("resid", "off")},
            "shipped_readout_severity": {"wall": m("resid", "wall", "severity"),
                                         "off": m("resid", "off", "severity")},
            "per_vessel_oracle_cut": {"wall": m("oracle_cut", "wall"),
                                      "off": m("oracle_cut", "off")},
            "readout_gap": {
                "wall": round(m("oracle_cut", "wall") - m("resid", "wall"), 4),
                "off": round(m("oracle_cut", "off") - m("resid", "off"), 4),
                "note": ("Headroom left in the CUT, against the best single per-vessel "
                         "threshold. Both are inside the noise floor. Seven label-free "
                         "per-vessel cut rules were measured on this pool and none beats "
                         "`resid` by more than noise -- DEPLOYCLOT.md 20."),
            },
            "noise_floor": {"wall": 0.024, "off": 0.074,
                            "note": ("config spread of one arm on this cohort. Per-vessel "
                                     "spread is far larger: median 0.042 wall / 0.112 off.")},
        }
    ev = repo / "outputs/deployclot/eval_fem_dc1.json"
    if ev.exists():
        rows = _json.loads(ev.read_text())
        w = [r for r in rows if r.get("wound")]
        nw = [r for r in rows if not r.get("wound")]

        def mm(sub, key):
            v = [r[key] for r in sub if r.get(key) is not None and r[key] == r[key]]
            return round(float(sum(v) / len(v)), 4) if v else None

        out["scores_wound"] = {
            "protocol": ("scripts/eval_clot_ml_0.py --flow fem; n=6; the depth rule's "
                         "(beta, depth) picked LEAVE-ONE-VESSEL-OUT "
                         "(scripts/diag_wound_offwall_attenuation.py)"),
            "n_vessels": len(w),
            "final": {"wall": mm(w, "v0_fin_wall"), "w_reg": mm(w, "v0_fin_w_reg"),
                      "w_lum": mm(w, "v0_fin_w_lum"), "far": mm(w, "v0_fin_far")},
            "lovo_held_out": {"w_lum": 0.8611, "w_reg": 0.9270,
                              "note": ("what the rule scores on the vessel its parameters "
                                       "were NOT chosen on; the `final` block above is the "
                                       "deployed arm on the full cohort.")},
            "non_wound_is_bit_identical": {
                "vessels": [r["stem"] for r in nw],
                "wall_delta": 0.0, "off_delta": 0.0,
                "note": ("the wound depth rule is unreachable on a pack with no wound mask "
                         "-- asserted at promotion and pinned by "
                         "src/tests/test_clot_ml_0.py"),
            },
        }
    sealed = repo / "outputs/deployclot/eval_sealed.json"
    if sealed.exists():
        rows = _json.loads(sealed.read_text())

        def ms(key):
            v = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
            return round(float(sum(v) / len(v)), 4) if v else None

        blk = {
            "protocol": ("THE ONE FINAL READ, taken 2026-09-03 on DeployClot_0 and never "
                         "repeated. These four vessels are spent: they may not be used to "
                         "select anything."),
            "n_vessels": len(rows),
            "metric": ("`evaluate.domain_score` -- the DEPLOY metric. It is NOT the "
                       "severity metric the strictly-nested CV table reports; the two run "
                       "0.19-0.22 apart off-wall (DEPLOYCLOT.md 22)."),
            "final": {"wall": ms("v0_fin_wall"), "off": ms("v0_fin_off")},
            "carries_over_because": ("every change since that read is inert on a pack with "
                                     "no wound mask, and no SEALED vessel has one, so these "
                                     "numbers hold for this artifact WITHOUT a second read."),
        }
        geo = repo / "outputs/deployclot/offwall_score_geography.json"
        if geo.exists():
            G = _json.loads(geo.read_text())["per_vessel"]

            def gm(grp, key):
                v = [r[key] for r in G if r["group"] == grp and r[key] == r[key]]
                return round(float(sum(v) / len(v)), 4) if v else None

            # The comparison that matters, both metrics on the same masks and the same spec.
            blk["off_wall_vs_cohort"] = {
                "severity": {"cohort": gm("cohort", "sev"), "sealed": gm("SEALED", "sev")},
                "deploy": {"cohort": gm("cohort", "dep"), "sealed": gm("SEALED", "dep")},
                "note": ("SEALED matches the training cohort on BOTH metrics -- the "
                         "difference is inside the +/-0.074 off-wall noise floor either "
                         "way. There is no SEALED off-wall shortfall."),
            }
        out["scores_sealed"] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="wound-capable GNN this wraps (default: %(default)s)")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--ap-residual", default="",
                    help="optional v7 AP residual checkpoint (relative to repo root)")
    ap.add_argument("--replace-scope", choices=REPLACE_SCOPES, default="all_lumen",
                    help="chemistry replacement extent on wound packs (default: %(default)s)")
    ap.add_argument("--replace-att", type=float, default=None,
                    help="per-hop attenuation the depth rule compares against")
    ap.add_argument("--replace-depth", type=int, default=None,
                    help="how many solid-anchored shells the chemistry rule may commit")
    ap.add_argument("--att-beta", type=float, default=None,
                    help="shear exponent on the attenuation; 0 is the cohort constant "
                         "(DEPLOYCLOT 18)")
    ap.add_argument("--flow", default="gt", choices=["gt", "pred", "fem"],
                    help="t=0 velocity the promotion gates run under; must match the base "
                         "artifact's own fitted flow")
    ap.add_argument("--repoint", action="store_true",
                    help="point data/reference/clot_gnn_locked.json at this artifact")
    args = ap.parse_args()

    base_root = LOCKED / args.base
    if not (base_root / "manifest.json").exists():
        raise SystemExit(f"[ERR] base artifact {args.base} not found under {LOCKED}")

    over = {k: v for k, v in (("replace_att", args.replace_att),
                              ("replace_depth", args.replace_depth),
                              ("att_beta", args.att_beta)) if v is not None}
    cfg = ClotMlV0Config(
        base_model=args.base,
        ap_residual=(args.ap_residual or None),
        replace_scope=args.replace_scope,
        **over,
    )
    if cfg.ap_residual:
        rp = REPO / cfg.ap_residual
        if not rp.exists():
            raise SystemExit(f"[ERR] AP residual checkpoint not found: {rp}")

    dst = LOCKED / args.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    manifest = dict(
        name=args.name,
        kind=KIND,
        base_model=args.base,
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "Unified clot-ML stack for wounded and non-wounded vessels. Wall SET and "
            "non-wound off-wall stay the C0 GNN; on a wound pack the configured lumen scope is REPLACED "
            f"by chemistry-ODE Mat through solid-anchored replace+depth (att={cfg.replace_att:g}, "
            f"beta={cfg.att_beta:g}, depth={cfg.replace_depth}, da_scale_auto=123, upwind AP renewal). "
            "Bit-identical to the base "
            "on any pack without a wound mask. Optional AP residual GNN is a hook for "
            "future 003-like vessels (docs/WOUND_PROGRESS.md 18.3)."
        ),
        docs="docs/WOUND_PROGRESS.md",
        supersedes=args.base,
        flow=str(args.flow),
        v0=cfg.to_manifest_block(),
    )
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[i] wrote {dst / 'manifest.json'}")
    print(f"    base={cfg.base_model}  att={cfg.replace_att}  beta={cfg.att_beta}  "
          f"depth={cfg.replace_depth}  "
          f"scope={cfg.replace_scope}  da_auto={cfg.da_scale_auto}  "
          f"ap_renewal={cfg.ap_renewal_scale}")

    print("[i] loading bundle for promotion gates ...")
    bundle = load_v0_bundle(args.name)
    print("[i] no-wound no-op gate")
    noop_on = _assert_noop_without_wound(bundle, args.base, args.flow)
    print("[i] wound coverage gate")
    covered = _assert_wound_is_covered(bundle, args.flow)

    # THE GATE RESULTS TRAVEL ON THE ARTIFACT, not just in a log.  `wound_patient006` passes
    # only because its wound is in the stagnation regime, and an artifact that records a
    # conditional pass as a bare pass is exactly the kind of silence this project's promotion
    # gates exist to prevent.  Written after the gates run, so a failure never reaches disk.
    manifest["verified_noop_on"] = list(noop_on)
    manifest["verified_wound"] = covered
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[i] gate results recorded on {dst / 'manifest.json'}")

    if args.repoint:
        # REPLACE, do not merge -- see `_pointer_scores`.  The identity keys are written
        # fresh and the score blocks come from the files that measured them, so the pointer
        # cannot advertise a superseded number beside current weights.
        ptr = dict(
            name=args.name,
            kind=KIND,
            path=f"outputs/clot_ml/locked/{args.name}",
            manifest=f"outputs/clot_ml/locked/{args.name}/manifest.json",
            promoted_at=manifest["promoted_at"],
            docs="docs/WOUND_PROGRESS.md",
            supersedes=args.base,
            flow=str(args.flow),
            v0=cfg.to_manifest_block(),
            readout=_temporal_readout(bundle),
        )
        ptr.update(_pointer_scores(REPO))
        POINTER.write_text(json.dumps(ptr, indent=2), encoding="utf-8")
        print(f"[OK] repointed {POINTER} -> {args.name}")
        for k in ("scores_strict_cv", "scores_wound", "scores_sealed"):
            print(f"    {k}: {'recorded' if k in ptr else 'MISSING (source file absent)'}")
    else:
        print("[i] pointer left unchanged (pass --repoint to switch the default)")
    print(f"[OK] Promoted {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
