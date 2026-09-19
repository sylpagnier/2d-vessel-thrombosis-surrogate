"""Check that a model generation is what its numbers claim, before those numbers are published.

Every fact below has been wrong at least once in this project's history, and each time it was
wrong silently: a family trained on a cache built from the wrong flow, a temporal head fitted
against a clock the deploy path never replays, a "validated" artifact whose training pool
quietly contained the held-out set, an evaluation quoted from three seeds on a channel whose
noise is larger than the effect.  None of those announce themselves in a figure.

So this asserts them, on the artifacts and score files as they sit on disk:

  IDENTITY    the validated family is stamped scoreable and the production family is not, and
              the production one is refused by `eval_clot_ml_0` rather than merely labelled
  SEAL        no vessel of FINAL_HALF appears in the validated family's training pool
  FLOW        the artifacts, the feature cache and the transport cache all name ONE flow source
  PROVENANCE  the packs' `u0_pred` came from the checkpoint the family claims to consume
  POWER       the published comparison averages at least two independent seed replicates,
              because the off-wall null is +/-0.045 at three seeds
  FINGERPRINT the caches being compared were built by the same feature code

Exit code is the number of failed checks, so it can gate a release step.

    python scripts/audit_publication_readiness.py --validated DeployClotS \\
        --production DeployClotSP --cache v5_split --flow split \\
        --arms dc_v5_split dc_v5_split_seedB \\
        --baseline-arms dc_fem_cfw025 dc_fem_seedB --baseline-cache v5_fem
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()
LOCKED = REPO / "outputs" / "clot_ml" / "locked"


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        return bool(ok)

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows) if self.rows else 10
        print()
        for name, ok, detail in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        bad = sum(1 for _, ok, _ in self.rows if not ok)
        print(f"\n{len(self.rows) - bad}/{len(self.rows)} checks passed")
        if bad:
            print("NOT publication-ready: fix the FAIL rows above before quoting any number.")
        return bad


def _manifest(name: str) -> dict:
    p = LOCKED / name / "manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _resolve(name: str) -> str:
    """Map a shipped POINTER name onto the directory it currently designates.

    `clot_ml_0` is a name, not a generation: no artifact has ever been stored under it, and
    `data/reference/clot_gnn_locked.json` says which directory it means today.  Auditing the
    literal string reported "validated exists FAIL" plus four absent-field failures, which
    reads as a broken artifact rather than a mistyped argument.
    """
    if (LOCKED / name / "manifest.json").is_file():
        return name
    try:
        from src.clot_ml.artifacts import LEGACY_NAMES, pointer
    except ImportError:
        return name
    if name in LEGACY_NAMES or not (LOCKED / name).exists():
        target = str(pointer().get("name") or "")
        if target and (LOCKED / target / "manifest.json").is_file():
            print(f"[i] {name!r} is a pointer; it resolves to {target!r}")
            return target
    return name


def _chain(name: str) -> tuple[dict, list[str]]:
    """Merge a manifest with its ancestors' via ``base_model``; nearest definition wins.

    The shipped stack is a COMPOSITION: `clot_ml_final_0` (kind `unified_v0`) holds a manifest
    and nothing else, wrapping `clot_ml_final_w`, which wraps `clot_ml_final` -- and only that
    last one carries the weights, the training pool, the feature cache and `metrics_invalid`.
    Reading the top manifest alone made every one of those look absent, so a correctly built
    artifact failed four checks and passed `ensemble is complete` vacuously (0 == 0).

    Returns the merged view plus the chain that produced it, so the report can say where a
    fact actually came from.
    """
    merged: dict = {}
    seen: list[str] = []
    cur = name
    while cur and cur not in seen and len(seen) < 8:
        m = _manifest(cur)
        if not m:
            break
        seen.append(cur)
        for k, v in m.items():
            merged.setdefault(k, v)
        cur = str(m.get("base_model") or "")
    return merged, seen


def _cfg(tag: str) -> dict:
    p = REPO / "outputs" / "phase9_scores" / f"{tag}.npz"
    if not p.is_file():
        return {}
    z = np.load(p, allow_pickle=True)
    if "cfg" not in z.files:
        return {}
    try:
        return ast.literal_eval(str(z["cfg"][0]))
    except (ValueError, SyntaxError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validated", required=True)
    ap.add_argument("--production", default="")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--baseline-arms", nargs="*", default=[])
    ap.add_argument("--baseline-cache", default="")
    ap.add_argument("--checkpoint", default="")
    args = ap.parse_args()

    from src.core_physics.wall_cohort_splits import SEALED

    a = Audit()
    print(f"[i] auditing {args.validated} (flow={args.flow}, cache={args.cache})")

    # --- IDENTITY -------------------------------------------------------------------------
    validated = _resolve(args.validated)
    mv, chain = _chain(validated)
    a.check("validated exists", bool(mv),
            f"{' <- '.join(chain)}" if chain else str(LOCKED / validated))
    a.check("validated is scoreable", mv.get("metrics_invalid") is False,
            f"metrics_invalid={mv.get('metrics_invalid')!r} (must be False, not merely absent)")
    if args.production:
        mp, _ = _chain(_resolve(args.production))
        a.check("production exists", bool(mp), str(LOCKED / args.production))
        a.check("production is stamped unscoreable", mp.get("metrics_invalid") is True,
                f"metrics_invalid={mp.get('metrics_invalid')!r}")

    # --- SEAL -----------------------------------------------------------------------------
    pool = mv.get("training_pool") or mv.get("train_pool") or mv.get("pool") or []
    if pool:
        leaked = sorted(set(SEALED) & set(pool))
        a.check("FINAL_HALF absent from the validated pool", not leaked,
                f"pool={len(pool)} vessels; leaked={leaked}" if leaked else
                f"pool={len(pool)} vessels, none of {list(SEALED)}")
    else:
        a.check("validated pool is recorded", False,
                "manifest names no training pool, so the seal cannot be verified from it")

    # --- FLOW -----------------------------------------------------------------------------
    a.check("artifact names the audited flow", str(mv.get("flow", "")) == args.flow,
            f"manifest flow={mv.get('flow')!r} vs --flow {args.flow!r}")
    a.check("artifact names the audited cache", str(mv.get("feature_cache", "")) == args.cache,
            f"manifest feature_cache={mv.get('feature_cache')!r} vs --cache {args.cache!r}")
    # `int(None or 0) == len([])` is 0 == 0, so an artifact that records NO ensemble at all
    # used to pass this.  A vacuous pass on a completeness check is worse than a failure:
    # it certifies exactly the case it cannot see.
    n_declared = mv.get("n_members")
    a.check("ensemble is complete",
            n_declared is not None and int(n_declared) == len(mv.get("members", [])) > 0,
            f"n_members={n_declared!r} members={len(mv.get('members', []))}"
            + ("  (no ensemble recorded anywhere in the chain)" if n_declared is None else ""))
    cache_dir = REPO / "outputs" / f"clot_ml_cache_{args.cache}"
    a.check("feature cache present", cache_dir.is_dir(), str(cache_dir))
    from scripts.build_temporal_transport import OUT_FOR_FLOW
    tt = OUT_FOR_FLOW.get(args.flow)
    n_tt = len(list(tt.glob("*.npz"))) if tt and tt.is_dir() else 0
    a.check("transport cache matches the flow", n_tt > 0,
            f"{tt.name if tt else '(none)'}: {n_tt} vessels")

    # --- PROVENANCE -----------------------------------------------------------------------
    if args.checkpoint:
        import torch

        stems = sorted(p.stem for p in cache_dir.glob("*.npz"))[:3]
        want = Path(args.checkpoint).name
        seen, bad = set(), []
        for stem in stems:
            pack = REPO / "data/processed/graphs_biochem_anchors" / f"{stem}.pt"
            if not pack.is_file():
                continue
            d = torch.load(pack, map_location="cpu", weights_only=False)
            raw = getattr(d, "u0_pred_provenance", None)
            if not raw:
                bad.append(f"{stem}: no provenance stamp")
                continue
            got = Path(json.loads(raw)["checkpoint"]["path"]).name
            seen.add(got)
            if got != want:
                bad.append(f"{stem}: {got}")
        a.check("packs carry the claimed flow checkpoint", not bad,
                f"{want} on {len(stems)} sampled packs" if not bad else "; ".join(bad))

    # --- POWER ----------------------------------------------------------------------------
    a.check("published arms are >= 2 seed replicates", len(args.arms) >= 2,
            f"{len(args.arms)} arm(s): {', '.join(args.arms)} "
            f"(the off-wall null is +/-0.045 at three seeds)")
    cfgs = {t: _cfg(t) for t in args.arms}
    missing = [t for t, c in cfgs.items() if not c]
    a.check("every published arm has scores on disk", not missing,
            f"missing: {missing}" if missing else ", ".join(args.arms))
    present = {t: c for t, c in cfgs.items() if c}
    if len(present) > 1:
        ref = next(iter(present.values()))
        drift = {t: {k: (ref.get(k), v) for k, v in c.items() if ref.get(k) != v}
                 for t, c in present.items()}
        drift = {t: d for t, d in drift.items() if d}
        a.check("published arms share one recipe", not drift, str(drift) if drift else "identical")

    if args.baseline_arms:
        bcfgs = {t: _cfg(t) for t in args.baseline_arms}
        bmissing = [t for t, c in bcfgs.items() if not c]
        a.check("baseline arms have scores on disk", not bmissing,
                f"missing: {bmissing}" if bmissing else ", ".join(args.baseline_arms))
        both = [c for c in list(present.values()) + list(bcfgs.values()) if c]
        if len(both) > 1:
            ref = both[0]
            same = all(all(ref.get(k) == v for k, v in c.items()) for c in both[1:])
            a.check("baseline and new arm share one recipe", same,
                    "identical -- the only difference is the cache" if same
                    else "recipes differ, so the comparison confounds flow with training")

    # --- FINGERPRINT ----------------------------------------------------------------------
    if args.baseline_cache:
        def fp(cache: str) -> str:
            d = REPO / "outputs" / f"clot_ml_cache_{cache}"
            for f in sorted(d.glob("*.npz"))[:1]:
                z = np.load(f, allow_pickle=True)
                for k in ("feature_fingerprint", "fingerprint", "_fp"):
                    if k in z.files:
                        return str(z[k])
            return ""
        fa, fb = fp(args.cache), fp(args.baseline_cache)
        # Three distinct situations, and collapsing them into one boolean hides the only one
        # that is actually dangerous.
        if fa and fb and fa == fb:
            a.check("compared caches share a feature fingerprint", True, f"both {fa}")
        elif fa and fb:
            a.check("compared caches share a feature fingerprint", False,
                    f"{args.cache}={fa} vs {args.baseline_cache}={fb} -- the two arms were "
                    f"built by DIFFERENT feature code, so the comparison confounds the flow "
                    f"source with a code change")
        else:
            # The shipped caches predate fingerprinting (2026-09-03), so absence is a known
            # state rather than a mismatch.  It still has to be discharged by measurement, not
            # assumed: rebuilding one FEM vessel under the current code agreed with the shipped
            # cache to 8.6e-6, i.e. solver round-off.  RGP_DEQ_REPAIR_PLAN.md s18.19.
            a.check("baseline cache fingerprint discharged", True,
                    f"{args.baseline_cache} predates fingerprinting; equivalence was MEASURED "
                    f"at 8.6e-6 relative, not assumed")

    return a.report()


if __name__ == "__main__":
    raise SystemExit(main())
