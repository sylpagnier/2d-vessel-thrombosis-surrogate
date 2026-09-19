"""Re-derive every headline number in the publication docs from the artifacts on disk.

WHY THIS EXISTS.  `docs/PUBLICATION_NOTES.md` §7 states the rule already -- *"if a row has no
command, it is not yet a claim"* -- and §6 is a list of eight numbers that were quoted and
later found wrong.  Every one of them was wrong **silently**: the figure rendered, the table
had the right shape, and the value came from a superseded model, a different metric, or a
smaller cohort than the sentence around it implied.  A prose rule cannot catch that.  This
can, because it reads the artifact rather than the paragraph.

WHAT A ROW IS.  `(claim id, the doc that states it, the value it states, the artifact and the
path inside it that produces that value, a tolerance)`.  The script resolves each artifact and
compares.  Three outcomes:

    PASS    the doc's number is what the artifact says
    FAIL    the doc's number is NOT what the artifact says -- fix one of them before quoting
    STALE   the artifact is missing, so the claim currently rests on nothing re-derivable

Exit code is the number of FAIL + STALE rows, so a release step can gate on it.

**A tolerance is not a fudge factor.**  It is the rounding the doc itself uses: a doc that says
"0.9127" is checked to 5e-5, one that says "~2,934x" to 1.0, one that says "~73%" to 0.01.  If a
doc rounds harder, widen the tolerance in the row and say so -- never widen it to make a
mismatch go away.

    python scripts/publication/verify_claims.py
    python scripts/publication/verify_claims.py --paper docs/publication/STORY.md
    python scripts/publication/verify_claims.py --json outputs/ablation/claims.json

DOC POINTERS.  The `doc:section` field of a row is a human label, not a resolved path.  Rows
written before 2026-09-11 say `PAPER n` / `PUBLICATION_NOTES n` / `PUBLICATION_PLAN n`; those
documents were superseded by `docs/publication/` (STORY / EVIDENCE / FIGURES / EXPERIMENTS) and
the tracked ones now live under `docs/archive/`.  The labels are left alone deliberately -- they
record where a number was first stated, which is provenance, and rewriting them would lose it.
`--paper` takes whichever document you are checking; point it at STORY.md.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()

#: the paper's central comparison, written by scripts/publication/diag_physics_vs_learned.py.
#: TWO files exist and they are NOT interchangeable: `PVLF` is the SHIPPED generation (v5_fem,
#: the family `data/reference/clot_gnn_locked.json` points at) and is what the manuscript
#: quotes; `PVL` is the earlier v5_split ladder, kept for provenance only.  They were dealt
#: DIFFERENT geometry-stratified folds -- `geometry_class.py`/`geometry_splits.py` changed on
#: 2026-09-07 14:06, between the two runs -- so a split-vs-fem delta mixes generation with
#: fold assignment and must never be presented as a generation effect (PAPER 5.4).
PVL = "outputs/ablation/physics_vs_learned.json"
PVLF = "outputs/ablation/physics_vs_learned_fem.json"
#: the SAME four-point protocol with a depth-matched MeshGraphNet-STYLE control in the `naive`
#: slot instead of `A_naive` (2026-09-10).  Same cache, same folds, same `plain`/`full` arms, so
#: only the floor row differs.  See the `A_mgn` scope note in `src/clot_ml/ablation.py`: it is
#: NOT a MeshGraphNet reimplementation and must never be quoted as "MeshGraphNet".
PVLM = "outputs/ablation/physics_vs_learned_mgn.json"
#: the shipped-generation ablation report (severity, both readouts)
ABLF = "outputs/ablation/ablation_report_ablfem.json"
#: the shipped model's spec, read from the locked artifact by generate_architecture_table.py
ARCH = "outputs/publication/data/architecture_table.json"
#: the ODE-timing ablation, paired on the shipped generation
ODET_W = "outputs/ablation/temporal_odetiming_with_ode.json"
ODET_N = "outputs/ablation/temporal_odetiming_no_ode.json"
#: the DEPLOY-PATH wall-clock ablation (2026-09-10), written by `scratch/go_wall_clock.sh`.
#: `--wall-clock` is a different switch from `--lag-anchor` and answers a different question:
#: it swaps the WALL commit series between the learned temporal head and the ODE's own crossing,
#: which is the mechanism the shipped artifact uses (`readout.lag_anchor: "ode"`) and which
#: PAPER 5.3 recorded as having no switch to ablate.
#:
#: THREE ARMS, AND THE THIRD IS THE ONE TO QUOTE FOR THE CLOCK.  `ode_wall_series` is the only
#: wall path with no forced final commit, so `ode` vs `head` mixes a clock effect with a SET
#: effect (final-time wall falls -0.0210 on 24/27 because nodes whose integrated `Mat` never
#: crosses inside the horizon are never committed).  `ode_commit` restores the forced final
#: commit the other paths already have, so the final set is identical (delta 0.0000, 0/27
#: moved) and the mean-over-time delta is the clock alone.
WC_H = "outputs/ablation/temporal_wallclock_head.json"
WC_O = "outputs/ablation/temporal_wallclock_ode.json"
WC_C = "outputs/ablation/temporal_wallclock_ode_commit.json"


# ---------------------------------------------------------------------------
# resolvers -- each returns a float, or raises/returns None when the artifact
# cannot produce the quantity (which is a STALE row, never a silent pass)
# ---------------------------------------------------------------------------
#: Artifacts NOT yet re-scored under the mirror-branch evaluation rule (`src/clot_ml/mirror_branch.py`,
#: 2026-09-16).  Each is a strictly-nested temporal evaluation (~45 min) whose cohort includes
#: comsol045/046; re-running them was deferred.  Every row that reads one is reported STALE, never
#: PASS, until it is regenerated -- then delete its line here.  Regenerate with the commands in
#: `scratch/go_wall_clock.sh`, `go_ode_timing.sh`, `go_temporal_correct.sh` and
#: `go_temporal_ablation.sh` (the `A4 / ode` spec).
PRE_MIRROR_RULE: frozenset = frozenset()   # all temporal ablations re-scored 2026-09-16


class PreMirrorArtifact(RuntimeError):
    """Raised by `_load` for an artifact scored before the mirror-branch rule -> STALE row."""


def _load(rel: str):
    if rel in PRE_MIRROR_RULE:
        raise PreMirrorArtifact(f"{rel} was scored before the mirror-branch rule; re-run it")
    p = REPO / rel
    if not p.is_file():
        return None
    txt = p.read_text(encoding="utf-8")
    return json.loads(txt) if txt.strip() else None


def _apps_rows(sweep: str) -> list:
    return (_load("outputs/publication/data/applications.json") or {}).get(sweep, [])


def _apps_at(sweep: str, strength_pct: float):
    """Final total clot mass (% of vessel) at the sweep point whose MEASURED strength is ``strength_pct``."""
    row = min(_apps_rows(sweep), key=lambda r: abs(r["strength_measured_pct"] - strength_pct), default=None)
    return None if row is None or abs(row["strength_measured_pct"] - strength_pct) > 0.2 else row["vessel_clot_pct_final"]


def _apps_last_zero(sweep: str):
    """Largest measured strength before the first point with any clot."""
    last = None
    for r in _apps_rows(sweep):
        if r["vessel_clot_pct_final"] > 0:
            break
        last = r["strength_measured_pct"]
    return last


def _apps_cost_min():
    import glob
    secs = [json.loads(Path(f).read_text(encoding="utf-8"))["rollout"]["elapsed_s"]
            for sid in ("01_stenosis_strength", "02_aneurysm_strength")
            for f in glob.glob(str(REPO / "outputs" / "research_sweeps" / sid / "arm_*.json"))]
    return sum(secs) / 60 if len(secs) == 80 else None


def _cplfreq_f1(stem: str, dom: str, field: str):
    """One arm's strict-F1 comparison from the coupling_frequency figure's own data file."""
    rows = _load("outputs/publication/data/coupling_frequency.json") or []
    row = next((r for r in rows if r.get("stem") == stem), None)
    return dig(row, "result_strict_f1", dom, field)


def dig(d, *path):
    """Walk a nested dict/list by key or index; None if any step is missing."""
    for k in path:
        if d is None:
            return None
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return None
    return d


def timing(*path):
    return lambda: dig(_load("outputs/publication/data/timing.json"), *path)


def timing_share(stage: str):
    def f():
        d = _load("outputs/publication/data/timing.json")
        num = dig(d, "summary", stage, "median")
        den = dig(d, "summary", "deploy_s", "median")
        return None if (num is None or not den) else num / den
    return f


def table4(cls: str, *path, profile: str = "publication"):
    return lambda: dig(_load(f"outputs/{profile}/data/table4_kfold.json"),
                       "per_class", cls, *path)


def sealed(field: str):
    def f():
        rows = _load("outputs/deployclot/eval_sealed.json")
        if not rows:
            return None
        vals = [r[field] for r in rows if r.get(field) is not None
                and r[field] == r[field]]
        return float(np.mean(vals)) if vals else None
    return f


def sealed_one(stem: str, field: str):
    def f():
        rows = _load("outputs/deployclot/eval_sealed.json") or []
        for r in rows:
            if r.get("stem") == stem:
                return r.get(field)
        return None
    return f


def paired(rel: str, *path):
    return lambda: dig(_load(rel), "paired", *path)


def flowreq(metric: str, field: str = "r"):
    return lambda: dig(_load("outputs/publication/data/flow_requirement.json"),
                       "panel_a", "correlations", metric, field)


def flowreq_pair(field: str, how: str = "median"):
    """Median (or sum) of a per-vessel RGP-DEQ diagnostic over the 33-vessel cohort."""
    def f():
        pr = dig(_load("outputs/publication/data/flow_requirement.json"), "panel_a", "pairs")
        if not pr:
            return None
        v = [p[field] for p in pr
             if isinstance(p.get(field), (int, float)) and p[field] == p[field]]
        if not v:
            return None
        return float(np.sum(v)) if how == "sum" else float(np.median(v))
    return f


def femaudit_med(field: str):
    """Median of a per-vessel FEM-vs-GT audit field -- the classical solve's own accuracy."""
    def f():
        rows = _load("outputs/deployclot/fem_flow_audit.json")
        if not rows:
            return None
        v = [r[field] for r in rows
             if isinstance(r.get(field), (int, float)) and r[field] == r[field]]
        return float(np.median(v)) if v else None
    return f


def ablation(arm: str, *path, report: str = "outputs/ablation/ablation_report.json"):
    return lambda: dig(_load(report), "arms", arm, *path)


def temporal(field: str, rel: str = "outputs/ablation/temporal_A4_ode.json"):
    """Cohort mean of one field in a temporal-evaluation artifact, NaNs dropped."""
    def f():
        d = _load(rel)
        if not d:
            return None
        v = [r[field] for r in d.values()
             if isinstance(r.get(field), (int, float)) and r[field] == r[field]]
        return float(np.mean(v)) if v else None
    return f


def v3(*path, rel="outputs/clot_ml/locked/clot_gnn_v3/manifest.json"):
    """The `clot_gnn_v3` era's time-conditioned CV numbers, from its own manifest.

    HISTORICAL AND NOT COMPARABLE to anything else in the paper: `flow="gt"`, a 19-vessel
    pool, 11 timesteps and the old 56-column v2 feature block, against the shipped family's
    FEM flow / 27 vessels / 69 columns.  Pinned here because it is the ONLY measurement in
    the repo of what temporal conditioning is worth, and a number that important should not
    live only in prose -- but it must never be averaged or differenced against a current one.
    """
    return lambda: dig(_load(rel), "scores_out_of_fold_cv", *path)


def temporal_ol(anchor: str, field: str):
    """Mean over vessels from the OWNER-LAG temporal runs (ode vs learned anchor).

    SCOPE, and it is narrow: `--lag-anchor` is consumed only inside `fit_lag_model`, which
    regresses the per-node lag **on off-wall GT nodes**.  The wall columns are bit-identical by
    construction and are NOT evidence about the ODE -- they are evidence the flag never varied
    the wall clock.  An earlier run without `--owner-lag` made the flag a no-op entirely, which
    is how that mistake announced itself (all eight fields equal).

    CORRECTED 2026-09-09.  This docstring used to say "wall commit times come from
    `ode_wall_series` unconditionally, so the ODE is the wall clock in BOTH arms".  That is only
    true when the tuner selects a `resid` family, where `wall_by_residual` dates each node by the
    ODE onset `oon` plus a LEARNED integer offset.  Otherwise the wall clock is `series_masks`,
    a pure threshold on the learned temporal head with no ODE in it at all.  `docs/PAPER.md` 5.3
    used to assert the opposite half with equal confidence.  Which one holds is a per-fold tuner
    outcome, so neither sentence was safe; the bit-identity has the same cause either way.
    The genuine deploy-path ablation is `--wall-clock {head,ode}` (added 2026-09-09,
    `scratch/go_wall_clock.sh`), NOT this pair.
    """
    import numpy as _np
    rel = "outputs/ablation/temporal_A4_ownerlag_%s.json" % anchor

    def f():
        d = _load(rel)
        if not d:
            return None
        v = [r[field] for r in d.values()
             if isinstance(r.get(field), (int, float)) and r[field] == r[field]]
        return float(_np.mean(v)) if v else None
    return f


def preflight(arm: str, what: str):
    """Pre-flight gate detector, as the CURRENT artifact reports it.

    Deliberately counted from the lists rather than read from a summary field: the "5 FAIL /
    5-of-5 caught" figure in PUBLICATION_NOTES 7.9 was measured on the PRE-CROSS-FIT arm and
    the current artifact cannot reproduce it -- the shipped arm never empties a gate, so there
    is nothing left to detect.  Counting keeps the doc honest about which arm it is quoting.
    """
    def f():
        d = dig(_load("outputs/runs/preflight_validation.json"), arm)
        if d is None:
            return None
        if what == "n":
            return d.get("n")
        return len(d.get(what, []))
    return f


# ---------------------------------------------------------------------------
# THE LEDGER.  One row per number a reader could copy out of the docs.
# ---------------------------------------------------------------------------
#: (id, doc:section, stated value, resolver, tolerance, note)
CLAIMS: list[tuple] = [
    # --- Table 3, cost -----------------------------------------------------------------
    ("timing.median_s", "PUBLICATION_NOTES 7.2 / PLAN 3", 58.9,
     timing("summary", "deploy_s", "median"), 0.05,
     "median end-to-end deploy seconds"),
    ("timing.n", "PUBLICATION_NOTES 7.2", 34,
     timing("summary", "deploy_s", "n"), 0, "vessels timed"),
    ("timing.speedup", "PUBLICATION_NOTES 7.2 / PLAN 3", 2934,
     timing("speedup_vs_comsol_median"), 1.0, "vs COMSOL 48 h"),
    ("timing.q1", "PUBLICATION_NOTES 7.2", 40.7,
     timing("summary", "deploy_s", "q1"), 0.05, "IQR low"),
    ("timing.q3", "PUBLICATION_NOTES 7.2", 67.3,
     timing("summary", "deploy_s", "q3"), 0.05, "IQR high"),
    ("timing.share_fem", "PUBLICATION_NOTES 7.2", 0.08,
     timing_share("fem_s"), 0.01, "FEM t=0 as a fraction of end-to-end"),
    ("timing.share_rollout", "PUBLICATION_NOTES 7.2", 0.73,
     timing_share("rollout_s"), 0.01, "rollout as a fraction of end-to-end"),

    # --- Table 4, geometry generalization (SHIPPED profile = fem) -----------------------
    ("t4.baseline.n", "PUBLICATION_NOTES 7.2", 20, table4("baseline", "n"), 0, ""),
    ("t4.baseline.wall", "PUBLICATION_NOTES 7.2", 0.9127,
     table4("baseline", "wall_final", "mean"), 5e-5, ""),
    ("t4.baseline.wall_sem", "PUBLICATION_NOTES 7.2", 0.0239,
     table4("baseline", "wall_final", "sem"), 5e-5, ""),
    ("t4.baseline.off", "PUBLICATION_NOTES 7.2", 0.6437,
     table4("baseline", "off_final", "mean"), 5e-5, ""),
    ("t4.baseline.off_n", "PUBLICATION_NOTES 7.2", 13,
     table4("baseline", "off_final", "n"), 0, "off-wall scored vessels"),
    ("t4.stenosis.n", "PUBLICATION_NOTES 7.2", 5, table4("stenosis", "n"), 0, ""),
    ("t4.stenosis.wall", "PUBLICATION_NOTES 7.2", 0.8608,
     table4("stenosis", "wall_final", "mean"), 5e-5, ""),
    ("t4.stenosis.wall_sem", "PUBLICATION_NOTES 7.2", 0.0310,
     table4("stenosis", "wall_final", "sem"), 5e-5, ""),
    ("t4.stenosis.off", "PUBLICATION_NOTES 7.2", 0.7938,
     table4("stenosis", "off_final", "mean"), 5e-5, ""),
    ("t4.aneurysm.n", "PUBLICATION_NOTES 7.2", 2, table4("aneurysm", "n"), 0, ""),
    ("t4.aneurysm.wall", "PUBLICATION_NOTES 7.2", 0.9723,
     table4("aneurysm", "wall_final", "mean"), 5e-5, ""),
    ("t4.aneurysm.off", "PUBLICATION_NOTES 7.2", 0.9204,
     table4("aneurysm", "off_final", "mean"), 5e-5, ""),
    ("t4.noise_floor.wall", "PUBLICATION_NOTES 7.2", 0.0037,
     lambda: dig(_load("outputs/publication/data/table4_kfold.json"),
                 "noise_floor", "wall"), 1e-6, "config floor, NOT the SEM"),
    ("t4.noise_floor.off", "PUBLICATION_NOTES 7.2", 0.0432,
     lambda: dig(_load("outputs/publication/data/table4_kfold.json"),
                 "noise_floor", "off"), 1e-6, "config floor, NOT the SEM"),

    # --- Table 5, the one sealed read ---------------------------------------------------
    ("sealed.wall_mean", "PUBLICATION_NOTES 6 / DEPLOYCLOT 13", 0.9572,
     sealed("v0_fin_wall"), 5e-5, "spent once, 2026-09-03"),
    ("sealed.off_mean", "PUBLICATION_NOTES 6", 0.6180,
     sealed("v0_fin_off"), 5e-5, "n=3; comsol031 has no off-wall GT"),
    ("sealed.comsol007.wall", "PUBLICATION_NOTES 6", 0.8968,
     sealed_one("comsol007", "v0_fin_wall"), 5e-5, ""),
    ("sealed.comsol013.wall", "PUBLICATION_NOTES 6", 0.9786,
     sealed_one("comsol013", "v0_fin_wall"), 5e-5, ""),
    ("sealed.comsol031.wall", "PUBLICATION_NOTES 6", 0.9533,
     sealed_one("comsol031", "v0_fin_wall"), 5e-5, ""),
    ("sealed.comsol043.wall", "PUBLICATION_NOTES 6", 1.0000,
     sealed_one("comsol043", "v0_fin_wall"), 5e-5, ""),

    # --- §5a / Table 6, the RGP-DEQ arm -------------------------------------------------
    ("rgp.wall_delta", "PUBLICATION_NOTES 2.3 / PLAN 3", -0.0086,
     paired("outputs/deployclot/crossfit5_vs_shipped.json", "wall", "delta"), 5e-5,
     "leak-free K=5 cross-fit vs shipped FEM"),
    ("rgp.wall_p", "PUBLICATION_NOTES 2.3", 0.4860,
     paired("outputs/deployclot/crossfit5_vs_shipped.json", "wall", "p"), 5e-5, "n.s."),
    ("rgp.wall_n", "PUBLICATION_NOTES 2.3", 27,
     paired("outputs/deployclot/crossfit5_vs_shipped.json", "wall", "n"), 0, ""),
    ("rgp.off_delta", "PUBLICATION_NOTES 2.3 / PLAN 3", -0.1185,
     paired("outputs/deployclot/crossfit5_vs_shipped.json", "off", "delta"), 5e-5, ""),
    ("rgp.off_p", "PUBLICATION_NOTES 2.3", 0.0005,
     paired("outputs/deployclot/crossfit5_vs_shipped.json", "off", "p"), 5e-5, ""),
    ("rgp.off_n", "PUBLICATION_NOTES 2.3", 20,
     paired("outputs/deployclot/crossfit5_vs_shipped.json", "off", "n"), 0, ""),

    # --- STORY 2.1, the oracle closed loop -- an upper bound, not a trained model --------
    # MEASURED 2026-09-12 by `scripts/eval_closed_loop_oracle.py --flow fem --every 4`, on the
    # DEPLOYED flow. These REPLACE the hand-computed values transcribed from PUBLICATION_NOTES
    # 1 (which read +0.0093 wall / +0.0143 off on n=5 under GT flow) -- the wall result holds in
    # sign and size, the OFF-WALL RESULT DOES NOT: it is -0.1604, not +0.0143. Do not re-quote
    # the old pair from any doc.
    #
    # THE n=3 TRAP ON THE OFF-WALL ROWS.  Only 3 of the 8 non-wound vessels carry off-wall GT.
    # The paired sign-flip permutation test has 2^3 = 8 arrangements, so its smallest possible
    # two-sided p is 2/8 = 0.25.  The reported p = 0.2498 is therefore the FLOOR, reached
    # because all three vessels move the same way -- it is not evidence of non-significance and
    # must never be read as "n.s.".  Quote the interval and the uniform sign instead.
    ("closedloop.nonwound_wall_delta", "STORY 2.1", 0.0002,
     paired("outputs/deployclot/closed_loop_oracle.json", "nonwound_wall", "delta"), 5e-4,
     "GT clot fed back into flow every step vs open loop; inside +/-0.024 wall floor; "
     "5 of 8 vessels are bit-identical"),
    ("closedloop.nonwound_wall_n", "STORY 2.1", 8,
     paired("outputs/deployclot/closed_loop_oracle.json", "nonwound_wall", "n"), 0, ""),
    ("closedloop.nonwound_off_delta", "STORY 2.1", -0.1604,
     paired("outputs/deployclot/closed_loop_oracle.json", "nonwound_off", "delta"), 5e-4,
     "the oracle makes off-wall WORSE on all three vessels that have off-wall GT "
     "(-0.2709 / -0.1979 / -0.0125)"),
    ("closedloop.nonwound_off_n", "STORY 2.1", 3,
     paired("outputs/deployclot/closed_loop_oracle.json", "nonwound_off", "n"), 0,
     "only 3 non-wound vessels carry off-wall GT -- see the n=3 permutation-floor note"),
    ("closedloop.wound_wreg_delta", "STORY 2.1", 0.0046,
     paired("outputs/deployclot/closed_loop_oracle.json", "wound_w_reg", "delta"), 5e-4,
     "flat"),
    ("closedloop.wound_wlum_delta", "STORY 2.1", 0.0000,
     paired("outputs/deployclot/closed_loop_oracle.json", "wound_w_lum", "delta"), 5e-4,
     "exactly flat on all six wound vessels"),
    ("closedloop.wound_wall_delta", "STORY 2.1", 0.0126,
     paired("outputs/deployclot/closed_loop_oracle.json", "wound_wall", "delta"), 5e-4,
     "inside the +/-0.024 wall floor"),
    ("closedloop.wound_n", "STORY 2.1", 6,
     paired("outputs/deployclot/closed_loop_oracle.json", "wound_w_reg", "n"), 0, ""),
    # The p the story argues ABOUT rather than from: at n=3 the sign-flip test's floor is
    # 2/8 = 0.25, so this value being 0.2498 means all three vessels moved the same way.
    ("closedloop.nonwound_off_p", "STORY 2.1", 0.2498,
     paired("outputs/deployclot/closed_loop_oracle.json", "nonwound_off", "p"), 5e-4,
     "the n=3 permutation FLOOR, not evidence of non-significance"),

    # --- STORY 0.3, the BATC vs BATC_0 knob decomposition ---------------------------------
    # `scripts/publication/diag_batc_decomposition.py`, 2026-09-12, on the 6 wound vessels'
    # saved predictions. NOT the deploy-cohort off-wall set DEPLOYCLOT 0.4's "+0.192" was
    # quoted on -- same effect, different set, and it does NOT reproduce those figures. The
    # beta knob in particular is NEGATIVE here on two of three domains, against the +0.002
    # claimed. Do not present these as confirming +0.192; that number is still unverified on
    # its own cohort.
    ("batcdec.wreg.compounded", "STORY 0.3", 0.0668,
     lambda: dig(_load("outputs/publication/data/batc_decomposition.json"),
                 "per_domain", "w_reg", "compounded"), 5e-5,
     "BATC - BATC_0 on identical predictions; NOT the sum of the singles"),
    ("batcdec.wreg.tolerance", "STORY 0.3", 0.0272,
     lambda: dig(_load("outputs/publication/data/batc_decomposition.json"),
                 "per_domain", "w_reg", "knobs", "tolerance (hops 2->4)", "gain"), 5e-5, ""),
    ("batcdec.wreg.shape", "STORY 0.3", 0.0522,
     lambda: dig(_load("outputs/publication/data/batc_decomposition.json"),
                 "per_domain", "w_reg", "knobs", "shape weight (0.5->0.2)", "gain"), 5e-5, ""),
    ("batcdec.wreg.graces", "STORY 0.3", 0.0045,
     lambda: dig(_load("outputs/publication/data/batc_decomposition.json"),
                 "per_domain", "w_reg", "knobs", "graces (0 -> shipped)", "gain"), 5e-5, ""),
    ("batcdec.wreg.beta", "STORY 0.3", -0.0195,
     lambda: dig(_load("outputs/publication/data/batc_decomposition.json"),
                 "per_domain", "w_reg", "knobs", "beta (0.5->1.0)", "gain"), 5e-5,
     "NEGATIVE -- F1 penalises a high-precision prediction that F0.5 rewarded"),

    # --- STORY 2.1b, the FAIR coupling test ----------------------------------------------
    # `scripts/eval_coupling_ceiling.py --flow fem --every 8`, n=3, 2026-09-12. Each arm is
    # scored at its OWN best committed-set size (top-k over its own score ranking), which is
    # what removes the readout-tuning mismatch that made E1's off-wall number uninterpretable.
    # CEILINGS ARE NOT MODEL SCORES: k is chosen knowing the answer. Only the DIFFERENCE is
    # quotable, and only as "designing for coupling would/would not raise the ceiling".
    ("coupling.wall_delta", "STORY 2.1b", 0.0036,
     lambda: dig(_load("outputs/deployclot/coupling_ceiling.json"), "result", "wall", "delta"),
     5e-5, "closed-loop ceiling minus open-loop ceiling, wall"),
    ("coupling.off_delta", "STORY 2.1b", -0.0072,
     lambda: dig(_load("outputs/deployclot/coupling_ceiling.json"), "result", "off", "delta"),
     5e-5, "and off-wall -- opposite sign, both negligible"),
    ("coupling.wall_open", "STORY 2.1b", 0.9045,
     lambda: dig(_load("outputs/deployclot/coupling_ceiling.json"), "result", "wall",
                 "mean_ceiling_open"), 5e-5, "NOT a model score -- an oracle-k ceiling"),
    ("coupling.wall_closed", "STORY 2.1b", 0.9081,
     lambda: dig(_load("outputs/deployclot/coupling_ceiling.json"), "result", "wall",
                 "mean_ceiling_closed"), 5e-5, ""),
    ("coupling.n", "STORY 2.1b", 3,
     lambda: dig(_load("outputs/deployclot/coupling_ceiling.json"), "n_vessels"), 0, ""),

    # --- STORY 2.1c, a network TRAINED on perfectly-coupled features ----------------------
    # `run_phase9_cv.py --arm A4 --folds 3 --seeds 1` on two caches differing ONLY by
    # CLOT_ML_ORACLE_BLOCKAGE, twice each (--seed-offset 0/1) so the seed null is measured.
    # THIS PARTLY OVERTURNS 2.1b: the ceiling test said the coupled field carries no extra
    # REACHABLE signal for the SHIPPED representation, and at the wall a network trained from
    # scratch on it does better by +0.0343 -- reproducibly, and far outside the seed noise.
    # NOT DEPLOYABLE: the oracle reads GT clot at t>0. This is a ceiling, not a gain.
    ("cpltrain.wall_delta", "STORY 2.1c", 0.0343,
     lambda: dig(_load("outputs/deployclot/coupling_trained.json"),
                 "result", "wall", "seed_averaged", "delta"), 5e-5,
     "seed-averaged; both seeds agree in sign (+0.0304, +0.0383)"),
    ("cpltrain.wall_noise_open", "STORY 2.1c", -0.0009,
     lambda: dig(_load("outputs/deployclot/coupling_trained.json"),
                 "result", "wall", "seed_noise_open"), 5e-5,
     "same cache, disjoint seeds -- the null the signal must beat"),
    ("cpltrain.wall_noise_closed", "STORY 2.1c", 0.0069,
     lambda: dig(_load("outputs/deployclot/coupling_trained.json"),
                 "result", "wall", "seed_noise_closed"), 5e-5, ""),
    ("cpltrain.off_delta", "STORY 2.1c", -0.0910,
     lambda: dig(_load("outputs/deployclot/coupling_trained.json"),
                 "result", "off", "seed_averaged", "delta"), 5e-5,
     "INSIDE the closed arm's own seed noise -- not established"),
    ("cpltrain.off_noise_closed", "STORY 2.1c", 0.1234,
     lambda: dig(_load("outputs/deployclot/coupling_trained.json"),
                 "result", "off", "seed_noise_closed"), 5e-5,
     "9x the open arm's seed noise: coupled training is far less stable off-wall"),
    ("cpltrain.n", "STORY 2.1c", 36,
     lambda: dig(_load("outputs/deployclot/coupling_trained.json"), "protocol", "n_vessels"),
     0, ""),

    # --- STORY 2.1h, the PHYSICS of coupling: the gate churns ----------------------------
    # Two FEM solves per vessel, identical but for a clot viscosity elevation on GT-clotted
    # nodes. GT is safe here: physics diagnostic, nothing trains on it. Measured at
    # delta_mu = 0.10 Pa.s, which is a LOWER BOUND -- the solver diverges at 0.68 and COMSOL's
    # own step is ~0.51.
    # Re-measured 2026-09-12 at COMSOL's PHYSICAL mu1 step (0.5135 Pa.s).  The earlier 0.2609 /
    # 0.0699 / 183 / n=3 came from a solve whose quadrature viscosity went negative (P2 ringing
    # of the clot step) and ran at 0.10 Pa.s; that artifact is kept as
    # coupled_flow_delta_PRE_CLIP_negative_viscosity.json and must not be quoted.
    ("cplflow.gate_jaccard", "STORY 2.1h", 0.4096,
     lambda: dig(_load("outputs/deployclot/coupled_flow_delta.json"),
                 "summary", "median_gate_jaccard"), 5e-5,
     "~41% of the firing-set union is shared between clot-free and clot-loaded flow"),
    ("cplflow.vel_rel_l2", "STORY 2.1h", 0.0912,
     lambda: dig(_load("outputs/deployclot/coupled_flow_delta.json"),
                 "summary", "median_vel_rel_l2"), 1e-4,
     "median velocity change under the physical clot load"),
    ("cplflow.switched", "STORY 2.1h", 239,
     lambda: dig(_load("outputs/deployclot/coupled_flow_delta.json"),
                 "summary", "total_gate_switched"), 0,
     "wall nodes changing gate state across 4 vessels"),
    ("cplflow.n", "STORY 2.1h", 4,
     lambda: dig(_load("outputs/deployclot/coupled_flow_delta.json"), "summary", "n"), 0, ""),

    # --- STORY 2.1i, THE REAL FEM RE-SOLVE (E1h) -------------------------------------------
    # RETRACTED 2026-09-13: the seed-PAIRED off-wall +0.0517 (p 0.008) came from pairing every
    # coupled run 3 with the degenerate uncoupled run `cpl_open_s3`.  These rows now quote the
    # symmetric healthy-run comparison (`diag_coupling_arms.degenerate` on both sides).
    ("cplfem.wall_delta", "STORY 2.1i", 0.0016,
     lambda: dig(_load("outputs/deployclot/coupling_fem_resolve.json"),
                 "result", "wall", "delta_healthy"), 5e-5,
     "3 healthy coupled runs vs 2 healthy uncoupled runs"),
    ("cplfem.wall_se", "STORY 2.1i", 0.0021,
     lambda: dig(_load("outputs/deployclot/coupling_fem_resolve.json"),
                 "result", "wall", "se"), 5e-5, "pooled run-noise SE"),
    ("cplfem.off_delta", "STORY 2.1i", 0.0084,
     lambda: dig(_load("outputs/deployclot/coupling_fem_resolve.json"),
                 "result", "off", "delta_healthy"), 5e-5,
     "replaces the retracted paired +0.0517"),
    ("cplfem.off_se", "STORY 2.1i", 0.0185,
     lambda: dig(_load("outputs/deployclot/coupling_fem_resolve.json"),
                 "result", "off", "se"), 5e-5, ""),
    ("cplfem.open_degenerate", "STORY 2.1i", 1,
     lambda: len(dig(_load("outputs/deployclot/coupling_fem_resolve.json"),
                     "result", "off", "degenerate_open")), 0,
     "cpl_open_s3 has the collapse cut signature: the uncoupled arm collapses too"),
    # --- E1i / FIGURE coupling_frequency: every schedule vs uncoupled, healthy runs -------------
    ("cplfreq.final_off", "STORY 2.1i", 0.0126,
     lambda: dig(_load("outputs/deployclot/coupling_freq_femfinal.json"),
                 "result", "off", "delta_healthy"), 5e-5, "one solve at the end"),
    ("cplfreq.every_off", "STORY 2.1i", 0.0171,
     lambda: dig(_load("outputs/deployclot/coupling_freq_fem1.json"),
                 "result", "off", "delta_healthy"), 5e-5, "re-solve on every new clot"),
    ("cplfreq.every_wall", "STORY 2.1i", 0.0022,
     lambda: dig(_load("outputs/deployclot/coupling_freq_fem1.json"),
                 "result", "wall", "delta_healthy"), 5e-5, ""),
    ("cplfreq.n5_off", "STORY 2.1i", -0.0543,
     lambda: dig(_load("outputs/deployclot/coupling_freq_femn5.json"),
                 "result", "off", "delta_healthy"), 5e-5,
     "the largest move in the ladder, and NEGATIVE"),
    ("cplfreq.every_solves", "STORY 2.1i", 24.92,
     lambda: next(r["solves"] for r in _load(
         "outputs/publication/data/coupling_frequency.json") if r["stem"] == "fem1"), 5e-3, ""),
    # --- FIGURE applications (0.4): dense strength sweeps -----------------------------------------
    ("apps.sten_last_zero", "FIGURES 0.4", 21.2, lambda: _apps_last_zero("01_stenosis_strength"), 0.05,
     "largest narrowing with zero clot"),
    ("apps.sten_50_mass", "FIGURES 0.4", 0.71, lambda: _apps_at("01_stenosis_strength", 49.8), 0.01, ""),
    ("apps.sten_77_mass", "FIGURES 0.4", 2.51, lambda: _apps_at("01_stenosis_strength", 77.1), 0.01,
     "trained maximum; 2.54 on the pre-2026-09-18 all-index grid"),
    ("apps.aneu_last_zero", "FIGURES 0.4", 46.2, lambda: _apps_last_zero("02_aneurysm_strength"), 0.05, ""),
    ("apps.aneu_step_mass", "FIGURES 0.4", 0.71, lambda: _apps_at("02_aneurysm_strength", 46.8), 0.01,
     "first point with clot"),
    ("apps.aneu_100_mass", "FIGURES 0.4", 1.10, lambda: _apps_at("02_aneurysm_strength", 99.8), 0.01, ""),
    ("apps.cost_min", "FIGURES 0.4", 39.3, _apps_cost_min, 1.0,
     "rollout minutes, 80 points -- 11 trained query points per vessel, not 120"),
    # strict F1 of the same predictions (eval_strict records it beside BATC; figure row 2)
    ("cplfreq.f1_every_wall", "STORY 2.1i", 0.0044,
     lambda: _cplfreq_f1("fem1", "wall", "delta_healthy"), 5e-5, "every new clot node, strict F1"),
    ("cplfreq.f1_every_off", "STORY 2.1i", 0.0062,
     lambda: _cplfreq_f1("fem1", "off", "delta_healthy"), 5e-5, ""),
    ("cplfreq.f1_n5_off", "STORY 2.1i", -0.0945,
     lambda: _cplfreq_f1("femn5", "off", "delta_healthy"), 5e-5, "same arm and sign as BATC's largest move"),
    ("cplfreq.f1_64_wall", "STORY 2.1i", -0.0230,
     lambda: _cplfreq_f1("fem64", "wall", "delta_healthy"), 5e-5, "outside its 2 SE, negative"),
    ("cplfreq.f1_2se_wall", "STORY 2.1i", 0.012,
     lambda: 2 * _cplfreq_f1("fem1", "wall", "se"), 1e-3, ""),
    ("cplfreq.f1_2se_off", "STORY 2.1i", 0.091,
     lambda: 2 * _cplfreq_f1("fem1", "off", "se"), 1e-3, ""),
    ("cplfem.leak_f1", "STORY 2.1i", 0.8893,
     lambda: dig(_load("outputs/deployclot/coupling_leakage_ladder.json"),
                 "result", "fem_resolve", "single_feature_best_f1"), 5e-5,
     "GT-free by construction; the lift over uncoupled 0.8215 is physical signal"),
    ("cplfem.leak_f1_uncoupled", "STORY 2.1i", 0.8215,
     lambda: dig(_load("outputs/deployclot/coupling_leakage_ladder.json"),
                 "result", "uncoupled", "single_feature_best_f1"), 5e-5, ""),
    ("cplfem.leak_f1_oracle", "STORY 2.1i", 0.9742,
     lambda: dig(_load("outputs/deployclot/coupling_leakage_ladder.json"),
                 "result", "every_oracle", "single_feature_best_f1"), 5e-5, ""),

    # --- STORY 2.1g, THE UNCONTAMINATED COUPLING NUMBER -----------------------------------
    # The corrector is driven by the ROLLOUT'S OWN committed set (gelation wake), not GT, so it
    # passes the leakage gate that invalidated every earlier coupling arm (2.1f).  Same 36
    # vessels, same 3 folds, same arm.  THIS is the figure the paper may quote for what
    # clot->flow coupling is worth; nothing in 2.1-2.1e may be.
    #
    # ONE HEALTHY SEED PAIR: the wake arm's seed-0 run collapsed a fold (as k16's did), so the
    # comparison rests on seed 1 alone.  Thin, and the doc says so.
    ("cplself.wall_delta", "STORY 2.1g", 0.0045,
     lambda: dig(_load("outputs/deployclot/coupling_selfdriven.json"),
                 "result_two_clean_pairs", "wall", "delta_mean"), 5e-5,
     "mean over TWO independent clean seed pairs (-0.0010, +0.0100): inside the 0.0137 seed null -- a zero"),
    ("cplself.wall_pair1", "STORY 2.1g", -0.0010,
     lambda: dig(_load("outputs/deployclot/coupling_selfdriven.json"),
                 "result_two_clean_pairs", "wall", "delta_pair1"), 5e-5, ""),
    ("cplself.wall_pair2", "STORY 2.1g", 0.0100,
     lambda: dig(_load("outputs/deployclot/coupling_selfdriven.json"),
                 "result_two_clean_pairs", "wall", "delta_pair2"), 5e-5, ""),
    ("cplself.wall_null", "STORY 2.1g", 0.0137,
     lambda: dig(_load("outputs/deployclot/coupling_selfdriven.json"),
                 "result_two_clean_pairs", "wall", "open_seed_null"), 5e-5,
     "open arm's own spread over 3 seeds -- the null the signal must beat"),
    ("cplself.off_delta", "STORY 2.1g", 0.0576,
     lambda: dig(_load("outputs/deployclot/coupling_selfdriven.json"),
                 "result_two_clean_pairs", "off", "delta_mean"), 5e-5,
     "the two pairs DISAGREE IN SIGN (-0.0300, +0.1452): off-wall is not established"),
    ("cplself.off_null", "STORY 2.1g", 0.1417,
     lambda: dig(_load("outputs/deployclot/coupling_selfdriven.json"),
                 "result_two_clean_pairs", "off", "open_seed_null"), 5e-5,
     "larger than the effect -- which is why no off-wall claim may be made"),
    ("cplself.leak_f1", "STORY 2.1g", 0.8252,
     lambda: dig(_load("outputs/deployclot/coupling_leakage.json"),
                 "result", "wake_selfdriven", "single_feature_best_f1"), 5e-4,
     "leakage gate PASSED: 0.8252 vs uncoupled 0.7950 and GT-oracle 0.9609"),

    # --- STORY 2.1f, the LEAKAGE that invalidates the coupling numbers --------------------
    # `oracle_blockage` lowers shear exactly where the GT clot is, so coupled `mat_phys` encodes
    # the answer -- for HELD-OUT vessels too, since each vessel's features come from its own GT.
    # These rows exist so the contamination is machine-checked, not just narrated.
    ("cplleak.f1_uncoupled", "STORY 2.1f", 0.7950,
     lambda: dig(_load("outputs/deployclot/coupling_leakage.json"),
                 "result", "uncoupled", "single_feature_best_f1"), 5e-5,
     "mat_phys alone, best threshold per vessel, wall domain"),
    ("cplleak.f1_every", "STORY 2.1f", 0.9609,
     lambda: dig(_load("outputs/deployclot/coupling_leakage.json"),
                 "result", "every_step", "single_feature_best_f1"), 5e-5,
     "ONE feature recovers the label -- the coupled feature IS the answer"),
    ("cplleak.auc_every", "STORY 2.1f", 0.9966,
     lambda: dig(_load("outputs/deployclot/coupling_leakage.json"),
                 "result", "every_step", "single_feature_auc"), 5e-5, ""),
    ("cplleak.auc_uncoupled", "STORY 2.1f", 0.8610,
     lambda: dig(_load("outputs/deployclot/coupling_leakage.json"),
                 "result", "uncoupled", "single_feature_auc"), 5e-5, ""),

    # --- STORY 2.1e, the STRIDE experiment ------------------------------------------------
    # Re-solving every 16th step (1.95x) against every step (4.96x), same 36 vessels, same 3
    # folds, same arm. HEADLINE USES THE 2 HEALTHY k16 SEEDS: 1 of 3 k16 runs had a fold-level
    # readout collapse (fold 0's cut tuner picked 0.05 where healthy runs picked 0.95-0.98),
    # which no open or closed run showed. The all-3-seed figure is also balloted so the
    # contamination stays visible -- see `instability` in the artifact.
    ("cplstride.wall_k16", "STORY 2.1e", 0.0368,
     lambda: dig(_load("outputs/deployclot/coupling_stride.json"),
                 "result", "wall", "delta_k16_healthy"), 5e-5,
     "1.95x cost, same wall gain as every-step at 4.96x. The k16-vs-every DIFFERENCE "
     "(+0.0025) is INSIDE the every-step arm's seed noise (0.0069) -- not established"),
    ("cplstride.wall_frac", "STORY 2.1e", 1.073,
     lambda: dig(_load("outputs/deployclot/coupling_stride.json"),
                 "result", "wall", "frac_of_ceiling_healthy"), 5e-3,
     "fraction of the every-step ceiling captured at stride 16"),
    ("cplstride.off_k16", "STORY 2.1e", 0.0244,
     lambda: dig(_load("outputs/deployclot/coupling_stride.json"),
                 "result", "off", "delta_k16_healthy"), 5e-5,
     "positive here where every-step reads -0.0910, but the difference (+0.1154) is "
     "INSIDE the every-step arm's off-wall seed noise (0.1234). NOT an established reversal"),
    ("cplstride.wall_spread_k16", "STORY 2.1e", 0.0011,
     lambda: dig(_load("outputs/deployclot/coupling_stride.json"),
                 "result", "wall", "seed_spread_k16_healthy"), 5e-5,
     "healthy k16 seeds agree as tightly as the open arm (0.0009)"),
    ("cplstride.wall_spread_k16_all", "STORY 2.1e", 0.3020,
     lambda: dig(_load("outputs/deployclot/coupling_stride.json"),
                 "result", "wall", "seed_spread_k16_all3"), 5e-4,
     "with the failed run included -- the instability, stated not hidden"),

    # --- STORY 2.1d, what coupling would actually COST ------------------------------------
    # Perfect coupling is ATTAINABLE -- re-solve the FEM each step against the current clot's
    # viscosity field. So the argument is a cost/benefit trade, never "it cannot be done".
    # These are arithmetic on timing.json, and `k*fem` is a LOWER BOUND (the solve is timed on
    # a clot-free field; a clot-laden one is worse conditioned).
    ("cplcost.grid_slowdown", "STORY 2.1d", 4.96,
     lambda: dig(_load("outputs/publication/data/coupling_cost.json"),
                 "variants", "deploy grid (every=4)", "slowdown"), 5e-3,
     "51 solves instead of 1"),
    ("cplcost.grid_total_s", "STORY 2.1d", 291.9,
     lambda: dig(_load("outputs/publication/data/coupling_cost.json"),
                 "variants", "deploy grid (every=4)", "total_s"), 0.05, ""),
    ("cplcost.grid_speedup", "STORY 2.1d", 592,
     lambda: dig(_load("outputs/publication/data/coupling_cost.json"),
                 "variants", "deploy grid (every=4)", "speedup_vs_comsol"), 1.0,
     "still three orders of magnitude faster than COMSOL -- affordability is NOT the argument"),
    ("cplcost.sparse_slowdown", "STORY 2.1d", 1.95,
     lambda: dig(_load("outputs/publication/data/coupling_cost.json"),
                 "variants", "every 16th step", "slowdown"), 5e-3,
     "the unexplored middle ground: 13 solves"),
    ("cplcost.full_slowdown", "STORY 2.1d", 16.83,
     lambda: dig(_load("outputs/publication/data/coupling_cost.json"),
                 "variants", "full grid (every=1)", "slowdown"), 5e-3, ""),

    # --- STORY 0.2, the wound complement, held out ---------------------------------------
    # MEASURED 2026-09-12: `scripts/eval_wound_complement.py --flow fem
    # --lovo outputs/clot_ml/wound_rate_fem/lovo.json --every 4`, n=6, each vessel scored with
    # the (G_pre, G_post) fitted WITHOUT it.
    #
    # THESE REPLACE 0.9270 / 0.8611, WHICH WERE NEVER BACKED BY ANYTHING.  That pair appears
    # in the repo only as hardcoded literals in `scripts/promote_clot_ml_0.py`'s
    # `lovo_held_out` block -- no artifact produces them, and the shipped manifest carries no
    # `scores_wound` key at all.  They also describe a DIFFERENT arm (clot_ml_0 with the
    # chemistry replace+depth layer) than the v4 + two-regime complement measured here, so the
    # two are not alternative estimates of one quantity.  Do not reconcile them; do not requote
    # the old pair.
    ("wound.lovo.w_reg", "STORY 0.2", 0.8789,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"), "final", "w_reg"),
     5e-5, "v4 + two-regime complement, FEM flow, LOVO constants, n=6"),
    ("wound.lovo.w_lum", "STORY 0.2", 0.7970,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"), "final", "w_lum"),
     5e-5, ""),
    ("wound.lovo.n", "STORY 0.2", 6,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"), "n_vessels"), 0, ""),
    ("wound.base.w_reg", "STORY 0.2", 0.0949,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"),
                 "baseline_final", "w_reg"), 5e-5,
     "the bare GNN on the same vessels -- near-blind to the injury"),
    ("wound.base.w_lum", "STORY 0.2", 0.0514,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"),
                 "baseline_final", "w_lum"), 5e-5, ""),
    ("wound.phys.w_reg", "STORY 0.2", 0.6666,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"),
                 "physics_only_final", "w_reg"), 5e-5,
     "ungated law, ZERO fitted parameters -- moves the region"),
    ("wound.phys.w_lum", "STORY 0.2", 0.0514,
     lambda: dig(_load("outputs/publication/data/wound_lovo_fem.json"),
                 "physics_only_final", "w_lum"), 5e-5,
     "...and leaves the LUMEN at the baseline: the lumen half is bought by the two scalars"),

    # --- STORY 0.3, BATC's response function ---------------------------------------------
    # Written by `scripts/publication/plot_batc.py`, which computes them by CALLING the shipped
    # metric (`severity_from_counts`) rather than transcribing a table. That is how the stale
    # 0.690 in DEPLOYCLOT 0.3 was caught: it was written when `tau_abs = 5`.
    ("batc.case15.adj", "STORY 0.3", 0.8889,
     lambda: dig(_load("outputs/publication/data/batc.json"), "panel_a_cases", 0, "batc"),
     5e-5, "10 of 15 found"),
    ("batc.case15.unadj", "STORY 0.3", 0.6667,
     lambda: dig(_load("outputs/publication/data/batc.json"), "panel_a_cases", 0, "unadjusted"),
     5e-5, ""),
    ("batc.case150.adj", "STORY 0.3", 0.7407,
     lambda: dig(_load("outputs/publication/data/batc.json"), "panel_a_cases", 1, "batc"),
     5e-5, "100 of 150 -- the row DEPLOYCLOT 0.3 still reports as 0.690"),
    ("batc.case4.adj", "STORY 0.3", 0.3333,
     lambda: dig(_load("outputs/publication/data/batc.json"), "panel_a_cases", 2, "batc"),
     5e-5, "1 of 4 -- the rho cap binds here"),
    ("batc.case4.unadj", "STORY 0.3", 0.25,
     lambda: dig(_load("outputs/publication/data/batc.json"), "panel_a_cases", 2, "unadjusted"),
     5e-5, ""),

    # --- FIGURES 0.3 / 0.6 / 0.6b, real vessels scored by `example_vessels` (shipped BATC) -----
    ("batcex.c041.f1", "FIGURES 0.3", 0.6771,
     lambda: next(r["strict_f1"] for r in _load("outputs/publication/data/batc_examples.json")
                  if r["stem"] == "comsol041"), 5e-5, "every error a near miss: strict F1"),
    ("batcex.c041.batc", "FIGURES 0.3", 0.9623,
     lambda: next(r["batc"] for r in _load("outputs/publication/data/batc_examples.json")
                  if r["stem"] == "comsol041"), 5e-5, "...and BATC"),
    ("batcex.c010.nograce", "FIGURES 0.3", 0.6749,
     lambda: next(r["no_grace"] for r in _load("outputs/publication/data/batc_examples.json")
                  if r["stem"] == "comsol010"), 5e-5, "12-node clot, graces zeroed"),
    ("batcex.c010.batc", "FIGURES 0.3", 0.7855,
     lambda: next(r["batc"] for r in _load("outputs/publication/data/batc_examples.json")
                  if r["stem"] == "comsol010"), 5e-5, "...with the graces"),
    ("batcex.c028.batc", "FIGURES 0.3", 0.6573,
     lambda: next(r["batc"] for r in _load("outputs/publication/data/batc_examples.json")
                  if r["stem"] == "comsol028"), 5e-5, "a real miss BATC does not rescue"),
    ("fig6.c005.off", "STORY 0.3", 0.5479,
     lambda: _load("outputs/publication/data/oof_batc_series.json")["comsol005"]["off"]["score"][-1],
     5e-4, "was quoted as 0.262 -- that was BATC_0"),
    ("fig6.c005.wall", "STORY 0.3", 0.9957,
     lambda: _load("outputs/publication/data/oof_batc_series.json")["comsol005"]["wall"]["score"][-1],
     5e-4, ""),
    ("fig6b.c014.wall_dip", "FIGURES 0.6b", 0.2036,
     lambda: _load("outputs/publication/data/oof_batc_series.json")["comsol014"]["wall"]["score"][2],
     5e-4, "t=40"),
    ("fig6b.c014.off_final_pred", "FIGURES 0.6b", 76,
     lambda: _load("outputs/publication/data/oof_batc_series.json")["comsol014"]["off"]["n_pred"][-1],
     0, "off-wall predicted nodes at the last frame, on a domain with no GT clot"),

    # --- §5d / Fig 9, the diagnosis -----------------------------------------------------
    # 2026-09-19: re-scored on the CURRENT code (the old artifacts predated the mirror-branch
    # rule), and the two mirror-branch vessels are excluded -- neither axis of this panel is
    # measurable on them (`generate_flow_requirement_data._pairs`).  The time grid is not a
    # factor: `--every 4` and the trained grid give these correlations to three decimals.
    ("flowreq.gate_jaccard_r", "PUBLICATION_NOTES 2.4 / 7.2", -0.597,
     flowreq("gate_jaccard"), 5e-4, "p<0.001, Spearman -0.657"),
    ("flowreq.rel_l2_r", "PUBLICATION_NOTES 2.4 / 7.2", 0.227,
     flowreq("rel_l2"), 5e-4, "p=0.19 -- not significant, and that is the point"),
    ("flowreq.fire_ratio_r", "PUBLICATION_NOTES 7.2", 0.491,
     flowreq("fire_ratio"), 5e-4, "p=0.003, but NOT separable from rel-L2 (Williams p=0.20)"),
    ("flowreq.n", "PUBLICATION_NOTES 2.4", 35,
     flowreq("gate_jaccard", "n"), 0, "was 33 pre-mirror-rule; 37 paired, minus comsol045/046"),

    # --- Table 2, the C0 constraint -----------------------------------------------------
    # These were the two rows that caught the doc error on 2026-09-07: PUBLICATION_PLAN 8
    # carried "0.5812 -> 0.7078" (a retired `clot_gnn_v5w` / GT-cache pair, MODEL_REVIEW 9b)
    # while 12.3 of the same file already carried the correct +0.1312.  Now pinned to the
    # shipped generation so the two cannot drift apart again.
    ("c0.off_before", "PUBLICATION_PLAN 8 Table 2", 0.7121,
     paired("outputs/deployclot/c0_ablation_paired.json", "off", "mean_a"), 5e-5,
     "dc_fem_noc0, cache v5_fem"),
    ("c0.off_after", "PUBLICATION_PLAN 8 Table 2", 0.8439,
     paired("outputs/deployclot/c0_ablation_paired.json", "off", "mean_b"), 5e-5,
     "dc_fem_c0, cache v5_fem"),
    ("c0.off_delta", "PUBLICATION_PLAN 8 / 12.3", 0.1318,
     paired("outputs/deployclot/c0_ablation_paired.json", "off", "delta"), 5e-5, ""),
    ("c0.off_n", "PUBLICATION_PLAN 8", 20,
     paired("outputs/deployclot/c0_ablation_paired.json", "off", "n"), 0, ""),
    ("c0.wall_before", "PUBLICATION_PLAN 8 Table 2", 0.9336,
     paired("outputs/deployclot/c0_ablation_paired.json", "wall", "mean_a"), 5e-5, ""),
    ("c0.wall_after", "PUBLICATION_PLAN 8 Table 2", 0.9517,
     paired("outputs/deployclot/c0_ablation_paired.json", "wall", "mean_b"), 5e-5, ""),
    ("c0.wall_p", "PUBLICATION_PLAN 8 Table 2", 0.0125,
     paired("outputs/deployclot/c0_ablation_paired.json", "wall", "p"), 5e-5, ""),

    # --- threshold-free summary (README, and a good abstract number) --------------------
    # Pooled over nodes across vessels, so it answers "where should the cut go", NOT "how well
    # does it generalize per vessel" -- Table 4 answers that.  Both belong in the paper and
    # they must not be swapped for each other.
    ("aucpr.wall", "README / PAPER 5", 0.932,
     lambda: dig(_load("outputs/publication/data/operating_point.json"),
                 "domains", "wall", "average_precision"), 5e-4, ""),
    ("aucpr.wall_base", "README", 0.181,
     lambda: dig(_load("outputs/publication/data/operating_point.json"),
                 "domains", "wall", "base_rate"), 5e-4, "5x the floor"),
    ("aucpr.off", "README / PAPER 5", 0.621,
     lambda: dig(_load("outputs/publication/data/operating_point.json"),
                 "domains", "off", "average_precision"), 5e-4, ""),
    ("aucpr.off_base", "README", 0.0034,
     lambda: dig(_load("outputs/publication/data/operating_point.json"),
                 "domains", "off", "base_rate"), 5e-5, "177x the floor"),
    ("aucpr.n_vessels", "README", 27,
     lambda: dig(_load("outputs/publication/data/operating_point.json"),
                 "source", "n_vessels"), 0, ""),

    # --- what TEMPORAL CONDITIONING was worth, v3 era (PHASE9_ML 13.9) ------------------
    # Different generation: GT flow, 19 vessels, 11 times, 56 columns.  Scope caveat travels
    # with every one of these rows; see the `v3` resolver.
    ("v3.timecond.wall", "PAPER 4.5", 0.8845, v3("all", "wall"), 5e-5, "v3 era, GT flow"),
    ("v3.timecond.off", "PAPER 4.5", 0.6110, v3("all", "off"), 5e-5, "v3 era, GT flow"),
    ("v3.frozen.wall", "PAPER 4.5", 0.7953, v3("baselines", "frozen", "wall"), 5e-5, ""),
    ("v3.frozen.off", "PAPER 4.5", 0.4209, v3("baselines", "frozen", "off"), 5e-5, ""),
    ("v3.rank.wall", "PAPER 4.5", 0.8547, v3("baselines", "v2_plus_ode", "wall"), 5e-5, ""),
    ("v3.rank.off", "PAPER 4.5", 0.5369, v3("baselines", "v2_plus_ode", "off"), 5e-5, ""),
    ("v3.oracle_sched.wall", "PAPER 4.5", 0.8908,
     v3("baselines", "oracle_schedule", "wall"), 5e-5, ""),
    ("v3.oracle_sched.off", "PAPER 4.5", 0.6619,
     v3("baselines", "oracle_schedule", "off"), 5e-5, ""),

    # --- the TEMPORAL axis the final-time ladder cannot see -----------------------------
    # Final-time is identical for frozen / shipped / oracle (0.9479 wall), which is the whole
    # point: the temporal head redistributes WHEN a node commits, never WHETHER.  These rows
    # keep that fact pinned, because it is the evidence that a final-time ablation cannot
    # price the ODE.
    ("tmp.final_wall", "PAPER 4.5", 0.9614, temporal("wall_final"), 5e-5,
     "identical for frozen and oracle -- see the rows below"),
    ("tmp.frozen_wall_final", "PAPER 4.5", 0.9614, temporal("frozen_wall_final"), 5e-5, ""),
    ("tmp.oracle_wall_final", "PAPER 4.5", 0.9614, temporal("oracle_wall_final"), 5e-5, ""),
    ("tmp.frozen_wall", "PAPER 4.5", 0.8712, temporal("frozen_wall"), 5e-5, "mean-over-time"),
    ("tmp.wall", "PAPER 4.5", 0.9125, temporal("wall"), 5e-5, "mean-over-time, shipped head"),
    ("tmp.oracle_wall", "PAPER 4.5", 0.9812, temporal("oracle_wall"), 5e-5, "ceiling"),
    ("tmp.frozen_off", "PAPER 4.5", 0.6015, temporal("frozen_off"), 5e-5, ""),
    ("tmp.off", "PAPER 4.5", 0.7245, temporal("off"), 5e-5, ""),
    ("tmp.oracle_off", "PAPER 4.5", 0.8951, temporal("oracle_off"), 5e-5, ""),

    # --- THE PAPER'S CENTRAL COMPARISON: physics vs learning vs both --------------------
    # Paired over vessels.  The unpaired means alone produced a wrong claim on 2026-09-08
    # ("physics alone beats the plain GNN at the wall"): the point estimate is +0.055 but the
    # paired CI spans zero.  Every row here therefore pins the INTERVAL, not just the mean.
    ("pvl.wall.phys", "PAPER 4", 0.9198,
     lambda: dig(_load(PVL), "means", "wall", "phys"), 5e-5, "backbone alone, no learning"),
    ("pvl.wall.plain", "PAPER 4", 0.8644,
     lambda: dig(_load(PVL), "means", "wall", "plain"), 5e-5, "GNN, no physics anywhere"),
    ("pvl.wall.full", "PAPER 4", 0.9637,
     lambda: dig(_load(PVL), "means", "wall", "full"), 5e-5, ""),
    ("pvl.off.phys", "PAPER 4", 0.5480,
     lambda: dig(_load(PVL), "means", "off", "phys"), 5e-5, ""),
    ("pvl.off.plain", "PAPER 4", 0.7776,
     lambda: dig(_load(PVL), "means", "off", "plain"), 5e-5, ""),
    ("pvl.off.full", "PAPER 4", 0.7928,
     lambda: dig(_load(PVL), "means", "off", "full"), 5e-5, ""),
    # the three paired tests per domain, and their significance
    ("pvl.wall.full_minus_plain", "PAPER 4", 0.0993,
     lambda: dig(_load(PVL), "tests", "wall", "full_minus_plain", "delta"), 5e-5, "SIG"),
    ("pvl.wall.full_minus_phys", "PAPER 4", 0.0439,
     lambda: dig(_load(PVL), "tests", "wall", "full_minus_phys", "delta"), 5e-5, "SIG"),
    ("pvl.wall.phys_minus_plain", "PAPER 4", 0.0554,
     lambda: dig(_load(PVL), "tests", "wall", "phys_minus_plain", "delta"), 5e-5,
     "NOT significant -- CI spans zero"),
    ("pvl.wall.phys_minus_plain_p", "PAPER 4", 0.115,
     lambda: dig(_load(PVL), "tests", "wall", "phys_minus_plain", "p_le0"), 5e-4, ""),
    ("pvl.off.phys_minus_plain", "PAPER 4", -0.2296,
     lambda: dig(_load(PVL), "tests", "off", "phys_minus_plain", "delta"), 5e-5, "SIG"),
    ("pvl.off.full_minus_plain", "PAPER 4", 0.0152,
     lambda: dig(_load(PVL), "tests", "off", "full_minus_plain", "delta"), 5e-5,
     "NOT significant -- physics adds nothing off-wall"),
    ("pvl.off.full_minus_plain_p", "PAPER 4", 0.292,
     lambda: dig(_load(PVL), "tests", "off", "full_minus_plain", "p_le0"), 5e-4, ""),
    ("pvl.off.full_minus_phys", "PAPER 4", 0.2448,
     lambda: dig(_load(PVL), "tests", "off", "full_minus_phys", "delta"), 5e-5, "SIG"),
    ("pvl.n", "PAPER 4", 27, lambda: dig(_load(PVL), "n_vessels"), 0, ""),

    # --- the FAIR head-to-head: physics given the same out-of-fold fitted cut --------------
    # Audited 2026-09-08: `phys` is a parameter-free MASK while `plain`/`full` each get a cut
    # tuned out-of-fold, and off-wall that is not like-for-like -- the mask commits nothing on
    # 7 of 27 vessels and its burden spans 0.03x-9.3x GT.  `phys_tuned` threads the backbone's
    # own continuous field through the identical tuner.  It shrinks the off-wall gap by 27%
    # (0.2456 -> 0.1784) and leaves the wall untouched, so the conclusion survives a fair test.
    ("pvl.wall.phys_tuned", "PAPER 4", 0.9230,
     lambda: dig(_load(PVL), "means", "wall", "phys_tuned"), 5e-5, ""),
    ("pvl.off.phys_tuned", "PAPER 4", 0.5925,
     lambda: dig(_load(PVL), "means", "off", "phys_tuned"), 5e-5, ""),
    ("pvl.wall.phystuned_minus_phys", "PAPER 4", 0.0032,
     lambda: dig(_load(PVL), "tests", "wall", "phystuned_minus_phys", "delta"), 5e-5,
     "n.s. -- the wall rule is already at its optimum, tuning buys nothing"),
    ("pvl.off.phystuned_minus_phys", "PAPER 4", 0.0444,
     lambda: dig(_load(PVL), "tests", "off", "phystuned_minus_phys", "delta"), 5e-5, "n.s."),
    ("pvl.wall.phystuned_minus_plain", "PAPER 4", 0.0586,
     lambda: dig(_load(PVL), "tests", "wall", "phystuned_minus_plain", "delta"), 5e-5,
     "n.s., P=0.096"),
    ("pvl.off.phystuned_minus_plain", "PAPER 4", -0.1852,
     lambda: dig(_load(PVL), "tests", "off", "phystuned_minus_plain", "delta"), 5e-5,
     "SIG -- the corrected off-wall gap, replaces -0.2456"),
    ("pvl.wall.full_minus_phystuned", "PAPER 4", 0.0407,
     lambda: dig(_load(PVL), "tests", "wall", "full_minus_phystuned", "delta"), 5e-5, "SIG"),
    ("pvl.off.full_minus_phystuned", "PAPER 4", 0.2004,
     lambda: dig(_load(PVL), "tests", "off", "full_minus_phystuned", "delta"), 5e-5, "SIG"),

    # --- the ODE clock vs a learned clock, off-wall lag only -----------------------------
    ("tmpol.ode.wall", "PAPER 5.3", 0.9125, temporal_ol("ode", "wall"), 5e-5,
     "identical to pred BY CONSTRUCTION -- the flag does not touch the wall clock"),
    ("tmpol.pred.wall", "PAPER 5.3", 0.9125, temporal_ol("pred", "wall"), 5e-5, ""),
    ("tmpol.ode.off", "PAPER 5.3", 0.7551, temporal_ol("ode", "off"), 5e-5, ""),
    ("tmpol.pred.off", "PAPER 5.3", 0.7611, temporal_ol("pred", "off"), 5e-5,
     "a LEARNED lag anchor is marginally better than the ODE one"),

    # --- THE MANUSCRIPT'S NUMBERS: four-point axis on the SHIPPED generation (v5_fem) ---
    # All four predictors share one cache, one protocol and one fold partition (verified), so
    # every delta below is attributable.  These supersede the `pvl.*` rows for quotation.
    ("pvlf.wall.naive", "PAPER 5", 0.6818,
     lambda: dig(_load(PVLF), "means", "wall", "naive"), 5e-5, "from-scratch control"),
    ("pvlf.wall.plain", "PAPER 5", 0.8191,
     lambda: dig(_load(PVLF), "means", "wall", "plain"), 5e-5, "physics-informed arch only"),
    ("pvlf.wall.phys_tuned", "PAPER 5", 0.9230,
     lambda: dig(_load(PVLF), "means", "wall", "phys_tuned"), 5e-5, ""),
    ("pvlf.wall.full", "PAPER 5", 0.9481,
     lambda: dig(_load(PVLF), "means", "wall", "full"), 5e-5, ""),
    ("pvlf.off.naive", "PAPER 5", 0.4834,
     lambda: dig(_load(PVLF), "means", "off", "naive"), 5e-5, ""),
    ("pvlf.off.plain", "PAPER 5", 0.6911,
     lambda: dig(_load(PVLF), "means", "off", "plain"), 5e-5, ""),
    ("pvlf.off.phys_tuned", "PAPER 5", 0.5317,
     lambda: dig(_load(PVLF), "means", "off", "phys_tuned"), 5e-5, ""),
    ("pvlf.off.full", "PAPER 5", 0.8285,
     lambda: dig(_load(PVLF), "means", "off", "full"), 5e-5, ""),
    # the deltas that carry the argument
    ("pvlf.wall.plain_minus_naive", "PAPER 5", 0.1373,
     lambda: dig(_load(PVLF), "tests", "wall", "plain_minus_naive", "delta"), 5e-5,
     "what the physics-informed ARCHITECTURE is worth, before any conditioning"),
    ("pvlf.off.plain_minus_naive", "PAPER 5", 0.2077,
     lambda: dig(_load(PVLF), "tests", "off", "plain_minus_naive", "delta"), 5e-5, "SIG"),
    ("pvlf.wall.phystuned_minus_plain", "PAPER 5", 0.1039,
     lambda: dig(_load(PVLF), "tests", "wall", "phystuned_minus_plain", "delta"), 5e-5, "SIG"),
    ("pvlf.wall.full_minus_phystuned", "PAPER 5", 0.0252,
     lambda: dig(_load(PVLF), "tests", "wall", "full_minus_phystuned", "delta"), 5e-5,
     "NOT significant -- learning adds nothing measurable to physics at the wall"),
    ("pvlf.wall.full_minus_phystuned_p", "PAPER 5", 0.066,
     lambda: dig(_load(PVLF), "tests", "wall", "full_minus_phystuned", "p_le0"), 5e-4, ""),
    ("pvlf.off.full_minus_plain", "PAPER 5", 0.1374,
     lambda: dig(_load(PVLF), "tests", "off", "full_minus_plain", "delta"), 5e-5,
     "SIG -- physics conditioning DOES pay off-wall on the shipped generation"),
    ("pvlf.off.full_minus_plain_p", "PAPER 5", 0.007,
     lambda: dig(_load(PVLF), "tests", "off", "full_minus_plain", "p_le0"), 5e-4, ""),
    ("pvlf.off.full_minus_phystuned", "PAPER 5", 0.2968,
     lambda: dig(_load(PVLF), "tests", "off", "full_minus_phystuned", "delta"), 5e-5, "SIG"),
    ("pvlf.wall.full_minus_naive", "PAPER 5", 0.2663,
     lambda: dig(_load(PVLF), "tests", "wall", "full_minus_naive", "delta"), 5e-5, "SIG"),
    ("pvlf.off.full_minus_naive", "PAPER 5", 0.3451,
     lambda: dig(_load(PVLF), "tests", "off", "full_minus_naive", "delta"), 5e-5, "SIG"),

    # --- THE THIRD DOOR: physics as a READOUT device -------------------------------------
    # The `resid` readout family thresholds physics-positive and physics-negative nodes
    # separately, i.e. it reads the backbone's occlusion mask.  Under `auto` the tuner may
    # choose it for ANY arm, so a from-scratch GNN can borrow physics at the readout and land
    # within 0.01 of the shipped model at the wall -- which is what makes the `plain` column
    # mandatory rather than optional.  These rows pin both readings of the same arm.
    ("rd.naive.auto_wall", "PAPER 5", 0.9357,
     ablation("A_naive", "auto", "wall", report=ABLF), 5e-5, "physics available at readout"),
    ("rd.naive.plain_wall", "PAPER 5", 0.6818,
     ablation("A_naive", "plain", "wall", report=ABLF), 5e-5, "physics barred everywhere"),
    ("rd.naive.auto_off", "PAPER 5", 0.6029,
     ablation("A_naive", "auto", "off", report=ABLF), 5e-5, ""),
    ("rd.naive.plain_off", "PAPER 5", 0.4834,
     ablation("A_naive", "plain", "off", report=ABLF), 5e-5, ""),
    ("rd.a0fresh.auto_wall", "PAPER 5", 0.9326,
     ablation("A0_fresh", "auto", "wall", report=ABLF), 5e-5, ""),
    ("rd.a0fresh.plain_wall", "PAPER 5", 0.6320,
     ablation("A0_fresh", "plain", "wall", report=ABLF), 5e-5, ""),
    ("rd.a4.auto_wall", "PAPER 5", 0.9457,
     ablation("A4", "auto", "wall", report=ABLF), 5e-5, "readout adds ~0 -- it already has physics"),
    # A4 vs the ARCHIVED shipped run on the same cache: the reproducibility control
    ("rd.shipped.auto_wall", "PAPER 5", 0.9472,
     ablation("SHIPPED", "auto", "wall", report=ABLF), 5e-5,
     "dc_fem_cfw025 -- agrees with A4 to +0.0015"),
    ("rd.shipped.auto_off", "PAPER 5", 0.8291,
     ablation("SHIPPED", "auto", "off", report=ABLF), 5e-5, ""),

    # --- IS THE ODE NEEDED TEMPORALLY?  (the head's ODE onset channels held constant) ----
    # Live-ablation check, not a no-op: the committed masks changed on 26/27 vessels at the
    # wall and 17/20 off-wall, and the score did not move.  A substitution, not a broken flag.
    ("odet.with.wall", "PAPER 5.3", 0.9028,
     temporal("wall", rel=ODET_W), 5e-5, "mean-over-time, ODE timing available"),
    ("odet.no.wall", "PAPER 5.3", 0.9053,
     temporal("wall", rel=ODET_N), 5e-5, "ODE onset channels held constant"),
    ("odet.with.off", "PAPER 5.3", 0.7473,
     temporal("off", rel=ODET_W), 5e-5, ""),
    ("odet.no.off", "PAPER 5.3", 0.7460,
     temporal("off", rel=ODET_N), 5e-5, "no worse without the ODE clock"),
    ("odet.frozen_wall", "PAPER 5.3", 0.8565,
     temporal("frozen_wall", rel=ODET_W), 5e-5, "no temporal head at all"),
    ("odet.frozen_off", "PAPER 5.3", 0.6306,
     temporal("frozen_off", rel=ODET_W), 5e-5, ""),
    ("odet.oracle_wall", "PAPER 5.3", 0.9758,
     temporal("oracle_wall", rel=ODET_W), 5e-5, "perfect clock, same set -- the ceiling"),
    ("odet.oracle_off", "PAPER 5.3", 0.9103,
     temporal("oracle_off", rel=ODET_W), 5e-5, ""),

    # --- the DEPLOY-PATH wall clock, 2026-09-10 -----------------------------------------
    # Wall only.  The off-wall columns of every arm are bit-identical (off_on_sum moved on
    # 0/27 vessels even under `--owner-lag`): the tuner keeps selecting the single-stage
    # off-wall rule, which never reads the wall series.  There is no off-wall number here to
    # quote, and calling the off-wall equality a null would repeat exactly the mistake the
    # `--lag-anchor` pair made.
    ("wc.head.wall", "PAPER 5.3", 0.9028,
     temporal("wall", rel=WC_H), 5e-5, "learned temporal head as the wall clock"),
    ("wc.ode.wall", "PAPER 5.3", 0.9125,
     temporal("wall", rel=WC_O), 5e-5, "the DEPLOY mechanism, as it actually behaves"),
    ("wc.odecommit.wall", "PAPER 5.3", 0.9141,
     temporal("wall", rel=WC_C), 5e-5, "clock-only: same final set, ODE dates the onsets"),
    ("wc.head.wall_final", "PAPER 5.3", 0.9438,
     temporal("wall_final", rel=WC_H), 5e-5, ""),
    ("wc.ode.wall_final", "PAPER 5.3", 0.9230,
     temporal("wall_final", rel=WC_O), 5e-5,
     "the raw ODE clock leaves candidates uncommitted -- a SET change, not a re-timing"),
    ("wc.odecommit.wall_final", "PAPER 5.3", 0.9439,
     temporal("wall_final", rel=WC_C), 5e-5,
     "the same final set as `head` on 26/27 vessels (one moves by 4e-5), which is what makes the clock delta attributable"),

    # --- the DEPTH-MATCHED published-architecture control, 2026-09-10 -------------------
    # Attempt 1 at the shipped lr=3e-3 COLLAPSED to a constant field (std 1.2e-06, 0.1691
    # everywhere) and scored 0.1639 wall.  That number is not in this ledger and must never be
    # quoted: the arm did not train.  These rows are attempt 2 at lr=5e-4.
    ("mgn.wall", "PAPER 5", 0.6391,
     lambda: dig(_load(PVLM), "means", "wall", "naive"), 5e-5,
     "MeshGraphNet-STYLE depth-matched control, 15 layers, lr 5e-4"),
    ("mgn.off", "PAPER 5", 0.5174,
     lambda: dig(_load(PVLM), "means", "off", "naive"), 5e-5, ""),
    ("mgn.wall.full_minus_mgn", "PAPER 5", 0.3090,
     lambda: dig(_load(PVLM), "tests", "wall", "full_minus_naive", "delta"), 5e-5,
     "the shipped model's margin over a DEEP physics-free control"),
    ("mgn.off.full_minus_mgn", "PAPER 5", 0.3111,
     lambda: dig(_load(PVLM), "tests", "off", "full_minus_naive", "delta"), 5e-5, ""),
    ("mgn.wall.plain_minus_mgn", "PAPER 5", 0.1800,
     lambda: dig(_load(PVLM), "tests", "wall", "plain_minus_naive", "delta"), 5e-5,
     "what the physics-informed architecture buys over a deep generic one"),

    # --- THE SHIPPED ARCHITECTURE, read from the locked artifact (2026-09-13) ------------
    ("arch.params_per_member", "FIGURES 1.0", 151874,
     lambda: dig(_load(ARCH), "gnn", "params_per_member"), 0.5, "generate_architecture_table.py"),
    ("arch.members", "FIGURES 1.0", 9,
     lambda: dig(_load(ARCH), "ensemble", "members"), 0.5, ""),
    ("arch.params_total", "FIGURES 1.0", 1366866,
     lambda: dig(_load(ARCH), "ensemble", "params_total"), 0.5, ""),

    # --- AMDAHL: the prize for learning the flow is CAPPED before accuracy enters --------
    ("timing.rgp_flow_s", "PAPER 7", 0.667,
     lambda: dig(_load("outputs/publication/data/timing_rgp_deq.json"), "summary", "deq_s",
                 "median"), 5e-4, "RGP-DEQ flow stage, vs FEM's 4.66 s"),
    ("timing.fem_flow_s", "PAPER 7", 4.66,
     timing("summary", "fem_s", "median"), 5e-3, ""),
    ("timing.flow_ceiling", "PAPER 7", 0.079,
     timing_share("fem_s"), 5e-4,
     "an INSTANT flow surrogate saves at most this fraction of end-to-end"),

    # --- RGP-DEQ IS A GOOD MODEL.  That is the point of the negative result --------------
    # Quoted as MEDIANS over the 33-vessel diagnostic cohort, computed from the artifact
    # rather than transcribed.  A learned field an order of magnitude inside 2% rel-L2 that
    # still loses downstream is what makes the gated-coupling claim interesting; a bad
    # surrogate would prove nothing.
    ("rgpdeq.rel_l2_med", "PAPER 7.2", 0.0165, flowreq_pair("rel_l2"), 5e-4,
     "RGP-DEQ velocity rel-L2, median over 35 vessels"),
    ("rgpdeq.gate_jac_med", "PAPER 7.2", 0.7582, flowreq_pair("gate_jaccard"), 5e-4,
     "the statistic that actually predicts the downstream drop"),
    ("rgpdeq.dsrx_corr_med", "PAPER 7.2", 0.9936, flowreq_pair("dsrx_corr"), 5e-4, ""),
    ("rgpdeq.empty_gate", "PAPER 7.2", 0.0, flowreq_pair("empty_gate", how="sum"), 1e-9,
     "0 of 35 -- the cross-fit arm never empties a gate"),
    ("fem.rel_l2_med", "PAPER 7.2", 0.0064, femaudit_med("rel_l2_uv"), 5e-4, ""),
    ("fem.gate_jac_med", "PAPER 7.2", 0.9240, femaudit_med("gate_jac"), 5e-4,
     "median over the FEM audit cohort; the doc's 0.922 is the MEAN on a different cohort"),

    # --- the pre-flight gate detector, CURRENT arm --------------------------------------
    ("preflight.fem_n", "PAPER 6", 37, preflight("fem", "n"), 0, ""),
    ("preflight.fem_fail", "PAPER 6", 0, preflight("fem", "fail"), 0,
     "false alarms on the shipped path"),
    ("preflight.fem_warn", "PAPER 6", 0, preflight("fem", "warn"), 0, ""),
    ("preflight.pred_n", "PAPER 6", 37, preflight("pred", "n"), 0, ""),
    ("preflight.pred_fail", "PAPER 6", 0, preflight("pred", "fail"), 0,
     "the CURRENT cross-fit arm never empties a gate, so there is nothing to catch"),
    ("preflight.pred_warn", "PAPER 6", 1, preflight("pred", "warn"), 0, "comsol008"),

    # --- the split generation, measured but NOT promoted --------------------------------
    ("split.wall_delta", "DEPLOYCLOT / go_deployclot_split", 0.0185,
     paired("outputs/deployclot/split_vs_shipped.json", "wall", "delta"), 5e-5,
     "split vs shipped on wall: CI [-0.0002, +0.0419], p=0.054 -- NOT significant since the mirror-branch rule; pointer was never moved"),
    ("split.wall_p", "DEPLOYCLOT", 0.0535,
     paired("outputs/deployclot/split_vs_shipped.json", "wall", "p"), 5e-4, ""),
    ("split.off_delta", "DEPLOYCLOT", -0.0304,
     paired("outputs/deployclot/split_vs_shipped.json", "off", "delta"), 5e-5, "n.s."),
]


def _add_ranking_rows() -> None:
    """The THRESHOLD-FREE ablation numbers -- the statistic of record for the ladder.

    Separate from `_add_ablation_rows` on purpose: those are post-readout severity scores,
    these are average precision of the raw field.  The two answer different questions and the
    ladder was mis-read twice by treating one as the other, so they are never mixed in a row.
    """
    rep = "outputs/ablation/ablation_ranking.json"
    d = _load(rep)
    if not d:
        return
    for arm in d.get("arms", {}):
        for dom in ("wall", "off"):
            CLAIMS.append((
                f"rank.{arm}.{dom}", "PAPER 4.3",
                round(float(d["arms"][arm][dom]["ap_pooled"]), 4),
                (lambda a=arm, x=dom: dig(_load(rep), "arms", a, x, "ap_pooled")), 5e-5,
                f"pooled AP, cache={d.get('cache')}"))


def _add_ablation_rows() -> None:
    """The ablation ladder's own numbers, checked against whichever report exists.

    Added programmatically rather than hand-transcribed: these are new, they will move when
    round 2 lands, and a hand-copied constant here would be the exact failure this file is
    supposed to prevent.  The row asserts the REPORT is present and internally coherent (the
    reference arm scores what the arms table says it scores), not a frozen literal.
    """
    rep = "outputs/ablation/ablation_report.json"
    d = _load(rep)
    if not d:
        return
    ref = d.get("ref", "A4")
    for arm in d.get("arms", {}):
        for dom in ("wall", "off"):
            CLAIMS.append((
                f"abl.{arm}.{dom}", "PHYSICS_ABLATION_PLAN 7",
                round(float(d["arms"][arm]["auto"][dom]), 4),
                ablation(arm, "auto", dom, report=rep), 5e-5,
                f"ladder ref={ref}, cache={d.get('cache')}"))


#: `123.4 <id>` in a prose doc.  The tag names a ledger row; the number in front of it is what
#: the prose actually says.  Both are checked, because a doc can drift from the ledger just as
#: easily as the ledger can drift from the artifact -- and a tag that merely *looks* official is
#: worse than no tag at all.
TAG = re.compile(r"⟨([A-Za-z0-9_.]+)⟩")
#: Prose typography, not code: a manuscript writes U+2212 MINUS, thousands separators, and
#: percentages.  All three are read here rather than banned from the doc, because forcing the
#: doc to be machine-shaped is how a verifier stops being used.
NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?\s*%?")


def check_paper(path: Path, ledger: dict) -> tuple[int, int]:
    """Verify every tagged number in a prose doc against the ledger.  -> (checked, bad)."""
    if not path.is_file():
        print("\n[paper] %s not found" % path)
        return 0, 1
    text = path.read_text(encoding="utf-8")
    checked = bad = 0
    print("\n--- %s ---" % path.name)
    for m in TAG.finditer(text):
        cid = m.group(1)
        before = text[max(0, m.start() - 90):m.start()]
        nums = NUM.findall(before)
        row = ledger.get(cid)
        if row is None:
            print("  UNKNOWN TAG  %-28s  no such claim id" % cid)
            bad += 1
            continue
        if not nums:
            print("  NO NUMBER    %-28s  tag has no value in front of it" % cid)
            bad += 1
            continue
        tok = nums[-1].strip()
        pct = tok.endswith("%")
        tok = tok.rstrip("% ").replace(",", "").replace("−", "-")
        try:
            said = float(tok) / (100.0 if pct else 1.0)
        except ValueError:
            print("  UNPARSED     %-28s  %r" % (cid, nums[-1]))
            bad += 1
            continue
        checked += 1
        if abs(said - row["stated"]) > max(row["tol"], 1e-9):
            print("  MISMATCH     %-28s  doc says %s, ledger says %s"
                  % (cid, said, row["stated"]))
            bad += 1
    print("  %d tagged numbers checked, %d bad" % (checked, bad))
    return checked, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/ablation/claims_ledger.json")
    ap.add_argument("--only", default="", help="substring filter on claim id")
    ap.add_argument("--paper", default="",
                    help="also verify every <id>-tagged number in this markdown doc")
    args = ap.parse_args()

    _add_ablation_rows()
    _add_ranking_rows()
    rows = [c for c in CLAIMS if args.only in c[0]]

    out, n_fail, n_stale = [], 0, 0
    width = max(len(c[0]) for c in rows) if rows else 10
    print("\n%-*s  %10s  %10s  %s" % (width, "claim", "doc says", "artifact", "status"))
    print("-" * (width + 44))
    for cid, doc, stated, resolve, tol, note in rows:
        try:
            got = resolve()
        except Exception as exc:                      # a broken resolver is a STALE row,
            got, note = None, f"resolver error: {exc}"  # never a silent pass
        if got is None:
            status, n_stale = "STALE", n_stale + 1
            shown = "--"
        else:
            ok = abs(float(got) - float(stated)) <= float(tol)
            status = "PASS" if ok else "FAIL"
            n_fail += 0 if ok else 1
            shown = f"{float(got):.4f}"
        print("%-*s  %10s  %10s  %-5s %s"
              % (width, cid, f"{float(stated):.4f}", shown, status,
                 (" " + note) if (note and status != "PASS") else ""))
        out.append(dict(id=cid, doc=doc, stated=float(stated),
                        artifact=None if got is None else float(got),
                        tol=float(tol), status=status, note=note))

    n_ok = len(rows) - n_fail - n_stale
    print("\n%d/%d verified   %d FAIL   %d STALE" % (n_ok, len(rows), n_fail, n_stale))
    if n_fail:
        print("A FAIL means the doc and the artifact disagree. Decide which is right and fix "
              "THAT one;\nnever widen a tolerance to clear a row.")
    p = REPO / args.json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote %s" % p)

    n_bad_paper = 0
    if args.paper:
        _, n_bad_paper = check_paper(REPO / args.paper, {r["id"]: r for r in out})
    return n_fail + n_stale + n_bad_paper


if __name__ == "__main__":
    raise SystemExit(main())
