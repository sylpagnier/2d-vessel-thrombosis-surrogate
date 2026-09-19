"""The 8 empty-GT vessels: they must reach the model, and be scored on their own terms.

The 2026-08-22 cohort decision (docs/SEALED_SPLIT.md, MODEL_REVIEW_2026-08-22 8b) admitted
`wall_cohort_splits.CLOT_FREE` to training and to scoring.  Half of it landed:
`SeverityScorer.score(..., empty_gt="score")` existed and **nothing could call it**, because
`build_clot_ml_cache.py` iterated `FIT + DEV` and then dropped any vessel whose GT was empty.

Two rules this file pins, and they are easy to conflate:

* an empty **DOMAIN** on a clot-CARRYING vessel -- 6 of 19 have no off-wall GT -- stays
  ``nan`` and drops out of the mean, or every off-wall number in the project changes meaning;
* an empty **VESSEL** grades false positives, ``1/(1 + n_pred/8)``, and is never averaged into
  a recall-bearing mean.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.clot_ml.severity_metric import DEFAULT, SeverityScorer, dilation_operator
from src.clot_ml.softmetric import EMPTY_GT_FP_TOL, soft_empty_gt_score
from src.core_physics.wall_cohort_splits import CLOT_FREE


# --------------------------------------------------------------------- the soft branch
def test_soft_empty_gt_score_is_one_at_zero_and_decreases():
    n = 20
    dom = torch.ones(n)
    assert float(soft_empty_gt_score(torch.zeros(n), dom)) == pytest.approx(1.0)
    p = torch.zeros(n)
    p[:4] = 1.0
    mid = float(soft_empty_gt_score(p, dom))
    p[:16] = 1.0
    lots = float(soft_empty_gt_score(p, dom))
    assert 1.0 > mid > lots > 0.0
    # exactly the hard metric's branch at a hard mask
    assert mid == pytest.approx(1.0 / (1.0 + 4 / EMPTY_GT_FP_TOL))
    assert lots == pytest.approx(1.0 / (1.0 + 16 / EMPTY_GT_FP_TOL))


def test_soft_empty_gt_score_is_differentiable_and_pushes_predictions_down():
    p = torch.full((10,), 0.5, requires_grad=True)
    (1.0 - soft_empty_gt_score(p, torch.ones(10))).backward()
    assert p.grad is not None
    assert torch.all(p.grad > 0), "minimising 1-score must push every probability DOWN"


def test_soft_empty_gt_score_respects_the_domain():
    p = torch.ones(10)
    dom = torch.zeros(10)
    dom[:2] = 1.0
    assert float(soft_empty_gt_score(p, dom)) == pytest.approx(
        1.0 / (1.0 + 2 / EMPTY_GT_FP_TOL))


@pytest.mark.parametrize("which", ["soft_score", "soft_severity"])
def test_empty_domain_defaults_to_dropping_the_term(which):
    """The default must stay `None` -- 6 clot-carrying vessels rely on it."""
    from src.clot_ml.severity_metric import soft_severity
    from src.clot_ml.softmetric import soft_score, to_torch_sparse

    n = 12
    ei = np.array([[i, i + 1] for i in range(n - 1)]).T
    D = to_torch_sparse(dilation_operator(ei, n, 2), torch.device("cpu"))
    p = torch.full((n,), 0.3)
    gt = torch.zeros(n)
    dom = torch.ones(n)
    fn = soft_score if which == "soft_score" else soft_severity
    assert fn(p, gt, D, dom, torch.zeros(n)) is None
    got = fn(p, gt, D, dom, torch.zeros(n), empty_gt="score")
    assert got is not None and 0.0 < float(got) < 1.0
    with pytest.raises(ValueError, match="empty_gt"):
        fn(p, gt, D, dom, torch.zeros(n), empty_gt="nonsense")


# --------------------------------------------------------------------- the hard branch
def test_scorer_separates_an_empty_domain_from_an_empty_vessel():
    n = 12
    ei = np.array([[i, i + 1] for i in range(n - 1)]).T
    gt = np.zeros(n, dtype=bool)
    gt[:3] = True                       # clot-carrying vessel
    wall = np.zeros(n, dtype=bool)
    wall[:6] = True                     # ... with all its GT on the wall
    vs = SeverityScorer(ei, gt, n, DEFAULT)
    pred = np.zeros(n, dtype=bool)

    off = vs.score(pred, ~wall)                       # empty off-wall DOMAIN
    assert off != off, "an empty domain on a carrying vessel must stay nan"
    assert vs.score(pred, ~wall, empty_gt="score") == pytest.approx(1.0)

    free = SeverityScorer(ei, np.zeros(n, dtype=bool), n, DEFAULT)
    assert free.score(pred, None) != free.score(pred, None)   # nan by default
    assert free.score(pred, None, empty_gt="score") == pytest.approx(1.0)
    p2 = np.zeros(n, dtype=bool)
    p2[:8] = True
    assert free.score(p2, None, empty_gt="score") == pytest.approx(1.0 / 2.0)


def test_bound_scorer_picks_the_convention_from_the_vessel():
    from scripts.eval_strict import BoundScorer  # noqa: PLC0415

    n = 10
    ei = np.array([[i, i + 1] for i in range(n - 1)]).T
    empty = np.zeros(n, dtype=bool)
    free = BoundScorer(ei, empty, n, DEFAULT, "score")
    carrying_empty_dom = BoundScorer(ei, empty, n, DEFAULT, "nan")
    assert free.score(empty) == pytest.approx(1.0)
    assert carrying_empty_dom.score(empty) != carrying_empty_dom.score(empty)


# --------------------------------------------------------------------- the data path
def test_the_cache_builder_no_longer_drops_the_clot_free_vessels():
    """The exact bug: `FIT + DEV` never listed them, and `y.sum() == 0` dropped them."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2]
           / "scripts/build_clot_ml_cache.py").read_text(encoding="utf-8")
    assert "CLOT_FREE" in src, "the clot-free vessels are not in the build list"
    assert re.search(r"todo\s*=\s*list\(FIT\)\s*\+\s*list\(DEV\)\s*\+\s*list\(CLOT_FREE\)",
                     src), "the build list must include CLOT_FREE"
    assert 'S["y"].sum() == 0 and a not in CLOT_FREE' in src, (
        "the empty-GT drop must exempt the clot-free list, or they are cached and then "
        "thrown away")


def test_eligible_pool_stays_clot_carrying():
    """Every caller of `eligible_pool` averages a RECALL-bearing score."""
    from src.clot_ml.geometry_splits import eligible_pool

    assert not (set(eligible_pool()) & set(CLOT_FREE))


def test_prepare_flags_a_clot_free_vessel_from_its_labels():
    """`train_one` selects the metric branch off this flag, not off a cohort import."""
    from scripts.train_clot_gnn import prepare  # noqa: PLC0415

    n = 40
    ei = np.array([[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]).T
    wall = np.zeros(n, dtype=bool)
    wall[:10] = True

    def sample(gt_nodes):
        y = np.zeros(n, np.float32)
        y[gt_nodes] = 1.0
        return dict(X=np.zeros((n, 3), np.float32), y=y,
                    mat_gt=np.zeros(n, np.float32), wall=wall, owner=np.zeros(n, np.int64),
                    edge_index=ei, pos=np.stack([np.arange(n), np.zeros(n)], 1).astype(
                        np.float32),
                    u=np.ones(n, np.float32), v=np.zeros(n, np.float32),
                    mat_phys=np.zeros(n, np.float32), gate=np.zeros(n, np.float32),
                    sr=np.ones(n, np.float32), spd=np.ones(n, np.float32),
                    phys_mask=np.zeros(n, bool))

    cache = {"free": sample([]), "carrying": sample([1, 2, 3])}
    mu = np.zeros(3, np.float32)
    sd = np.ones(3, np.float32)
    G = prepare(cache, ["free", "carrying"], mu, sd, torch.device("cpu"), need_soft=True)
    assert G["free"]["empty_gt"] is True
    assert G["carrying"]["empty_gt"] is False


def test_the_empty_gt_metric_term_is_off_by_default():
    """It has NO measured effect (MODEL_REVIEW 8f.4); "none" is the default on parsimony.

    Paired, per configuration: `v5a` off +0.0694 [-0.043, +0.186] P(<=0)=0.116, `v5b` off
    -0.0150 [-0.089, +0.051] P(<=0)=0.651.  Both intervals cross zero and the sign flips, so
    it is not distinguishable from noise on this cohort -- consistent with 8c, where the
    clot-free vessels sit three orders of magnitude from any decision boundary.

    This test pins the DEFAULT, not a claim of harm.  Turning it on is a legitimate
    experiment; turning it on silently, so that promotion trains on a different objective
    from the CV that selected the design, is not.
    """
    from scripts.run_phase9_cv import BASE

    assert BASE["empty_gt_loss"] == "none", (
        "the clot-free metric term is on by default again -- it has no measured benefit, and "
        "at +/-0.074 off-wall this cohort cannot resolve a global-bias term at all")

    # `promote_clot_gnn_v4.py` lives at `scripts/`, not under the era archive -- the
    # 2026-09-02 script reorg left this import pointing at an empty directory and the
    # test has been failing on the ModuleNotFoundError ever since, which is not the
    # thing it is meant to catch.
    from scripts.promote_clot_gnn_v4 import BASE as PBASE

    assert PBASE["empty_gt_loss"] == BASE["empty_gt_loss"], (
        "promotion trains with a different objective from the CV that selected the design")


def test_promotion_will_not_reuse_a_member_from_a_different_generation():
    """`os.path.exists` is not a resume check.  MODEL_REVIEW 8f.6.

    `--name` defaults to `clot_gnn_v4`, so a re-promotion after the 2026-08-22 pack repair
    found the PREVIOUS generation's nine member files, "kept" all of them, and rewrote the
    manifest -- new strict-CV scores wrapped around superseded weights, silently.  `in_dim`
    does not catch it: the channel count did not change.
    """
    import inspect

    from scripts import promote_clot_gnn_v4 as P   # lives at scripts/, not the era archive

    src = inspect.getsource(P.main)
    assert "_fingerprint" in src, "the resume path has no fingerprint check"
    assert 'get("fingerprint")' in src, (
        "an existing member must be VALIDATED, not merely detected")
    # everything a member's weights depend on must be in the hash
    fp = inspect.getsource(P.main)
    fp = fp[fp.index("def _fingerprint"):fp.index("members, t0")]
    for must in ("pool", "cols", "cfg.items()", "mu", "sd"):
        assert must in fp, f"fingerprint ignores {must}: a change to it would go unnoticed"


def test_promotion_and_cv_agree_on_the_c0_constraint():
    """`shape_w` is a TRAINING term, so the CV that measures it and the weights that ship it
    must use the same value -- exactly the `empty_gt_loss` rule, for the same reason.

    Its default is 0 (off) in both, because the shipped v5a/v5b/v5c tags predate C0.  The
    2026-08-23 artifact is promoted with `--shape-w 2.0` explicitly and records it in the
    manifest, so the value that produced the weights always travels with them
    (MODEL_REVIEW 9b).
    """
    # `promote_clot_gnn_v4.py` lives at `scripts/`, not under the era archive -- the
    # 2026-09-02 script reorg left this import pointing at an empty directory and the
    # test has been failing on the ModuleNotFoundError ever since, which is not the
    # thing it is meant to catch.
    from scripts.promote_clot_gnn_v4 import BASE as PBASE
    from scripts.run_phase9_cv import BASE

    assert "shape_w" in BASE and "shape_w" in PBASE, (
        "the C0 constraint must be a declared config key in both, or promotion cannot record "
        "which value trained the weights")
    assert PBASE["shape_w"] == BASE["shape_w"], (
        "promotion defaults to a different C0 weight from the CV driver")
