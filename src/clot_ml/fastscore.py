"""A fast, exactly-equivalent copy of the domain-restricted deploy score.

The reference path (`compute_clot_relaxed_metrics`) re-does a 2-hop graph dilation on a
15k-node mesh for every call.  Threshold selection needs O(10^4) calls per arm, so the
dilation operator is precomputed once per vessel and the score becomes two sparse matvecs.

`assert_matches_reference` pins it against the real implementation; run it in the tests.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

RELAX_HOPS = 2
F_BETA = 0.5
IOU_W = F05_W = 0.5
EMPTY_GT_FP_TOL = 8.0


def _dilator(ei: np.ndarray, n: int, hops: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(ei.shape[1], np.int8), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    D = (A + sp.eye(n, format="csr", dtype=np.int8)).astype(np.int8)
    out = D
    for _ in range(hops - 1):
        out = ((out @ D) > 0).astype(np.int8)
    return out


from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer


def active_severity_config():
    """The `SeverityConfig` matching the ACTIVE deploy score mode.

    `LEGACY` reproduces `clot_guiding` exactly (verified numerically), so following the
    same switch `clot_score_from_deploy_dict` reads keeps this fast path and the reference
    path on the same metric.  Without this the two silently diverge the moment Deploy Score
    v2 becomes the default, which is what `assert_matches_reference` exists to catch.
    """
    from src.evaluation.clot_relaxed_metrics import species_continuous_clout_score_mode

    return DEFAULT if species_continuous_clout_score_mode() == "severity" else LEGACY


class VesselScorer:
    """Precomputed scorer for one vessel.  ``score(pred, domain)`` matches the reference."""

    def __init__(self, ei: np.ndarray, gt: np.ndarray, n: int, cfg=None):
        self.scorer = SeverityScorer(ei, gt, n, cfg or active_severity_config())

    def score(self, pred: np.ndarray, domain: np.ndarray | None = None) -> float:
        g = self.scorer.gt & domain if domain is not None else self.scorer.gt
        if int(g.sum()) == 0:
            return float("nan")
        return self.scorer.score(pred, domain)


def assert_matches_reference(ei, gt, pred, wall, n, *, atol=1e-9):
    import torch

    from src.clot_ml.evaluate import domain_score

    vs = VesselScorer(ei, gt, n)
    for domain in (wall, ~wall):
        ref = domain_score(pred, gt, torch.tensor(ei), domain, wall)
        got = vs.score(pred, domain)
        if ref != ref and got != got:
            continue
        assert abs(ref - got) < atol, (ref, got)
    return True
