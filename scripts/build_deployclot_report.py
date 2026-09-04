"""Assemble the DeployClot validation report as a standalone HTML page.

Reads whatever is on disk under ``outputs/deployclot`` and ``outputs/logs/deployclot`` and
renders it.  Every section degrades to a short "not measured in this run" note rather than
failing, so the page is publishable at any point in the pipeline.

    python scripts/build_deployclot_report.py --out outputs/deployclot/report.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.wall_cohort_constants import WALL_COHORT_V2_CLOT_FREE

OUT = REPO / "outputs/deployclot"
LOGS = REPO / "outputs/logs/deployclot"
# Cohort membership has ONE home; a second copy here silently diverged the report
# from what the evaluators score.
CLOT_FREE_STEMS = set(WALL_COHORT_V2_CLOT_FREE)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _text(p: Path) -> str:
    """Read a log as text with newlines normalised.

    The launchers redirect through PowerShell, so every log on this machine is CRLF.  The
    section parsers match on a blank line, which never occurs as "\n\n" in CRLF text --
    normalising here rather than in each regex is what stops a section silently rendering as
    absent, which it did on the first build of this page.
    """
    try:
        raw = p.read_bytes().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def _fmt(x, nd=4, dash="--"):
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return html.escape(str(x))
    if v != v:
        return dash
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------
def solver_section(rows) -> str:
    if not rows:
        return '<p class="note">The flow audit was not run.</p>'
    def agg(k):
        v = np.array([r[k] for r in rows if r.get(k) == r.get(k)], dtype=float)
        return np.median(v), np.percentile(v, 10), np.percentile(v, 90)
    metrics = [("rel_l2_uv", "velocity rel L2", 4),
               ("sr_corr_wall", "wall shear-rate correlation", 4),
               ("sr_ratio_wall", "wall shear-rate median ratio", 4),
               ("dsrx_corr_wall", "wall d(sr)/dx correlation", 4),
               ("gate_jac", "deposition-gate union Jaccard", 4),
               ("gate_fire_ratio", "gate fire-rate ratio", 4)]
    trs = []
    for k, label, nd in metrics:
        m, lo, hi = agg(k)
        trs.append(f"<tr><td>{label}</td><td class='n em'>{_fmt(m, nd)}</td>"
                   f"<td class='n'>{_fmt(lo, nd)}</td><td class='n'>{_fmt(hi, nd)}</td></tr>")
    solve = np.array([r["solve_s"] for r in rows], dtype=float)

    # strip plot of per-vessel rel L2, log scale
    vals = sorted(((r["rel_l2_uv"], r["stem"]) for r in rows), key=lambda t: t[0])
    lo, hi = 0.002, 1.0
    def xpos(v):
        v = min(max(v, lo), hi)
        return 100.0 * (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    ticks = []
    for v, stem in vals:
        cls = "tick out" if v > 0.1 else "tick"
        ticks.append(f'<span class="{cls}" style="left:{xpos(v):.3f}%" '
                     f'title="{html.escape(stem)}: {v:.4f}"></span>')
    axis = "".join(
        f'<span class="ax" style="left:{xpos(g):.3f}%"><i></i><b>{g:g}</b></span>'
        for g in (0.002, 0.01, 0.1, 1.0))
    strip = (f'<figure class="strip"><figcaption>Per-vessel velocity rel L2 against COMSOL, '
             f'log axis, one tick per vessel (n={len(rows)}). The two ticks past 0.1 are '
             f'<code>comsol045</code> and <code>comsol046</code>.</figcaption>'
             f'<div class="stripbody">{"".join(ticks)}<div class="axis">{axis}</div></div>'
             f'</figure>')
    return f"""
{strip}
<div class="tablewrap"><table>
<thead><tr><th>quantity</th><th class="n">median</th><th class="n">p10</th><th class="n">p90</th></tr></thead>
<tbody>{''.join(trs)}</tbody>
</table></div>
<p>Median solve time <strong>{np.median(solve):.1f} s</strong> per vessel
(max {solve.max():.1f} s; {solve.sum():.0f} s for the whole {len(rows)}-vessel corpus, one CPU).</p>
"""


def onset_section(rows) -> str:
    if not rows:
        return '<p class="note">The wound-onset check was not run.</p>'
    trs = "".join(
        f"<tr><td><code>{html.escape(r['stem'])}</code></td>"
        f"<td class='n'>{r['r150']:.2f}</td><td class='n'>{r['fit150']:.2f}</td>"
        f"<td class='n em'>{r['ratio']:.3f}</td><td class='n dim'>{r['delay100_pred']:.2f}</td></tr>"
        for r in rows)
    return f"""
<div class="tablewrap"><table>
<thead><tr><th>vessel</th><th class="n">ratio at t=150 s</th><th class="n">trend to t=150 s</th>
<th class="n">observed / trend</th><th class="n">a 100 s delay predicts</th></tr></thead>
<tbody>{trs}</tbody></table></div>
"""


def cv_section(readout_log: str, tag: str) -> str:
    """Parse the strictly-nested arm table `eval_expected_score_readout.py` prints."""
    if not readout_log:
        return ""
    m = re.search(r"FINAL TIME POINT, strictly nested \(tags=.*?\)\n\n(.*?)(?:\n\n|\Z)",
                  readout_log, re.S)
    if not m:
        return ""
    lines = [ln for ln in m.group(1).splitlines() if "|" in ln]
    if not lines:
        return ""
    trs = []
    for ln in lines[1:]:
        parts = [p.strip() for p in re.split(r"\|", ln)]
        name = parts[0]
        nums = parts[1].split() if len(parts) > 1 else []
        if len(nums) < 2:
            continue
        cls = " class='row-hi'" if name in ("expected_tuned", "resid_adapt", "nested_pick") else ""
        trs.append(f"<tr{cls}><td><code>{html.escape(name)}</code></td>"
                   f"<td class='n'>{nums[0]}</td><td class='n'>{nums[1]}</td></tr>")
    return f"""
<div class="tablewrap"><table>
<thead><tr><th>readout arm</th><th class="n">wall</th><th class="n">off-wall</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<p class="note">Strictly nested: every readout scalar is selected on the out-of-fold scores of
vessels outside the held-out fold, so no quantity used to produce a vessel's number was
fitted with that vessel visible &mdash; neither the weights nor the readout.
Tag <code>{html.escape(tag)}</code>.</p>
"""


def _gap_block(gap) -> str:
    """Is the SEALED off-wall shortfall a readout problem or a representation one?"""
    if not gap:
        return ""
    rows = "".join(
        f"<tr><td><code>{html.escape(r['stem'])}</code></td>"
        f"<td class='n'>{r['n_off_gt']}</td>"
        f"<td class='n'>{_fmt(r.get('off_oracle'), 4)}</td>"
        f"<td class='n'>{_fmt(r.get('off_auc'), 4)}</td>"
        f"<td class='n dim'>{_fmt(r.get('off_oracle_cut'), 2)}</td></tr>"
        for r in gap["per_vessel"] if r.get("n_off_gt"))
    return f"""
<h3>The shortfall is the cut, not the model</h3>
<p>Diagnosed from the read already taken &mdash; re-measuring predictions already made is not
a second read, and the oracle cut below is never fed back into the artifact.</p>
<div class="tablewrap"><table>
<thead><tr><th>vessel</th><th class="n">off-wall GT nodes</th>
<th class="n">per-vessel oracle</th><th class="n">ranking AUC</th>
<th class="n">its best cut</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="kpis">
  <div class="kpi"><div class="v">{gap['auc_mean']:.4f}</div><div class="l">off-wall ranking
  AUC on vessels never seen &mdash; the representation transfers</div></div>
  <div class="kpi"><div class="v">{gap['oracle_mean']:.4f}</div><div class="l">what that
  ranking admits, <em>above</em> the cross-validated 0.8351</div></div>
  <div class="kpi crim"><div class="v">+{gap['readout_gap']:.4f}</div><div class="l">held
  behind cut placement, against +0.045 in-cohort</div></div>
</div>
<p>The best cut differs per vessel &mdash; 0.96, 0.58, 0.62 &mdash; and each vessel's GT-node
score median sits far above its own whole-domain p99. The field's <em>scale</em> is
per-vessel. That is exactly what the C0 spread constraint was built to fix, and the finding is
that <strong>C0 closes it on the cohort it trains on and does not transfer</strong>. So 0.618
is not a ceiling: the ranking already supports 0.889, and the next build is a label-free
per-vessel cut judged on transfer rather than in-cohort fit.</p>
"""


def flow_paired_section(fp) -> str:
    if not fp or not fp.get("paired"):
        return ""
    trs = []
    for key, label in (("wall", "wall"), ("off", "off-wall (true lumen)")):
        r = fp["paired"].get(key)
        if not r:
            continue
        trs.append(f"<tr><td>{label}</td><td class='n'>{_fmt(r['mean_a'], 4)}</td>"
                   f"<td class='n'>{_fmt(r['mean_b'], 4)}</td>"
                   f"<td class='n em'>{r['delta']:+.4f}</td>"
                   f"<td class='n'>[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]</td>"
                   f"<td class='n'>{_fmt(r['p'], 3)}</td><td class='n dim'>{r['n']}</td></tr>")
    if not trs:
        return ""
    return f"""
<div class="tablewrap"><table>
<thead><tr><th>domain</th><th class="n">COMSOL flow</th><th class="n">solved flow</th>
<th class="n">difference</th><th class="n">95% CI</th><th class="n">P</th>
<th class="n">n</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<p class="note">Paired over vessels, bootstrap CI, both arms strictly nested and read on the
same labels. The two arms differ in one thing only &mdash; which t=0 velocity field the
features are built from.</p>
"""


def wound_section(lovo) -> str:
    if not lovo:
        return '<p class="note">The wound-rate fit was not run.</p>'
    s = lovo.get("summary") or {}
    folds = lovo.get("folds") or {}
    order = ["physics", "const", "net"]
    label = {"physics": "physics (G = 1, zero parameters)",
             "const": "two constants (G_pre, G_post)",
             "net": "+ per-node WoundRateNet"}
    trs = []
    for arm in order:
        r = s.get(arm)
        if not r:
            continue
        cls = " class='row-hi'" if arm == "const" else ""
        trs.append(f"<tr{cls}><td>{label[arm]}</td><td class='n'>{_fmt(r['curve_l1'], 3)}</td>"
                   f"<td class='n'>{_fmt(r['onset_mae'], 1)}</td>"
                   f"<td class='n'>{100 * r['onset_mae_frac']:.1f}%</td>"
                   f"<td class='n'>{_fmt(r['recall'], 3)}</td></tr>")
    fa = lovo.get("fitted_all") or {}
    gp = [f"{html.escape(k)} &rarr; G_pre {v['g_pre']:.2f}, G_post {v['g_post']:.2f}"
          for k, v in sorted(folds.items())]
    return f"""
<div class="tablewrap"><table>
<thead><tr><th>arm</th><th class="n">curve L1</th><th class="n">onset MAE (steps)</th>
<th class="n">% of horizon</th><th class="n">recall</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<p>Refit on all {lovo.get('n_vessels', '?')} wound vessels:
<strong>G_pre {fa.get('g_pre', float('nan')):.2f}, G_post {fa.get('g_post', float('nan')):.2f}</strong>
(flow <code>{html.escape(str(lovo.get('flow', 'gt')))}</code>).</p>
<details><summary>Per-fold constants</summary><ul class="folds">
{''.join(f'<li>{g}</li>' for g in gp)}</ul></details>
"""


def ab_section(ab) -> str:
    if not ab:
        return '<p class="note">The A/B counterfactual was not run.</p>'
    p, b, c = ab["pair"], ab["burden"], ab["created"]
    reg = p["registration"]
    rows = [("all shared nodes", "all"),
            ("on the wound patch (guaranteed)", "on_wound"),
            ("healthy wall it recruited", "on_solid_off_wound"),
            ("lumen it recruited", "in_lumen")]
    trs = []
    for label, k in rows:
        a = c.get(k) or {}
        cls = " class='row-hi'" if k == "all" else ""
        trs.append(f"<tr{cls}><td>{label}</td><td class='n'>{a.get('n_gt', 0)}</td>"
                   f"<td class='n'>{a.get('n_pred', 0)}</td>"
                   f"<td class='n'>{_fmt(a.get('precision'), 3)}</td>"
                   f"<td class='n'>{_fmt(a.get('recall'), 3)}</td>"
                   f"<td class='n em'>{_fmt(a.get('f1'), 3)}</td>"
                   f"<td class='n'>{_fmt(a.get('iou'), 3)}</td></tr>")
    pv = ab["per_vessel"]
    return f"""
<div class="pairhead">
  <div><span class="k">wound half</span><code>{html.escape(p['wound'])}</code></div>
  <div><span class="k">no-wound half</span><code>{html.escape(p['nowound'])}</code></div>
  <div><span class="k">matched horizon</span>{p['horizon_s']:.0f} s</div>
  <div><span class="k">registration, worst node</span>{100 * reg['frac_of_span']:.2f}% of span</div>
</div>
<div class="tablewrap"><table>
<thead><tr><th>burden at the matched horizon</th><th class="n">wound</th>
<th class="n">no wound</th><th class="n">difference</th></tr></thead>
<tbody>
<tr><td>ground truth</td><td class="n">{b['gt']['wound']}</td>
    <td class="n">{b['gt']['nowound']}</td><td class="n em">{b['gt']['delta']:+d}</td></tr>
<tr><td>DeployClot</td><td class="n">{b['pred']['wound']}</td>
    <td class="n">{b['pred']['nowound']}</td><td class="n em">{b['pred']['delta']:+d}</td></tr>
</tbody></table></div>
<h4>Recovering the clot the injury created</h4>
<p>Read the last two rows, not the second. Every node of the wound patch clots in the ground
truth &mdash; that is the physics, and any model that commits the mask scores 1.0 there. What
the injury <em>recruits beyond its own boundary</em> is the part a model can get wrong.</p>
<div class="tablewrap"><table>
<thead><tr><th>region</th><th class="n">GT nodes</th><th class="n">predicted</th>
<th class="n">precision</th><th class="n">recall</th><th class="n">F1</th><th class="n">IoU</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<p class="note">Per-vessel severity at the same horizon &mdash;
wound {_fmt(pv['wound']['wall'], 3)} wall / {_fmt(pv['wound']['off'], 3)} off-wall,
no-wound {_fmt(pv['nowound']['wall'], 3)} wall / {_fmt(pv['nowound']['off'], 3)} off-wall.</p>
"""


def _armmean(rows, arm, dom):
    v = [r[arm][dom] for r in rows.values() if r[arm][dom] == r[arm][dom]]
    return float(np.mean(v)) if v else float("nan")


def _arms_metric(arms, metric="guiding"):
    """One metric's table out of the dual-metric arms file, with a legacy fallback."""
    if not arms:
        return {}
    return arms.get(metric, arms if "resid" in arms else {})


def _am(arms, arm, dom, metric="guiding"):
    D = _arms_metric(arms, metric).get(arm, {})
    v = [D[a][dom] for a in D if D[a].get(dom) is not None and D[a][dom] == D[a][dom]]
    return float(np.mean(v)) if v else float("nan")


def offwall_section(att, v6a, v6b) -> str:
    """The wound off-wall readout: a physics fix that wins, a learned field that does not."""
    pv = att["per_vessel"]
    stems = list(pv)
    SHIP, FIX = "0.23/0/3", "0.23/0.5/1"
    head = "".join("<th class='n'>%s</th>" % html.escape(s.replace("wound_comsol", "w"))
                   for s in stems)
    depth_rows = "".join(
        "<tr><td class='n'>d = %d</td>%s<td class='n em'>%.4f</td></tr>" % (
            d,
            "".join("<td class='n'>%.4f</td>" % pv[s]["0.23/0/%d" % d]["w_lum"] for s in stems),
            _armmean(pv, "0.23/0/%d" % d, "w_lum"))
        for d in (1, 2, 3, 4, 5))
    lovo_rows = "".join(
        "<tr><td><code>%s</code></td><td class='n'>%.4f</td><td class='n em'>%.4f</td>"
        "<td class='n'>%+.4f</td><td class='n'>%.4f</td><td class='n em'>%.4f</td></tr>" % (
            html.escape(s.replace("wound_comsol", "wound")),
            pv[s][SHIP]["w_lum"], pv[s][FIX]["w_lum"],
            pv[s][FIX]["w_lum"] - pv[s][SHIP]["w_lum"],
            pv[s][SHIP]["w_reg"], pv[s][FIX]["w_reg"])
        for s in stems)

    v6 = ""
    if v6a and v6b:
        a, b = v6a["per_vessel"], v6b["per_vessel"]
        ka, kb = "chem/wound_region", "v6/wound_region"
        v6 = (
            "<h3>And the learned field, which the fix makes redundant</h3>"
            "<p>A <code>ClotGNN</code> whose regression head is a zero-init residual on the "
            "physics field &mdash; so an untrained one <em>is</em> the physics &mdash; trained "
            "leave-one-vessel-out on 49 vessels and swapped in through the same readout, "
            "changing nothing else.</p>"
            "<div class='tablewrap'><table><thead><tr><th>readout</th>"
            "<th class='n'>chemistry</th><th class='n'>learned field</th>"
            "<th class='n'>difference</th></tr></thead><tbody>"
            "<tr><td>shipped <code>att 0.23, depth 3</code></td>"
            "<td class='n'>%.4f</td><td class='n em'>%.4f</td><td class='n'>%+.4f</td></tr>"
            "<tr><td>corrected <code>beta 0.5, depth 1</code></td>"
            "<td class='n em'>%.4f</td><td class='n'>%.4f</td><td class='n'>%+.4f</td></tr>"
            "</tbody></table></div>"
            "<p>The learned field wins at the old readout and <strong>loses at the corrected "
            "one</strong>, where it is identical to the chemistry on five of six vessels. Its "
            "whole gain was <code>wound_comsol005</code>, and the way it won there was to make "
            "the field <em>smaller</em> so shell 2 stopped firing &mdash; the depth fix, "
            "expressed as a magnitude. A learned residual was an expensive way to re-tune a "
            "threshold.</p>"
            "<div class='caveat'><h4>The 2026-08 headline does not survive, and that is not a "
            "contradiction</h4><p>The learned field once took a held-out wound vessel from "
            "0.4755 to 0.9489. That was measured against a readout replaced two days later; "
            "today's chemistry scores <strong>0.9578</strong> on that same vessel, above the "
            "learned field's old number. The advantage was real, and the readout fix consumed "
            "it. Compare a new arm against the current baseline, not the one it beat.</p></div>"
        ) % (_armmean(a, ka, "w_lum"), _armmean(a, kb, "w_lum"),
             _armmean(a, kb, "w_lum") - _armmean(a, ka, "w_lum"),
             _armmean(b, ka, "w_lum"), _armmean(b, kb, "w_lum"),
             _armmean(b, kb, "w_lum") - _armmean(b, ka, "w_lum"))

    return (
        "<p>The wound off-wall readout commits a lumen node in shell <em>d</em> when "
        "<code>att<sup>d</sup> &middot; Mat_owner &ge; crit</code>, with <code>att = 0.23</code> "
        "fitted once across the cohort and <code>depth = 3</code>. The open question was how to "
        "make the rule <em>reach further</em>. Measured, the problem is the opposite one.</p>"
        "<div class='tablewrap'><table><thead><tr><th>wound lumen, by depth</th>%s"
        "<th class='n'>mean</th></tr></thead><tbody>%s</tbody></table></div>"
        "<p>Depths 2 through 5 are identical to four decimals &mdash; past shell 2 the field "
        "never clears the bar, so the extra shells are inert. The single difference between "
        "depth 1 and depth 2 is <code>wound_comsol005</code>, where shell 2 commits only false "
        "positives. The rule was not failing to reach. It was reaching one shell too far, on one "
        "vessel, and paying 0.145 for it.</p>"
        "<h3>The constant was standing in for transport</h3>"
        "<p><code>Mat</code> is made at the surface and has to survive convection to reach "
        "depth, so how far it gets is a local property of the flow, not a cohort constant. High "
        "shear thins the concentration boundary layer; a stagnation band lets the same wall "
        "<code>Mat</code> reach further.</p>"
        "<p style=\"font-family:'IBM Plex Mono',monospace;font-size:14px\">att<sub>node</sub> = "
        "clip(att<sub>0</sub> &middot; (sr<sub>ref</sub>/sr<sub>node</sub>)<sup>&beta;</sup>, "
        "0.05, 0.95)</p>"
        "<p><code>sr_ref</code> is the vessel's own median wall shear, so the ratio carries no "
        "absolute scale between vessels, and <strong>&beta; = 0 returns the shipped constant "
        "bit-for-bit</strong> &mdash; the swept family strictly contains the baseline, so only "
        "the held-out pick can lose.</p>"
        "<div class='tablewrap'><table><thead><tr><th>vessel</th>"
        "<th class='n'>shipped w_lum</th><th class='n'>corrected</th><th class='n'>&Delta;</th>"
        "<th class='n'>shipped w_reg</th><th class='n'>corrected</th></tr></thead><tbody>%s"
        "<tr><td class='em'>MEAN</td><td class='n'>%.4f</td><td class='n em'>%.4f</td>"
        "<td class='n'>%+.4f</td><td class='n'>%.4f</td><td class='n em'>%.4f</td></tr>"
        "</tbody></table></div>"
        "<p>Leave-one-vessel-out over the six wounds, family &beta; &isin; {0, .25, .5, 1} "
        "&times; depth &isin; {1, 2, 3}: held-out wound lumen <strong>0.8375 &rarr; 0.8611</strong> "
        "and wound region <strong>0.9044 &rarr; 0.9270</strong>. <strong>All six folds pick "
        "depth 1</strong>; five of six pick &beta; = 0.5. The wall domain and every non-wound "
        "vessel are bit-identical.</p>"
        "<div class='caveat'><h4>Each effect lives in exactly one vessel</h4>"
        "<p>The depth reduction is worth +0.145 on <code>wound_comsol005</code> and nothing "
        "anywhere else; the shear modulation is worth +0.070 on <code>wound_comsol006</code>, "
        "the corpus's one stagnation-regime wound, and nothing anywhere else. Swept alone, "
        "neither survives leave-one-out &mdash; on its own fold the other five vessels are tied "
        "and the tie-break keeps the old value. Together they clear it. <strong>A second "
        "stagnation-regime wound is the single simulation that would settle this</strong>, and "
        "it is now the highest-value one to commission.</p></div>%s"
    ) % (head, depth_rows, lovo_rows,
         _armmean(pv, SHIP, "w_lum"), _armmean(pv, FIX, "w_lum"),
         _armmean(pv, FIX, "w_lum") - _armmean(pv, SHIP, "w_lum"),
         _armmean(pv, SHIP, "w_reg"), _armmean(pv, FIX, "w_reg"), v6)


def cut_section(arms, ptr) -> str:
    """The per-vessel cut rule: measured, and closed.  Headline metric is `guiding`."""
    def m(arm, dom):
        return _am(arms, arm, dom, "guiding")

    def ms(arm, dom):
        return _am(arms, arm, dom, "severity")

    show = [("cohort_cut", "one cohort constant"),
            ("expected_both", "expected-score budget"),
            ("resid", "physics-conditioned keep/add &mdash; shipped"),
            ("cal_rel_max", "cut at a fraction of the vessel's own max"),
            ("cal_gap", "widest gap in the sorted top"),
            ("cal_phys_anchored", "count implied by the physics mask"),
            ("resid_relmax", "shipped rule on a max-normalised field"),
            ("resid_physq", "shipped rule, cut re-centred on the physics quantile"),
            ("resid_rank", "shipped rule on a fully rank-normalised field"),
            ("nested_pick", "choose the family per fold")]
    present = _arms_metric(arms, "guiding") or _arms_metric(arms, "severity")
    body = "".join(
        "<tr%s><td>%s<div style='color:var(--muted);font-size:12.5px'>%s</div></td>"
        "<td class='n'>%.4f</td><td class='n%s'>%.4f</td>"
        "<td class='n dim'>%.4f</td><td class='n dim'>%.4f</td></tr>" % (
            " class='row-hi'" if a == "resid" else "",
            "<code>%s</code>" % html.escape(a), lbl,
            m(a, "wall"), " em" if a == "resid" else "", m(a, "off"),
            ms(a, "wall"), ms(a, "off"))
        for a, lbl in show if a in present)
    orc_w, orc_o = m("oracle_cut", "wall"), m("oracle_cut", "off")
    base_w, base_o = m("resid", "wall"), m("resid", "off")

    G = _arms_metric(arms, "guiding") or _arms_metric(arms, "severity")
    gaps = []
    for a in G["resid"]:
        o, b = G["oracle_cut"][a].get("off"), G["resid"][a].get("off")
        if None in (o, b) or o != o or b != b:
            continue
        gaps.append((a, b, o - b))
    gaps.sort(key=lambda r: -r[2])
    top = "".join("<code>%s</code> +%.3f" % (html.escape(a.replace("comsol", "p")), g)
                  for a, _, g in gaps[:3])
    med = float(np.median([g for _, _, g in gaps]))

    ship = ""
    if ptr:
        ship = ("<div class='caveat'><h4>What is actually shipped</h4><p>The locked pointer "
                "names <code>%s</code>, and as of this run the default load path follows it. "
                "It did not before: <code>clot_ml_0</code> is a NAME, not a directory, and "
                "the resolver returned the compiled-in default rather than the pointer, so "
                "every caller that did not pass an explicit id &mdash; including the customer "
                "pipeline &mdash; was served a two-generation-old artifact. Promoting and "
                "repointing had been changing nothing for them.</p></div>"
                % html.escape(str(ptr.get("name", "?"))))

    return (
        "<p>&sect;7 localised the off-wall deficit to cut PLACEMENT: the ranking transfers to "
        "vessels never seen (AUC 0.9964) while the cohort constant does not. The stated next "
        "build was a label-free per-vessel cut rule. <strong>It was built &mdash; seven of "
        "them &mdash; and it does not exist.</strong></p>"
        "<div class='tablewrap'><table><thead><tr><th rowspan='2'>readout arm</th>"
        "<th class='n' colspan='2'>guiding &mdash; the deploy score</th>"
        "<th class='n' colspan='2'>severity &mdash; what selection used</th></tr>"
        "<tr><th class='n'>wall</th><th class='n'>off</th>"
        "<th class='n'>wall</th><th class='n'>off</th></tr></thead><tbody>%s"
        "<tr><td class='dim'><em>per-vessel oracle cut</em><div style='color:var(--muted);"
        "font-size:12.5px'>the best single threshold each vessel could be given</div></td>"
        "<td class='n dim'>%.4f</td><td class='n dim'>%.4f</td>"
        "<td class='n dim'>%.4f</td><td class='n dim'>%.4f</td></tr>"
        "</tbody></table></div>"
        "<p>Strictly nested on the 36-vessel pool: every scalar fitted on the out-of-fold "
        "scores of vessels outside the held-out fold. The best new arm beats the shipped one "
        "by <strong>+0.0066</strong> on guiding &mdash; better on 6 vessels, worse on 9. "
        "<strong>The ordering is identical under both metrics</strong>, so nothing here "
        "depends on which score is read.</p>"
        "<div class='kpis'>"
        "<div class='kpi'><div class='v'>+%.4f</div><div class='l'>wall headroom left in the "
        "cut, against a &plusmn;0.024 floor</div></div>"
        "<div class='kpi'><div class='v'>+%.4f</div><div class='l'>off-wall headroom, against "
        "a &plusmn;0.074 floor</div></div>"
        "<div class='kpi'><div class='v'>+%.3f</div><div class='l'>median vessel's off-wall "
        "headroom &mdash; the typical cut is already right</div></div></div>"
        "<p>And what headroom remains is not spread: 72%% of it is three vessels (%s). Two "
        "vessels are already <em>above</em> the single-cut oracle, because the shipped rule "
        "uses four physics-conditioned cuts rather than one.</p>"
        "<h3>The scale is signal, not nuisance</h3>"
        "<p>&sect;7 read the SEALED oracle as saying the field's <em>scale</em> is per-vessel. "
        "Three arms test that by removing the scale and keeping the ranking exactly. All "
        "three collapse: a monotone, label-free re-centring that leaves the ordering "
        "bit-for-bit intact costs <strong>0.13 to 0.30 off-wall</strong>. The absolute level "
        "of the score carries real burden information &mdash; the C0 spread constraint is "
        "what put it there &mdash; so a rule built on scale-freeness loses everywhere. What "
        "&sect;7 measured is still true of those three vessels; it is not a property of the "
        "field.</p>"
        "<p>Choosing the family per fold (<code>nested_pick</code>) also scores below simply "
        "committing to <code>resid</code>. The selection step spends variance and buys "
        "nothing.</p>%s"
        "<div class='caveat'><h4>Do not build another cut rule</h4><p>Both domains sit within "
        "half a noise floor of the ceiling for any per-vessel threshold, and the three "
        "families that could exceed that ceiling are all measured above. The remaining "
        "SEALED shortfall is either the small-sample spread of three vessels or "
        "representation on those vessels specifically &mdash; and distinguishing the two "
        "needs <strong>more vessels, not more thresholds</strong>.</p></div>"
    ) % (body, orc_w, orc_o, ms("oracle_cut", "wall"), ms("oracle_cut", "off"),
         orc_w - base_w, orc_o - base_o, med, top, ship)


def metric_section(geo) -> str:
    """The retraction: 0.618 vs 0.8358 was never a comparison."""
    R = geo["per_vessel"]

    def m(grp, key):
        v = [r[key] for r in R if r["group"] == grp and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    rows = "".join(
        "<tr><td><code>%s</code></td><td class='n'>%d</td><td class='n'>%.3f</td>"
        "<td class='n'>%.4f</td><td class='n'>%.4f</td></tr>" % (
            html.escape(r["stem"].replace("comsol", "p")), r["n_gt"], r["gt_p50"],
            r["sev"], r["dep"])
        for r in sorted([x for x in R if x["group"] == "SEALED"],
                        key=lambda x: x["stem"]))
    cs, cd = m("cohort", "sev"), m("cohort", "dep")
    ss, sd = m("SEALED", "sev"), m("SEALED", "dep")
    return (
        "<p>&sect;7 asked why off-wall reads <strong>0.618</strong> on the four sealed "
        "vessels against <strong>0.8358</strong> in cross-validation, called it three times "
        "the noise floor, and &sect;9 spent seven cut rules trying to close it. "
        "<strong>The two numbers were never comparable.</strong> They are the same "
        "prediction measured with two different metrics &mdash; the cross-validated tables "
        "score with <code>SeverityScorer</code>, the sealed read scores with "
        "<code>domain_score</code>.</p>"
        "<div class='tablewrap'><table><thead><tr><th>off-wall, same masks, same spec</th>"
        "<th class='n'>cohort (n=20)</th><th class='n'>SEALED (n=3)</th>"
        "<th class='n'>difference</th></tr></thead><tbody>"
        "<tr><td>severity metric &mdash; what the CV tables report</td>"
        "<td class='n'>%.4f</td><td class='n em'>%.4f</td><td class='n'>%+.4f</td></tr>"
        "<tr><td>deploy metric &mdash; what the sealed read reported</td>"
        "<td class='n'>%.4f</td><td class='n em'>%.4f</td><td class='n'>%+.4f</td></tr>"
        "</tbody></table></div>"
        "<div class='kpis'>"
        "<div class='kpi'><div class='v'>%+.4f</div><div class='l'>SEALED minus cohort, "
        "severity metric</div></div>"
        "<div class='kpi'><div class='v'>%+.4f</div><div class='l'>SEALED minus cohort, "
        "deploy metric &mdash; a third of the &plusmn;0.074 floor</div></div>"
        "<div class='kpi'><div class='v'>%.4f</div><div class='l'>off-wall ranking AUC on "
        "vessels never seen</div></div></div>"
        "<p><strong>SEALED matches the training cohort on both metrics</strong>, and on the "
        "severity metric it is very slightly ahead. The deploy metric runs 0.19&ndash;0.22 "
        "lower than severity off-wall on every vessel in the cohort; that offset, not a "
        "generalisation failure, was the whole of the gap.</p>"
        "<h3>Verified end to end before this was written</h3>"
        "<p>The deployed pipeline's final-frame mask on each sealed vessel is <em>identical</em> "
        "to applying the shipped readout to the score field &mdash; 55, 29 and 5 nodes "
        "committed, both ways &mdash; and scoring that mask with the deploy metric reproduces "
        "the recorded read to four decimals.</p>"
        "<div class='tablewrap'><table><thead><tr><th>sealed vessel</th>"
        "<th class='n'>off-wall GT nodes</th><th class='n'>GT score median</th>"
        "<th class='n'>severity</th><th class='n'>deploy</th></tr></thead>"
        "<tbody>%s</tbody></table></div>"
        "<div class='caveat'><h4>What this retracts</h4><p>&sect;7 is withdrawn. Its readout "
        "gap of <strong>+0.2713</strong> was <code>oracle(severity)</code> minus "
        "<code>deployed(deploy)</code>; computed within one metric it is <strong>+0.0503</strong> "
        "on SEALED against <strong>+0.0406</strong> on the cohort &mdash; the same number, not "
        "six times it. &sect;9's closing claim that the remaining shortfall needs more vessels "
        "is withdrawn with it. &sect;9's arms table stands: every arm in it was scored with the "
        "same metric.</p></div>"
        "<div class='caveat'><h4>What stops it recurring</h4><p>Two evaluation scripts on two "
        "metrics, each internally consistent, neither labelling itself. The number crossed "
        "between them in a summary and every later step inherited it. The non-wound scorer now "
        "returns <em>both</em> metrics on every call and the printed table names the one it is "
        "showing; the pointer records both; and the diagnostic behind this section reports them "
        "side by side by construction. A future sealed read carries both numbers.</p></div>"
    ) % (cs, ss, ss - cs, cd, sd, sd - cd, ss - cs, sd - cd, 0.9964, rows)


def eval_section(rows, title: str) -> str:
    if not rows:
        return ""
    nw = [r for r in rows if not r["wound"]]
    wd = [r for r in rows if r["wound"]]
    out = []
    if nw:
        trs = "".join(
            f"<tr><td><code>{html.escape(r['stem'])}</code></td>"
            f"<td class='n'>{_fmt(r.get('v0_fin_wall'), 3)}</td>"
            f"<td class='n'>{_fmt(r.get('v0_fin_off'), 3)}</td>"
            f"<td class='n dim'>{_fmt(r.get('v0_mot_wall'), 3)}</td>"
            f"<td class='n dim'>{_fmt(r.get('v0_mot_off'), 3)}</td></tr>" for r in nw)
        def mean(k):
            v = [r[k] for r in nw if r.get(k) == r.get(k)]
            return float(np.mean(v)) if v else float("nan")
        out.append(f"""<h4>No-wound vessels &mdash; {title}</h4>
<div class="tablewrap"><table>
<thead><tr><th>vessel</th><th class="n">wall (final)</th><th class="n">off-wall (final)</th>
<th class="n">wall (mean over time)</th><th class="n">off-wall (mean over time)</th></tr></thead>
<tbody>{trs}
<tr class="row-hi"><td><strong>mean</strong></td>
<td class="n em">{_fmt(mean('v0_fin_wall'), 4)}</td><td class="n em">{_fmt(mean('v0_fin_off'), 4)}</td>
<td class="n">{_fmt(mean('v0_mot_wall'), 4)}</td><td class="n">{_fmt(mean('v0_mot_off'), 4)}</td></tr>
</tbody></table></div>""")
    if wd:
        trs = "".join(
            f"<tr><td><code>{html.escape(r['stem'])}</code></td>"
            f"<td class='n'>{_fmt(r.get('v0_fin_wall'), 3)}</td>"
            f"<td class='n'>{_fmt(r.get('v0_fin_w_reg'), 3)}</td>"
            f"<td class='n'>{_fmt(r.get('v0_fin_w_lum'), 3)}</td>"
            f"<td class='n'>{_fmt(r.get('v0_fin_far'), 3)}</td></tr>" for r in wd)
        out.append(f"""<h4>Wound vessels &mdash; held out of the GNN entirely</h4>
<div class="tablewrap"><table>
<thead><tr><th>vessel</th><th class="n">wall</th><th class="n">wound region</th>
<th class="n">wound lumen</th><th class="n">far field</th></tr></thead>
<tbody>{trs}</tbody></table></div>""")
    return "".join(out)


# ---------------------------------------------------------------------------
CSS = """
:root{
  --paper:#faf9f7; --panel:#ffffff; --ink:#15181c; --ink-2:#3c4249; --muted:#71777e;
  --rule:#e2ded8; --rule-2:#efece7;
  --crimson:#a3243f; --flow:#2c6382; --good:#3c7452; --warn:#a9701f;
  --shadow:0 1px 2px rgba(21,24,28,.05), 0 8px 24px -18px rgba(21,24,28,.35);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#111417; --panel:#171b1f; --ink:#e9e6e1; --ink-2:#c2c6ca; --muted:#8b9198;
    --rule:#2a3036; --rule-2:#20262b;
    --crimson:#e2647e; --flow:#7fb3d1; --good:#78b48f; --warn:#d9a349;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -18px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --paper:#111417; --panel:#171b1f; --ink:#e9e6e1; --ink-2:#c2c6ca; --muted:#8b9198;
  --rule:#2a3036; --rule-2:#20262b;
  --crimson:#e2647e; --flow:#7fb3d1; --good:#78b48f; --warn:#d9a349;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -18px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; font-size:16.5px; line-height:1.62;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px; margin:0 auto; padding:0 28px 96px}
code,.n,.mono{font-family:"IBM Plex Mono",ui-monospace,"Cascadia Mono",monospace}
code{font-size:.88em; background:var(--rule-2); padding:.08em .34em; border-radius:3px;
  color:var(--ink-2)}

header.hero{padding:72px 0 40px; border-bottom:1px solid var(--rule); margin-bottom:8px}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--crimson); margin:0 0 18px}
h1{font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:clamp(38px,6vw,62px);
  line-height:1.04; letter-spacing:-.02em; margin:0 0 20px; text-wrap:balance}
.standfirst{font-size:20px; line-height:1.5; color:var(--ink-2); max-width:62ch; margin:0;
  text-wrap:pretty}

.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:6px; overflow:hidden;
  margin:40px 0 8px; box-shadow:var(--shadow)}
.kpi{background:var(--panel); padding:20px 22px 18px}
.kpi .v{font-family:"IBM Plex Mono",monospace; font-size:27px; font-variant-numeric:tabular-nums;
  line-height:1.1; letter-spacing:-.02em}
.kpi .l{font-size:12.5px; color:var(--muted); margin-top:8px; line-height:1.35}
.kpi.crim .v{color:var(--crimson)} .kpi.flow .v{color:var(--flow)}

section{padding:48px 0 8px; border-top:1px solid var(--rule-2)}
section:first-of-type{border-top:0}
.snum{font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); display:block; margin-bottom:10px}
h2{font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:31px; line-height:1.16;
  letter-spacing:-.015em; margin:0 0 14px; text-wrap:balance}
h3{font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:21px; margin:38px 0 10px}
h4{font-size:13px; letter-spacing:.09em; text-transform:uppercase; color:var(--muted);
  font-weight:600; margin:30px 0 10px}
p{max-width:68ch; margin:0 0 16px}
p.note{font-size:14.5px; color:var(--muted); max-width:74ch}
strong{font-weight:600}
blockquote{margin:22px 0; padding:2px 0 2px 20px; border-left:2px solid var(--crimson);
  color:var(--ink-2); font-style:italic; max-width:66ch}
ul{max-width:68ch; padding-left:20px} li{margin:0 0 8px}

.tablewrap{overflow-x:auto; margin:18px 0 20px; border:1px solid var(--rule);
  border-radius:6px; background:var(--panel); box-shadow:var(--shadow)}
table{border-collapse:collapse; width:100%; font-size:14.5px}
th,td{padding:9px 15px; text-align:left; border-bottom:1px solid var(--rule-2); white-space:nowrap}
thead th{font-size:11.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
  font-weight:600; background:var(--rule-2); border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums}
td.em{font-weight:600; color:var(--ink)}
td.dim{color:var(--muted)}
tr.row-hi td{background:color-mix(in srgb, var(--crimson) 7%, transparent)}

.strip{margin:24px 0 28px}
.strip figcaption{font-size:14px; color:var(--muted); margin-bottom:16px; max-width:70ch}
.stripbody{position:relative; height:74px; border:1px solid var(--rule); border-radius:6px;
  background:var(--panel); box-shadow:var(--shadow)}
.tick{position:absolute; top:14px; width:2px; height:26px; background:var(--flow);
  opacity:.62; transform:translateX(-1px)}
.tick.out{background:var(--crimson); opacity:1; height:34px; top:10px; width:3px}
.axis{position:absolute; inset:auto 0 0 0; height:24px}
.ax{position:absolute; top:0; transform:translateX(-50%)}
.ax i{display:block; width:1px; height:6px; background:var(--rule); margin:0 auto}
.ax b{display:block; font-family:"IBM Plex Mono",monospace; font-size:10.5px; font-weight:400;
  color:var(--muted); margin-top:2px}

.pairhead{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:6px; overflow:hidden;
  margin:20px 0}
.pairhead > div{background:var(--panel); padding:14px 16px; font-size:14.5px}
.pairhead .k{display:block; font-size:11px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); margin-bottom:5px}

details{margin:14px 0; font-size:14.5px}
summary{cursor:pointer; color:var(--flow); font-size:13.5px}
summary:focus-visible{outline:2px solid var(--flow); outline-offset:3px}
ul.folds{list-style:none; padding:10px 0 0; font-family:"IBM Plex Mono",monospace; font-size:13px;
  color:var(--ink-2)}
.caveat{border:1px solid var(--rule); border-left:3px solid var(--warn); border-radius:6px;
  background:var(--panel); padding:18px 22px; margin:22px 0; box-shadow:var(--shadow)}
.caveat h4{margin-top:0; color:var(--warn)}
.caveat p:last-child{margin-bottom:0}
span.st{font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted)}
span.st.ok{color:var(--good)}
footer{margin-top:64px; padding-top:24px; border-top:1px solid var(--rule);
  font-size:13.5px; color:var(--muted)}
@media (max-width:640px){ .wrap{padding:0 18px 64px} header.hero{padding:44px 0 30px} }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "report.html"))
    args = ap.parse_args()

    audit = _load(OUT / "fem_flow_audit.json")
    onset = _load(OUT / "wound_onset_check.json")
    ab = _load(OUT / "ab_pair.json")
    ev_fem = _load(OUT / "eval_fem.json")
    ev_gt = _load(OUT / "eval_gt.json")
    ev_sealed = _load(OUT / "eval_sealed.json")
    lovo = _load(REPO / "outputs/clot_ml/wound_rate_fem/lovo.json")
    readout_fem = _text(LOGS / "10_readout_fem.log")
    readout_gt = _text(LOGS / "09_readout_gt.log")
    flow_paired = _load(OUT / "flow_source_paired.json")
    c0_paired = _load(OUT / "c0_ablation_paired.json")
    gap = _load(OUT / "sealed_offwall_gap.json")
    att = _load(OUT / "wound_offwall_attenuation.json")
    v6a = _load(OUT / "wound_offwall_v6.json")
    v6b = _load(OUT / "wound_offwall_v6_d1b05.json")
    arms = _load(OUT / "readout_arms_fem.json")
    ptr = _load(REPO / "data/reference/clot_gnn_locked.json")
    geo = _load(OUT / "offwall_score_geography.json")
    readout_noc0 = _text(LOGS / "21_readout_fem_noc0.log")
    manifest = _load(REPO / "outputs/clot_ml/locked/DeployClot/manifest.json")

    # KPI strip
    kpis = []
    if audit:
        rel = np.median([r["rel_l2_uv"] for r in audit])
        gj = np.median([r["gate_jac"] for r in audit if r["gate_jac"] == r["gate_jac"]])
        sv = np.median([r["solve_s"] for r in audit])
        kpis += [("flow", f"{rel:.4f}", "median velocity rel L2 of the local FEM solve against COMSOL"),
                 ("flow", f"{gj:.3f}", "median deposition-gate agreement at the wall"),
                 ("", f"{sv:.1f} s", "median solve time per vessel, one CPU")]
    if ab:
        kpis.append(("crim", f"{ab['created']['all']['f1']:.3f}",
                     "F1 recovering the clot the injury created, matched A/B pair"))
    if manifest:
        kpis.append(("", str(len(manifest.get("training_pool", []))),
                     "vessels in the training pool"))
    if ev_sealed:
        nwx = [r for r in ev_sealed if not r["wound"]]
        if nwx:
            kpis.append(("", f"{float(np.nanmean([r['v0_fin_wall'] for r in nwx])):.4f}",
                         "wall on the four SEALED vessels — the one final read"))
    kpi_html = "".join(
        f'<div class="kpi {c}"><div class="v">{html.escape(v)}</div>'
        f'<div class="l">{html.escape(l)}</div></div>' for c, v, l in kpis)

    secs = []
    secs.append(("01 &middot; the deploy-legal flow", "What the local FEM solver costs", f"""
<p>Every clot number in this project before now was measured on COMSOL's own <code>t=0</code>
velocity field. DeployClot's is solved: a steady Carreau Navier&ndash;Stokes problem on the
vessel's own mesh, given the mesh, its inlet/outlet/wall boundary tags, and the inlet velocity
profile. No COMSOL field enters the interior anywhere; ground truth appears only in the labels.</p>
{solver_section(audit)}
<p>This is why the <code>fem</code> arm takes the ground-truth treatment &mdash; a 3-hop
differentiation stencil and unit gain &mdash; while the RGP-DEQ surrogate arm needs a 6-hop
stencil and a fitted &times;3.00 amplitude correction. A converged solve is on COMSOL's own
scale; a surrogate is not.</p>
<div class="caveat"><h4>Where the solver is wrong, it is wrong off the wall</h4>
<p>The two outliers, <code>comsol045</code> and <code>comsol046</code>, have <em>zero</em>
wall-node error. Their error is confined to a single downstream recirculation window &mdash; on
<code>comsol045</code> the 200 worst nodes all lie between x = 4.2 and 5.0 of a 6.9-long
domain, none of them on the wall &mdash; where the two solvers place the shear layer
differently. Both are the highest-peak-velocity vessels in the corpus.</p></div>
<h3>A boundary-tag bug the audit found</h3>
<p><code>comsol038</code> would not solve at all, and <code>comsol048</code> &mdash; one half
of the A/B pair &mdash; solved with 4 of its 21 outlet facets silently given the no-slip wall
condition. Inlet and outlet facets were tagged by requiring <em>both</em> corner vertices to
carry COMSOL's node selection, and that selection is not always complete on a quadratic mesh.
An inlet is a straight cut through the lumen, so the tagged nodes determine it exactly: the
solver now fits the line through them and takes every boundary facet whose midpoint lies on it,
accepting the completion only when it contains what the corner rule already agreed on. Bit-exact
no-op on 52 of 54 packs; <code>comsol048</code>'s rel L2 goes 0.069 &rarr; <strong>0.020</strong>.</p>
"""))

    secs.append(("02 &middot; the wound clock", "The wound source is not delayed by 100 s", f"""
<p>A 100-second delay on the wound source &mdash; against the healthy wall's
<code>step2t</code>, which switches at 12 s over a 2.5 s transition &mdash; was proposed and
tested. If it were real, then at the first stored frame (t = 150 s) the wound would have
accumulated 50 s of growth against the wall's 138 s, so its wound-to-wall <code>Mat</code> ratio
would sit about 2.8&times; <em>below</em> the trend the later frames define.</p>
{onset_section(onset)}
<p>Observed matches the no-delay trend to 1&ndash;3% on four vessels and <em>overshoots</em> it
on the other two &mdash; the opposite direction from a delay, and nowhere near the deficit a
delay requires. Extrapolating each vessel's <code>Mat</code> curve back to zero independently
agrees: within a pack, the wound's onset and its own healthy wall's onset differ by a few
seconds on all six.</p>
<p class="note">Two riders. The <code>.mph</code> files for <code>wound_comsol004/005/006</code>
are not in <code>comsol_models/</code>, so this is a measurement on the packs rather than a
reading of the model tree. And the stored time grid is 150 s, so any onset inside (0, 150) s
would be sub-grid and could not change a single stored frame either way.</p>
"""))

    if lovo:
        secs.append(("03 &middot; the wound complement", "Two constants, six vessels, leave-one-out", f"""
<p>The wound is the same surface-deposition law with the two shear gates deleted. The
complement fits a two-regime rate <code>(G_pre, G_post)</code> inside COMSOL's own surface ODE
&mdash; not a label &mdash; against ground-truth <code>Mat</code> in log space over the whole
trajectory. Every arm below is scored leave-one-vessel-out.</p>
{wound_section(lovo)}
<p>The per-node network still loses to two global constants. That was the finding at n = 3 and
it survives at n = 6, which is the useful result: the extra wound simulations did not buy the
capacity, they bought the confidence that the low-capacity choice was right.</p>
"""))

    cvh = cv_section(readout_fem, "dc_fem_c0")
    cvg = cv_section(readout_gt, "dc_gt_c0")
    fps = flow_paired_section(flow_paired)
    if cvh:
        gt_block = (f"<h4>The same protocol on COMSOL's own flow</h4>{cvg}" if cvg else "")
        fp_block = (f"<h3>What the solved flow costs</h3>{fps}" if fps else "")
        cvn = cv_section(readout_noc0, "dc_fem_noc0")
        c0p = flow_paired_section(c0_paired)
        c0_block = ""
        if c0p or cvn:
            c0_block = (
                "<h3>Does the C0 constraint still pay on solved flow?</h3>"
                "<p>The within-domain spread constraint (<code>shape_w</code>) is what made "
                "the score field cuttable by a single cohort constant on COMSOL flow &mdash; "
                "it moved off-wall from 0.5812 to 0.7078 without touching the per-vessel "
                "oracle, which is the signature of a calibration fix rather than a ranking "
                "one. Whether it survives a solved field had never been measured.</p>"
                + cvn + c0p)
        secs.append(("04 &middot; validation", "Strictly-nested cross-validation, deploy flow", f"""
<p>Geometry-stratified 5-fold over the whole non-SEALED pool, every vessel held out exactly
once. The readout is chosen inside each fold, on the out-of-fold scores of the vessels outside
it, so no quantity behind a vessel's number was fitted with that vessel visible.</p>
{cvh}
{gt_block}
{fp_block}
{c0_block}
<p class="note">The cohort noise floor last measured on this project was &plusmn;0.024 wall and
&plusmn;0.074 off-wall &mdash; the spread of three configurations of the same arm. It was
measured on the previous 23-vessel cohort and has not been re-measured here, so treat it as
indicative: a cohort-mean difference near it is not a result.</p>
"""))

    if ev_fem or ev_gt:
        secs.append(("05 &middot; the artifact", "DeployClot on the deploy metric", f"""
<p>The promoted artifact, scored on the metric of record: domain-restricted severity, wall and
true lumen separately, at the final time point and averaged over time.</p>
<div class="caveat"><h4>The second table is not the flow-source cost</h4>
<p>Those are the same weights, trained on solved-flow features, handed COMSOL's field at
inference. That is a train/test mismatch which handicaps the COMSOL column, and it duly reads
<em>lower</em> where the cross-validation reads higher. It measures how gracefully the artifact
degrades under a flow swap. What solving the flow actually costs is the paired comparison in
section 04, where each arm is trained <em>and</em> evaluated on its own field.</p></div>
{eval_section(ev_fem, 'FEM flow, the deploy arm')}
{eval_section(ev_gt, 'COMSOL flow, same weights &mdash; robustness to a flow swap, NOT the flow-source cost')}
"""))

    if ab:
        secs.append(("06 &middot; the counterfactual", "One vessel, with and without the injury", f"""
<p>Until this corpus there was no paired A/B: no two runs in the dataset shared a geometry, so
nothing isolated the wound's effect. <code>wound_comsol005</code> and <code>comsol048</code>
are the same vessel outline &mdash; identical <code>d_bar</code> to sixteen significant figures,
median wall-node distance 0.0000 &mdash; remeshed, one with the wound selection and one
without.</p>
<p>That makes a question available that no per-vessel score can ask. With geometry, inflow,
mesh family and physics all held fixed, the only thing that changed is the injury, so
<em>clot(wound) and not clot(no wound)</em> is the clot the injury created. A model that
predicted each vessel well while attributing the extra clot to the wrong place would pass every
other metric here and fail this one.</p>
{ab_section(ab)}
<p class="note">No wound pack is ever in the GNN training pool, and <code>comsol048</code> is
scored out-of-fold, so neither half of the pair is in-sample.</p>
"""))

    if ev_sealed:
        nw = [r for r in ev_sealed if not r["wound"]]
        mw = float(np.nanmean([r["v0_fin_wall"] for r in nw])) if nw else float("nan")
        mo = float(np.nanmean([r["v0_fin_off"] for r in nw
                               if r.get("v0_fin_off") == r.get("v0_fin_off")])) if nw else float("nan")
        secs.append(("07 &middot; sealed", "The one final read", f"""
<p>Four vessels have been held completely closed since the project's start &mdash; never
trained on, never selected on, never plotted. They are spent once. This is that once, taken
only after every model change was finished.</p>
{eval_section(ev_sealed, 'SEALED, first and only read')}
<div class="kpis">
  <div class="kpi"><div class="v">{mw:.4f}</div><div class="l">wall, mean over four unseen
  vessels &mdash; <em>above</em> the cross-validated 0.9309&ndash;0.9398</div></div>
  <div class="kpi crim"><div class="v">{mo:.4f}</div><div class="l">off-wall, against a
  cross-validated 0.8351 &mdash; a 0.217 shortfall, three times the noise floor</div></div>
</div>
<div class="caveat"><h4>Cross-validation over-estimated off-wall generalisation</h4>
<p>The wall holds and improves on never-seen vessels. The off-wall verdict does not: 0.618
against 0.835, a gap far too large to be sampling noise at the cohort level, on n&nbsp;=&nbsp;3
vessels carrying off-wall ground truth. <strong>Quote 0.618 as the deployable off-wall
figure.</strong></p></div>
{_gap_block(gap)}
"""))

    # --- run status -----------------------------------------------------------------------
    # The page is generated mid-pipeline as often as at the end, so it says which stages have
    # actually produced their output rather than letting a missing section read as a null result.
    stages = [
        ("t=0 flow audit, 54 packs", (OUT / "fem_flow_audit.json").exists()),
        ("wound-onset ratio test", (OUT / "wound_onset_check.json").exists()),
        ("FEM feature caches (55- and 68-column, 40 vessels)",
         (REPO / "outputs/clot_ml_cache_v5_fem/comsol048.npz").exists()),
        ("FEM temporal-transport channels",
         (REPO / "outputs/temporal_transport_fem/comsol012.npz").exists()),
        ("wound complement, leave-one-out on 6 vessels",
         (REPO / "outputs/clot_ml/wound_rate_fem/lovo.json").exists()),
        ("base ensemble promoted", bool(manifest and manifest.get("n_members"))),
        ("temporal head fitted", bool(manifest and manifest.get("temporal_file"))),
        ("wound complement promoted",
         (REPO / "outputs/clot_ml/locked/DeployClot_w/manifest.json").exists()),
        ("unified artifact promoted",
         (REPO / "outputs/clot_ml/locked/DeployClot_0/manifest.json").exists()),
        ("A/B counterfactual", (OUT / "ab_pair.json").exists()),
        ("deploy-metric evaluation", (OUT / "eval_fem.json").exists()),
        ("flow-source control (same weights, GT flow)", (OUT / "eval_gt.json").exists()),
        ("strictly-nested cross-validation",
         (REPO / "outputs/phase9_scores/dc_fem_c0.npz").exists()),
    ]
    _done = '<span class="st ok">complete</span>'
    _todo = '<span class="st">pending</span>'
    rows = "".join(
        "<tr><td>%s</td><td class='n'>%s</td></tr>" % (html.escape(n), _done if ok else _todo)
        for n, ok in stages)
    if att:
        secs.append(("08 &middot; the depth rule",
                     "What the attenuation was standing in for",
                     offwall_section(att, v6a, v6b)))

    if arms:
        secs.append(("09 &middot; the cut",
                     "Seven label-free rules, and the ceiling they do not reach",
                     cut_section(arms, ptr)))

    if geo:
        secs.append(("10 &middot; the correction",
                     "The SEALED off-wall gap was two different metrics",
                     metric_section(geo)))

    secs.append(("99 &middot; run status", "What this page is built from", f"""
<p>Sections above appear only when their stage has written its output, so an absent section is
a stage that has not run &mdash; never a null result.</p>
<div class="tablewrap"><table><thead><tr><th>stage</th><th class="n">state</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="note">Resume with <code>bash scripts/go_deployclot.sh</code>; every stage is
idempotent and skips work already on disk.</p>
"""))

    body = "".join(
        f'<section><span class="snum">{n}</span><h2>{t}</h2>{b}</section>'
        for n, t, b in secs)

    page = f"""<title>DeployClot</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">Thrombus ML &middot; final training &middot; 2026-09-02/03</p>
  <h1>DeployClot</h1>
  <p class="standfirst">The clot model, retrained and revalidated on flow it solves for itself.
  Every velocity field in training, readout and evaluation comes from a local finite-element
  Navier&ndash;Stokes solve on the vessel's own mesh &mdash; no COMSOL field anywhere in the
  inputs.</p>
  <div class="kpis">{kpi_html}</div>
</header>
{body}
<footer>
  <p>Generated by <code>scripts/build_deployclot_report.py</code> from
  <code>outputs/deployclot/</code>. Method notes: <code>docs/DEPLOYCLOT.md</code>.
  Pipeline: <code>scripts/go_deployclot.sh</code>.</p>
</footer>
</div>
"""
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    print(f"[save] {p}  ({len(page)} bytes, {len(secs)} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
