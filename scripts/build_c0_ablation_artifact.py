"""Viz the real 'v4 -> v5' story: the C0 distributional constraint (shape_w 0 -> 2).

The clot_gnn_v4 and clot_gnn_v5 directories currently locked in this repo are byte-identical
(same GNN member weights, same temporal.pkl) -- verified by sha256. Neither is what an
earlier report meant by "v4 alone" vs "v5": that comparison is the C0 training-time
distributional constraint (docs/PHASE10_V4.md, MODEL_REVIEW_2026-08-22.md 9b), which was
already baked into BOTH on-disk directories by the time either was promoted (2026-08-23).
The "before" arm (shape_w=0, tags v5a/v5b/v5c) was never kept as its own loadable locked
artifact -- only its per-vessel scores survive, in outputs/strict_v5_2026-08-22.json, paired
against the "after" arm in outputs/strict_c0shape_2026-08-23.json. This script diffs those
two caches directly rather than re-deriving the numbers by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OLD_PATH = Path("outputs/strict_v5_2026-08-22.json")   # shape_w=0.0, tags v5a/v5b/v5c
NEW_PATH = Path("outputs/strict_c0shape_2026-08-23.json")  # shape_w=2.0, shipped
OUT_PATH = Path("outputs/c0_ablation_report.html")


def main() -> None:
    old = json.loads(OLD_PATH.read_text())["rows"]
    new = json.loads(NEW_PATH.read_text())["rows"]
    assert set(old) == set(new), "vessel sets differ between the two caches"

    rows = []
    for p in sorted(old):
        o, n = old[p], new[p]

        def clean(v):
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 4)

        rows.append(dict(
            p=p, cls=o["cls"], free=bool(o["free"]),
            wall_old=clean(o["wall"]), wall_new=clean(n["wall"]),
            off_old=clean(o["off"]), off_new=clean(n["off"]),
            n_off_old=o["n_pred_off"], n_off_new=n["n_pred_off"],
        ))
    for r in rows:
        r["wall_d"] = None if r["wall_old"] is None or r["wall_new"] is None else round(r["wall_new"] - r["wall_old"], 4)
        r["off_d"] = None if r["off_old"] is None or r["off_new"] is None else round(r["off_new"] - r["off_old"], 4)

    carrying = [r for r in rows if not r["free"]]
    free = [r for r in rows if r["free"]]
    wall_vals = [(r["wall_old"], r["wall_new"]) for r in carrying if r["wall_old"] is not None]
    off_vals = [(r["off_old"], r["off_new"]) for r in carrying if r["off_old"] is not None]
    summary = dict(
        n_carrying=len(carrying), n_free=len(free), n_off_scored=len(off_vals),
        wall_old=round(sum(v[0] for v in wall_vals) / len(wall_vals), 4),
        wall_new=round(sum(v[1] for v in wall_vals) / len(wall_vals), 4),
        off_old=round(sum(v[0] for v in off_vals) / len(off_vals), 4),
        off_new=round(sum(v[1] for v in off_vals) / len(off_vals), 4),
    )

    data_json = json.dumps(dict(rows=rows, summary=summary)).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data_json)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")
    print(f"wall {summary['wall_old']} -> {summary['wall_new']}  "
          f"off {summary['off_old']} -> {summary['off_new']}  (n_off={summary['n_off_scored']})")


TEMPLATE = r"""<title>The C0 Ablation</title>
<style>
:root {
  --bg: #f5f1ef; --surface: #ffffff; --surface-2: #f0e8e5; --ink: #2a1f1c; --muted: #7a655f;
  --line: #e5d8d3; --accent: #b8442f; --accent-ink: #ffffff;
  --old-c: #2d6e5e; --new-c: #b8442f; --old-far: #bfe0d5; --new-far: #edcdc4;
  --gain: #2d6e5e; --loss: #b8442f; --free-c: #6b5a52;
  --shadow: 0 1px 2px rgba(42,31,28,0.06), 0 8px 24px -12px rgba(42,31,28,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #191412; --surface: #211a17; --surface-2: #2b211d; --ink: #f0e6e2; --muted: #b5a099;
    --line: #3a2d27; --accent: #e0704f; --accent-ink: #1a0d08;
    --old-c: #57bda3; --new-c: #e0704f; --old-far: #1f4d43; --new-far: #5c3226;
    --gain: #57bda3; --loss: #e0704f; --free-c: #b5a099;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #191412; --surface: #211a17; --surface-2: #2b211d; --ink: #f0e6e2; --muted: #b5a099;
  --line: #3a2d27; --accent: #e0704f; --accent-ink: #1a0d08;
  --old-c: #57bda3; --new-c: #e0704f; --old-far: #1f4d43; --new-far: #5c3226;
  --gain: #57bda3; --loss: #e0704f; --free-c: #b5a099;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1100px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }
.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 0 0 0.5rem; }
h1 { font-family: var(--serif); font-size: clamp(1.7rem, 3vw, 2.3rem); font-weight: 600; margin: 0 0 1rem; text-wrap: balance; letter-spacing: -0.01em; }
h2.section { font-family: var(--serif); font-size: 1.3rem; font-weight: 600; margin: 2.6rem 0 0.7rem; }
.lede { font-size: 1.03rem; color: var(--ink); max-width: 74ch; margin: 0 0 1.4rem; }
.lede code, p code, li code, .callout code { background: var(--surface-2); border-radius: 4px; padding: 0.08rem 0.32rem; font-family: var(--mono); font-size: 0.88em; }
.callout { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--new-c); border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.92rem; max-width: 78ch; margin: 0 0 1rem; box-shadow: var(--shadow); }
.callout b { color: var(--new-c); }
.section-note { color: var(--muted); font-size: 0.92rem; max-width: 74ch; margin: 0 0 1.2rem; }

.stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; margin: 0 0 1.6rem; box-shadow: var(--shadow); }
.stat-cell { background: var(--surface); padding: 1rem 1.1rem; text-align: center; }
.stat-cell .label { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.stat-cell .value { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.5rem; font-weight: 700; }
.stat-cell .arrow { color: var(--muted); font-size: 1rem; margin: 0 0.15rem; }
.stat-cell .delta { display: block; font-family: var(--mono); font-size: 0.78rem; margin-top: 0.25rem; }
.delta.pos { color: var(--gain); } .delta.neg { color: var(--loss); }
@media (max-width: 760px) { .stat-strip { grid-template-columns: 1fr 1fr; } }

.legend-row { display: flex; gap: 1.4rem; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted); margin: 0 0 1rem; }
.legend-row .sw { display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px; margin-right: 0.4rem; vertical-align: -1px; }

.chart-wrap { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 1.3rem 1.4rem 1rem; box-shadow: var(--shadow); overflow-x: auto; }
.bar-row { display: grid; grid-template-columns: 8.5rem 1fr 4.2rem; align-items: center; gap: 0.6rem; min-height: 1.65rem; }
.bar-row + .bar-row { border-top: 1px solid var(--line); }
.bar-row .name { font-family: var(--mono); font-size: 0.76rem; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-row .name .cls { color: var(--muted); font-size: 0.68rem; margin-left: 0.3rem; }
.bar-track { position: relative; height: 1rem; background: var(--surface-2); border-radius: 3px; }
.bar-track .zero { position: absolute; top: -3px; bottom: -3px; width: 1px; background: var(--line); }
.bar-fill { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.bar-fill.gain { background: var(--gain); }
.bar-fill.loss { background: var(--loss); }
.bar-row .num { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.76rem; text-align: right; }
.bar-row .num.pos { color: var(--gain); } .bar-row .num.neg { color: var(--loss); }
.bar-row .num.nan { color: var(--muted); }

table.mover-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; box-shadow: var(--shadow); }
table.mover-table th, table.mover-table td { padding: 0.55rem 0.9rem; text-align: right; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
table.mover-table th:first-child, table.mover-table td:first-child { text-align: left; font-family: var(--mono); }
table.mover-table th { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); font-weight: 600; background: var(--surface-2); }
table.mover-table td.delta.pos { color: var(--gain); font-weight: 700; }
table.mover-table td.delta.neg { color: var(--loss); font-weight: 700; }
.table-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1.4rem; }
@media (max-width: 760px) { .table-pair { grid-template-columns: 1fr; } }

.free-strip { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.6rem 0 1.4rem; }
.free-chip { font-family: var(--mono); font-size: 0.74rem; background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 0.7rem; color: var(--muted); }
.free-chip b { color: var(--free-c); }

.foot-note { color: var(--muted); font-size: 0.82rem; max-width: 78ch; margin-top: 2.4rem; border-top: 1px solid var(--line); padding-top: 1.2rem; }
</style>

<main>
  <p class="eyebrow">Model provenance correction &middot; the C0 distributional constraint</p>
  <h1>v4 and v5 are the same weights. The real split is C0.</h1>
  <p class="lede">
    <code>clot_gnn_v4</code> and <code>clot_gnn_v5</code>, as currently locked in this repo,
    are byte-identical &mdash; same GNN member weights (sha256-verified), same temporal
    readout. What actually moved the numbers is the <b>C0 training-time distributional
    constraint</b> (<code>shape_w</code>: 0 &rarr; 2), which was already baked into both
    directories by the time either was promoted on 2026-08-23. This page diffs the two
    per-vessel score caches directly &mdash; <code>outputs/strict_v5_2026-08-22.json</code>
    (<code>shape_w=0</code>) against <code>outputs/strict_c0shape_2026-08-23.json</code>
    (<code>shape_w=2</code>, shipped) &mdash; on the strict, nested, geometry-stratified
    5-fold CV cohort: 23 clot-carrying vessels plus 8 clot-free.
  </p>

  <div class="callout">
    <b>Neither arm is separately loadable as a model today.</b> The pre-C0 configuration
    (tags <code>v5a</code>/<code>v5b</code>/<code>v5c</code>, <code>shape_w=0</code>) was
    never kept as its own locked, promotable artifact &mdash; only its per-vessel scores
    survive in the cache above. So a live per-vessel spatial replay (model vs ground truth,
    the two-window format used elsewhere in this project) isn't possible for the "before"
    arm without retraining it; this page shows the paired score deltas that <i>are</i> on
    disk instead.
  </div>

  <h2 class="section">The cohort mean</h2>
  <div class="stat-strip" id="stat-strip"></div>
  <p class="section-note">
    Paired per configuration (<code>docs/PHASE10_V4.md</code> / <code>MODEL_REVIEW_2026-08-22.md</code>
    9b): wall +0.0194 [+0.0011, +0.0405] P=0.018; off-wall +0.1266 [+0.0356, +0.2252] P=0.002.
    The wall gain sits inside this cohort's own noise floor (&plusmn;0.024) and is not claimed;
    the off-wall gain does not. Off-wall is scored only on the 16 of 23 clot-carrying vessels
    that have any off-wall GT clot &mdash; the other 7 are undefined (no GT positives in that
    domain), not zero.
  </p>

  <h2 class="section">Per-vessel off-wall delta, sorted</h2>
  <p class="section-note">
    The shape of the gain, not just its mean: a tail fix that rescues vessels that were
    failing while vessels already scoring well barely move. Wall delta is the small dot;
    off-wall delta is the bar.
  </p>
  <div class="legend-row">
    <span><span class="sw" style="background:var(--gain)"></span>C0 gains off-wall</span>
    <span><span class="sw" style="background:var(--loss)"></span>C0 loses off-wall</span>
    <span><span class="sw" style="background:var(--muted); border-radius:50%; width:0.5rem; height:0.5rem; display:inline-block; margin-right:0.4rem;"></span>wall delta (dot position)</span>
  </div>
  <div class="chart-wrap"><div id="chart"></div></div>

  <h2 class="section">Biggest movers</h2>
  <div class="table-pair">
    <div>
      <table class="mover-table" id="gains-table">
        <thead><tr><th>vessel</th><th>off, before</th><th>off, after</th><th>&Delta;</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div>
      <table class="mover-table" id="losses-table">
        <thead><tr><th>vessel</th><th>off, before</th><th>off, after</th><th>&Delta;</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <p class="section-note">
    <code>patient032</code> committed <b>zero</b> off-wall nodes before C0 &mdash; its score
    field never crossed the cohort cut. That's the failure mode C0 was aimed at, and the
    vessel that moved most. <code>patient019</code> (wall &minus;0.080, largest single wall
    regression) and <code>patient005</code> (off-wall 0.250 &rarr; 0.168 on just 4 GT nodes,
    already the worst-served vessel) are the two to watch if the wall side or the
    low-burden vessels are touched again.
  </p>

  <h2 class="section">The clot-free tripwire</h2>
  <p class="section-note">
    8 vessels with <code>maxMat = 0</code> and empty ground truth, scored on the
    false-positive branch only. C0's extra off-wall commitment did not leak into them.
  </p>
  <div class="free-strip" id="free-strip"></div>

  <p class="foot-note">
    Source: <code>outputs/strict_v5_2026-08-22.json</code> (shape_w=0, tags
    v5a/v5b/v5c) vs <code>outputs/strict_c0shape_2026-08-23.json</code> (shape_w=2, tags
    c0shape/c0shape_b/c0shape_c) &mdash; both from <code>scripts/eval_strict.py</code>'s
    strict, nested, geometry-stratified 5-fold protocol. Generated by
    <code>scripts/build_c0_ablation_artifact.py</code>. Full detail:
    <code>docs/PHASE10_V4.md</code>, <code>docs/MODEL_REVIEW_2026-08-22.md</code> &sect;9b.
  </p>
</main>

<script id="viz-data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('viz-data').textContent);
  const rows = DATA.rows, s = DATA.summary;
  const fmt = v => v === null ? 'n/a' : v.toFixed(3);
  const fmtd = v => v === null ? 'n/a' : (v >= 0 ? '+' : '') + v.toFixed(3);

  const strip = document.getElementById('stat-strip');
  const stats = [
    ['wall (mean)', s.wall_old, s.wall_new, s.wall_new - s.wall_old],
    ['off-wall (mean)', s.off_old, s.off_new, s.off_new - s.off_old],
  ];
  strip.innerHTML = stats.map(([label, o, n, d]) => `
    <div class="stat-cell">
      <div class="label">${label}</div>
      <div class="value">${o.toFixed(3)}<span class="arrow">&rarr;</span>${n.toFixed(3)}</div>
      <span class="delta ${d >= 0 ? 'pos' : 'neg'}">${fmtd(d)}</span>
    </div>`).join('') + `
    <div class="stat-cell">
      <div class="label">clot-carrying</div>
      <div class="value">${s.n_carrying}</div>
    </div>
    <div class="stat-cell">
      <div class="label">clot-free (tripwire)</div>
      <div class="value">${s.n_free}</div>
    </div>`;

  const carrying = rows.filter(r => !r.free);
  const scored = carrying.filter(r => r.off_d !== null).sort((a, b) => b.off_d - a.off_d);
  const maxAbs = Math.max(...scored.map(r => Math.abs(r.off_d)), 0.05);

  const chart = document.getElementById('chart');
  chart.innerHTML = scored.map(r => {
    const w = Math.abs(r.off_d) / maxAbs * 50;
    const isGain = r.off_d >= 0;
    const left = isGain ? 50 : 50 - w;
    const dotPct = r.wall_d === null ? null : 50 + (r.wall_d / maxAbs * 50);
    return `<div class="bar-row">
      <div class="name">${r.p}<span class="cls">${r.cls}</span></div>
      <div class="bar-track">
        <div class="zero" style="left:50%"></div>
        <div class="bar-fill ${isGain ? 'gain' : 'loss'}" style="left:${left}%; width:${w}%"></div>
        ${dotPct === null ? '' : `<div style="position:absolute; top:50%; left:${dotPct}%; width:6px; height:6px; margin:-3px 0 0 -3px; border-radius:50%; background:var(--muted); border:1px solid var(--surface);"></div>`}
      </div>
      <div class="num ${isGain ? 'pos' : 'neg'}">${fmtd(r.off_d)}</div>
    </div>`;
  }).join('');

  const gains = scored.slice(0, 5);
  const losses = scored.slice(-5).reverse();
  function fillTable(id, list) {
    document.querySelector(`#${id} tbody`).innerHTML = list.map(r => `
      <tr><td>${r.p}</td><td>${fmt(r.off_old)}</td><td>${fmt(r.off_new)}</td>
      <td class="delta ${r.off_d >= 0 ? 'pos' : 'neg'}">${fmtd(r.off_d)}</td></tr>`).join('');
  }
  fillTable('gains-table', gains);
  fillTable('losses-table', losses);

  const free = rows.filter(r => r.free);
  document.getElementById('free-strip').innerHTML = free.map(r =>
    `<span class="free-chip">${r.p} <b>${fmt(r.wall_new)} / ${fmt(r.off_new)}</b></span>`).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
