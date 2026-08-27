"""Assemble the unified ``clot_ml_v0`` wound time-lapse visualization."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("outputs/clot_ml_v0_temporal_data.json")
OUT_PATH = Path("outputs/clot_ml_v0_temporal_report.html")

VESSEL_LABELS = {
    "wound_patient001": "wound_patient001 — LOVO",
    "wound_patient002": "wound_patient002 — LOVO",
    "wound_patient003": "wound_patient003 — LOVO, known outlier",
}


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    meta = payload.pop("_meta", {})
    model_name = meta.get("model", "clot_gnn_v4w")
    base_name = meta.get("base_model", "clot_gnn_v4")
    sw = meta.get("scores_wound") or {}
    data_json = json.dumps(payload).replace("</", "<\\/")
    labels_json = json.dumps(VESSEL_LABELS)
    order = list(payload.keys())

    def badge(v):
        b = '<span class="tab-badge badge-lovo">LOVO</span>'
        if payload[v].get("outlier"):
            b += '<span class="tab-badge badge-outlier">OUTLIER</span>'
        return b

    tabs = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-vessel="{v}">{v}{badge(v)}</button>'
        for i, v in enumerate(order)
    )

    def r3(v):
        return "&mdash;" if v is None else f"{v:.3f}"

    fin, mot = sw.get("final", {}), sw.get("mean_over_time", {})
    arms = [
        (base_name, "v4", ""),
        (f"+ wound physics, G=1 (set only)", "v4w_physics_only", ""),
        (model_name, "v4w", "hero"),
    ]
    rows = []
    for label, key, cls in arms:
        f, m = fin.get(key, {}), mot.get(key, {})
        cells = "".join(
            f'<td class="{cls}">{r3(f.get(dom))}</td>' for dom in ("wall", "wnd", "w_reg")
        ) + f'<td class="{cls}">{r3(f.get("w_lum"))}</td>' + "".join(
            f'<td class="{cls}">{r3(m.get(dom))}</td>' for dom in ("wall", "wnd", "w_reg", "w_lum")
        )
        rows.append(f'      <tr><td class="{cls}">{label}</td>{cells}</tr>')
    table_rows = "\n".join(rows) if rows else "      <tr><td colspan=8>no scores_wound in locked manifest</td></tr>"

    base_mot_wreg = mot.get("v4", {}).get("w_reg")
    full_mot_wreg = mot.get("v4w", {}).get("w_reg")
    if base_mot_wreg is not None and full_mot_wreg is not None:
        table_note = (
            f"Wall stays fixed across arms &mdash; the complement never touches it. Wound "
            f"region goes <b>{base_mot_wreg:.3f} &rarr; {full_mot_wreg:.3f} mean-over-time</b> "
            f"(the two fitted scalars earning their keep), read off the current shipped "
            f"model's own eval cache. n=3, so treat the magnitude as indicative rather than "
            f"final -- the sign and mechanism are the finding."
        )
    else:
        table_note = "scores_wound not found in data/reference/clot_gnn_locked.json."

    html = TEMPLATE.replace("__DATA__", data_json).replace("__LABELS__", labels_json).replace(
        "<!--TABS-->", tabs
    ).replace("__FIRST__", order[0]).replace("__MODEL__", model_name).replace(
        "__BASE__", base_name
    ).replace("__TABLE_ROWS__", table_rows).replace("__TABLE_NOTE__", table_note)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<title>__MODEL__ &mdash; Wound Model vs Ground Truth</title>
<style>
:root {
  --bg: #f5f1ef; --surface: #ffffff; --surface-2: #f0e8e5; --ink: #2a1f1c; --muted: #7a655f;
  --line: #e5d8d3; --accent: #b8442f; --accent-ink: #ffffff;
  --model-c: #b8442f; --gt-c: #2d6e5e;
  --model-far: #edcdc4; --gt-far: #bfe0d5;
  --score-wall: #6b5a52; --score-wound: #b8442f; --score-wound-off: #a8781f; --score-off: #2d6e5e;
  --site-zone: #8a7a9c;
  --shadow: 0 1px 2px rgba(42,31,28,0.06), 0 8px 24px -12px rgba(42,31,28,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #191412; --surface: #211a17; --surface-2: #2b211d; --ink: #f0e6e2; --muted: #b5a099;
    --line: #3a2d27; --accent: #e0704f; --accent-ink: #1a0d08;
    --model-c: #e0704f; --gt-c: #57bda3;
    --model-far: #5c3226; --gt-far: #1f4d43;
    --score-wall: #b5a099; --score-wound: #e0704f; --score-wound-off: #dba648; --score-off: #57bda3;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #191412; --surface: #211a17; --surface-2: #2b211d; --ink: #f0e6e2; --muted: #b5a099;
  --line: #3a2d27; --accent: #e0704f; --accent-ink: #1a0d08;
  --model-c: #e0704f; --gt-c: #57bda3;
  --model-far: #5c3226; --gt-far: #1f4d43;
  --score-wall: #b5a099; --score-wound: #e0704f; --score-off: #57bda3;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f5f1ef; --surface: #ffffff; --surface-2: #f0e8e5; --ink: #2a1f1c; --muted: #7a655f;
  --line: #e5d8d3; --accent: #b8442f; --accent-ink: #ffffff;
  --model-c: #b8442f; --gt-c: #2d6e5e;
  --model-far: #edcdc4; --gt-far: #bfe0d5;
  --score-wall: #6b5a52; --score-wound: #b8442f; --score-wound-off: #a8781f; --score-off: #2d6e5e;
  --shadow: 0 1px 2px rgba(42,31,28,0.06), 0 8px 24px -12px rgba(42,31,28,0.18);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1260px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }

.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.7rem, 3.1vw, 2.4rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 34ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 72ch; margin: 0 0 1.1rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }

.callout { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--model-c); border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 0.87rem; color: var(--ink); max-width: 78ch; margin: 0 0 1rem; }
.callout b { color: var(--model-c); }
.callout + .callout { border-left-color: var(--gt-c); }
.callout + .callout b { color: var(--gt-c); }
.callout + .callout + .callout { border-left-color: var(--score-wall); margin-bottom: 2rem; }
.callout + .callout + .callout b { color: var(--score-wall); }

.tabs { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }
.tab { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.7rem 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s; display: inline-flex; align-items: center; gap: 0.5rem; }
.tab:hover { border-color: var(--accent); color: var(--ink); }
.tab.active { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
.tab-badge { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.04em; padding: 0.12rem 0.4rem; border-radius: 4px; }
.badge-lovo { background: var(--surface-2); color: var(--muted); }
.badge-outlier { background: #a83a3a; color: #ffffff; }
.tab.active .badge-lovo { background: rgba(255,255,255,0.3); color: var(--accent-ink); }
.tab-group-label { font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin: 0 0 1.3rem; }

.spatial-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-bottom: 1.1rem; }
.panel-box { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); padding: 1rem 1.1rem 1.2rem; }
.panel-box h2 { font-family: var(--serif); font-size: 1.0rem; font-weight: 600; margin: 0 0 0.6rem; display: flex; align-items: center; gap: 0.5rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
canvas { width: 100%; display: block; border-radius: 8px; background: var(--surface-2); }
.spatial-canvas { aspect-ratio: 1 / 1; cursor: grab; touch-action: none; }
.spatial-canvas:active { cursor: grabbing; }
.panel-head-row { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.6rem; flex-wrap: wrap; }
.panel-head-row h2 { margin: 0; }
.zoom-hint { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); white-space: nowrap; }

.score-strip { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; margin: 0 0 1.2rem; box-shadow: var(--shadow); }
.score-cell { background: var(--surface); padding: 0.9rem 1.1rem; text-align: center; }
.score-cell.hero { background: var(--surface-2); }
.score-cell .label { font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.35rem; }
.score-cell .value { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em; }
.score-cell.wall .value { color: var(--score-wall); }
.score-cell.wound .value { color: var(--score-wound); }
.score-cell.wound-off .value { color: var(--score-wound-off); }
.score-cell.off .value { color: var(--score-off); }

.transport { display: flex; align-items: center; gap: 0.9rem; margin: 0.9rem 0 1.8rem; }
.play-btn { font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); cursor: pointer; flex: none; min-width: 4.6rem; }
.play-btn:hover { filter: brightness(1.06); }
.reset-btn { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.8rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; flex: none; }
.reset-btn:hover { border-color: var(--accent); color: var(--ink); }
input[type="range"] { flex: 1; accent-color: var(--accent); }
.time-readout { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.76rem; color: var(--muted); min-width: 9rem; text-align: right; flex: none; }
.time-readout b { color: var(--ink); }

.score-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1.1rem; margin-bottom: 1rem; }
.chart-canvas { aspect-ratio: 4 / 3; }

.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 2.4rem; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 0.45rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.sq { border-radius: 2px; }
.swatch.di { border-radius: 2px; transform: rotate(45deg); width: 9px; height: 9px; }
.swatch.model { background: var(--model-c); } .swatch.gt { background: var(--gt-c); }
.grad-swatch { width: 34px; height: 11px; border-radius: 3px; flex: none; }
.grad-swatch.grad-model { background: linear-gradient(90deg, var(--model-c), var(--model-far)); }
.grad-swatch.grad-gt { background: linear-gradient(90deg, var(--gt-c), var(--gt-far)); }
.ring-swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; border: 1.6px solid; opacity: 0.7; }

h2.section { font-family: var(--serif); font-weight: 600; font-size: 1.3rem; margin: 2.6rem 0 0.4rem; }
.section-note { color: var(--muted); font-size: 0.92rem; max-width: 72ch; margin: 0 0 1.2rem; }
.finding-box { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--score-wound); border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.9rem; color: var(--ink); max-width: 78ch; }
.finding-box b { color: var(--score-wound); }
.compare-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.8rem; margin: 1rem 0; }
.compare-table th, .compare-table td { text-align: right; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--line); }
.compare-table th:first-child, .compare-table td:first-child { text-align: left; }
.compare-table th { color: var(--muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.03em; }
.compare-table td.hero { color: var(--score-wound); font-weight: 700; }
.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 76ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }

@media (max-width: 1100px) { .score-row { grid-template-columns: 1fr 1fr; } .score-strip { grid-template-columns: 1fr 1fr; } }
@media (max-width: 900px) { .score-row { grid-template-columns: 1fr; } }
@media (max-width: 760px) { .spatial-row { grid-template-columns: 1fr; } .score-strip { grid-template-columns: 1fr; } }
</style>

<main>
  <p class="eyebrow">Unified baseline &middot; __MODEL__ &middot; model vs ground truth</p>
  <h1>One baseline for wounded and unwounded vessels.</h1>
  <p class="lede">
    COMSOL's own wound law is the healthy-wall law with the shear gates <strong>deleted</strong>
    &mdash; on all 3 known wound vessels the deposition gate fires on <b>0%</b> of wound
    nodes at t=0, while <b>100%</b> of them clot. <code>__MODEL__</code> is the unified
    baseline: a C0-trained GNN plus the wound boundary-condition branch and its two timing
    constants. On a pack with no wound mask it returns its base GNN output bit-for-bit.
    This page therefore shows the wound cases where the routing is active.
  </p>

  <div class="callout">
    <b>Only 3 wound vessels exist, and none are SEALED yet.</b>
    <code>wound_patient001/002/003</code> are a separate, tiny cohort from the 19-vessel
    pool the GNN backbone trained on -- the backbone itself has never seen a wound. The two
    wound-specific scalars (<code>G_pre</code>, <code>G_post</code>) are fit
    <b>leave-one-vessel-out</b> (LOVO, n=3): each vessel's tab shows a model that never saw
    that vessel's wound-timing data. A genuinely new, held-out wound vessel is expected
    soon; this page is built to grow a fourth, SEALED-badged tab when it lands.
  </div>
  <div class="callout">
    <b><code>wound_patient003</code> is a known outlier.</b> Per
    <code>docs/WOUND_PROGRESS.md</code> &sect;11, it is externally triggered rather than
    following the self-triggered mechanism the other two show, and it carries most of the
    remaining timing error. Shown, not hidden -- its tab is marked.
  </div>
  <div class="callout">
    <b>Research baseline only &mdash; not a cold-deploy claim.</b> This artifact was trained
    and strictly validated using GT t=0 flow. Its no-wound strict nested-CV result is
    0.920 wall / 0.708 off-wall at final time; predicted-flow evaluation is still required
    before calling it deploy-ready.
  </div>

  <h2 class="section">What the complement actually buys</h2>
  <p class="section-note">
    <code>scripts/eval_wound_complement.py</code>, LOVO constants, GT t=0 flow, mean over
    the 3 wound vessels. <code>wnd</code> is coverage only (the boundary is ~100% GT clot,
    so getting the SET right is nearly free, per <code>docs/WOUND_PROGRESS.md</code>);
    <code>w_reg</code>/<code>w_lum</code> are the real result &mdash; region and lumen-only,
    entirely about timing.
  </p>
  <div style="overflow-x:auto">
  <table class="compare-table">
    <thead><tr><th>arm</th><th>FIN wall</th><th>FIN wnd</th><th>FIN w_reg</th><th>FIN w_lum</th><th>MOT wall</th><th>MOT wnd</th><th>MOT w_reg</th><th>MOT w_lum</th></tr></thead>
    <tbody>
__TABLE_ROWS__
    </tbody>
  </table>
  </div>
  <p class="section-note">__TABLE_NOTE__</p>

  <p class="tab-group-label">all 3 known wound vessels, LOVO constants &mdash; no SEALED wound vessel exists yet</p>
  <div class="tabs">
    <!--TABS-->
  </div>

  <div class="spatial-row">
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--model-c)"></span>Model (__MODEL__)</h2>
        <span class="zoom-hint" id="zoom-readout">1.0&times;</span>
      </div>
      <canvas id="canvas-model" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--gt-c)"></span>Ground truth</h2>
        <span class="zoom-hint">scroll/drag, synced</span>
      </div>
      <canvas id="canvas-gt" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
  </div>

  <div class="score-strip">
    <div class="score-cell wall"><div class="label">wall score</div><div class="value" id="val-wall">&mdash;</div></div>
    <div class="score-cell wound"><div class="label">wound score</div><div class="value" id="val-wound">&mdash;</div></div>
    <div class="score-cell wound-off"><div class="label">off-wound lumen</div><div class="value" id="val-wound-off">&mdash;</div></div>
    <div class="score-cell off"><div class="label">off-wall score</div><div class="value" id="val-off">&mdash;</div></div>
  </div>
  <p class="section-note" style="margin-top:-0.6rem">
    <b>Wound score</b> is domain-restricted to the wound region (boundary + lumen out to
    8 mesh-graph hops), not the raw wound boundary alone &mdash; that boundary is 100% GT
    clot on every vessel, so scoring it directly can't tell a model from the ungated law.
    <b>Off-wound lumen</b> is the off-boundary subset of that same region: clot the wound
    pushed into the flow, scored separately from healthy-wall off-wall behaviour.
  </p>

  <div class="transport">
    <button class="play-btn" id="play-btn">&#9654; Play</button>
    <button class="reset-btn" id="reset-zoom-btn" title="Reset zoom/pan (or double-click either window)">&#8635; Reset view</button>
    <input type="range" id="frame-slider" min="0" max="12" value="0" step="1" />
    <div class="time-readout" id="time-readout">t = 0 s <b>(0%)</b></div>
  </div>

  <div class="score-row">
    <div class="panel-box">
      <h2>Wall score over time</h2>
      <canvas id="chart-wall" class="chart-canvas" width="480" height="360"></canvas>
    </div>
    <div class="panel-box">
      <h2>Wound score over time</h2>
      <canvas id="chart-wound" class="chart-canvas" width="480" height="360"></canvas>
    </div>
    <div class="panel-box">
      <h2>Off-wound lumen over time</h2>
      <canvas id="chart-wound-off" class="chart-canvas" width="480" height="360"></canvas>
    </div>
    <div class="panel-box">
      <h2>Off-wall score over time</h2>
      <canvas id="chart-off" class="chart-canvas" width="480" height="360"></canvas>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch model"></span> model</div>
    <div class="legend-item"><span class="swatch gt"></span> ground truth</div>
    <div class="legend-item">&#9679; healthy wall</div>
    <div class="legend-item"><span class="ring-swatch" style="border-color:var(--site-zone); border-style:dashed; background:var(--site-zone); opacity:1"></span> wound site (always shown)</div>
    <div class="legend-item"><span class="swatch di" style="background:var(--score-wound)"></span> wound, clotted</div>
    <div class="legend-item">&#9632; off-wall lumen</div>
    <div class="legend-item"><span class="grad-swatch grad-model"></span> depth into lumen (model)</div>
    <div class="legend-item"><span class="grad-swatch grad-gt"></span> depth into lumen (GT)</div>
    <div class="legend-item">gaps in a chart = no GT clot in that domain at that time</div>
  </div>

  <h2 class="section">What to read off this</h2>
  <div class="finding-box">
    Watch the <b>diamonds</b> &mdash; that's the wound collar, and it's the only domain
    where the complement changes anything. Early frames should show the wound diamonds
    lighting up together (or close to it) in Ground Truth, then Model's diamonds should
    follow at roughly the right pace, not all at once. If Model's wound score chart is flat
    near 1.0 for most of the run, the timing scalars are doing their job; if it lags GT
    badly on one vessel, that vessel is absorbing more of the LOVO error than the others
    &mdash; check whether it's <code>wound_patient003</code> before concluding anything, since
    that vessel is already a known outlier for reasons unrelated to the wound mechanism
    itself.
  </div>

  <p class="foot-note">
    Model: <code>src/clot_ml/v0.py::load_v0_bundle</code> +
    <code>predict_clot_ml_v0</code>. The locked <code>clot_ml_v0</code> manifest identifies
    this release as behaviorally identical to <code>clot_gnn_v6w</code>; its configured
    baseline path uses the GNN plus wound complement, with no chemistry-ODE replacement
    branch. GT from
    <code>gt_clot_phi_at_time</code> at every real simulated timestep. Scores from
    <code>src.clot_ml.evaluate.domain_score</code> (the same
    <code>compute_clot_relaxed_metrics</code> + <code>clot_score_from_deploy_dict</code>
    convention used throughout this project), domain-restricted to wall / wound region /
    off-wound lumen / off-wall (<code>wound_region_masks</code> in
    <code>src/clot_ml/wound.py</code>).
    Full detail: <code>docs/WOUND_PROGRESS.md</code>.
  </p>
</main>

<script id="viz-data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('viz-data').textContent);
  const LABELS = __LABELS__;
  const order = Object.keys(DATA);
  let vessel = "__FIRST__";
  let frame = 0;
  let playing = false;
  let timer = null;

  const canvasModel = document.getElementById('canvas-model');
  const canvasGt = document.getElementById('canvas-gt');
  const cmModel = canvasModel.getContext('2d');
  const cmGt = canvasGt.getContext('2d');
  const wallChart = document.getElementById('chart-wall');
  const woundChart = document.getElementById('chart-wound');
  const woundOffChart = document.getElementById('chart-wound-off');
  const offChart = document.getElementById('chart-off');
  const wctx = wallChart.getContext('2d');
  const wdctx = woundChart.getContext('2d');
  const wodctx = woundOffChart.getContext('2d');
  const octx = offChart.getContext('2d');
  const slider = document.getElementById('frame-slider');
  const readout = document.getElementById('time-readout');
  const playBtn = document.getElementById('play-btn');
  const resetZoomBtn = document.getElementById('reset-zoom-btn');
  const zoomReadout = document.getElementById('zoom-readout');
  const valWall = document.getElementById('val-wall');
  const valWound = document.getElementById('val-wound');
  const valWoundOff = document.getElementById('val-wound-off');
  const valOff = document.getElementById('val-off');

  const view = { k: 1, panX: 0, panY: 0 };
  const MIN_K = 1, MAX_K = 25;

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function bbox(pts) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    return [x0, x1, y0, y1];
  }

  function canvasPoint(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const cx = (evt.clientX !== undefined ? evt.clientX : evt.touches[0].clientX);
    const cy = (evt.clientY !== undefined ? evt.clientY : evt.touches[0].clientY);
    return { x: (cx - rect.left) * (canvas.width / rect.width), y: (cy - rect.top) * (canvas.height / rect.height) };
  }

  function resetView() {
    view.k = 1; view.panX = 0; view.panY = 0;
    zoomReadout.textContent = '1.0×';
    redraw();
  }

  function zoomAt(canvas, screenX, screenY, factor) {
    const newK = Math.max(MIN_K, Math.min(MAX_K, view.k * factor));
    const worldX = (screenX - view.panX) / view.k;
    const worldY = (screenY - view.panY) / view.k;
    view.panX = screenX - worldX * newK;
    view.panY = screenY - worldY * newK;
    view.k = newK;
    zoomReadout.textContent = view.k.toFixed(1) + '×';
    redraw();
  }

  function attachZoomPan(canvas) {
    canvas.addEventListener('wheel', (evt) => {
      evt.preventDefault();
      const p = canvasPoint(canvas, evt);
      zoomAt(canvas, p.x, p.y, evt.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });

    let dragging = false, start = null;
    canvas.addEventListener('mousedown', (evt) => {
      dragging = true;
      start = { x: evt.clientX, y: evt.clientY, panX: view.panX, panY: view.panY };
    });
    window.addEventListener('mousemove', (evt) => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
      view.panX = start.panX + (evt.clientX - start.x) * scaleX;
      view.panY = start.panY + (evt.clientY - start.y) * scaleY;
      redraw();
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    canvas.addEventListener('dblclick', () => resetView());

    let pinchDist = null;
    canvas.addEventListener('touchstart', (evt) => {
      if (evt.touches.length === 1) {
        dragging = true;
        start = { x: evt.touches[0].clientX, y: evt.touches[0].clientY, panX: view.panX, panY: view.panY };
      } else if (evt.touches.length === 2) {
        dragging = false;
        const [a, b] = evt.touches;
        pinchDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      }
    }, { passive: true });
    canvas.addEventListener('touchmove', (evt) => {
      if (evt.touches.length === 1 && dragging) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
        view.panX = start.panX + (evt.touches[0].clientX - start.x) * scaleX;
        view.panY = start.panY + (evt.touches[0].clientY - start.y) * scaleY;
        redraw();
      } else if (evt.touches.length === 2 && pinchDist !== null) {
        const [a, b] = evt.touches;
        const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        const mid = canvasPoint(canvas, { clientX: (a.clientX + b.clientX) / 2, clientY: (a.clientY + b.clientY) / 2 });
        zoomAt(canvas, mid.x, mid.y, d / pinchDist);
        pinchDist = d;
      }
    }, { passive: true });
    canvas.addEventListener('touchend', () => { dragging = false; pinchDist = null; });
  }

  function hexToRgb(hex) {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function lerpColor(hexNear, hexFar, t) {
    const a = hexToRgb(hexNear), b = hexToRgb(hexFar);
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' + Math.round(a[2] + (b[2] - a[2]) * t) + ')';
  }

  function drawDiamond(ctx, x, y, r) {
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
    ctx.closePath(); ctx.fill();
  }

  // Andrew's monotone chain -- convex hull of the wound points, so the wound *site* can be
  // drawn as a single soft region rather than a scatter of per-node rings (which reads too
  // much like a field of already-clotted dots).
  function convexHull(points) {
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (pts.length <= 2) return pts;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }

  function drawWindow(ctx, canvas, wallHot, woundHot, lumenHot, nearColor, farColor, woundColor) {
    const d = DATA[vessel];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);

    const all = d.bg.concat(d.wall_pos).concat(d.wound_pos).concat(d.lumen_pos);
    const [x0, x1, y0, y1] = bbox(all);
    const pad = 24;
    const sx = (w - 2 * pad) / Math.max(x1 - x0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(y1 - y0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = pad + ((w - 2 * pad) - s * (x1 - x0)) / 2;
    const oy = pad + ((h - 2 * pad) - s * (y1 - y0)) / 2;
    function px(x) { return (ox + (x - x0) * s) * view.k + view.panX; }
    function py(y) { return (oy + (y1 - y) * s) * view.k + view.panY; }

    // Wound site: a single soft region behind everything else, so it reads as "this is
    // where the wound is" -- a landmark, not a scatter of already-clotted-looking dots.
    if (d.wound_pos.length >= 3) {
      const hull = convexHull(d.wound_pos.map(([x, y]) => [px(x), py(y)]));
      const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
      const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
      const pad = 10;
      ctx.beginPath();
      hull.forEach(([x, y], i) => {
        const dx = x - cx, dy = y - cy;
        const dist = Math.hypot(dx, dy) || 1;
        const ex = x + (dx / dist) * pad, ey = y + (dy / dist) * pad;
        if (i === 0) ctx.moveTo(ex, ey); else ctx.lineTo(ex, ey);
      });
      ctx.closePath();
      ctx.fillStyle = css('--site-zone');
      ctx.globalAlpha = 0.09;
      ctx.fill();
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = css('--site-zone');
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    ctx.fillStyle = css('--muted');
    ctx.globalAlpha = 0.22;
    for (const [x, y] of d.bg) { ctx.beginPath(); ctx.arc(px(x), py(y), 1.1, 0, Math.PI * 2); ctx.fill(); }

    ctx.globalAlpha = 1;
    ctx.fillStyle = nearColor;
    for (let i = 0; i < d.wall_pos.length; i++) {
      if (!wallHot[i]) continue;
      const [x, y] = d.wall_pos[i];
      ctx.beginPath(); ctx.arc(px(x), py(y), 2.9, 0, Math.PI * 2); ctx.fill();
    }
    for (let i = 0; i < d.lumen_pos.length; i++) {
      if (!lumenHot[i]) continue;
      const [x, y] = d.lumen_pos[i];
      const r = 2.9;
      ctx.fillStyle = lerpColor(nearColor, farColor, d.lumen_dist[i]);
      ctx.fillRect(px(x) - r, py(y) - r, r * 2, r * 2);
    }
    ctx.fillStyle = woundColor;
    ctx.strokeStyle = css('--surface');
    ctx.lineWidth = 1;
    for (let i = 0; i < d.wound_pos.length; i++) {
      if (!woundHot[i]) continue;
      const [x, y] = d.wound_pos[i];
      drawDiamond(ctx, px(x), py(y), 4.6);
      ctx.beginPath();
      ctx.moveTo(px(x), py(y) - 4.6); ctx.lineTo(px(x) + 4.6, py(y)); ctx.lineTo(px(x), py(y) + 4.6); ctx.lineTo(px(x) - 4.6, py(y));
      ctx.closePath(); ctx.stroke();
    }
  }

  function drawScoreChart(ctx, canvas, series, tSeries, tFinal, curT, color) {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);
    const padL = 32, padR = 12, padT = 10, padB = 26;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    function px(t) { return padL + (t / tFinal) * plotW; }
    function py(v) { return padT + plotH - Math.max(0, Math.min(1, v)) * plotH; }

    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1; ctx.font = '9px ' + css('--mono'); ctx.fillStyle = css('--muted');
    for (let i = 0; i <= 4; i++) {
      const v = i / 4, y = py(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), 2, y + 3);
    }
    for (let i = 0; i <= 3; i++) {
      const tv = (tFinal / 3) * i, x = px(tv);
      ctx.fillText(Math.round(tv / 1000) + 'k', x - 8, h - padB + 13);
    }

    ctx.strokeStyle = color; ctx.lineWidth = 2;
    let started = false;
    for (let i = 0; i < tSeries.length; i++) {
      const v = series[i];
      if (v === null || v === undefined) { started = false; continue; }
      const x = px(tSeries[i]), y = py(v);
      if (!started) { ctx.beginPath(); ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const cx = px(curT);
    ctx.strokeStyle = css('--ink'); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function scoreAtTime(scoreT, series, t) {
    let best = null, bestDist = Infinity;
    for (let i = 0; i < scoreT.length; i++) {
      if (series[i] === null || series[i] === undefined) continue;
      const dist = Math.abs(scoreT[i] - t);
      if (dist < bestDist) { bestDist = dist; best = series[i]; }
    }
    return best;
  }

  function fmt(v) { return v === null ? 'n/a' : v.toFixed(3); }

  function updateReadout() {
    const d = DATA[vessel];
    const t = d.frame_t[frame];
    const pct = Math.round((t / d.t_final) * 100);
    readout.innerHTML = 't = ' + Math.round(t) + ' s <b>(' + pct + '%)</b>';
    valWall.textContent = fmt(scoreAtTime(d.score_t, d.score_wall, t));
    valWound.textContent = fmt(scoreAtTime(d.score_t, d.score_wound, t));
    valWoundOff.textContent = fmt(scoreAtTime(d.score_t, d.score_wound_off, t));
    valOff.textContent = fmt(scoreAtTime(d.score_t, d.score_off, t));
  }

  function redraw() {
    const d = DATA[vessel];
    drawWindow(cmModel, canvasModel, d.frame_model_wall[frame], d.frame_model_wound[frame], d.frame_model_lumen[frame], css('--model-c'), css('--model-far'), css('--model-c'));
    drawWindow(cmGt, canvasGt, d.frame_gt_wall[frame], d.frame_gt_wound[frame], d.frame_gt_lumen[frame], css('--gt-c'), css('--gt-far'), css('--gt-c'));
    drawScoreChart(wctx, wallChart, d.score_wall, d.score_t, d.t_final, d.frame_t[frame], css('--score-wall'));
    drawScoreChart(wdctx, woundChart, d.score_wound, d.score_t, d.t_final, d.frame_t[frame], css('--score-wound'));
    drawScoreChart(wodctx, woundOffChart, d.score_wound_off, d.score_t, d.t_final, d.frame_t[frame], css('--score-wound-off'));
    drawScoreChart(octx, offChart, d.score_off, d.score_t, d.t_final, d.frame_t[frame], css('--score-off'));
    updateReadout();
  }

  function setVessel(v) {
    vessel = v;
    frame = 0;
    slider.value = 0;
    view.k = 1; view.panX = 0; view.panY = 0;
    zoomReadout.textContent = '1.0×';
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.vessel === v));
    redraw();
  }

  attachZoomPan(canvasModel);
  attachZoomPan(canvasGt);
  resetZoomBtn.addEventListener('click', resetView);

  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => { stopPlay(); setVessel(t.dataset.vessel); });
  });

  slider.addEventListener('input', () => { stopPlay(); frame = parseInt(slider.value, 10); redraw(); });

  function stopPlay() {
    playing = false; playBtn.innerHTML = '&#9654; Play';
    if (timer) { clearInterval(timer); timer = null; }
  }

  playBtn.addEventListener('click', () => {
    if (playing) { stopPlay(); return; }
    playing = true; playBtn.innerHTML = '&#10074;&#10074; Pause';
    timer = setInterval(() => { frame = (frame + 1) % 13; slider.value = frame; redraw(); }, 750);
  });

  window.addEventListener('resize', redraw);
  setVessel(vessel);
})();
</script>
"""

if __name__ == "__main__":
    main()
