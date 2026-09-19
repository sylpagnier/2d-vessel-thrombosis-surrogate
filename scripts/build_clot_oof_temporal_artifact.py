"""Build the standard two-window temporal viz from a clot_ml_0 OOF JSON payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_PATH = Path("outputs/clot_ml_0_temporal_data.json")
OUT_PATH = Path("outputs/clot_ml_0_oof_temporal_report.html")

CLASS_COLOR = {"aneurysm": "badge-aneurysm", "stenosis": "badge-stenosis",
              "baseline": "badge-baseline", "unknown": "badge-baseline"}
CLASS_TEXT = {"aneurysm": "ANEURYSM", "stenosis": "STENOSIS",
             "baseline": "BASELINE", "unknown": "GEOM N/A"}
CLASS_NAME = {"aneurysm": "aneurysm", "stenosis": "stenosis",
             "baseline": "baseline geometry", "unknown": "geometry unclassified"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA_PATH),
                    help="VIZ_STANDARD JSON payload (default: clot_ml_0 combined OOF data)")
    ap.add_argument("--out", default=str(OUT_PATH), help="HTML report path")
    args = ap.parse_args()
    data_path, out_path = Path(args.data), Path(args.out)
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    meta = payload.pop("_meta", {})
    data_json = json.dumps(payload).replace("</", "<\\/")
    order = list(payload.keys())
    model_name = str(meta.get("model") or payload[order[0]].get("model", "clot_ml_0"))
    mode_vessels = {str(k): list(v) for k, v in (meta.get("mode_vessels") or {}).items()}
    if not mode_vessels:
        mode_vessels = {str(meta.get("mode", "all")): order}
    mode_labels = {str(k): str(v) for k, v in (meta.get("mode_labels") or {}).items()}
    mode_labels = {k: mode_labels.get(k, k) for k in mode_vessels}
    default_mode = str(meta.get("default_mode", next(iter(mode_vessels))))
    if default_mode not in mode_vessels:
        default_mode = next(iter(mode_vessels))
    mode_options = "".join(
        f'<option value="{k}"{" selected" if k == default_mode else ""}>{mode_labels[k]}</option>'
        for k in mode_vessels
    ) if len(mode_vessels) > 1 else ""
    mode_meta_json = json.dumps({"vessels": mode_vessels, "labels": mode_labels,
                                 "default": default_mode})

    # Held-out status is read off training_pool per-model (gen_v4_temporal_data.py), not
    # hardcoded here -- clot_gnn_v3/v4/v4w never trained on VIZ_HALF, but v5/v5w did
    # (docs/SEALED_SPLIT.md, 2026-08-22 amendment), so the same 7 vessels can be validation
    # evidence for one model generation and in-sample for the next.
    def vessel_label(v):
        p = payload[v]
        cls = CLASS_NAME[p["geom_class"]]
        if p.get("out_of_fold"):
            return (f"{v} — OUT-OF-FOLD (fold {p.get('fold', '?')}), {cls}; "
                    f"base model and temporal readout excluded this vessel")
        if p.get("sealed_viz"):
            return f"{v} — VALIDATION (held out), {cls}, never trained on by {model_name}"
        if p.get("was_viz_half"):
            return f"{v} — in-sample for {model_name} (released from held-out 2026-08-22), {cls}"
        return f"{v} — {cls}, training pool (in-sample)"

    labels_json = json.dumps({v: vessel_label(v) for v in order})
    options = "\n".join(
        f'<option value="{v}">{v} · fold {payload[v].get("fold", "?")} · '
        f'{CLASS_NAME[payload[v]["geom_class"]]}</option>'
        for v in order
    )

    def badge(v):
        cls = payload[v]["geom_class"]
        b = f'<span class="tab-badge {CLASS_COLOR[cls]}">{CLASS_TEXT[cls]}</span>'
        if payload[v].get("out_of_fold"):
            b = '<span class="tab-badge badge-sealed">OOF</span>' + b
        elif payload[v].get("sealed_viz"):
            b = '<span class="tab-badge badge-sealed">VALIDATION</span>' + b
        elif payload[v].get("was_viz_half"):
            b = '<span class="tab-badge badge-outlier">IN-SAMPLE NOW</span>' + b
        return b

    tabs = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-vessel="{v}">{v}{badge(v)}</button>'
        for i, v in enumerate(order)
    )

    held_out = [v for v in order if payload[v].get("out_of_fold") or payload[v].get("sealed_viz")]
    ex_viz_half = [v for v in order if payload[v].get("was_viz_half") and not payload[v].get("sealed_viz")]
    in_sample = [v for v in order if not payload[v].get("out_of_fold")
                 and not payload[v].get("sealed_viz") and not payload[v].get("was_viz_half")]

    if any(payload[v].get("out_of_fold") for v in held_out):
        callout = (
            '<div class="callout">'
            f'<b><span class="tab-badge badge-sealed" style="margin-left:0">OOF</span> '
            f'Every displayed vessel is an outer-fold generalization prediction.</b> Its '
            'GNN member weights and its temporal readout were fitted without that vessel; '
            'the split is embedded in the input payload. FINAL_HALF is excluded by assertion '
            'and is not opened for this report.</div>'
        )
    elif held_out:
        callout = (
            '<div class="callout">'
            f'<b><span class="tab-badge badge-sealed" style="margin-left:0">VALIDATION</span> '
            f'{" / ".join(held_out)} {"is" if len(held_out) == 1 else "are"} genuinely '
            f'never-trained-on.</b> Per <code>docs/SEALED_SPLIT.md</code>, these sit outside '
            f'<code>{model_name}</code>\'s own training pool (read straight off its manifest, '
            f'not a hardcoded list) -- the closest thing to a real generalization test this '
            f'page can show for this exact model, though at n={len(held_out)} it is not a '
            'substitute for the CV estimate below.</div>'
        )
    else:
        callout = (
            '<div class="callout"><b>No genuinely held-out vessel is shown for '
            f'<code>{model_name}</code>.</b> Per <code>docs/SEALED_SPLIT.md</code>\'s '
            '2026-08-22 amendment, <code>comsol001/010/014/042</code> (VIZ_HALF) moved from '
            'held-out into the training pool for this model generation -- they were the only '
            'genuinely held-out evidence the earlier <code>clot_gnn_v4</code>/<code>v4w</code> '
            'viz had. <code>comsol007/013/031/043</code> (FINAL_HALF) remain sealed by '
            'project policy, reserved for the one true final read, and are deliberately not '
            'opened here. The strict, nested CV number below is the real generalization '
            f'estimate for {model_name}.</div>'
        )
    if ex_viz_half:
        callout += (
            '<div class="callout"><b><span class="tab-badge badge-outlier" '
            f'style="margin-left:0">IN-SAMPLE NOW</span> {" / ".join(ex_viz_half)} used to be '
            f'held-out.</b> They were VIZ_HALF -- genuinely never-trained-on for '
            '<code>clot_gnn_v3</code>/<code>v4</code>/<code>v4w</code> -- but were released '
            f'into the training pool on 2026-08-22, and <code>{model_name}</code> was trained '
            'after that date, so it has seen them. Shown for continuity with the earlier '
            'report, not as validation evidence.</div>'
        )
    if in_sample:
        callout += (
            f'<div class="callout"><b>The remaining {len(in_sample)} tab'
            f'{"s are" if len(in_sample) != 1 else " is"} in-sample</b> for '
            f'<code>{model_name}</code>: {" / ".join(in_sample)}. Badged by geometry class '
            'only. Scroll down for the CV number that actually estimates generalization for '
            'this group.</div>'
        )

    off_label = str(meta.get("off_label", "off-wall"))
    html = TEMPLATE.replace("__DATA__", data_json).replace("__LABELS__", labels_json).replace(
        "<!--TABS-->", tabs
    ).replace("__FIRST__", order[0]).replace("__MODEL__", model_name).replace(
        "<!--HELDOUT_CALLOUT-->", callout
    ).replace("__OPTIONS__", options).replace("__OOF_N__", str(len(order))).replace(
        "__FLOW__", str(meta.get("flow", "unknown")).upper()
    ).replace("__OFF_LABEL__", off_label).replace("__MODE_OPTIONS__", mode_options).replace(
        "__MODE_META__", mode_meta_json).replace("__DEFAULT_MODE__", default_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


LEGACY_TEMPLATE = r"""<title>Phase 10 &mdash; __MODEL__ vs Ground Truth</title>
<style>
:root {
  --bg: #f6f4f0; --surface: #ffffff; --surface-2: #efeae2; --ink: #241f18; --muted: #6b6255;
  --line: #e2dbcd; --accent: #a8681a; --accent-ink: #ffffff;
  --model-c: #a8681a; --gt-c: #2b6f6b;
  --model-far: #ecd9bb; --gt-far: #b9dbd8;
  --score-wall: #a8681a; --score-off: #2b6f6b;
  --shadow: 0 1px 2px rgba(36,31,24,0.06), 0 8px 24px -12px rgba(36,31,24,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17140f; --surface: #201b14; --surface-2: #2a2318; --ink: #f1ece0; --muted: #b0a48d;
    --line: #382f21; --accent: #e0a03e; --accent-ink: #1a1206;
    --model-c: #e0a03e; --gt-c: #5fbcb6;
    --model-far: #5c481f; --gt-far: #1f4d49;
    --score-wall: #e0a03e; --score-off: #5fbcb6;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #17140f; --surface: #201b14; --surface-2: #2a2318; --ink: #f1ece0; --muted: #b0a48d;
  --line: #382f21; --accent: #e0a03e; --accent-ink: #1a1206;
  --model-c: #e0a03e; --gt-c: #5fbcb6;
  --model-far: #5c481f; --gt-far: #1f4d49;
  --score-wall: #e0a03e; --score-off: #5fbcb6;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f6f4f0; --surface: #ffffff; --surface-2: #efeae2; --ink: #241f18; --muted: #6b6255;
  --line: #e2dbcd; --accent: #a8681a; --accent-ink: #ffffff;
  --model-c: #a8681a; --gt-c: #2b6f6b;
  --model-far: #ecd9bb; --gt-far: #b9dbd8;
  --score-wall: #a8681a; --score-off: #2b6f6b;
  --shadow: 0 1px 2px rgba(36,31,24,0.06), 0 8px 24px -12px rgba(36,31,24,0.18);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1220px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }

.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.7rem, 3.1vw, 2.4rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 32ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 70ch; margin: 0 0 1.1rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }

.callout { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--model-c); border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 0.87rem; color: var(--ink); max-width: 76ch; margin: 0 0 1rem; }
.callout b { color: var(--model-c); }
.callout + .callout { border-left-color: var(--gt-c); margin-bottom: 2rem; }
.callout + .callout b { color: var(--gt-c); }

.tabs { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }
.tab { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.7rem 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s; display: inline-flex; align-items: center; gap: 0.5rem; }
.tab:hover { border-color: var(--accent); color: var(--ink); }
.tab.active { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
.tab-badge { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.04em; padding: 0.12rem 0.4rem; border-radius: 4px; margin-left: 0.3rem; }
.tab-badge:first-of-type { margin-left: 0; }
.badge-sealed { background: #c9a227; color: #241f18; }
.badge-outlier { background: #6b5a52; color: #ffffff; }
.badge-aneurysm { background: #8a4fc7; color: #ffffff; }
.badge-stenosis { background: #c7504f; color: #ffffff; }
.badge-baseline { background: var(--surface-2); color: var(--muted); }
.tab.active .badge-baseline { background: rgba(255,255,255,0.3); color: var(--accent-ink); }
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
.panel-meta { display: flex; align-items: center; gap: 0.6rem; }
.zoom-hint { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); white-space: nowrap; }
.score-badge { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.5rem; font-weight: 700; padding: 0.35rem 0.85rem; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line); white-space: nowrap; letter-spacing: -0.01em; }
.score-badge.wall { color: var(--score-wall); }
.score-badge.off { color: var(--score-off); }
.score-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-bottom: 1rem; }
.chart-canvas { aspect-ratio: 16 / 9; }

.transport { display: flex; align-items: center; gap: 0.9rem; margin: 0.9rem 0 1.8rem; }
.play-btn { font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); cursor: pointer; flex: none; min-width: 4.6rem; }
.play-btn:hover { filter: brightness(1.06); }
.reset-btn { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.8rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; flex: none; }
.reset-btn:hover { border-color: var(--accent); color: var(--ink); }
input[type="range"] { flex: 1; accent-color: var(--accent); }
.time-readout { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.76rem; color: var(--muted); min-width: 9rem; text-align: right; flex: none; }
.time-readout b { color: var(--ink); }

.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 2.4rem; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 0.45rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.model { background: var(--model-c); } .swatch.gt { background: var(--gt-c); }
.grad-swatch { width: 34px; height: 11px; border-radius: 3px; flex: none; }
.grad-swatch.grad-model { background: linear-gradient(90deg, var(--model-c), var(--model-far)); }
.grad-swatch.grad-gt { background: linear-gradient(90deg, var(--gt-c), var(--gt-far)); }
.lineswatch { width: 18px; height: 0; border-top: 2px solid var(--score-wall); }
.lineswatch.off { border-top-color: var(--score-off); }

h2.section { font-family: var(--serif); font-weight: 600; font-size: 1.3rem; margin: 2.6rem 0 0.4rem; }
.section-note { color: var(--muted); font-size: 0.92rem; max-width: 70ch; margin: 0 0 1.2rem; }
.finding-box { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--gt-c); border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.9rem; color: var(--ink); max-width: 76ch; }
.finding-box b { color: var(--gt-c); }
.compare-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.82rem; margin: 1rem 0; }
.compare-table th, .compare-table td { text-align: right; padding: 0.4rem 0.7rem; border-bottom: 1px solid var(--line); }
.compare-table th:first-child, .compare-table td:first-child { text-align: left; }
.compare-table th { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
.compare-table td.win { color: var(--gain, #1f8a4c); font-weight: 600; }
.compare-table td.trust { color: var(--model-c); font-weight: 700; }
.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 74ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }

@media (max-width: 760px) { .spatial-row, .score-row { grid-template-columns: 1fr; } }
</style>

<main>
  <p class="eyebrow">Phase 10 &middot; __MODEL__ &middot; model vs ground truth</p>
  <h1>The stricter protocol, and what survives it.</h1>
  <p class="lede">
    <code>clot_gnn_v4</code> keeps v3's "time is a direct model input" design and adds a
    live advective-transport feature (COMSOL's own advection operator, solved fresh on the
    mesh for every query time) plus a properly nested evaluation protocol
    (<code>docs/PHASE10_V4.md</code>) after three selection leaks were found in how v3 was
    scored. Both windows below animate from the model's own per-timestep output;
    predictions are forced monotone in time and an off-wall node can never fire before its
    own wall owner does. These vessels carry no wound mask, so <code>__MODEL__</code> (the
    currently shipped model) runs its base ensemble here unchanged &mdash; any wound
    complement on top is inert by construction.
  </p>

  <!--HELDOUT_CALLOUT-->

  <h2 class="section">The number that matters: strict, nested, geometry-stratified 5-fold CV</h2>
  <p class="section-note">
    <code>docs/PHASE10_V4.md</code> &sect;1 found v3's published numbers were ~0.02
    optimistic from three selection leaks, and fixed the protocol so every readout scalar
    is chosen only on out-of-fold scores of vessels outside the held-out fold
    (<code>scripts/eval_strict.py</code>). This is the number that selected v4's design,
    and the cohort's own noise floor is reported alongside it rather than left implicit.
  </p>
  <div style="overflow-x:auto">
  <table class="compare-table">
    <thead><tr><th>FINAL-TIME &mdash; the fully-formed clot</th><th>wall</th><th>off-wall</th></tr></thead>
    <tbody>
      <tr><td>physics backbone</td><td>0.8766</td><td>0.4141</td></tr>
      <tr><td>v3, same strict protocol</td><td>0.9014</td><td>0.7011</td></tr>
      <tr><td class="trust">v4, strict CV, all 19 &mdash; <b>the comparator for the two rows below</b></td><td class="trust">0.9176</td><td class="trust">0.7366</td></tr>
      <tr><td>v4, strict CV, per-vessel spread (median / 10th pct / min)</td><td>0.959 / 0.821 / 0.613</td><td>0.772 / 0.496 / 0.267</td></tr>
      <tr><td><b>v4, comsol042 (SEALED, stenosis)</b></td><td><b>0.716</b></td><td><b>0.696</b></td></tr>
      <tr><td><b>v4, comsol001 (SEALED)</b></td><td><b>0.741</b></td><td><b>0.785</b></td></tr>
      <tr><td>v4, training-pool tabs below (in-sample, for contrast)</td><td>~0.96</td><td>~0.93</td></tr>
    </tbody>
    <thead><tr><th>MEAN-OVER-TIME &mdash; averaged across the run</th><th>wall</th><th>off-wall</th></tr></thead>
    <tbody>
      <tr><td>v3, same strict protocol</td><td>0.8687</td><td>0.6389</td></tr>
      <tr><td class="trust">v4, strict CV, all 19 &mdash; the design-selection number</td><td class="trust">0.8750</td><td class="trust">0.7188</td></tr>
      <tr><td class="trust">v4, strict CV, priority class (n=3)</td><td class="trust">0.8798</td><td class="trust">0.8411</td></tr>
      <tr><td>v4, strict CV, baseline class (n=16)</td><td>0.8741</td><td>0.6821</td></tr>
      <tr><td>oracle timing, same set (ceiling)</td><td>0.9662</td><td>0.8709</td></tr>
    </tbody>
  </table>
  </div>
  <p class="section-note">
    The two SEALED rows are <b>final-time</b> numbers and are printed against the
    <b>final-time</b> CV row, not the mean-over-time one. An earlier version of this page
    compared them across metrics, which made the off-wall result look worse than it is.
  </p>
  <p class="section-note">
    <b>The off-wall gain over v3 is statistically significant; the wall gain is not.</b>
    Paired vessel bootstrap: off-wall +0.0685 [+0.0281, +0.1116], P(diff&le;0) = 0.000; wall
    +0.0063 [&minus;0.0150, +0.0353], P(diff&le;0) = 0.342. The cohort's noise floor is
    &plusmn;0.024 wall / &plusmn;0.091 off-wall (three configs of the same arm spread that
    much) &mdash; most 0.01&ndash;0.03 differences at n=19 are not measurable, and v4's own
    wall gain sits inside that floor.
  </p>
  <p class="section-note">
    <b>Off-wall generalized as advertised; the wall did not.</b> Against the final-time CV
    row, the two held-out off-wall scores (0.696, 0.785 vs 0.7366) land at the 31st and
    54th percentile of the pool's own out-of-fold spread &mdash; exact rank test p = 0.40,
    i.e. indistinguishable from two ordinary draws. The two wall scores (0.716, 0.741 vs
    0.9176) rank 2nd and 4th of 21; <b>p = 0.029</b>, which is too low to attribute to
    draw variance alone.
  </p>
  <p class="section-note">
    <b>The wall shortfall is two vessels failing for unrelated reasons</b>
    (<code>scripts/diag_sealed_wall_readout.py</code>). On <code>comsol001</code> the
    network ranks correctly &mdash; a per-vessel oracle cut scores <b>0.980</b> against the
    shipped readout's 0.741 &mdash; but its mean wall score, 0.448, sits far outside the
    0.11&ndash;0.39 range the cohort cut was calibrated on, so a fixed cut commits 252
    nodes against 183 true ones. On <code>comsol042</code> the oracle cut itself only
    reaches <b>0.768</b>: no readout recovers it, the network simply does not separate
    that vessel. The first is extrapolation past the calibration range, the second is a
    plain model failure, and averaging them into one number hides both.
  </p>
  <p class="section-note">
    <b>A cohort mean is not a per-vessel promise.</b> The strict-CV wall mean has a
    standard error of 0.023, but the standard deviation of a <i>single new vessel</i> is
    0.103 &mdash; an 80% interval of [0.79, 1.00]. The 0.9176 mean is also not a typical
    vessel: the median is 0.959, and <code>comsol028</code> (0.613) and
    <code>comsol018</code> (0.731) carry the whole left tail. <code>comsol042</code>
    joins that tail rather than departing from it. The training-pool tabs below read far
    higher (~0.96/0.93) because they are in-sample &mdash; the gap between those numbers
    and these two is most of what "in-sample" actually costs you.
  </p>

  <p class="tab-group-label">tabs badged by geometry class, not held-out status &mdash; every vessel is in-sample</p>
  <div class="tabs">
    <!--TABS-->
  </div>

  <div class="spatial-row">
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--model-c)"></span>Model (__MODEL__)</h2>
        <div class="panel-meta">
          <span class="score-badge wall" id="val-wall">wall &mdash;</span>
          <span class="zoom-hint" id="zoom-readout">1.0&times;</span>
        </div>
      </div>
      <canvas id="canvas-model" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--gt-c)"></span>Ground truth</h2>
        <div class="panel-meta">
          <span class="score-badge off" id="val-off">off-wall &mdash;</span>
          <span class="zoom-hint">scroll/drag, synced</span>
        </div>
      </div>
      <canvas id="canvas-gt" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
  </div>

  <div class="transport">
    <button class="play-btn" id="play-btn">&#9654; Play</button>
    <button class="reset-btn" id="reset-zoom-btn" title="Reset zoom/pan (or double-click either window)">&#8635; Reset view</button>
    <input type="range" id="frame-slider" min="0" max="12" value="0" step="1" />
    <div class="time-readout" id="time-readout">t = 0 s <b>(0%)</b></div>
  </div>

  <div class="score-row">
    <div class="panel-box">
      <h2>Wall score over time</h2>
      <canvas id="chart-wall" class="chart-canvas" width="620" height="349"></canvas>
    </div>
    <div class="panel-box">
      <h2>Off-wall score over time</h2>
      <canvas id="chart-off" class="chart-canvas" width="620" height="349"></canvas>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch model"></span> model</div>
    <div class="legend-item"><span class="swatch gt"></span> ground truth</div>
    <div class="legend-item">&#9679; wall &nbsp; &#9632; lumen</div>
    <div class="legend-item"><span class="grad-swatch grad-model"></span> depth into lumen (model)</div>
    <div class="legend-item"><span class="grad-swatch grad-gt"></span> depth into lumen (GT)</div>
    <div class="legend-item"><span class="lineswatch"></span> wall score</div>
    <div class="legend-item"><span class="lineswatch off"></span> off-wall score</div>
  </div>

  <h2 class="section">What to read off this</h2>
  <div class="finding-box">
    Start with <code>comsol042</code> and <code>comsol001</code> &mdash; they're first
    for a reason, and they're the only two tabs whose score means anything about
    generalization on their own. Then flip to <code>comsol040</code>/<code>041</code>/<code>044</code>/<code>012</code>/<code>032</code>
    and notice how much cleaner the model/GT match looks &mdash; that visual gap <em>is</em>
    the in-sample effect, not a difference in vessel difficulty. Both panels genuinely
    move now, so watch the <em>Model</em> window's own front advance, not just Ground
    Truth's; where the two shapes diverge in time is a real read on the learned schedule.
    Off-wall is where v4 earned its place over v3 in CV (statistically significant), and
    off-wall is the domain that held up out of sample. On the wall, watch
    <code>comsol001</code> specifically: the model paints <em>more</em> clot than the
    ground truth, not less. That over-commitment is the cut landing in the wrong place on
    a vessel outside the range it was calibrated on, not the network mistaking where the
    clot is &mdash; the same score field, cut per-vessel, would read 0.980.
  </div>

  <p class="foot-note">
    Model: <code>src/clot_ml/locked.py::load_default</code> +
    <code>predict_default_series</code> (dispatches to <code>predict_temporal_v4</code>),
    reusing the locked <code>clot_gnn_v4</code> GNN ensemble plus a gradient-boosted
    temporal head with a live advective-transport feature, solved fresh per query time
    (<code>docs/PHASE10_V4.md</code> &sect;5/&sect;8b). GT from
    <code>gt_clot_phi_at_time</code> at every real simulated timestep. Scores from
    <code>compute_clot_relaxed_metrics</code> + <code>clot_score_from_deploy_dict</code>,
    domain-restricted to wall or off-wall -- computed fresh here, independent of the
    manifest's own strict-CV numbers shown above.
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
  const offChart = document.getElementById('chart-off');
  const wctx = wallChart.getContext('2d');
  const octx = offChart.getContext('2d');
  const slider = document.getElementById('frame-slider');
  const readout = document.getElementById('time-readout');
  const playBtn = document.getElementById('play-btn');
  const resetZoomBtn = document.getElementById('reset-zoom-btn');
  const zoomReadout = document.getElementById('zoom-readout');
  const valWall = document.getElementById('val-wall');
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

  function drawWindow(ctx, canvas, wallHot, lumenHot, nearColor, farColor) {
    const d = DATA[vessel];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);

    const all = d.bg.concat(d.wall_pos).concat(d.lumen_pos);
    const [x0, x1, y0, y1] = bbox(all);
    const pad = 24;
    const sx = (w - 2 * pad) / Math.max(x1 - x0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(y1 - y0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = pad + ((w - 2 * pad) - s * (x1 - x0)) / 2;
    const oy = pad + ((h - 2 * pad) - s * (y1 - y0)) / 2;
    function px(x) { return (ox + (x - x0) * s) * view.k + view.panX; }
    function py(y) { return (oy + (y1 - y) * s) * view.k + view.panY; }

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
    ctx.beginPath();
    for (let i = 0; i < tSeries.length; i++) {
      const x = px(tSeries[i]), y = py(series[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const cx = px(curT);
    ctx.strokeStyle = css('--ink'); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function scoreAtTime(scoreT, series, t) {
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < scoreT.length; i++) {
      const dist = Math.abs(scoreT[i] - t);
      if (dist < bestDist) { bestDist = dist; best = i; }
    }
    return series[best];
  }

  function updateReadout() {
    const d = DATA[vessel];
    const t = d.frame_t[frame];
    const pct = Math.round((t / d.t_final) * 100);
    readout.innerHTML = 't = ' + Math.round(t) + ' s <b>(' + pct + '%)</b>';
    const sw = scoreAtTime(d.score_t, d.score_wall, t);
    const so = scoreAtTime(d.score_t, d.score_offwall, t);
    valWall.textContent = 'wall ' + sw.toFixed(3);
    valOff.textContent = 'off-wall ' + so.toFixed(3);
  }

  function redraw() {
    const d = DATA[vessel];
    drawWindow(cmModel, canvasModel, d.frame_model_wall[frame], d.frame_model_lumen[frame], css('--model-c'), css('--model-far'));
    drawWindow(cmGt, canvasGt, d.frame_gt_wall[frame], d.frame_gt_lumen[frame], css('--gt-c'), css('--gt-far'));
    drawScoreChart(wctx, wallChart, d.score_wall, d.score_t, d.t_final, d.frame_t[frame], css('--score-wall'));
    drawScoreChart(octx, offChart, d.score_offwall, d.score_t, d.t_final, d.frame_t[frame], css('--score-off'));
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

TEMPLATE = r"""<title>__MODEL__ — OOF temporal generalization</title>
<style>
:root {
  --bg:#f6f4f0; --surface:#fff; --surface-2:#efeae2; --ink:#241f18; --muted:#6b6255;
  --line:#e2dbcd; --accent:#a8681a; --accent-ink:#fff;
  --model-c:#a8681a; --gt-c:#2b6f6b; --model-far:#ecd9bb; --gt-far:#b9dbd8;
  --score-wall:#a8681a; --score-off:#2b6f6b;
  --serif:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code","Roboto Mono",Consolas,monospace;
}
@media (prefers-color-scheme:dark) {
  :root { --bg:#17140f; --surface:#201b14; --surface-2:#2a2318; --ink:#f1ece0; --muted:#b0a48d;
    --line:#382f21; --accent:#e0a03e; --accent-ink:#1a1206; --model-c:#e0a03e;
    --gt-c:#5fbcb6; --model-far:#5c481f; --gt-far:#1f4d49; --score-wall:#e0a03e; --score-off:#5fbcb6; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); line-height:1.45; }
main { max-width:1220px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1,h2,p { margin-top:0; }
h1 { font-family:var(--serif); font-size:clamp(1.55rem,3vw,2.25rem); font-weight:600; line-height:1.15; margin-bottom:.35rem; }
h2 { font-family:var(--serif); font-size:1.05rem; font-weight:600; margin-bottom:.45rem; }
.eyebrow,.meta,.provenance { font-family:var(--mono); font-size:.72rem; color:var(--muted); }
.eyebrow { letter-spacing:.1em; text-transform:uppercase; color:var(--accent); margin-bottom:.55rem; }
.subtitle { color:var(--muted); margin-bottom:1.35rem; }
.stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin:0 0 1.25rem; }
.stat { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:.7rem .85rem; }
.stat label { display:block; color:var(--muted); font-size:.7rem; margin-bottom:.15rem; }
.stat strong { display:block; font:600 1.38rem var(--mono); letter-spacing:-.04em; }
.stat span { display:block; color:var(--muted); font: .67rem var(--mono); margin-top:.1rem; }
.cohort-grid,.spatial-row,.score-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.cohort-grid { margin:0 0 2rem; }
.chart-pane { min-width:0; }
.chart-pane h2,.panel-head { display:flex; justify-content:space-between; align-items:baseline; gap:.7rem; }
.chart-pane h2 span { color:var(--muted); font: .67rem var(--mono); }
.hist-canvas,.chart-canvas { width:100%; display:block; background:var(--surface); border:1px solid var(--line); border-radius:8px; }
.hist-canvas { aspect-ratio:16/7; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.85rem; min-width:0; }
.spatial-row { margin-bottom:.85rem; }
.spatial-canvas { width:100%; display:block; aspect-ratio:1; background:var(--surface-2); border-radius:6px; cursor:grab; touch-action:none; }
.spatial-canvas:active { cursor:grabbing; }
.panel-head { margin-bottom:.55rem; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.35rem; }
.score-pair { display:flex; gap:.45rem; align-items:center; flex-wrap:wrap; }
.score-badge { border:1px solid var(--line); border-radius:6px; padding:.28rem .52rem; font:600 .82rem var(--mono); white-space:nowrap; }
.score-badge.wall { color:var(--score-wall); }.score-badge.off { color:var(--score-off); }
.controls { display:flex; gap:.75rem; align-items:center; flex-wrap:wrap; margin:.85rem 0 1rem; }
.controls label { font: .76rem var(--mono); color:var(--muted); }
select,button,input[type=range] { accent-color:var(--accent); }
select { min-width:270px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--ink); padding:.42rem .55rem; font-family:var(--mono); font-size:.76rem; }
button { border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--ink); padding:.42rem .65rem; cursor:pointer; font-family:var(--mono); font-size:.76rem; }
button.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
input[type=range] { flex:1 1 240px; }
.time-readout { min-width:8.5rem; text-align:right; font: .74rem var(--mono); color:var(--muted); }
.time-readout b { color:var(--ink); }
.provenance { margin:.15rem 0 1rem; }
.legend { display:flex; gap:1rem; flex-wrap:wrap; color:var(--muted); font-size:.74rem; margin:0 0 1.3rem; }
.swatch { display:inline-block; width:10px; height:10px; margin-right:.3rem; vertical-align:-1px; }.swatch.wall{background:var(--score-wall)}.swatch.off{background:var(--score-off)}.swatch.model{background:var(--model-c)}.swatch.gt{background:var(--gt-c)}
.method { border-top:1px solid var(--line); padding-top:.85rem; color:var(--muted); font-size:.74rem; }
@media (max-width:760px) { .stat-grid { grid-template-columns:1fr 1fr; }.cohort-grid,.spatial-row,.score-row { grid-template-columns:1fr; } select { min-width:0; width:100%; }.controls { align-items:stretch; }.time-readout { text-align:left; } }
</style>
<main>
  <p class="eyebrow">__MODEL__ · outer-fold generalization</p>
  <h1>Generalization, vessel by vessel.</h1>
  <p class="subtitle"><b id="mode-summary">__OOF_N__ vessels</b> · GT flow at t=0 · FINAL_HALF excluded.</p>

  <section aria-label="OOF cohort performance">
    <div class="stat-grid">
      <div class="stat"><label>Final deploy · wall</label><strong id="final-wall-mean">—</strong><span id="final-wall-worst">—</span></div>
      <div class="stat"><label>Final deploy · __OFF_LABEL__</label><strong id="final-off-mean">—</strong><span id="final-off-worst">—</span></div>
      <div class="stat"><label>Mean over time · wall</label><strong id="mot-wall-mean">—</strong><span id="mot-wall-worst">—</span></div>
      <div class="stat"><label>Mean over time · __OFF_LABEL__</label><strong id="mot-off-mean">—</strong><span id="mot-off-worst">—</span></div>
      <div class="stat region-stat"><label>Final deploy · wound region</label><strong id="final-region-mean">—</strong><span id="final-region-worst">—</span></div>
      <div class="stat region-stat"><label>Mean over time · wound region</label><strong id="mot-region-mean">—</strong><span id="mot-region-worst">—</span></div>
    </div>
    <div class="cohort-grid">
      <section class="chart-pane"><h2>Final deploy score distribution <span>y = vessels</span></h2><canvas id="hist-final" class="hist-canvas" width="620" height="272" aria-label="Final deploy score distribution"></canvas></section>
      <section class="chart-pane"><h2>Mean-over-time distribution <span>y = vessels</span></h2><canvas id="hist-mot" class="hist-canvas" width="620" height="272" aria-label="Mean over time score distribution"></canvas></section>
    </div>
  </section>

  <div class="legend" aria-label="Legend"><span><i class="swatch wall"></i>wall score</span><span><i class="swatch off"></i>__OFF_LABEL__ score</span><span><i class="swatch model"></i>model</span><span><i class="swatch gt"></i>ground truth</span><span>● boundary &nbsp; ■ lumen; lighter squares = farther from wall</span></div>

  <section aria-label="Vessel viewer">
    <div class="controls">
      <label for="mode-select">View</label>
      <select id="mode-select">__MODE_OPTIONS__</select>
      <label for="vessel-select">Vessel</label>
      <select id="vessel-select">__OPTIONS__</select>
      <button id="play-btn" class="primary" type="button">Play</button>
      <button id="reset-zoom-btn" type="button">Reset view</button>
      <input id="frame-slider" type="range" min="0" max="12" value="0" step="1" aria-label="Simulation time" />
      <output id="time-readout" class="time-readout">t = —</output>
    </div>
    <p class="provenance" id="provenance"></p>
    <div class="spatial-row">
      <section class="panel"><div class="panel-head"><h2><i class="dot" style="background:var(--model-c)"></i>Model</h2><span class="score-pair"><span id="val-wall" class="score-badge wall">wall —</span><span id="val-off" class="score-badge off">__OFF_LABEL__ —</span></span></div><canvas id="canvas-model" class="spatial-canvas" width="560" height="560" aria-label="Model prediction spatial view"></canvas></section>
      <section class="panel"><div class="panel-head"><h2><i class="dot" style="background:var(--gt-c)"></i>Ground truth</h2><span class="meta" id="zoom-readout">1.0× · synced zoom</span></div><canvas id="canvas-gt" class="spatial-canvas" width="560" height="560" aria-label="Ground truth spatial view"></canvas></section>
    </div>
    <div class="score-row">
      <section class="chart-pane"><h2>Wall score over time</h2><canvas id="chart-wall" class="chart-canvas" width="620" height="349" aria-label="Wall deploy score over time"></canvas></section>
      <section class="chart-pane"><h2>__OFF_LABEL__ score over time</h2><canvas id="chart-off" class="chart-canvas" width="620" height="349" aria-label="__OFF_LABEL__ deploy score over time"></canvas></section>
    </div>
    <div class="score-row region-row">
      <section class="chart-pane"><h2>Wound-region score over time</h2><canvas id="chart-region" class="chart-canvas" width="620" height="349" aria-label="Wound-region deploy score over time"></canvas></section>
    </div>
  </section>
  <p class="method">Scores are domain-restricted canonical deploy scores. The temporal path is the strict OOF evaluation trajectory; it is shown to inspect timing, not to replace the final deployed readout.</p>
</main>
<script id="viz-data" type="application/json">__DATA__</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById('viz-data').textContent);
  const MODE = __MODE_META__;
  const allOrder = Object.keys(DATA);
  let mode = "__DEFAULT_MODE__";
  let order = (MODE.vessels[mode] || allOrder).slice();
  let vessel = order[0], frame = 0, playing = false, timer = null;
  const $ = (id) => document.getElementById(id);
  const canvasModel = $('canvas-model'), canvasGt = $('canvas-gt');
  const cmModel = canvasModel.getContext('2d'), cmGt = canvasGt.getContext('2d');
  const wallChart = $('chart-wall'), offChart = $('chart-off'), regionChart = $('chart-region');
  const wctx = wallChart.getContext('2d'), octx = offChart.getContext('2d'), rctx = regionChart.getContext('2d');
  const slider = $('frame-slider'), select = $('vessel-select'), modeSelect = $('mode-select'), playBtn = $('play-btn');
  const view = {k:1, panX:0, panY:0}; const MIN_K=1, MAX_K=25;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const mean = (xs) => xs.reduce((a,b)=>a+b,0) / Math.max(xs.length,1);
  const last = (xs) => xs[xs.length-1];
  let summary = [];
  function buildSummary() { summary = order.map(v => { const d=DATA[v], r=d.score_wound_region || []; return {v, finalWall:last(d.score_wall), finalOff:last(d.score_offwall), motWall:mean(d.score_wall), motOff:mean(d.score_offwall), finalRegion:r.length?last(r):null, motRegion:r.length?mean(r):null}; }); }
  function worst(key) { const valid=summary.filter(r=>Number.isFinite(r[key])); return valid.reduce((a,b) => b[key] < a[key] ? b : a, valid[0]); }
  function setText(id, value) { $(id).textContent = value; }
  function populateSummary() {
    const specs = [['finalWall','final-wall-mean','final-wall-worst'],['finalOff','final-off-mean','final-off-worst'],['motWall','mot-wall-mean','mot-wall-worst'],['motOff','mot-off-mean','mot-off-worst'],['finalRegion','final-region-mean','final-region-worst'],['motRegion','mot-region-mean','mot-region-worst']];
    const hasRegion = summary.some(r => Number.isFinite(r.finalRegion));
    document.querySelectorAll('.region-stat,.region-row').forEach(el => { el.style.display = hasRegion ? '' : 'none'; });
    specs.forEach(([key,meanId,worstId]) => { const lo=worst(key); if(!lo){setText(meanId,'—');setText(worstId,'not available');return;} setText(meanId, mean(summary.filter(r=>Number.isFinite(r[key])).map(r=>r[key])).toFixed(3)); setText(worstId, `worst ${lo.v} · ${lo[key].toFixed(3)}`); });
    setText('mode-summary', `${MODE.labels[mode] || mode} · ${order.length} vessels`);
  }
  function bbox(pts) { let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity; for(const [x,y] of pts){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);} return [x0,x1,y0,y1]; }
  function hexToRgb(hex){const h=hex.replace('#','');return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}
  function lerpColor(near,far,t){const a=hexToRgb(near),b=hexToRgb(far);return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;}
  function drawWindow(ctx,canvas,wallHot,lumenHot,near,far){
    const d=DATA[vessel], w=canvas.width,h=canvas.height; ctx.clearRect(0,0,w,h);ctx.fillStyle=css('--surface-2');ctx.fillRect(0,0,w,h);
    const [x0,x1,y0,y1]=bbox(d.bg.concat(d.wall_pos,d.lumen_pos)), pad=24, s=Math.min((w-2*pad)/Math.max(x1-x0,1e-9),(h-2*pad)/Math.max(y1-y0,1e-9));
    const ox=pad+((w-2*pad)-s*(x1-x0))/2, oy=pad+((h-2*pad)-s*(y1-y0))/2;
    const px=x=>(ox+(x-x0)*s)*view.k+view.panX, py=y=>(oy+(y1-y)*s)*view.k+view.panY;
    ctx.fillStyle=css('--muted');ctx.globalAlpha=.22;for(const [x,y] of d.bg){ctx.beginPath();ctx.arc(px(x),py(y),1.1,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
    ctx.fillStyle=near;for(let i=0;i<d.wall_pos.length;i++){if(!wallHot[i])continue;const[x,y]=d.wall_pos[i];ctx.beginPath();ctx.arc(px(x),py(y),2.9,0,Math.PI*2);ctx.fill();}
    for(let i=0;i<d.lumen_pos.length;i++){if(!lumenHot[i])continue;const[x,y]=d.lumen_pos[i],r=2.9;ctx.fillStyle=lerpColor(near,far,d.lumen_dist[i]);ctx.fillRect(px(x)-r,py(y)-r,2*r,2*r);}
  }
  function drawScoreChart(ctx,canvas,series,times,tFinal,curT,color){
    const w=canvas.width,h=canvas.height,pL=42,pR=12,pT=10,pB=30,pW=w-pL-pR,pH=h-pT-pB,px=t=>pL+(t/tFinal)*pW,py=v=>pT+pH-(Math.max(0,Math.min(1,v))*pH);
    ctx.clearRect(0,0,w,h);ctx.fillStyle=css('--surface');ctx.fillRect(0,0,w,h);ctx.strokeStyle=css('--line');ctx.fillStyle=css('--muted');ctx.lineWidth=1;ctx.font='9px '+css('--mono');
    for(let i=0;i<=4;i++){const v=i/4,y=py(v);ctx.beginPath();ctx.moveTo(pL,y);ctx.lineTo(w-pR,y);ctx.stroke();ctx.fillText(v.toFixed(2),4,y+3);}
    for(let i=0;i<=3;i++){const t=tFinal*i/3,x=px(t);ctx.fillText(Math.round(t/1000)+'k',x-7,h-12);}ctx.fillText('t (s)',w-30,h-2);ctx.save();ctx.translate(11,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('score',0,0);ctx.restore();
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();series.forEach((v,i)=>{const x=px(times[i]),y=py(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
    const x=px(curT);ctx.strokeStyle=css('--ink');ctx.globalAlpha=.35;ctx.beginPath();ctx.moveTo(x,pT);ctx.lineTo(x,pT+pH);ctx.stroke();ctx.globalAlpha=1;
  }
  function drawHistogram(canvas, wallVals, offVals){
    const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,pL=44,pR=10,pT=12,pB=36,pW=w-pL-pR,pH=h-pT-pB,bins=10;
    const counts=(vals)=>Array.from({length:bins},(_,i)=>vals.filter(v=>Math.min(bins-1,Math.floor(Math.max(0,Math.min(.999999,v))*bins))===i).length);const a=counts(wallVals),b=counts(offVals),max=Math.max(1,...a,...b),top=Math.ceil(max/2)*2;
    ctx.clearRect(0,0,w,h);ctx.fillStyle=css('--surface');ctx.fillRect(0,0,w,h);ctx.font='9px '+css('--mono');ctx.strokeStyle=css('--line');ctx.fillStyle=css('--muted');
    for(let i=0;i<=top;i++){const y=pT+pH-(i/top)*pH;ctx.globalAlpha=i===0?1:.7;ctx.beginPath();ctx.moveTo(pL,y);ctx.lineTo(w-pR,y);ctx.stroke();if(i===0||i===top||i%2===0)ctx.fillText(String(i),pL-14,y+3);}ctx.globalAlpha=1;
    const bw=pW/bins;for(let i=0;i<bins;i++){const x=pL+i*bw, gap=3, half=(bw-gap)/2;const bar=(n,col,dx)=>{const bh=(n/top)*pH;ctx.fillStyle=col;ctx.fillRect(x+dx,pT+pH-bh,half,bh);};bar(a[i],css('--score-wall'),gap/2);bar(b[i],css('--score-off'),gap/2+half);}
    [0,.25,.5,.75,1].forEach(v=>{const x=pL+v*pW;ctx.fillStyle=css('--muted');ctx.fillText(v.toFixed(2),x-9,h-18);});ctx.fillText('deploy score',w/2-29,h-3);ctx.save();ctx.translate(11,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('vessels',0,0);ctx.restore();
  }
  function scoreAt(times,series,t){let j=0,d=Infinity;times.forEach((x,i)=>{if(Math.abs(x-t)<d){j=i;d=Math.abs(x-t)}});return series[j];}
  function redraw(){const d=DATA[vessel],t=d.frame_t[frame];drawWindow(cmModel,canvasModel,d.frame_model_wall[frame],d.frame_model_lumen[frame],css('--model-c'),css('--model-far'));drawWindow(cmGt,canvasGt,d.frame_gt_wall[frame],d.frame_gt_lumen[frame],css('--gt-c'),css('--gt-far'));drawScoreChart(wctx,wallChart,d.score_wall,d.score_t,d.t_final,t,css('--score-wall'));drawScoreChart(octx,offChart,d.score_offwall,d.score_t,d.t_final,t,css('--score-off'));if(d.score_wound_region){drawScoreChart(rctx,regionChart,d.score_wound_region,d.score_t,d.t_final,t,css('--gt-c'));}$('time-readout').innerHTML=`t = ${Math.round(t)} s <b>(${Math.round(100*t/d.t_final)}%)</b>`;$('val-wall').textContent='wall '+scoreAt(d.score_t,d.score_wall,t).toFixed(3);$('val-off').textContent='__OFF_LABEL__ '+scoreAt(d.score_t,d.score_offwall,t).toFixed(3);}
  function setVessel(v){vessel=v;frame=0;slider.value=0;view.k=1;view.panX=0;view.panY=0;$('zoom-readout').textContent='1.0× · synced zoom';redraw();}
  function resetView(){view.k=1;view.panX=0;view.panY=0;$('zoom-readout').textContent='1.0× · synced zoom';redraw();}
  function zoomAt(canvas,x,y,factor){const k=Math.max(MIN_K,Math.min(MAX_K,view.k*factor)),wx=(x-view.panX)/view.k,wy=(y-view.panY)/view.k;view.panX=x-wx*k;view.panY=y-wy*k;view.k=k;$('zoom-readout').textContent=view.k.toFixed(1)+'× · synced zoom';redraw();}
  function canvasPoint(canvas,e){const r=canvas.getBoundingClientRect(),q=e.touches?e.touches[0]:e;return{x:(q.clientX-r.left)*(canvas.width/r.width),y:(q.clientY-r.top)*(canvas.height/r.height)}}
  function attachZoomPan(canvas){let drag=false,start=null,pinch=null;canvas.addEventListener('wheel',e=>{e.preventDefault();const p=canvasPoint(canvas,e);zoomAt(canvas,p.x,p.y,e.deltaY<0?1.15:1/1.15)},{passive:false});canvas.addEventListener('mousedown',e=>{drag=true;start={x:e.clientX,y:e.clientY,px:view.panX,py:view.panY}});window.addEventListener('mousemove',e=>{if(!drag)return;const r=canvas.getBoundingClientRect();view.panX=start.px+(e.clientX-start.x)*canvas.width/r.width;view.panY=start.py+(e.clientY-start.y)*canvas.height/r.height;redraw()});window.addEventListener('mouseup',()=>drag=false);canvas.addEventListener('dblclick',resetView);canvas.addEventListener('touchstart',e=>{if(e.touches.length===1){drag=true;start={x:e.touches[0].clientX,y:e.touches[0].clientY,px:view.panX,py:view.panY}}else{drag=false;pinch=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY)}},{passive:true});canvas.addEventListener('touchmove',e=>{if(e.touches.length===1&&drag){const r=canvas.getBoundingClientRect();view.panX=start.px+(e.touches[0].clientX-start.x)*canvas.width/r.width;view.panY=start.py+(e.touches[0].clientY-start.y)*canvas.height/r.height;redraw()}else if(e.touches.length===2&&pinch){const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY),p=canvasPoint(canvas,{clientX:(e.touches[0].clientX+e.touches[1].clientX)/2,clientY:(e.touches[0].clientY+e.touches[1].clientY)/2});zoomAt(canvas,p.x,p.y,d/pinch);pinch=d;}},{passive:true});canvas.addEventListener('touchend',()=>{drag=false;pinch=null});}
  function stop(){playing=false;playBtn.textContent='Play';if(timer){clearInterval(timer);timer=null;}}
  function fillVessels(){select.innerHTML=order.map(v=>`<option value="${v}">${v} · fold ${DATA[v].fold} · ${DATA[v].geom_class}</option>`).join('');}
  function refreshMode(){order=(MODE.vessels[mode]||allOrder).filter(v=>DATA[v]);if(!order.length)return;fillVessels();buildSummary();populateSummary();drawHistogram($('hist-final'),summary.map(r=>r.finalWall),summary.map(r=>r.finalOff));drawHistogram($('hist-mot'),summary.map(r=>r.motWall),summary.map(r=>r.motOff));setVessel(order[0]);}
  select.addEventListener('change',()=>{stop();setVessel(select.value)});if(modeSelect){modeSelect.addEventListener('change',()=>{stop();mode=modeSelect.value;refreshMode();});if(Object.keys(MODE.vessels).length<2)modeSelect.style.display='none';}
  slider.addEventListener('input',()=>{stop();frame=Number(slider.value);redraw()});playBtn.addEventListener('click',()=>{if(playing){stop();return;}playing=true;playBtn.textContent='Pause';timer=setInterval(()=>{frame=(frame+1)%DATA[vessel].frame_t.length;slider.value=frame;redraw()},750)});$('reset-zoom-btn').addEventListener('click',resetView);attachZoomPan(canvasModel);attachZoomPan(canvasGt);window.addEventListener('resize',redraw);
  refreshMode();
})();
</script>"""

if __name__ == "__main__":
    main()
