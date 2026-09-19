"""ClotML: local browser UI for customer vessel prediction.

The browser owns presentation and interaction; this process owns geometry parsing,
CUDA inference, and model files.  It deliberately uses only the Python standard
library so the customer launcher does not need a web-framework dependency.

Launch with::

    python -m src.tools.customer_predict_web
    python -m src.tools.customer_predict_web --cpu
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import multiprocessing as mp
import queue
import re
import threading
import time
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from src.clot_ml.locked import trained_time_axis
from src.config import VesselConfig
from src.data_gen.lib.customer_geometry_import import (
    ANEURYSM_DEPTH_GAIN,
    DEFAULT_RE,
    MESH_UNIT_SCALE,
    STENOSIS_DEPTH_GAIN,
    TRAINED_ANEURYSM_SHARPNESS,
    TRAINED_ANEURYSM_WIDENING,
    TRAINED_INLET_WIDTH_M,
    TRAINED_MAX_S_AMPLITUDE_M,
    TRAINED_PATHOLOGY_LOCATION,
    TRAINED_STENOSIS_NARROWING,
    TRAINED_STENOSIS_SHARPNESS,
    TRAINED_WOUND_CENTER_FRAC,
    TRAINED_WOUND_HORIZON_S,
    TRAINED_WOUND_WIDTH_FRAC,
    SUPPORTED_SUFFIXES,
    BoundaryNeedsInput,
    apply_customer_mirrored_wound,
    apply_customer_max_pathology,
    build_parametric_customer_graph,
    ensure_inbox,
    list_inbox,
    load_customer_geometry,
)
from src.data_gen.lib.vessel_generator import make_vessel_params
from src.inference.customer_pipeline import CustomerDeployPipeline
from src.inference.customer_retrain_pipeline import CustomerRetrainPipeline
from src.tools.customer_predict_metrics import (
    SECONDS_PER_HOUR,
    VALIDATED_HORIZON_S,
    trajectory_scientific_table,
    write_scientific_csv,
)
from src.utils.paths import get_project_root


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClotML — Customer vessel forecast</title>
<style>
/* One fixed "clinical" light theme on every OS. There is deliberately NO prefers-color-scheme
   block: the page used to flip to a dark palette whenever the OS was in dark mode, so the same
   release looked different on a dark-mode Mac than on a light-mode Windows machine. The canvas
   colors are the legend: lumen blue, wall slate, clot red (off-wall a lighter red, same family),
   wound orange -- chrome (buttons, tabs) uses a separate UI blue so it never reads as data. */
:root{color-scheme:light;--bg:#eef1f5;--surface:#ffffff;--surface2:#f5f7fa;--canvas:#fbfcfd;--line:#dde3ea;--ink:#1c2430;--muted:#5b6878;--accent:#1f67a8;--accent-ink:#ffffff;--lumen:#9fd0ee;--lumen-ink:#3a8fc6;--wall:#2b3440;--clot:#c0262d;--wound:#f08a24;--inlet:#2e9d6a;--outlet:#7a55c7;--green:#2f7d57;--red:#c0262d;--shadow:0 1px 2px rgba(28,36,48,.06),0 8px 24px -14px rgba(28,36,48,.18);--radius-lg:12px;--radius-md:8px;--radius-sm:7px;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;--serif:var(--sans);--mono:ui-monospace,"SF Mono","Cascadia Code","Roboto Mono",Consolas,monospace;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px}button,input,select{font:inherit;color:inherit}button,select,input{accent-color:var(--accent)}
.topbar{min-height:68px;margin:18px 22px 0;padding:14px 26px;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);display:flex;flex-wrap:wrap;align-items:center;gap:12px 18px;box-shadow:var(--shadow)}
.brand{font-size:22px;font-weight:700;color:var(--ink);white-space:nowrap}.brand-sub{font-size:22px;font-weight:400;color:var(--muted);white-space:nowrap}.topbar-note{font-family:var(--mono);color:var(--muted);font-size:12px;background:var(--surface2);padding:.2rem .5rem;border-radius:var(--radius-sm)}.status{font-family:var(--mono);margin-left:auto;color:var(--muted);font-size:11.5px;text-align:right;max-width:44%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.status.ok{color:var(--green)}.status.err{color:var(--red)}
.shell{display:grid;grid-template-columns:284px minmax(0,1fr);gap:18px;margin:18px 22px 24px}.rail,.workspace-card,.stat{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);box-shadow:var(--shadow)}.rail{padding:20px 18px;min-height:calc(100vh - 128px)}
.section{padding:0 0 18px;margin:0 0 18px;border-bottom:1px solid var(--line)}.section:last-child{border-bottom:0;margin-bottom:0}.section h2{margin:0 0 10px;font:700 11px var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--ink)}.section h2.collapsible{cursor:pointer;display:flex;align-items:center;justify-content:space-between;user-select:none}.section h2.collapsible .chev{font-size:11px;color:var(--muted);transition:transform .15s}.section h2.collapsible[aria-expanded="true"] .chev{transform:rotate(90deg)}label{display:block;color:var(--muted);font-size:12px;margin:10px 0 5px}.field{width:100%;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-sm);padding:9px 10px}.field:focus{outline:2px solid color-mix(in srgb,var(--accent) 35%,transparent);outline-offset:1px}.range-row{display:grid;grid-template-columns:1fr 52px;gap:10px;align-items:center}.range-row output{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink);text-align:right}.hint{font-size:11px;line-height:1.4;color:var(--muted);margin:10px 0 0}.hint-details{margin-top:8px}.hint-details>summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;gap:5px;color:var(--muted);font-size:11px;font-family:var(--mono)}.hint-details>summary::-webkit-details-marker{display:none}.hint-details>summary::before{content:'ⓘ';font-size:12px;line-height:1}.hint-details>summary:hover{color:var(--accent)}.hint-details[open]>summary{color:var(--accent)}.hint-details>.hint{margin-top:6px}.inlet-re{display:flex;align-items:baseline;gap:5px;margin:10px 0 5px;font-size:11px;color:var(--muted);white-space:nowrap}.inlet-re strong{font-family:var(--mono);font-size:12px;color:var(--ink);font-weight:600}.inline{display:flex;gap:8px}.inline>*{flex:1}.btn{font-family:var(--mono);font-size:12.5px;border:1px solid var(--line);border-radius:var(--radius-sm);background:var(--surface2);padding:9px 10px;cursor:pointer;transition:background .15s,border-color .15s,color .15s}.btn:hover{border-color:var(--accent);filter:brightness(.97)}.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600;width:100%;margin-top:5px}.btn.primary:hover{filter:brightness(1.06)}.btn:disabled{opacity:.55;cursor:wait}.preset{margin-top:8px}.hidden{display:none!important}
.workspace{min-width:0;display:grid;gap:14px}.workspace-card{padding:18px}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.stat{padding:13px 15px;border-radius:var(--radius-md);box-shadow:none}.stat label{margin:0 0 5px}.stat strong{display:block;font:700 22px/1.15 var(--mono);letter-spacing:-.01em;font-variant-numeric:tabular-nums}.stat span{display:block;color:var(--muted);font-size:11px;margin-top:3px}.viewer-head{display:flex;align-items:baseline;gap:12px;margin-bottom:12px}.viewer-head h1{margin:0;font:600 18px var(--sans)}.viewer-head span{font-family:var(--mono);margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.canvas-wrap{position:relative;background:var(--canvas);border:1px solid var(--line);border-radius:var(--radius-lg);overflow:hidden;min-height:430px}.canvas-wrap canvas{display:block;width:100%;height:430px}.view-reset-btn{position:absolute;top:10px;right:10px;z-index:2;width:30px;height:30px;border-radius:var(--radius-sm);border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer;font-size:15px;line-height:1;display:flex;align-items:center;justify-content:center;box-shadow:var(--shadow)}.view-reset-btn:hover{border-color:var(--accent);color:var(--accent)}.legend{display:flex;align-items:center;gap:8px 18px;flex-wrap:wrap;margin:12px 0 0;padding:10px 14px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-md);color:var(--muted);font-size:11px}.key{display:inline-flex;align-items:center;gap:5px}.swatch{width:11px;height:11px;display:inline-block;flex:none}.swatch.lumen{background:var(--lumen);border-radius:50%;width:8px;height:8px}.swatch.wall{background:var(--wall);border-radius:50%;width:8px;height:8px}.swatch.clot{background:var(--clot);border-radius:50%}.swatch.wound{border:2px solid var(--wound);border-radius:50%}.legend .key{color:var(--ink)}.legend .legend-note{margin-left:auto;color:var(--muted)}
.velocity-single .canvas-wrap{min-height:360px}.velocity-single canvas{height:360px}.flow-field-picker{display:flex;align-items:center;gap:8px;margin:0 0 10px}.flow-field-picker label{margin:0;color:var(--muted);font-size:12px}.field-select{font-family:var(--mono);font-size:12px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-sm);padding:6px 9px;cursor:pointer}.timeline{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin-top:14px;padding:12px 14px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-lg)}.timeline label{margin:0;font:700 11px var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--ink)}.timeline input{width:100%}.timeline output{font-family:var(--mono);min-width:112px;text-align:right;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.science{margin-top:14px}.science h2{margin:0 0 8px;font:700 11px var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--ink)}#science-canvas{width:100%;height:210px;display:block;background:var(--canvas);border:1px solid var(--line);border-radius:var(--radius-md)}.sci-caption{font-size:11px;color:var(--muted);margin:14px 0 6px}.sci-stats{display:flex;gap:26px;flex-wrap:wrap;padding:12px 18px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-md);margin:0}#axial-canvas{width:100%;height:120px;display:block;background:var(--canvas);border:1px solid var(--line);border-radius:var(--radius-md)}.download{font-family:var(--mono);display:inline-block;margin-top:9px;color:var(--accent);font-size:12px}.error{color:var(--red);font-size:12px;line-height:1.4;margin-top:10px}
.app-mode-select{font-family:var(--mono);font-size:12px;font-weight:600;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-sm);padding:6px 10px;cursor:pointer;white-space:nowrap}.mode-switch{display:flex;gap:2px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-sm);padding:3px}.mode-btn{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--muted);background:transparent;border:0;border-radius:5px;padding:6px 13px;cursor:pointer;white-space:nowrap;transition:background .15s,color .15s}.mode-btn:hover{color:var(--ink)}.mode-btn.active{background:var(--accent);color:var(--accent-ink)}.mode-hint{font-family:var(--mono);color:var(--muted);font-size:11px}.snapshot-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}.snap-tab{background:var(--surface2);border:1px solid var(--line);border-radius:20px;padding:5px 12px 5px 14px;cursor:pointer;font-size:11.5px;font-family:var(--mono);display:flex;align-items:center;gap:8px;white-space:nowrap;transition:background .15s,border-color .15s,color .15s}.snap-tab.active{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600}.snap-tab:not(.active):hover{border-color:var(--accent)}.snap-close{opacity:.5;cursor:pointer;font-size:14px;line-height:1;margin-left:2px}.snap-close:hover{opacity:1}.score-bar{display:flex;flex-wrap:wrap;gap:12px 30px;padding:12px 18px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-md);margin:0 0 14px}.score-item{display:flex;flex-direction:column;gap:3px}.score-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-family:var(--mono)}.score-badge{font:700 20px var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}.score-badge.wall{color:var(--clot)}.score-badge.vessel{color:var(--lumen-ink)}.score-badge.occ{color:var(--wall)}.score-def{font-size:10.5px;color:var(--muted)}.retrain-console{background:var(--surface2);color:var(--ink);border:1px solid var(--line);border-radius:var(--radius-md);padding:12px;font-family:var(--mono);white-space:pre-wrap;height:460px;overflow-y:auto;font-size:11.5px;line-height:1.5}
@media(max-width:900px){.shell{grid-template-columns:1fr}.rail{min-height:0}.topbar{margin-left:12px;margin-right:12px}.shell{margin-left:12px;margin-right:12px}.status{max-width:35%}}@media(max-width:560px){.topbar{padding:0 14px;gap:9px}.brand,.brand-sub{font-size:20px}.topbar-note{display:none}.stats{grid-template-columns:1fr}.canvas-wrap canvas{height:320px}.canvas-wrap{min-height:320px}.timeline{grid-template-columns:1fr}.timeline output{text-align:left}}
.label-row{display:flex;align-items:center;gap:6px;margin:10px 0 5px}.label-row label{margin:0}.info{position:relative;display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;border:1px solid var(--muted);color:var(--muted);font:italic 700 10px/1 Georgia,serif;cursor:help;user-select:none}.info:hover,.info:focus{border-color:var(--accent);color:var(--accent);outline:none}.info-pop{display:none;position:absolute;left:-12px;top:23px;z-index:20;width:252px;padding:10px 12px;background:var(--ink);color:#fff;border-radius:8px;font:400 11.5px/1.45 var(--sans);text-align:left;box-shadow:var(--shadow)}.info-pop b{font-weight:600}.info-pop p{margin:0 0 7px}.info-pop p:last-child{margin:0}.info:hover .info-pop,.info:focus .info-pop{display:block}.node-tip{position:absolute;z-index:3;pointer-events:none;display:flex;align-items:center;gap:6px;padding:5px 9px;background:var(--ink);color:#fff;border-radius:6px;font:500 11.5px/1.3 var(--sans);white-space:nowrap;box-shadow:var(--shadow)}.node-tip i{width:9px;height:9px;border-radius:50%;flex:none;box-shadow:0 0 0 1px rgba(255,255,255,.6)}.run-progress{margin:0 0 14px;padding:12px 16px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-md)}.rp-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px}.rp-label{font-weight:600;font-size:13px}.rp-head-right{display:flex;align-items:center;gap:10px}.rp-pct{font:700 13px var(--mono);font-variant-numeric:tabular-nums}.rp-cancel{padding:4px 11px;font-size:11.5px;width:auto;margin:0}#run-progress:not(.running) .rp-cancel{display:none}.rp-track{height:10px;background:#e1e7ee;border-radius:6px;overflow:hidden}.rp-fill{height:100%;width:0;background:var(--accent);border-radius:6px;transition:width .45s ease}.run-progress.running .rp-fill{background-image:linear-gradient(45deg,rgba(255,255,255,.22) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.22) 50%,rgba(255,255,255,.22) 75%,transparent 75%);background-size:20px 20px;animation:rp-stripe 1s linear infinite}@keyframes rp-stripe{to{background-position:20px 0}}.run-progress.done{background:#edf7f1;border-color:#a9d6bd}.run-progress.done .rp-fill{background:var(--green)}.run-progress.done .rp-label,.run-progress.done .rp-pct{color:var(--green)}.run-progress.error{background:#fdf0f0;border-color:#ebb4b6}.run-progress.error .rp-fill{background:var(--red)}.run-progress.error .rp-label{color:var(--red)}.run-progress.cancelled .rp-fill{background:var(--muted)}.rp-steps{display:flex;flex-wrap:wrap;gap:4px 14px;list-style:none;margin:10px 0 0;padding:0;font-size:11px;color:var(--muted)}.rp-steps li{display:flex;align-items:center;gap:6px}.rp-steps li::before{content:'';width:8px;height:8px;border-radius:50%;border:1.5px solid #b6c1cc;flex:none}.rp-steps li.active{color:var(--ink);font-weight:600}.rp-steps li.active::before{border-color:var(--accent);background:var(--accent)}.rp-steps li.done{color:var(--green)}.rp-steps li.done::before{border-color:var(--green);background:var(--green)}.rp-sub{font:11px var(--mono);color:var(--muted);margin-top:7px}.swatch.inlet{background:var(--inlet);border-radius:50%}.swatch.outlet{background:var(--outlet);border-radius:50%}.boundary-panel{margin-top:10px;padding:10px;background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-sm)}.boundary-msg{margin:0 0 8px;font-size:11.5px;line-height:1.45;color:var(--ink)}.boundary-msg.warn{color:#9a4d00}.boundary-msg.pick{color:var(--accent);font-weight:600}.boundary-msg .sub{display:block;color:var(--muted);font-weight:400;margin-top:3px}
</style></head>
<body>
<header class="topbar"><span class="brand">ClotML</span><span class="brand-sub">Predict</span><div class="mode-switch" role="tablist" aria-label="Mode"><button type="button" class="mode-btn active" data-mode="clot" role="tab" aria-selected="true">Clot</button><button type="button" class="mode-btn" data-mode="flow" role="tab" aria-selected="false">Flow</button><button type="button" class="mode-btn" data-mode="retrain" role="tab" aria-selected="false">Retrain</button></div><select id="app-mode" class="app-mode-select hidden" aria-hidden="true" tabindex="-1"><option value="clot">Clot Predictor</option><option value="flow">Flow Simulator</option><option value="retrain">Retrain Model</option></select><span id="mode-hint" class="mode-hint">Solves initial flow with a local FEM and then does a clot prediction rollout</span><span id="status" class="status" aria-live="polite">Ready — choose a geometry to begin.</span></header>
<main class="shell">
<aside class="rail" id="predict-rail">
  <section class="section"><h2>Geometry</h2>
    <div class="label-row"><label for="source">Source</label><span class="info" tabindex="0" role="button" aria-label="What the geometry sources mean" aria-describedby="source-info">i<span id="source-info" class="info-pop" role="tooltip"><p><b>Parametric vessel</b>: build a vessel with sliders (width, bend, S-curve, optional stenosis or aneurysm). Nothing to prepare; the app meshes it for you.</p><p><b>Upload geometry</b>: bring your own 2D vessel mesh (.msh or .nas) or a prepared .pt graph. The app finds the walls, inlet and outlet for you; if it can't tell, you click the two open ends on the preview.</p></span></span></div><select id="source" class="field"><option value="parametric" selected>Parametric vessel</option><option value="upload">Upload geometry</option></select>
    <div id="upload-controls" class="hidden"><label for="upload">Geometry file</label><input id="upload" class="field" type="file" accept=".msh,.nas,.pt"><div id="mesh-options" class="hidden"><label for="mesh-unit">Mesh units</label><select id="mesh-unit" class="field"><option value="auto" selected>Auto-detect</option><option value="mm">Millimetres</option><option value="cm">Centimetres</option><option value="m">Metres</option></select></div><div id="boundary-panel" class="boundary-panel hidden"><p id="boundary-msg" class="boundary-msg"></p><div id="boundary-actions" class="inline"><button id="swap-ends" class="btn" type="button">Swap ends</button><button id="pick-ends" class="btn" type="button">Pick ends</button></div><button id="auto-ends" class="btn hidden" type="button" style="margin-top:6px;width:100%">Back to automatic</button></div></div>
    <div id="param-controls" class="hidden"><label for="width">Width (m)</label><div class="range-row"><input id="width" type="range" min="0.0085" max="0.02" step="0.0005" value="0.015"><output id="width-out">0.015</output></div><label for="bend">Bend (deg)</label><div class="range-row"><input id="bend" type="range" min="0" max="90" step="1" value="20"><output id="bend-out">20</output></div><label for="amp">S-amp (m)</label><div class="range-row"><input id="amp" type="range" min="0" max="0.005" step="0.0005" value="0"><output id="amp-out">0.000</output></div><label for="pathology">Pathology preset</label><select id="pathology" class="field"><option value="none">None</option><option value="stenosis">Stenosis</option><option value="aneurysm">Aneurysm</option></select><div id="pathology-opts" class="hidden" style="margin-top:10px"><label for="p-strength">Strength (%)</label><div class="range-row"><input id="p-strength" type="range" min="0" max="77" step="1" value="77"><output id="p-strength-out">77</output></div><p id="p-strength-note" class="hint"></p><details class="hint-details"><summary>What does strength mean?</summary><p class="hint">The percent change of the vessel width at the centre of the pathology: a 50% stenosis leaves half the healthy width there, a 50% aneurysm makes it 1.5x as wide. The slider stops at the most severe pathology the model was trained on: 77% narrowing, or 99% widening (1.99x). Trained pathologies all sit at mid-length with one shape each, so position and taper are fixed to match.</p></details></div></div>
  </section>
  <section class="section"><h2>Wound</h2><label><input id="wound-enabled" type="checkbox"> Add mirrored wound</label><p class="hint">Optional. An injured segment on both walls, drawn in orange.</p><div id="wound-controls" class="hidden"><label for="wound-position">Position along vessel (%)</label><div class="range-row"><input id="wound-position" type="range" min="32" max="69" step="1" value="50"><output id="wound-position-out">50</output></div><label for="wound-width">Mirrored wound width (%)</label><div class="range-row"><input id="wound-width" type="range" min="6" max="15" step="1" value="10"><output id="wound-width-out">10</output></div><details class="hint-details"><summary>How wound placement works</summary><p class="hint">The same axial segment is placed on both vessel walls. Position and width stay within the wounds the model was trained on (32-69% along the vessel, 6-15% wide), and a wounded vessel can be simulated for up to 3.3 h (12,000 s).</p></details></div></section>
  <section class="section"><h2>Conditions</h2><p class="inlet-re">Inlet Re <strong>450</strong> <span>(fixed)</span></p><div id="hours-row"><label for="hours">Simulation time (hours)</label><div class="range-row"><input id="hours" type="range" min="1" max="8.33" step="0.01" value="8"><output id="hours-out">8.0</output></div><p id="hours-hint" class="hint">Up to 8.3 h (30,000 s), the longest simulated time the model was checked against COMSOL.</p></div></section>
  <section class="section"><h2>Run</h2><button id="run" class="btn primary" type="button">Run prediction</button><button id="snapshot-btn" class="btn" type="button" disabled style="margin-top:6px;width:100%">Save snapshot</button><p id="error" class="error hidden" role="alert"></p>
<p id="estimate-note" class="hint hidden">Runtime depends on mesh size and simulated time.</p><button id="estimate" class="btn hidden" type="button">Estimate runtime</button></section>
</aside>
<aside class="rail hidden" id="retrain-rail">
  <section class="section"><h2>Retrain model</h2><label for="retrain-dir">Dataset directory (absolute path)</label><input id="retrain-dir" class="field" type="text" placeholder="e.g. C:\data\graphs or /Users/you/graphs"><details class="hint-details"><summary>Which files retrain accepts</summary><p class="hint">Folder of .pt graphs (already-built, real-labeled vessels) and/or solved .mph COMSOL files. Each file is checked automatically — already-built graphs are used as-is, and .mph files are converted if a matching mesh is alongside them and COMSOL is available on this machine; anything that can't be used is skipped with a reason in the log below.</p></details><button id="run-retrain" class="btn primary" type="button" style="margin-top:16px">Start retrain</button><button id="cancel-retrain" class="btn" type="button" disabled style="margin-top:6px;width:100%">Cancel</button><details class="hint-details"><summary>What retraining actually does</summary><p class="hint">This trains a candidate model and scores it on a held-out split — it does not change the live model. A researcher reviews the results and promotes it separately.</p></details><p id="error-retrain" class="error hidden" role="alert"></p></section>
</aside>
<section class="workspace" id="main-workspace">
  <section class="workspace-card"><div id="run-progress" class="run-progress hidden" aria-live="polite"><div class="rp-head"><span id="rp-label" class="rp-label">Starting…</span><div class="rp-head-right"><span id="rp-pct" class="rp-pct">0%</span><button id="cancel-run" class="btn rp-cancel" type="button" disabled>Cancel</button></div></div><div id="rp-bar" class="rp-track" role="progressbar" aria-label="Prediction progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="rp-fill" class="rp-fill"></div></div><ol id="rp-steps" class="rp-steps"><li>Geometry</li><li>Blood flow</li><li>Clot growth</li><li>Metrics</li></ol><div id="rp-sub" class="rp-sub"></div></div><div id="snapshot-tabs" class="snapshot-tabs hidden"></div><div id="score-bar" class="score-bar hidden"><div class="score-item"><div class="score-lbl">Wall coverage</div><span class="score-badge wall" id="score-wall">—</span><span class="score-def">of wall with clot</span></div><div class="score-item"><div class="score-lbl">Lumen occlusion</div><span class="score-badge occ" id="score-occ">—</span><span class="score-def">deepest narrowing</span></div><div class="score-item"><div class="score-lbl">Vessel coverage</div><span class="score-badge vessel" id="score-vessel">—</span><span class="score-def">of all nodes with clot</span></div><div class="score-item"><div class="score-lbl">Clot span</div><span class="score-badge" id="score-span">—</span><span class="score-def">of vessel length</span></div><div class="score-item"><div class="score-lbl">Growth rate</div><span class="score-badge" id="score-rate">—</span><span class="score-def">span per hour</span></div></div><div class="viewer-head"><h1 id="viewer-title">Geometry preview</h1><span id="viewer-time">No prediction yet</span></div><div id="field-wrap" class="canvas-wrap"><canvas id="field-canvas" aria-label="Clot prediction field"></canvas><div class="node-tip hidden"></div><button id="reset-view" class="view-reset-btn" type="button" title="Reset view" aria-label="Reset view">&#9974;</button></div><div id="velocity-wrap" class="velocity-single hidden"><div class="flow-field-picker"><label for="flow-field">Field</label><select id="flow-field" class="field-select"><option value="velocity">Velocity magnitude</option><option value="shear">Shear rate</option></select></div><div class="canvas-wrap"><canvas id="vel0" aria-label="Vessel flow field"></canvas><div class="node-tip hidden"></div><button id="reset-view-flow" class="view-reset-btn" type="button" title="Reset view" aria-label="Reset view">&#9974;</button></div></div><div id="clot-legend" class="legend"><span class="key"><i class="swatch lumen"></i>lumen (no clot)</span><span class="key"><i class="swatch wall"></i>wall</span><span class="key"><i class="swatch clot"></i>clot</span><span class="key"><i class="swatch wound"></i>wound</span><span class="key caps-key hidden"><i class="swatch inlet"></i>inlet</span><span class="key caps-key hidden"><i class="swatch outlet"></i>outlet</span><span class="key legend-note">Hover a node to see its type</span></div><div id="timeline-row" class="timeline"><label for="time">Timeline</label><input id="time" type="range" min="0" max="0" step="1" value="0" disabled><output id="time-out">Preview</output></div><div id="science" class="science hidden"><h2>Scientific metrics</h2><p class="sci-caption">Coverage and occlusion over time</p><canvas id="science-canvas" aria-label="Wall coverage, vessel coverage, and occlusion depth over time"></canvas><p class="sci-caption">Clot distribution along the vessel (proximal → distal), selected frame</p><canvas id="axial-canvas" aria-label="Share of clot mass in the proximal, mid, and distal thirds of the vessel"></canvas><a id="download" class="download hidden" download>Download full metrics CSV</a></div></section>
</section>
<section class="workspace hidden" id="retrain-workspace"><section class="workspace-card"><div class="viewer-head"><h1 id="retrain-title">Retrain logs</h1><span id="retrain-time">Awaiting start…</span></div><div id="retrain-console" class="retrain-console"></div></section></section>
</main>
<script>
(() => {
const $=id=>document.getElementById(id), state={upload:{gen:0,id:null,hints:null,scale:1,preview:null,needs:false,picking:null,pending:null},data:null,mode:'clot',frame:0,job:null,jobKind:'predict',snapshots:[],liveResult:null,liveMode:'clot',liveAppMode:'clot',activeTab:'live',flowField:'velocity',viewAppMode:'clot'};
const cv=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
// Median nearest-neighbour spacing among screen-space points, via a coarse grid so it stays
// cheap on a few thousand nodes.  A bounding-box-area/count average (the obvious shortcut)
// badly overestimates spacing for wall nodes: they sit on a 1-D curve, not spread over the
// 2-D area the rest of the mesh fills, so that average is dominated by the sparser interior
// and every wall marker sized off it overlaps its neighbours into a solid painted band.
function nnPitch(xs,ys){const n=xs.length;if(n<2)return 4;let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(let k=0;k<n;k++){const x=xs[k],y=ys[k];if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y}const cell=Math.max(1,Math.sqrt(Math.max(1,(maxX-minX)*(maxY-minY))/n)),grid=new Map(),key=(cx,cy)=>cx+','+cy;for(let k=0;k<n;k++){const cx=Math.floor(xs[k]/cell),cy=Math.floor(ys[k]/cell),k2=key(cx,cy);if(!grid.has(k2))grid.set(k2,[]);grid.get(k2).push(k)}const dists=[];for(let k=0;k<n;k++){const cx=Math.floor(xs[k]/cell),cy=Math.floor(ys[k]/cell);let best=Infinity;for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){const arr=grid.get(key(cx+dx,cy+dy));if(!arr)continue;for(const j of arr){if(j===k)continue;const ddx=xs[k]-xs[j],ddy=ys[k]-ys[j],d2=ddx*ddx+ddy*ddy;if(d2<best)best=d2}}if(best<Infinity)dists.push(Math.sqrt(best))}if(!dists.length)return cell;dists.sort((a,b)=>a-b);return dists[Math.floor(dists.length/2)]}
const source=$('source'), status=$('status'), error=$('error'), run=$('run'), estimate=$('estimate'), time=$('time'), appMode=$('app-mode');
let retrainLogs=[];
window.view=window.view||{s:1,x:0,y:0,drag:0,sx:0,sy:0};
const setStatus=(msg,tone='')=>{status.textContent=msg;status.className='status '+tone};
const fmt=(x,d=1)=>Number.isFinite(Number(x))?Number(x).toFixed(d):'—';
function resetPreview(msg){const c=$('field-canvas'),[w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');c._hit=null;ctx.setTransform(scale,0,0,scale,0,0);ctx.fillStyle=cv('--canvas');ctx.fillRect(0,0,w,h);ctx.fillStyle=cv('--muted');ctx.font='13px '+cv('--sans');ctx.fillText((msg||'Choose a geometry file, then press Run prediction.'),18,24);$('field-wrap').classList.remove('hidden');$('velocity-wrap').classList.add('hidden');$('science').classList.add('hidden');$('viewer-title').textContent='Geometry preview';$('viewer-time').textContent='No prediction yet';time.disabled=true;$('time-out').textContent='Preview';$('score-bar').classList.add('hidden')}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// The status-polling loop alone hits this server every 700ms; on Windows a fresh TCP
// connect/accept every request occasionally loses the race and fetch() rejects outright
// ("Failed to fetch") even though the server is healthy. Retry a couple of times before
// surfacing an error -- only for network-level failures, never for a real HTTP error status,
// which resolves fetch() normally and is handled by each caller's own !r.ok check.
async function fetchRetry(url,opts,tries=3){let lastErr;for(let i=0;i<tries;i++){try{return await fetch(url,opts)}catch(e){lastErr=e;if(i<tries-1)await new Promise(res=>setTimeout(res,250*(i+1)))}}throw lastErr}
function markGeometryDirty(){state.activeTab='live';state.data=null;state.frame=0;$('download').classList.add('hidden');renderTabs();if(!rpTimer)$('run-progress').classList.add('hidden');if(source.value==='parametric')renderPreview();else if(source.value==='upload'){if(state.upload.preview)drawUploadPreview();else resetPreview();scheduleUploadPreview()}else resetPreview();setStatus('Geometry changed — ready for a new prediction.','ok')}
function syncWoundControls(){$('wound-controls').classList.toggle('hidden',!$('wound-enabled').checked)}
function toggleSource(){const v=source.value;$('upload-controls').classList.toggle('hidden',v!=='upload');$('param-controls').classList.toggle('hidden',v!=='parametric');state.data=null;state.frame=0;if(v==='parametric')renderPreview();else if(state.upload.preview)drawUploadPreview();else{resetPreview();scheduleUploadPreview()}}
// ---- Upload preview + inlet/outlet --------------------------------------------------------
// Choosing a file previews it straight away: the server reads the mesh outline, finds the
// walls, inlet and outlet, and returns the vessel with those marked. The file is sent once;
// later previews and the run refer to it by `upload_id`. When the ends are ambiguous the server
// says so and the user clicks the inlet, then the outlet (sent as points in metres).
const fileB64=f=>new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(String(r.result).split(',')[1]);r.onerror=rej;r.readAsDataURL(f)});
const uploadFile=()=>$('upload').files[0]||null, isMeshFile=f=>!!f&&/\.(msh|nas)$/i.test(f.name);
let uploadTimer=null;
function scheduleUploadPreview(){clearTimeout(uploadTimer);if(source.value==='upload'&&uploadFile())uploadTimer=setTimeout(loadUploadPreview,350)}
function fmtLen(m){m=Number(m);return !(m>0)?'':m<0.01?(m*1e3).toFixed(1)+' mm':m<1?(m*100).toFixed(1)+' cm':m.toFixed(2)+' m'}
const UNIT_NAMES={m:'metres',cm:'centimetres',mm:'millimetres'};
function setBoundary(msg,tone,sub,actions){$('boundary-panel').classList.remove('hidden');const el=$('boundary-msg');el.className='boundary-msg '+(tone||'');el.textContent=msg;if(sub){const sp=document.createElement('span');sp.className='sub';sp.textContent=sub;el.appendChild(sp)}$('boundary-actions').classList.toggle('hidden',!actions);$('swap-ends').classList.toggle('hidden',!actions||state.upload.needs);$('auto-ends').classList.toggle('hidden',!state.upload.hints)}
function describeBoundary(){const U=state.upload,res=U.preview;if(!res)return;const b=res.boundary||{},f=uploadFile();if(!isMeshFile(f)){setBoundary('Walls, inlet and outlet come from the file.','',null,false);return}
  const unitNote=b.unit?`Units: ${UNIT_NAMES[b.unit]||b.unit}${$('mesh-unit').value==='auto'?' (auto-detected)':''}`+(b.length_m?` · vessel ${fmtLen(b.length_m)} long`:'')+(b.inlet_width_m?`, inlet ${fmtLen(b.inlet_width_m)} wide`:''):null;
  if(U.picking){setBoundary(U.picking==='inlet'?'Click the inlet: the open end where blood flows in.':'Now click the outlet: the other open end.','pick',unitNote,true);return}
  if(res.needs_boundary){setBoundary(res.message||'Couldn\'t tell which ends are open.','warn',unitNote,true);return}
  const how=b.boundary_source==='user'?'Using the inlet and outlet you picked.':b.boundary_source==='tags'?'Inlet and outlet read from the mesh tags.':'Found the inlet (green) and outlet (purple) automatically.';
  const warn=(b.warnings||[]);setBoundary(how+' Wrong way round? Swap them.',warn.length?'warn':'',[unitNote].concat(warn.length?['Outside what the model was trained on: '+warn.join(' ')]:[]).filter(Boolean).join(' · '),true)}
function drawUploadPreview(){const res=state.upload.preview;if(!res)return;drawField($('field-canvas'),0,{pos:res.pos,masks:res.masks,phi:[res.pos.map(()=>0)],caps:true});$('field-wrap').classList.remove('hidden');$('velocity-wrap').classList.add('hidden');$('science').classList.add('hidden');const f=uploadFile();$('viewer-title').textContent='Geometry preview'+(f?' — '+f.name:'');$('viewer-time').textContent='Not simulated';time.disabled=true;$('time-out').textContent='Preview';describeBoundary()}
async function loadUploadPreview(){const f=uploadFile(),U=state.upload;$('mesh-options').classList.toggle('hidden',!isMeshFile(f));if(!f||source.value!=='upload'||state.data)return;const gen=++U.gen;U.picking=null;error.classList.add('hidden');
  if(!U.preview)resetPreview('Reading '+f.name+'…');setStatus('Reading geometry…');
  try{const p={kind:'preview',source:'upload',filename:f.name,mesh_unit:$('mesh-unit').value,wound_enabled:$('wound-enabled').checked,wound_position:Number($('wound-position').value),wound_width:Number($('wound-width').value)};if(U.hints){p.inlet_hint=U.hints.inlet;p.outlet_hint=U.hints.outlet}if(U.id)p.upload_id=U.id;else p.data_b64=await fileB64(f);
    const r=await fetchRetry('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),j=await r.json();if(!r.ok)throw new Error(j.error||'Could not read the geometry');
    let res=null;for(;;){const jj=await (await fetchRetry('/api/job/'+j.job_id)).json();if(jj.status==='done'){res=jj.result;break}if(jj.status==='error')throw new Error(jj.error||'Could not read the geometry');if(jj.status==='cancelled')return;await new Promise(ok=>setTimeout(ok,300))}
    if(gen!==U.gen||source.value!=='upload'||state.data)return;
    U.id=res.upload_id||U.id;U.scale=Number(res.pos_scale_m)||1;U.preview=res;U.needs=!!res.needs_boundary;if(U.needs&&!U.hints)U.picking='inlet';
    drawUploadPreview();setStatus(U.needs?'Pick the inlet and outlet on the preview.':'Geometry ready — press Run prediction.',U.needs?'':'ok')
  }catch(e){if(gen!==U.gen)return;U.preview=null;resetPreview();$('boundary-panel').classList.add('hidden');error.textContent=e.message;error.classList.remove('hidden');setStatus('Could not read the geometry','err')}}
function capCentroid(mask){const res=state.upload.preview;let x=0,y=0,n=0;res.pos.forEach((q,i)=>{if(mask[i]){x+=q[0];y+=q[1];n++}});return n?[x/n*state.upload.scale,y/n*state.upload.scale]:null}
$('swap-ends').addEventListener('click',()=>{const U=state.upload,res=U.preview;if(!res||U.needs)return;const a=capCentroid(res.masks.inlet),b=capCentroid(res.masks.outlet);if(!a||!b)return;U.hints={inlet:b,outlet:a};loadUploadPreview()});
$('pick-ends').addEventListener('click',()=>{const U=state.upload;if(!U.preview)return;U.picking='inlet';U.pending=null;describeBoundary()});
$('auto-ends').addEventListener('click',()=>{state.upload.hints=null;state.upload.picking=null;loadUploadPreview()});
(()=>{const c=$('field-canvas');let x0=0,y0=0;c.addEventListener('mousedown',e=>{x0=e.clientX;y0=e.clientY});c.addEventListener('click',e=>{const U=state.upload,res=U.preview,h=c._hit;if(!U.picking||!res||state.data||!h||Math.hypot(e.clientX-x0,e.clientY-y0)>5)return;const r=c.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,m=res.masks;let best=-1,bd=400;for(let i=0;i<h.xs.length;i++){if(!(m.wall[i]||m.inlet[i]||m.outlet[i]))continue;const dx=h.xs[i]-mx,dy=h.ys[i]-my,dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=i}}if(best<0)return;const pt=[res.pos[best][0]*U.scale,res.pos[best][1]*U.scale];if(U.picking==='inlet'){U.pending=pt;U.picking='outlet';describeBoundary()}else{U.hints={inlet:U.pending,outlet:pt};U.picking=null;loadUploadPreview()}})})();
function bindRange(id,out,digits=0){const el=$(id),o=$(out);const sync=()=>o.value=Number(el.value).toFixed(digits);el.addEventListener('input',sync);sync()}
bindRange('width','width-out',3);bindRange('bend','bend-out');bindRange('amp','amp-out',3);bindRange('hours','hours-out',1);bindRange('wound-position','wound-position-out');bindRange('wound-width','wound-width-out');bindRange('p-strength','p-strength-out',0);
function syncSeverityNote(){const el=$('p-strength'),k=$('pathology').value,cap=k==='aneurysm'?99:77;const wasAtCap=Number(el.value)>=Number(el.max);el.max=String(cap);if(wasAtCap||Number(el.value)>cap)el.value=String(cap);$('p-strength-out').textContent=el.value;const s=Number(el.value)/100;$('p-strength-note').textContent=k==='stenosis'?`Width at the centre: ${(1-s).toFixed(2)}x the healthy width (${Math.round(100*s)}% narrower).`:k==='aneurysm'?`Width at the centre: ${(1+s).toFixed(2)}x the healthy width (${Math.round(100*s)}% wider).`:''}$('pathology').addEventListener('change',()=>{$('pathology-opts').classList.toggle('hidden',$('pathology').value==='none');syncSeverityNote()});$('p-strength').addEventListener('input',syncSeverityNote);syncSeverityNote();function syncHoursCap(){const h=$('hours'),wound=$('wound-enabled').checked,max=wound?3.33:8.33;h.max=String(max);if(Number(h.value)>max){h.value=String(max);h.dispatchEvent(new Event('input'))}$('hours-hint').textContent=wound?'Up to 3.3 h (12,000 s) with a wound, the longest a wound was simulated in training.':'Up to 8.3 h (30,000 s), the longest simulated time the model was checked against COMSOL.'}$('wound-enabled').addEventListener('change',syncHoursCap);syncHoursCap();

source.addEventListener('change',toggleSource);['width','bend','amp','pathology','p-strength','wound-position','wound-width'].forEach(id=>{['input','change'].forEach(ev=>$(id).addEventListener(ev,markGeometryDirty))});$('wound-enabled').addEventListener('change',()=>{syncWoundControls();markGeometryDirty()});$('upload').addEventListener('change',()=>{state.upload={gen:state.upload.gen+1,id:null,hints:null,scale:1,preview:null,needs:false,picking:null,pending:null};$('boundary-panel').classList.add('hidden');markGeometryDirty()});$('mesh-unit').addEventListener('change',()=>{state.upload.hints=null;state.upload.preview=null;markGeometryDirty()});syncWoundControls();toggleSource();
function resizeCanvas(c){const r=c.getBoundingClientRect(),d=Math.max(1,window.devicePixelRatio||1);c.width=Math.max(1,Math.round(r.width*d));c.height=Math.max(1,Math.round(r.height*d));return [c.width/d,c.height/d,d]}
function extent(data){let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;for(const p of data.pos){x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1])}return [x0,x1,y0,y1]}
function renderPreview(){if(source.value!=='parametric'||state.data)return;const c=$('field-canvas'),[w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=cv('--canvas');ctx.fillRect(0,0,w,h);const width=Number($('width').value),bend=Number($('bend').value)*Math.PI/180,amp=Number($('amp').value),pathology=$('pathology').value,woundOn=$('wound-enabled').checked,woundCenter=Number($('wound-position').value)/100,woundWidth=Number($('wound-width').value)/100,pStrength=Number($('p-strength').value)/100,actualQ=0.5,stdQ=0.0357/(pathology==='stenosis'?0.77:0.49);
// Same centerline formulas as src/data_gen/lib/vessel_generator.py's _centerline_arc /
// _centerline_s_curve, at the customer path's fixed length (VesselConfig.base_length = 0.1 m)
// -- real physical coordinates, not a canvas-relative sketch, so the preview's shape and
// aspect ratio actually match what gets meshed and solved (was previously scaled off canvas
// height with an arbitrary constant, which drifted from the true length/amplitude ratio).
const L=0.1,N=96,curveType=amp>1e-9?'s_curve':(Math.abs(bend)>1e-9?'arc':'straight'),cx=[],cy=[];
if(curveType==='arc'){const radius=L/Math.max(bend,1e-3);for(let i=0;i<=N;i++){const theta=(i/N)*bend;cx.push(radius*Math.sin(theta));cy.push(radius*(Math.cos(theta)-1.0))}}
else if(curveType==='s_curve'){for(let i=0;i<=N;i++){const t=i/N;cx.push(t*L);cy.push(amp*Math.sin(2*Math.PI*t))}}
else{for(let i=0;i<=N;i++){cx.push((i/N)*L);cy.push(0)}}
// Half-width per station: the same Gaussian bump on wall offset apply_customer_max_pathology
// applies server-side, so strength/location/sharpness preview exactly what gets solved.
// pStrength is the fractional width change at the centre (0.5 -> 0.5x stenosis / 1.5x aneurysm),
// capped at the trained 77% / 99%, so the walls never cross.
const rad=[];for(let i=0;i<=N;i++){const q=i/N;let r=width/2;if(pathology==='stenosis'){const gauss=Math.exp(-0.5*Math.pow((q-actualQ)/stdQ,2));r*=Math.max(0,1-pStrength*gauss)}else if(pathology==='aneurysm'){const gauss=Math.exp(-0.5*Math.pow((q-actualQ)/stdQ,2));r*=1+pStrength*gauss}rad.push(r)}
// Perpendicular offset from the local tangent, not a vertical offset -- vertical and
// "perpendicular to the vessel" stop agreeing as soon as the centerline bends.
const topx=[],topy=[],botx=[],boty=[],nxs=[],nys=[];
for(let i=0;i<=N;i++){const i0=Math.max(0,i-1),i1=Math.min(N,i+1);let dx=cx[i1]-cx[i0],dy=cy[i1]-cy[i0];const dn=Math.hypot(dx,dy)||1;dx/=dn;dy/=dn;const nx=-dy,ny=dx;nxs.push(nx);nys.push(ny);topx.push(cx[i]+nx*rad[i]);topy.push(cy[i]+ny*rad[i]);botx.push(cx[i]-nx*rad[i]);boty.push(cy[i]-ny*rad[i])}
const allX=[...topx,...botx],allY=[...topy,...boty],x0=Math.min(...allX),x1=Math.max(...allX),y0=Math.min(...allY),y1=Math.max(...allY),pad=38,s=Math.min((w-2*pad)/Math.max(x1-x0,1e-9),(h-2*pad)/Math.max(y1-y0,1e-9)),ox=pad+((w-2*pad)-s*(x1-x0))/2,oy=pad+((h-2*pad)-s*(y1-y0))/2,basePx=x=>ox+(x-x0)*s,basePy=y=>oy+(y1-y)*s;
// Same fixed-radius-under-pan/zoom scheme as drawField: vx/vy/vs move node centers, never the
// node radius, so wheel-zoom on the preview spreads nodes apart instead of inflating them.
const vx=window.view.x/scale,vy=window.view.y/scale,vs=window.view.s,tx=x=>vx+basePx(x)*vs,ty=y=>vy+basePy(y)*vs;
let woundLineA=null,woundLineB=null;
if(woundOn){
// Two boundary lines (one per axial edge of the band, wall-to-wall) plus a very light fill of the
// enclosed strip -- traced from the curve's own top/bottom samples rather than a screen-aligned
// bounding box, so it follows the vessel's actual curvature on a bent or narrowed section instead
// of a straight-edged rectangle sitting crooked across it. Same shape language as drawField. The
// lines are held and drawn last (after the wall dots below), not here, so a dense background
// scatter can never bury them.
const top=[],bot=[];
for(let i=0;i<=N;i++){
  if(Math.abs(i/N-woundCenter)>woundWidth/2)continue;
  top.push([tx(topx[i]),ty(topy[i])]);
  bot.push([tx(botx[i]),ty(boty[i])]);
}
if(top.length&&bot.length){
  ctx.save();
  ctx.beginPath();ctx.moveTo(top[0][0],top[0][1]);
  for(let k=1;k<top.length;k++)ctx.lineTo(top[k][0],top[k][1]);
  for(let k=bot.length-1;k>=0;k--)ctx.lineTo(bot[k][0],bot[k][1]);
  ctx.closePath();ctx.globalAlpha=0.14;ctx.fillStyle=cv('--wound');ctx.fill();
  ctx.restore();
  woundLineA=[top[0],bot[0]];
  woundLineB=[top[top.length-1],bot[bot.length-1]];
}
}
ctx.globalAlpha=0.9;ctx.fillStyle=cv('--lumen');for(let i=0;i<=N;i++){for(let j=-3;j<=3;j++){const px_=cx[i]+nxs[i]*(j/3.5)*rad[i],py_=cy[i]+nys[i]*(j/3.5)*rad[i];ctx.beginPath();ctx.arc(tx(px_),ty(py_),1.6,0,Math.PI*2);ctx.fill()}}ctx.globalAlpha=1;ctx.fillStyle=cv('--wall');for(let i=0;i<=N;i++){ctx.beginPath();ctx.arc(tx(topx[i]),ty(topy[i]),2.8,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(tx(botx[i]),ty(boty[i]),2.8,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;if(woundLineA){ctx.save();ctx.globalAlpha=0.9;ctx.strokeStyle=cv('--wound');ctx.lineWidth=1.6;ctx.setLineDash([5,3]);ctx.beginPath();ctx.moveTo(woundLineA[0][0],woundLineA[0][1]);ctx.lineTo(woundLineA[1][0],woundLineA[1][1]);ctx.stroke();ctx.beginPath();ctx.moveTo(woundLineB[0][0],woundLineB[0][1]);ctx.lineTo(woundLineB[1][0],woundLineB[1][1]);ctx.stroke();ctx.restore()}{const hx=[],hy=[],lab=[];for(let i=0;i<=N;i++){for(let j=-3;j<=3;j++){hx.push(tx(cx[i]+nxs[i]*(j/3.5)*rad[i]));hy.push(ty(cy[i]+nys[i]*(j/3.5)*rad[i]));lab.push(0)}const inW=woundOn&&Math.abs(i/N-woundCenter)<=woundWidth/2?2:1;hx.push(tx(topx[i]),tx(botx[i]));hy.push(ty(topy[i]),ty(boty[i]));lab.push(inW,inW)}const PL=[['Lumen (preview sketch)',cv('--lumen')],['Wall (preview sketch)',cv('--wall')],['Wound (preview sketch)',cv('--wound')]];c._hit={xs:hx,ys:hy,label:k=>PL[lab[k]]}}ctx.fillStyle=cv('--muted');ctx.font='12px '+cv('--sans');ctx.fillText(woundOn?'Preview: wound site shaded orange. Press Run prediction to simulate clot.':'Preview only. Press Run prediction to simulate clot.',18,22);$('field-wrap').classList.remove('hidden');$('velocity-wrap').classList.add('hidden');$('science').classList.add('hidden');$('viewer-title').textContent='Parametric vessel preview';$('viewer-time').textContent='Not simulated';time.disabled=true;$('time-out').textContent='Preview'}
function drawField(c,idx,override){const d=override||state.data;if(!d)return;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=cv('--canvas');ctx.fillRect(0,0,w,h);const [x0,x1,y0,y1]=extent(d),pad=28,s=Math.min((w-2*pad)/Math.max(x1-x0,1e-9),(h-2*pad)/Math.max(y1-y0,1e-9)),ox=pad+((w-2*pad)-s*(x1-x0))/2,oy=pad+((h-2*pad)-s*(y1-y0))/2,vx=window.view.x/scale,vy=window.view.y/scale,vs=window.view.s,
// Positions pan/zoom with the view (vx/vy/vs); marker RADIUS never gets multiplied by vs. If the
// same scale drove both, dots would grow with zoom -- staying just as overlapped as at rest -- so
// zooming would never actually separate nodes. A fixed screen radius means zooming spreads node
// centers apart while the dots themselves stay put, which is what makes individual nodes legible.
basePx=p=>ox+(p[0]-x0)*s,basePy=p=>oy+(y1-p[1])*s,px=p=>vx+basePx(p)*vs,py=p=>vy+basePy(p)*vs,phi=d.phi[idx]||d.phi[0],wall=d.masks.wall||[],wound=d.masks.wound||[],clotCol=cv('--clot'),woundCol=cv('--wound'),HOT=0.08;
// Marker size scales to the mesh's actual node density instead of a fixed pixel size -- a P2
// mesh's node spacing varies a lot (dense near-wall mid-sides vs. a sparse interior), and a fixed
// size that looked like discrete dots on a coarse mesh tiles into a solid blob on a dense one
// (adjacent hot markers overlapping edge-to-edge). Density is measured in base (unzoomed)
// coordinates so it stays stable as the user zooms, instead of resampling every frame.
const areaPx=Math.max(1,s*s*Math.max(x1-x0,1e-9)*Math.max(y1-y0,1e-9)),pitch=Math.sqrt(areaPx/Math.max(d.pos.length,1)),bgR=Math.max(0.8,Math.min(1.7,pitch*0.36));
// The wound-cavity boundary is a real no-slip wall in the source mesh, but mask_wall and
// mask_wound are built disjoint (see apply_customer_mirrored_wound) -- so wound nodes must be
// OR'd in wherever "is this a wall node" drives the marker choice, or the wound's own boundary
// silently falls back to faint background scatter instead of the wall's black trace / clot circle.
const wallXs=[],wallYs=[];for(let i=0;i<d.pos.length;i++){if(!wall[i]&&!wound[i])continue;const p=d.pos[i];wallXs.push(basePx(p));wallYs.push(basePy(p))}
const wallPitch=nnPitch(wallXs,wallYs),hotBase=Math.max(1.3,Math.min(4.0,wallPitch*0.45));
// Wound sites are traced with two boundary lines (one per axial edge of the band, connecting the
// matching wall point on each side) plus a very light fill of the enclosed wall-to-wall strip --
// built from the wound nodes' own positions rather than a screen-aligned bounding box, so it
// follows the vessel's actual curvature on a bent or narrowed section instead of a straight-edged
// rectangle sitting crooked across it. The fill is faint enough, and every marker's color ramp now
// starts from its own "not clotted" baseline rather than the canvas background (see below), that
// nothing drawn over it loses contrast the way the old solid-wash halo did.
let woundLineA=null,woundLineB=null;
const woundPts=[];for(let i=0;i<d.pos.length;i++){if(!wound[i])continue;const p=d.pos[i];woundPts.push([px(p),py(p)])}
if(woundPts.length>=2){
  const midY=woundPts.reduce((sum,p)=>sum+p[1],0)/woundPts.length;
  const top=woundPts.filter(p=>p[1]<=midY).sort((a,b)=>a[0]-b[0]);
  const bot=woundPts.filter(p=>p[1]>midY).sort((a,b)=>a[0]-b[0]);
  if(top.length&&bot.length){
    ctx.save();
    ctx.beginPath();ctx.moveTo(top[0][0],top[0][1]);
    for(let k=1;k<top.length;k++)ctx.lineTo(top[k][0],top[k][1]);
    for(let k=bot.length-1;k>=0;k--)ctx.lineTo(bot[k][0],bot[k][1]);
    ctx.closePath();ctx.globalAlpha=0.12;ctx.fillStyle=woundCol;ctx.fill();
    ctx.restore();
    // The two cut-line strokes are held and drawn last (after every marker below), not here --
    // this region can be mesh-dense enough that background scatter alone stacks into a lighter
    // wash over anything drawn before it, and a boundary line is exactly the kind of thin, low-
    // alpha mark that wash could bury. Drawing on top guarantees the line is never the casualty.
    woundLineA=[top[0],bot[0]];
    woundLineB=[top[top.length-1],bot[bot.length-1]];
  }
}
// Sparse, low-opacity background scatter for the open vessel -- a dense solid fill reads as an
// illustration; a subtle point cloud reads as what it is, real mesh nodes.
ctx.globalAlpha=0.75;ctx.fillStyle=cv('--lumen');
for(let i=0;i<d.pos.length;i++){if(phi[i]>=HOT||wall[i]||wound[i])continue;const p=d.pos[i];ctx.beginPath();ctx.arc(px(p),py(p),bgR,0,Math.PI*2);ctx.fill()}
// The wall (wound boundary included) gets its own solid black trace instead of blending into that
// background scatter, so the vessel boundary stays visible before any clot forms and lights up in
// place once phi crosses HOT -- same position and radius, only the color changes.
ctx.globalAlpha=1;ctx.fillStyle=cv('--wall');
for(let i=0;i<d.pos.length;i++){if(phi[i]>=HOT||!(wall[i]||wound[i]))continue;const p=d.pos[i];ctx.beginPath();ctx.arc(px(p),py(p),hotBase*0.8,0,Math.PI*2);ctx.fill()}
// Clot is one uniform class: every node at or above HOT is the same red circle at the same
// fixed screen radius, wherever it sits (wall, wound or lumen). A fixed radius keeps dense clot
// legible as discrete nodes at any zoom instead of merging into a blob.
ctx.globalAlpha=1;ctx.fillStyle=clotCol;
for(let i=0;i<d.pos.length;i++){if(phi[i]<HOT)continue;const p=d.pos[i];ctx.beginPath();ctx.arc(px(p),py(p),hotBase,0,Math.PI*2);ctx.fill()}
ctx.globalAlpha=1;ctx.strokeStyle=woundCol;ctx.lineWidth=Math.max(0.9,hotBase*0.3);
for(let i=0;i<d.pos.length;i++){if(!wound[i])continue;const p=d.pos[i];ctx.beginPath();ctx.arc(px(p),py(p),hotBase*1.55,0,Math.PI*2);ctx.stroke()}
if(woundLineA){ctx.save();ctx.globalAlpha=0.9;ctx.strokeStyle=woundCol;ctx.lineWidth=1.6;ctx.setLineDash([5,3]);ctx.beginPath();ctx.moveTo(woundLineA[0][0],woundLineA[0][1]);ctx.lineTo(woundLineA[1][0],woundLineA[1][1]);ctx.stroke();ctx.beginPath();ctx.moveTo(woundLineB[0][0],woundLineB[0][1]);ctx.lineTo(woundLineB[1][0],woundLineB[1][1]);ctx.stroke();ctx.restore()}
ctx.globalAlpha=1;
// Upload previews mark the inlet and outlet the server found (or the user picked).
const caps=!!(override&&override.caps),inCol=cv('--inlet'),outCol=cv('--outlet');document.querySelectorAll('.caps-key').forEach(k=>k.classList.toggle('hidden',!caps));
if(caps){const inl0=d.masks.inlet||[],outl0=d.masks.outlet||[];let cxm=0,cym=0;for(const q of d.pos){cxm+=px(q);cym+=py(q)}cxm/=d.pos.length;cym/=d.pos.length;for(const [mask,col,name] of [[inl0,inCol,'Inlet'],[outl0,outCol,'Outlet']]){let sx=0,sy=0,n=0;ctx.fillStyle=col;for(let i=0;i<d.pos.length;i++){if(!mask[i])continue;const p=d.pos[i],x=px(p),y=py(p);ctx.beginPath();ctx.arc(x,y,hotBase*1.25,0,Math.PI*2);ctx.fill();sx+=x;sy+=y;n++}if(n){sx/=n;sy/=n;let ux=sx-cxm,uy=sy-cym;const un=Math.hypot(ux,uy)||1;ctx.font='600 12px '+cv('--sans');ctx.textAlign='center';ctx.textBaseline='middle';const lx=Math.min(w-30,Math.max(30,sx-ux/un*34)),ly=Math.min(h-10,Math.max(10,sy-uy/un*34));ctx.lineWidth=3;ctx.strokeStyle='#fff';ctx.strokeText(name,lx,ly);ctx.fillText(name,lx,ly);ctx.textAlign='left';ctx.textBaseline='alphabetic'}}}
// Hover targets: every node's screen position plus what it is, for the node tooltip.
const hx=new Float32Array(d.pos.length),hy=new Float32Array(d.pos.length);for(let i=0;i<d.pos.length;i++){hx[i]=px(d.pos[i]);hy[i]=py(d.pos[i])}
const inl=d.masks.inlet||[],outl=d.masks.outlet||[],lumenCol=cv('--lumen'),wallCol=cv('--wall');
c._hit={xs:hx,ys:hy,label:i=>phi[i]>=HOT?[wound[i]?'Clot node (wound site)':'Clot node',clotCol]:wound[i]?['Wound node',woundCol]:wall[i]?['Wall node',wallCol]:inl[i]?['Inlet node',caps?inCol:lumenCol]:outl[i]?['Outlet node',caps?outCol:lumenCol]:['Lumen node',lumenCol]}}
// Classic "jet"/rainbow CFD colormap (blue -> cyan -> green -> yellow -> red), the scale
// COMSOL and most flow solvers default to -- far more legible for a scalar field than a
// single-hue ramp, since it gives the eye distinct bands instead of one continuous shade.
function jetColor(t){t=Math.max(0,Math.min(1,t));const r=Math.max(0,Math.min(1,1.5-Math.abs(4*t-3))),g=Math.max(0,Math.min(1,1.5-Math.abs(4*t-2))),b=Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)));return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`}
function drawColorbar(ctx,w,maxVal,unit){const bw=130,bh=10,bx=w-bw-14-40,by=14;for(let i=0;i<bw;i++){ctx.fillStyle=jetColor(i/bw);ctx.fillRect(bx+i,by,1,bh)}ctx.strokeStyle=cv('--line');ctx.lineWidth=1;ctx.strokeRect(bx+0.5,by+0.5,bw,bh);ctx.fillStyle=cv('--muted');ctx.font='10px '+cv('--mono');ctx.textAlign='left';ctx.fillText('0',bx,by+bh+12);ctx.textAlign='right';ctx.fillText((maxVal>0?Number(maxVal).toPrecision(3):'0')+' '+unit,bx+bw,by+bh+12);ctx.textAlign='left'}
const FLOW_FIELDS={velocity:{label:'Flow field',unit:'m/s'},shear:{label:'Shear rate',unit:'1/s'}};
function drawFlowField(c,idx){const d=state.data;if(!d||!d.velocity)return;const wantShear=state.flowField==='shear'&&d.shear,which=wantShear?FLOW_FIELDS.shear:FLOW_FIELDS.velocity,arr=wantShear?d.shear:d.velocity;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=cv('--canvas');ctx.fillRect(0,0,w,h);const [x0,x1,y0,y1]=extent(d),pad=28,s=Math.min((w-2*pad)/Math.max(x1-x0,1e-9),(h-2*pad)/Math.max(y1-y0,1e-9)),ox=pad+((w-2*pad)-s*(x1-x0))/2,oy=pad+((h-2*pad)-s*(y1-y0))/2,vx=window.view.x/scale,vy=window.view.y/scale,vs=window.view.s,basePx=p=>ox+(p[0]-x0)*s,basePy=p=>oy+(y1-p[1])*s,px=p=>vx+basePx(p)*vs,py=p=>vy+basePy(p)*vs,v=arr[idx]||arr[0],mx=v.reduce((a,b)=>Math.max(a,b),1e-9);for(let i=0;i<d.pos.length;i++){const q=v[i]/mx,p=d.pos[i];ctx.fillStyle=jetColor(q);ctx.fillRect(px(p)-2,py(p)-2,4,4)}ctx.fillStyle=cv('--ink');ctx.font='12px '+cv('--sans');ctx.fillText(which.label+' ('+which.unit+')',18,22);drawColorbar(ctx,w,mx,which.unit);const m=d.masks||{},hx=new Float32Array(d.pos.length),hy=new Float32Array(d.pos.length);for(let i=0;i<d.pos.length;i++){hx[i]=px(d.pos[i]);hy[i]=py(d.pos[i])}c._hit={xs:hx,ys:hy,label:i=>{const t=(m.wound||[])[i]?'Wound node':(m.wall||[])[i]?'Wall node':(m.inlet||[])[i]?'Lumen node (inlet)':(m.outlet||[])[i]?'Lumen node (outlet)':'Lumen node';return [t+' · '+Number(v[i]).toPrecision(3)+' '+which.unit,jetColor(v[i]/mx)]}}}
function drawScience(){const d=state.data,c=$('science-canvas');if(!d||!d.metrics)return;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=cv('--canvas');ctx.fillRect(0,0,w,h);const rows=d.metrics,p={l:42,r:14,t:18,b:30},pw=w-p.l-p.r,ph=h-p.t-p.b,px=i=>p.l+(i/Math.max(rows.length-1,1))*pw,py=v=>p.t+ph-(Math.max(0,Math.min(100,v))/100)*ph;ctx.strokeStyle=cv('--line');ctx.fillStyle=cv('--muted');ctx.font='10px '+cv('--sans');for(let k=0;k<=4;k++){const y=py(k*25);ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText(k*25+'%',6,y+3)}const line=(key,col)=>{ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>{const x=px(i),y=py(Number(r[key])||0);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()};const series=[['wall_clot_pct',cv('--clot'),'Wall coverage'],['max_occlusion_pct',cv('--wall'),'Lumen occlusion'],['vessel_clot_pct',cv('--lumen-ink'),'Vessel coverage']];series.forEach(([k,c])=>line(k,c));const fx=px(state.frame);ctx.save();ctx.strokeStyle=cv('--muted');ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(fx,p.t);ctx.lineTo(fx,p.t+ph);ctx.stroke();ctx.restore();ctx.font='11px '+cv('--sans');let lx=p.l+8;series.forEach(([,c,label])=>{ctx.fillStyle=c;ctx.fillRect(lx,p.t+4,14,3);ctx.fillStyle=cv('--ink');ctx.fillText(label,lx+19,p.t+9);lx+=ctx.measureText(label).width+42});const T=d.times||[],tEnd=Number(T[T.length-1])||0;ctx.fillStyle=cv('--muted');ctx.font='10px '+cv('--sans');ctx.textAlign='center';for(let k=0;k<=4;k++){const i=Math.round(k/4*(rows.length-1)),x=px(i);ctx.fillText(fmt((Number(T[i])||k/4*tEnd)/3600,1)+' h',x,h-10)}ctx.textAlign='left'}
function drawAxial(){const d=state.data,c=$('axial-canvas');if(!d||!d.metrics)return;const m=d.metrics[state.frame];if(!m)return;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=cv('--canvas');ctx.fillRect(0,0,w,h);const rows=[['Proximal',Number(m.clot_mass_prox_pct)||0],['Mid',Number(m.clot_mass_mid_pct)||0],['Distal',Number(m.clot_mass_dist_pct)||0]];const p={l:66,r:44,t:8,b:8},rowH=(h-p.t-p.b)/rows.length,barH=rowH*0.5;ctx.font='11px '+cv('--sans');rows.forEach((row,i)=>{const [label,pct]=row,y=p.t+i*rowH+rowH/2,barW=(w-p.l-p.r)*Math.max(0,Math.min(100,pct))/100;ctx.fillStyle=cv('--muted');ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(label,p.l-8,y);ctx.fillStyle=cv('--surface2');ctx.fillRect(p.l,y-barH/2,w-p.l-p.r,barH);ctx.fillStyle=cv('--clot');ctx.fillRect(p.l,y-barH/2,barW,barH);ctx.fillStyle=cv('--ink');ctx.textAlign='left';ctx.fillText(fmt(pct,0)+'%',p.l+barW+6,y)})}
function updateStats(){const d=state.data,m=d.metrics&&d.metrics[state.frame];if(!m)return;const rate=Number(m.clot_front_speed_per_h);$('score-span').textContent=fmt((Number(m.clot_axis_span_norm)||0)*100,0)+'%';$('score-rate').textContent=Number.isFinite(rate)?fmt(rate,2):'—';$('score-wall').textContent=fmt(m.wall_clot_pct,1)+'%';$('score-vessel').textContent=fmt(m.vessel_clot_pct,1)+'%';$('score-occ').textContent=fmt(m.max_occlusion_pct,1)+'%';$('score-bar').classList.remove('hidden');$('viewer-time').textContent=`t = ${fmt(d.times[state.frame],0)} s (${fmt(Number(d.times[state.frame])/3600,2)} h)`;$('time-out').textContent=$('viewer-time').textContent}
function geometryLabel(d){const g=d.meta&&d.meta.geometry;if(!g||!g.curve_type)return '';const width=Number(g.width_m);const shape=`${g.curve_type} · ${Number.isFinite(width)?(width*1e3).toFixed(1)+' mm':'custom width'}`;return ` · ${shape}`}
function render(){const d=state.data;if(!d)return;const velocity=state.viewAppMode==='flow';$('field-wrap').classList.toggle('hidden',velocity);$('velocity-wrap').classList.toggle('hidden',!velocity);$('science').classList.toggle('hidden',velocity);$('timeline-row').classList.toggle('hidden',velocity);$('clot-legend').classList.toggle('hidden',velocity);time.disabled=velocity;time.max=Math.max(0,d.times.length-1);time.value=state.frame;const title=velocity?'Flow field':'Clot prediction';$('viewer-title').textContent=title+geometryLabel(d);if(velocity&&d.velocity){drawFlowField($('vel0'),0)}else{drawField($('field-canvas'),state.frame);drawScience();drawAxial()}updateStats()}
time.addEventListener('input',()=>{state.frame=Number(time.value);render()});window.addEventListener('resize',()=>{if(state.data)render();else if(source.value==='parametric')renderPreview();else if(source.value==='upload'&&state.upload.preview)drawUploadPreview();else resetPreview()});
function payloadFromForm(){return {source:source.value,filename:'',app_mode:appMode.value,hours:Number($('hours').value),width:Number($('width').value),bend:Number($('bend').value),amp:Number($('amp').value),pathology:$('pathology').value,pathology_strength:Number($('p-strength').value),wound_enabled:$('wound-enabled').checked,wound_position:Number($('wound-position').value),wound_width:Number($('wound-width').value)}}
async function readUploadInto(p){if(p.source!=='upload')return true;const f=uploadFile(),U=state.upload;if(!f){error.textContent='Choose a geometry file first.';return false}if(U.needs&&!U.hints){error.textContent='Click the inlet and outlet on the preview first.';return false}p.filename=f.name;p.mesh_unit=$('mesh-unit').value;if(U.hints){p.inlet_hint=U.hints.inlet;p.outlet_hint=U.hints.outlet}if(U.id)p.upload_id=U.id;else p.data_b64=await fileB64(f);return true}
// Run progress. The server tags each job with its stage (geometry, flow solve, clot rollout,
// metrics) as the pipeline reports it; there is no per-step hook inside the locked rollout, so
// within a stage the bar eases toward (never past) that stage's end: el/(el+tau) keeps it visibly
// moving even when a big mesh runs several times longer than the runtime estimate. It only reaches 100% when
// the job is actually done.
const STAGES=[{label:'Preparing geometry',w:.08,tau:4},{label:'Solving blood flow',w:.24,tau:13},{label:'Predicting clot growth',w:.60,tau:30},{label:'Computing metrics',w:.08,tau:3}];
let rpTimer=null,rpPct=0;
function setProgress(pct,label,sub,stage,cls){const el=$('run-progress');el.classList.remove('hidden','running','done','error','cancelled');el.classList.add(cls);$('rp-fill').style.width=pct.toFixed(1)+'%';$('rp-pct').textContent=Math.floor(pct)+'%';$('rp-bar').setAttribute('aria-valuenow',String(Math.floor(pct)));$('rp-label').textContent=label;$('rp-sub').textContent=sub;[...$('rp-steps').children].forEach((li,i)=>{li.classList.toggle('done',i<stage);li.classList.toggle('active',i===stage&&cls==='running')})}
function progressStart(){clearInterval(rpTimer);rpTimer=setInterval(progressTick,250);rpPct=0;state.rp={status:'queued',stage:0,created_at:Date.now()/1000};progressTick()}
function progressTick(){const j=state.rp;if(!j)return;const s=Math.max(0,Math.min(3,Number(j.stage)||0)),now=Date.now()/1000,t0=Number(j.stage_started_at||j.created_at)||now,el=Math.max(0,now-t0),exp=Number(j.expected_rollout_s),tau=s===2&&exp>0?Math.max(20,exp):STAGES[s].tau,base=STAGES.slice(0,s).reduce((a,x)=>a+x.w,0),within=el/(el+tau);rpPct=Math.max(rpPct,Math.min(99,(base+STAGES[s].w*within)*100));setProgress(rpPct,`Step ${s+1} of 4 · ${STAGES[s].label}…`,`${duration(now-(Number(j.created_at)||now))} elapsed`+(s===2&&el>2*tau?' · still working: large meshes can take several minutes on this step':''),s,'running')}
function progressUpdate(j){state.rp=j;if(j.status==='done'||j.status==='error'||j.status==='cancelled'){clearInterval(rpTimer);rpTimer=null;const s=Math.max(0,Math.min(3,Number(j.stage)||0));if(j.status==='done')setProgress(100,'Finished · prediction ready',`Completed in ${duration(j.elapsed_s)}`,4,'done');else if(j.status==='cancelled')setProgress(rpPct,'Cancelled','',s,'cancelled');else setProgress(Math.max(rpPct,4),'Failed: '+(j.error||'prediction failed'),'',s,'error')}else progressTick()}
function duration(s){s=Math.max(0,Number(s)||0);return s>=60?`${Math.floor(s/60)}m ${Math.round(s%60)}s`:`${Math.round(s)} s`}
async function startJob(kind){error.classList.add('hidden');run.disabled=true;estimate.disabled=true;$('cancel-run').disabled=true;state.jobKind=kind;if(kind==='predict')progressStart();setStatus(kind==='estimate'?'Inspecting geometry for runtime estimate…':'Preparing prediction…');const p=payloadFromForm();p.kind=kind;state.pendingAppMode=p.app_mode;try{if(!await readUploadInto(p)){error.classList.remove('hidden');setStatus('Ready','err');return}const r=await fetchRetry('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),j=await r.json();if(!r.ok)throw new Error(j.error||'Could not start job');state.job=j.job_id;$('cancel-run').disabled=false;poll()}catch(e){if(kind==='predict')progressUpdate({status:'error',error:e.message});error.textContent=e.message;error.classList.remove('hidden');setStatus('Ready','err');run.disabled=false;estimate.disabled=false}}
async function poll(){if(!state.job)return;try{const r=await fetchRetry('/api/job/'+state.job),j=await r.json();if(state.jobKind==='predict')progressUpdate(j);const elapsed=j.created_at?` · ${duration(Date.now()/1000-j.created_at)} elapsed`:'';setStatus((j.message||j.status)+((j.status==='running'||j.status==='queued')?elapsed:''),j.status==='error'?'err':j.status==='done'?'ok':'');if(j.status==='running'||j.status==='queued'){setTimeout(poll,700);return}$('cancel-run').disabled=true;if(j.status==='cancelled'){setStatus('Cancelled.','err');run.disabled=false;estimate.disabled=false;return}if(j.status==='error')throw new Error(j.error||'Prediction failed');if(j.result&&j.result.kind==='estimate'){$('estimate-note').textContent=`Estimated ${duration(j.result.estimate_low_s)}–${duration(j.result.estimate_high_s)} (central ${duration(j.result.estimate_mid_s)}): ${j.result.n_nodes.toLocaleString()} nodes, ${j.result.n_steps} rollout steps. ${j.result.note}`;setStatus('Runtime estimate ready.','ok')}else{state.data=j.result;state.liveResult=j.result;state.liveMode=j.result.meta.run_mode||'clot';state.mode=state.liveMode;state.liveAppMode=state.pendingAppMode;state.viewAppMode=state.liveAppMode;state.frame=0;state.activeTab='live';renderTabs();$('snapshot-btn').disabled=false;const dl=$('download');if(j.result.csv_url){dl.href=j.result.csv_url;dl.classList.remove('hidden')}else dl.classList.add('hidden');render();const actual=Number(j.elapsed_s);setStatus(`Prediction complete in ${duration(actual)}.`, 'ok');saveSnapshot()}run.disabled=false;estimate.disabled=false}catch(e){error.textContent=e.message;error.classList.remove('hidden');run.disabled=false;estimate.disabled=false;$('cancel-run').disabled=true}}
async function cancelJob(){if(!state.job)return;$('cancel-run').disabled=true;setStatus('Cancelling…');try{await fetchRetry('/api/job/'+state.job+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}}

function kindLabel(appMode){return appMode==='flow'?'Flow':'Clot'}
function snapshotLabel(d,appMode){const g=d&&d.meta&&d.meta.geometry;const wid=g&&g.width_m?((g.width_m||0)*1e3).toFixed(1)+' mm':'custom';const hrs=d&&d.times&&d.times.length?((d.times[d.times.length-1]||0)/3600).toFixed(1)+'h':'?';const shape=g&&g.curve_type?g.curve_type:'vessel';return `${kindLabel(appMode)} · ${shape} ${wid} ${hrs}`}
function renderTabs(){const el=$('snapshot-tabs');if(!state.snapshots.length){el.classList.add('hidden');return}el.classList.remove('hidden');el.innerHTML='';const mk=(label,idx)=>{const t=document.createElement('span');t.className='snap-tab'+(state.activeTab===idx?' active':'');t.innerHTML=`${esc(label)}<span class="snap-close" data-idx="${typeof idx==='number'?idx:'live'}">×</span>`;t.addEventListener('click',e=>{if(!e.target.classList.contains('snap-close'))switchTab(idx)});t.querySelector('.snap-close').addEventListener('click',e=>{e.stopPropagation();if(idx==='live')return;closeSnap(Number(e.target.dataset.idx))});el.appendChild(t)};mk('Live','live');state.snapshots.forEach((s,i)=>mk(s.label,i))}
function switchTab(idx){state.activeTab=idx;if(idx==='live'){state.data=state.liveResult;state.mode=state.liveMode||'clot';state.viewAppMode=state.liveAppMode||'clot'}else{const snap=state.snapshots[idx];if(!snap)return;state.data=snap.data;state.mode=snap.mode;state.viewAppMode=snap.appMode||'clot'}state.frame=0;renderTabs();if(state.data){render();updateStats()}}
function closeSnap(idx){state.snapshots.splice(idx,1);if(state.activeTab===idx){switchTab('live')}else{if(typeof state.activeTab==='number'&&state.activeTab>idx)state.activeTab--;renderTabs()}}
function saveSnapshot(){if(!state.liveResult)return;const label=snapshotLabel(state.liveResult,state.liveAppMode);state.snapshots.push({label,data:state.liveResult,mode:state.liveMode||state.mode,appMode:state.liveAppMode||'clot'});switchTab(state.snapshots.length-1)}
$('snapshot-btn').addEventListener('click',saveSnapshot);

const MODE_HINTS={clot:'Solves initial flow with a local FEM and then does a clot prediction rollout',flow:"Solve a vessel's flow field",retrain:'Train a candidate clot-prediction model on your own files — a researcher reviews it before it ships'};
document.querySelectorAll('.mode-btn').forEach(btn=>{btn.addEventListener('click',()=>{if(appMode.value===btn.dataset.mode)return;appMode.value=btn.dataset.mode;appMode.dispatchEvent(new Event('change'))})});
$('flow-field').addEventListener('change',()=>{state.flowField=$('flow-field').value;if(state.data)render()});
appMode.addEventListener('change',()=>{const m=appMode.value;document.querySelectorAll('.mode-btn').forEach(b=>{const on=b.dataset.mode===m;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on))});$('mode-hint').textContent=MODE_HINTS[m]||'';$('retrain-rail').classList.toggle('hidden',m!=='retrain');$('predict-rail').classList.toggle('hidden',m==='retrain');$('main-workspace').classList.toggle('hidden',m==='retrain');$('retrain-workspace').classList.toggle('hidden',m!=='retrain');const woundSection=$('wound-enabled').closest('.section');const isFlow=m==='flow';$('hours-row').classList.toggle('hidden',isFlow);$('timeline-row').classList.toggle('hidden',isFlow);if(isFlow){$('wound-enabled').checked=false;syncWoundControls();woundSection.classList.add('hidden')}else{woundSection.classList.remove('hidden')}if(m==='retrain'){setStatus('Ready — choose a dataset directory and start retrain.')}else{markGeometryDirty()}});

$('run-retrain').addEventListener('click',async()=>{error.classList.add('hidden');$('error-retrain').classList.add('hidden');$('run-retrain').disabled=true;$('cancel-retrain').disabled=true;setStatus('Preparing retrain…');retrainLogs=[];$('retrain-console').textContent='';const p={kind:'retrain',data_dir:$('retrain-dir').value};try{const r=await fetchRetry('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),j=await r.json();if(!r.ok)throw new Error(j.error||'Could not start job');state.job=j.job_id;$('cancel-retrain').disabled=false;pollRetrain()}catch(e){$('error-retrain').textContent=e.message;$('error-retrain').classList.remove('hidden');setStatus('Ready','err');$('run-retrain').disabled=false}});
async function pollRetrain(){if(!state.job)return;try{const r=await fetchRetry('/api/job/'+state.job),j=await r.json();const msg=j.message||j.status;setStatus(msg,j.status==='error'?'err':j.status==='done'?'ok':'');$('retrain-time').textContent=j.status==='done'?'Complete':(j.status==='error'?'Failed':(j.status==='cancelled'?'Cancelled':'Running…'));if(msg!==retrainLogs[retrainLogs.length-1]){retrainLogs.push(msg);$('retrain-console').textContent+=msg+'\n';$('retrain-console').scrollTop=$('retrain-console').scrollHeight}if(j.status==='running'||j.status==='queued'){setTimeout(pollRetrain,700);return}$('cancel-retrain').disabled=true;if(j.status==='cancelled'){$('run-retrain').disabled=false;return}if(j.status==='error')throw new Error(j.error||'Retrain failed');$('run-retrain').disabled=false}catch(e){$('error-retrain').textContent=e.message;$('error-retrain').classList.remove('hidden');$('run-retrain').disabled=false;$('cancel-retrain').disabled=true}}
async function cancelRetrainJob(){if(!state.job)return;$('cancel-retrain').disabled=true;setStatus('Cancelling…');try{await fetchRetry('/api/job/'+state.job+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}}

function repaint(){if(state.data)render();else if(source.value==='parametric')renderPreview();else if(source.value==='upload'&&state.upload.preview)drawUploadPreview()}
function bindPanZoom(canvas){canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;const mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;const old=window.view.s;window.view.s*=e.deltaY<0?1.1:0.9;window.view.s=Math.max(1,Math.min(window.view.s,20));window.view.x=mx-(mx-window.view.x)*(window.view.s/old);window.view.y=my-(my-window.view.y)*(window.view.s/old);repaint()});canvas.addEventListener('mousedown',e=>{window.view.drag=1;window.view.sx=e.clientX;window.view.sy=e.clientY});canvas.addEventListener('dblclick',()=>{window.view={s:1,x:0,y:0,drag:0,sx:0,sy:0};repaint()})}
window.addEventListener('mousemove',e=>{if(!window.view.drag)return;const d=window.devicePixelRatio||1;window.view.x+=(e.clientX-window.view.sx)*d;window.view.y+=(e.clientY-window.view.sy)*d;window.view.sx=e.clientX;window.view.sy=e.clientY;repaint()});
window.addEventListener('mouseup',()=>window.view.drag=0);
bindPanZoom($('field-canvas'));bindPanZoom($('vel0'));
// Node tooltip: nearest drawn node within 10 px of the pointer, from the targets each draw stores on its canvas.
function bindHover(c){const tip=c.parentElement.querySelector('.node-tip'),hide=()=>{tip.classList.add('hidden')};c.addEventListener('mousemove',e=>{const h=c._hit;if(!h||window.view.drag){hide();return}const r=c.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=100;for(let i=0;i<h.xs.length;i++){const dx=h.xs[i]-mx,dy=h.ys[i]-my,dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=i}}if(best<0){hide();return}const [text,col]=h.label(best);tip.innerHTML=`<i style="background:${col}"></i>${esc(text)}`;tip.style.left=h.xs[best]+'px';tip.style.top=h.ys[best]+'px';tip.style.transform=h.xs[best]>r.width-190?'translate(calc(-100% - 12px),-50%)':'translate(12px,-50%)';tip.classList.remove('hidden')});c.addEventListener('mouseleave',hide)}
bindHover($('field-canvas'));bindHover($('vel0'));
$('reset-view').addEventListener('click',()=>{window.view={s:1,x:0,y:0,drag:0,sx:0,sy:0};repaint()});
$('reset-view-flow').addEventListener('click',()=>{window.view={s:1,x:0,y:0,drag:0,sx:0,sy:0};repaint()});

run.addEventListener('click',()=>startJob('predict'));estimate.addEventListener('click',()=>startJob('estimate'));$('cancel-run').addEventListener('click',cancelJob);$('cancel-retrain').addEventListener('click',cancelRetrainJob);
})();
</script></body></html>'''


ROOT = get_project_root()
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
PIPELINE: CustomerDeployPipeline | None = None
REQUIRE_CUDA = True
WORKER_REQUESTS: Any | None = None
WORKER_EVENTS: Any | None = None
WORKER_PROCESS: Any | None = None
WORKER_LOCK = threading.Lock()
FINISHED_STATUSES = ("done", "error", "cancelled")
MAX_FINISHED_CSV_JOBS = 16
MAX_FINISHED_OTHER_JOBS = 32


def _bounded(req: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    """A numeric request field clamped to the range its UI control allows.

    The sliders bound every value in the browser, but a raw API call does not go through them:
    an unbounded width or wound fraction reaches the mesher and can hang or exhaust the worker.
    """
    raw = req.get(key, default)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.") from None
    if not math.isfinite(value):
        raise ValueError(f"'{key}' must be a finite number.")
    return min(hi, max(lo, value))


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"JSON constant {name} is not allowed.")


def _safe_name(name: str) -> str:
    raw = Path(str(name)).name
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return clean or "customer_geometry.pt"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _trajectory_payload(traj: Any, *, run_mode: str, csv_url: str | None = None) -> dict[str, Any]:
    times = [float(x) for x in np.asarray(traj.t_sec).reshape(-1)]
    frames = [traj.frame(i) for i in range(int(traj.n_steps))]
    phi = [np.round(np.asarray(fr["phi"], dtype=np.float32), 4).tolist() for fr in frames]
    include_velocity = bool((traj.meta or {}).get("include_velocity", False))
    velocity = None
    shear = None
    if include_velocity:
        velocity = [np.round(np.asarray(fr["vel_mag"], dtype=np.float32), 4).tolist() for fr in frames]
        shear_map = getattr(traj, "shear_mag", None) or {}
        if shear_map:
            zeros = np.zeros(len(traj.pos), dtype=np.float32)
            shear = [
                np.round(np.asarray(shear_map.get(i, zeros), dtype=np.float32), 4).tolist()
                for i in range(int(traj.n_steps))
            ]
    rows = trajectory_scientific_table(traj)
    masks = {
        "wall": np.asarray(traj.mask_wall if traj.mask_wall is not None else np.zeros(len(traj.pos)), dtype=bool).tolist(),
        "wound": np.asarray(
            traj.mask_wound if traj.mask_wound is not None else np.zeros(len(traj.pos)), dtype=bool
        ).tolist(),
        "inlet": np.asarray(traj.mask_inlet if traj.mask_inlet is not None else np.zeros(len(traj.pos)), dtype=bool).tolist(),
        "outlet": np.asarray(traj.mask_outlet if traj.mask_outlet is not None else np.zeros(len(traj.pos)), dtype=bool).tolist(),
    }
    meta = dict(traj.meta or {})
    meta.update({"run_mode": run_mode, "n_nodes": int(len(traj.pos)), "n_steps": int(traj.n_steps)})
    return _clean({
        "pos": np.round(np.asarray(traj.pos, dtype=np.float32), 5).tolist(),
        "times": times,
        "phi": phi,
        "velocity": velocity,
        "shear": shear,
        "masks": masks,
        "metrics": rows,
        "meta": meta,
        "csv_url": csv_url,
    })


# Control ranges, from the training envelope in customer_geometry_import.py (rounded inward to
# the slider step). test_customer_predict_web.py checks the page's sliders use exactly these.
PARAM_WIDTH_M = (math.ceil(TRAINED_INLET_WIDTH_M[0] * 2000) / 2000, math.floor(TRAINED_INLET_WIDTH_M[1] * 2000) / 2000)
PARAM_WIDTH_DEFAULT_M = 0.015   # the cohort's median inlet width
WOUND_POSITION_PCT = (math.ceil(TRAINED_WOUND_CENTER_FRAC[0] * 100), math.floor(TRAINED_WOUND_CENTER_FRAC[1] * 100))
WOUND_WIDTH_PCT = (math.ceil(TRAINED_WOUND_WIDTH_FRAC[0] * 100), math.floor(TRAINED_WOUND_WIDTH_FRAC[1] * 100))
WOUND_HORIZON_H = TRAINED_WOUND_HORIZON_S / SECONDS_PER_HOUR


def _parametric_params(req: dict[str, Any]) -> dict[str, Any]:
    """Build a clean, deterministic customer vessel from exactly the visible controls."""
    width = _bounded(req, "width", PARAM_WIDTH_DEFAULT_M, *PARAM_WIDTH_M)
    bend = math.radians(_bounded(req, "bend", 20.0, 0.0, 90.0))
    amp = _bounded(req, "amp", 0.0, 0.0, TRAINED_MAX_S_AMPLITUDE_M)
    cfg = VesselConfig(phase="kinematics")
    params = make_vessel_params(idx=0, level=0, cfg=cfg, width=width, angle_span=bend, amplitude=amp)
    # ``make_vessel_params`` begins from the training sampler, which intentionally includes
    # random pathology/noise.  A customer parametric vessel must instead be defined solely
    # by this UI, so clear that sampled state before applying an explicit preset below.
    n = int(cfg.num_ctrl_pts)
    params.update({
        "v_type": "straight",
        "offsets": [0.0] * n,
        "noise_top": [0.0] * n,
        "noise_bot": [0.0] * n,
        "tortuosity": [0.0] * max(n - 4, 0),
        "jitter": [],
        "path_loc": 2,
        "pathology_mode": "none",
    })
    if amp > 1e-9:
        params["curve_type"] = "s_curve"
    elif abs(bend) > 1e-9:
        params["curve_type"] = "arc"
    else:
        params["curve_type"] = "straight"
    pathology = str(req.get("pathology", "none"))
    if pathology in ("max_stenosis", "max_aneurysm", "stenosis", "aneurysm"):
        # ``pathology_strength`` is the percent change of the vessel width at the pathology's
        # centre: a 50% stenosis leaves 0.5x the healthy width there, a 50% aneurysm makes it
        # 1.5x. It is capped at the most severe pathology the model was TRAINED on (77% narrowing,
        # 99% widening), not the generator's geometric maximum (80% / 3x) -- and clamped here too,
        # since a raw API call can bypass the slider. Every trained pathology is centred with one
        # taper, so position and taper are fixed to the trained shape rather than exposed.
        stenosis = "stenosis" in pathology
        cap_pct = 100.0 * (TRAINED_STENOSIS_NARROWING if stenosis else TRAINED_ANEURYSM_WIDENING)
        change = _bounded(req, "pathology_strength", cap_pct, 0.0, cap_pct) / 100.0
        # Both walls move by the same offset, so the width changes by twice the per-wall offset.
        # Divided by the measured build gain so the BUILT throat has the requested depth.
        if stenosis:
            strength = change / (float(cfg.max_stenosis_diameter_occlusion) * STENOSIS_DEPTH_GAIN)
        else:
            # Every trained aneurysm bulges ONE wall (comsol040 top, comsol047 bottom, 2.975x the
            # half-width) and clots only in that sac; a symmetric two-wall bulge put 58% more
            # clot on comsol047's twin.  One wall carries the whole widening: offset = factor * width.
            strength = change / (float(cfg.max_aneurysm_factor) * ANEURYSM_DEPTH_GAIN)
        params = apply_customer_max_pathology(
            params,
            cfg,
            pathology,
            strength=strength,
            location=TRAINED_PATHOLOGY_LOCATION,
            sharpness=TRAINED_STENOSIS_SHARPNESS if stenosis else TRAINED_ANEURYSM_SHARPNESS,
            one_wall=not stenosis,
        )
    return params


def _parametric_data(req: dict[str, Any], *, re_target: float, t_final_s: float, n_steps: int):
    params = _parametric_params(req)
    try:
        data = build_parametric_customer_graph(
            re_target=re_target, t_final_s=t_final_s, n_steps=n_steps, params_override=params,
        )
    except Exception as exc:
        raise ValueError(
            f"{exc} (requested {float(params['width']) * 1e3:.2f} mm "
            f"{params['curve_type']} vessel)"
        ) from exc
    data.customer_parametric_spec = {
        "width_m": float(params["width"]),
        "bend_deg": 0.0 if str(params["curve_type"]) == "s_curve" else _bounded(req, "bend", 20.0, 0.0, 90.0),
        "s_amplitude_m": float(params["amplitude"]),
        "curve_type": str(params["curve_type"]),
        "pathology": str(params.get("pathology_mode", "none")),
    }
    return _apply_requested_wound(data, req)


def _apply_requested_wound(data: Any, req: dict[str, Any]) -> Any:
    # Toggle off means "add no wound", not "erase the geometry's own": disabling would restore
    # an uploaded pack's wound as healthy wall and silently run a wounded vessel as unwounded.
    # Parametric and mesh graphs are built fresh with no wound, so for them this is a no-op.
    if not bool(req.get("wound_enabled", False)):
        return data
    return apply_customer_mirrored_wound(
        data,
        enabled=bool(req.get("wound_enabled", False)),
        position_frac=_bounded(req, "wound_position", 50.0, *WOUND_POSITION_PCT) / 100.0,
        width_frac=_bounded(req, "wound_width", 10.0, *WOUND_WIDTH_PCT) / 100.0,
    )


def _get_pipeline() -> CustomerDeployPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = CustomerDeployPipeline(require_cuda=REQUIRE_CUDA)
    return PIPELINE


def _metrics_csv_path(job_id: str) -> Path:
    """Path of one job's metrics CSV.

    ``job_id`` reaches here from the request line, so it is rebuilt from its parsed UUID
    rather than interpolated: `uuid.UUID` rejects anything that is not 32 hex digits in the
    canonical layout, and `str()` of the result cannot contain a separator or a `..`.  The
    route's own regex and the `JOBS` membership check already make traversal unreachable --
    this is the belt that does not depend on either of them staying that way.
    """
    safe_id = str(uuid.UUID(str(job_id)))
    return ROOT / "outputs" / "customer_predict" / f"web_scientific_metrics_{safe_id}.csv"


def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(updates)
        if updates.get("status") in FINISHED_STATUSES:
            _prune_finished_jobs()


def _prune_finished_jobs() -> None:
    """Forget the oldest finished jobs so a long session does not grow without bound.

    A finished prediction holds every frame's fields, and each inbox selection adds a preview
    job, so JOBS is capped.  Jobs that produced a metrics CSV are counted separately: a burst of
    previews must not evict a prediction whose download link the page (or a snapshot) still shows.
    Caller holds JOBS_LOCK.
    """
    finished = sorted(
        (job.get("finished_at") or 0.0, job_id)
        for job_id, job in JOBS.items() if job.get("status") in FINISHED_STATUSES
    )
    with_csv = [j for _, j in finished if (JOBS[j].get("result") or {}).get("csv_url")]
    without_csv = [j for _, j in finished if not (JOBS[j].get("result") or {}).get("csv_url")]
    for job_id in with_csv[:-MAX_FINISHED_CSV_JOBS] + without_csv[:-MAX_FINISHED_OTHER_JOBS]:
        JOBS.pop(job_id, None)
        _metrics_csv_path(job_id).unlink(missing_ok=True)


def _remove_orphan_metrics_csvs() -> None:
    """Delete metrics CSVs left by an earlier server run: no job of this run can serve them."""
    for path in (ROOT / "outputs" / "customer_predict").glob("web_scientific_metrics_*.csv"):
        try:
            path.unlink()
        except OSError:
            pass


def _request_settings(req: dict[str, Any]) -> tuple[float, float, float, int, str]:
    # Scientific used to be a separate run mode; its metrics/figures are cheap enough that
    # every clot prediction now includes them, so there is nothing left to select here.
    # Every vessel clot_ml_0 trained on ran at Re 450, and the UI has no Re control; a raw API
    # value is still validated but clamped there (Re 250 put a quarter of a stenosis's clot at
    # the inlet, 2026-09-16).
    re_target = _bounded(req, "re", DEFAULT_RE, DEFAULT_RE, DEFAULT_RE)
    mode = "clot"
    if str(req.get("app_mode", "clot")) == "flow":
        # Flow Simulator shows the frozen t=0 local-FEM field only -- there is no clot
        # rollout, so the timeline collapses to a single instant regardless of the (hidden)
        # hours control's value.
        hours = 0.1
        t_final_s = hours * SECONDS_PER_HOUR
        return re_target, hours, t_final_s, 2, mode
    # Server-side too: the slider stops at the validated horizon, but a raw API call does not.
    # A wound was only ever simulated to ~12,000 s, so a wounded vessel stops there.
    max_s = TRAINED_WOUND_HORIZON_S if req.get("wound_enabled") else VALIDATED_HORIZON_S
    hours = _bounded(req, "hours", 8.0, 0.1, max_s / SECONDS_PER_HOUR)
    t_final_s = min(hours * SECONDS_PER_HOUR, max_s)
    # The model's own timeline -- not a resolution knob.  See `trained_time_axis`.
    n_steps = len(trained_time_axis(t_final_s))
    return re_target, hours, t_final_s, n_steps, mode


UPLOAD_ID_RE = re.compile(r"[0-9a-f]{32}\.(?:pt|msh|nas)")


def _upload_dir() -> Path:
    d = ensure_inbox() / "web_uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_upload(req: dict[str, Any]) -> Path:
    """Save an uploaded geometry under its content hash, or reuse one saved by an earlier request.

    A preview uploads the file once and hands back ``upload_id``; the run then sends only the id,
    so a large mesh is not re-sent. The id is matched against a strict pattern and resolved inside
    the upload folder only, so it can never name another path.
    """
    upload_id = req.get("upload_id")
    if upload_id and not req.get("data_b64"):
        if not isinstance(upload_id, str) or not UPLOAD_ID_RE.fullmatch(upload_id):
            raise ValueError("Invalid upload id.")
        path = _upload_dir() / upload_id
        if not path.is_file():
            raise ValueError("The uploaded file is no longer available. Choose it again.")
        return path
    suffix = Path(_safe_name(str(req.get("filename", "")))).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Unsupported geometry type. Use .msh, .nas, or .pt.")
    raw = base64.b64decode(str(req.get("data_b64", "")), validate=True)
    if not raw:
        raise ValueError("Choose a geometry file first.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Geometry file is too large (80 MB limit).")
    path = _upload_dir() / f"{hashlib.sha256(raw).hexdigest()[:32]}{suffix}"
    if not path.is_file():
        path.write_bytes(raw)
    return path


def _point(req: dict[str, Any], key: str) -> list[float] | None:
    """An optional [x, y] point in metres (a user-picked inlet/outlet location)."""
    raw = req.get(key)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"'{key}' must be an [x, y] pair.")
    try:
        out = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be an [x, y] pair of numbers.") from None
    if not all(math.isfinite(v) and abs(v) < 1e3 for v in out):
        raise ValueError(f"'{key}' must be finite.")
    return out


def _load_request_data(
    req: dict[str, Any], *, re_target: float, t_final_s: float, n_steps: int, progress: Any
) -> Any:
    source = str(req.get("source", "inbox"))
    if source == "parametric":
        progress("Building parametric geometry…")
        return _parametric_data(req, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps)
    if source == "upload":
        path = _store_upload(req)
        req["upload_id"] = path.name
        unit = str(req.get("mesh_unit") or "auto")
        if unit not in ("auto", *MESH_UNIT_SCALE):
            raise ValueError("'mesh_unit' must be auto, m, cm or mm.")
        progress(f"Loading {_safe_name(str(req.get('filename') or path.name))}…")
        data = load_customer_geometry(
            path, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps, mesh_unit=unit,
            inlet_hint=_point(req, "inlet_hint"), outlet_hint=_point(req, "outlet_hint"),
        )
        return _apply_requested_wound(data, req)
    filename = _safe_name(str(req.get("filename", "")))
    candidates = {p.name: p for p in list_inbox()}
    if filename not in candidates:
        raise ValueError("Select a geometry from the Inbox or upload one first.")
    progress(f"Loading {filename}…")
    data = load_customer_geometry(candidates[filename], re_target=re_target, t_final_s=t_final_s, n_steps=n_steps)
    return _apply_requested_wound(data, req)


def _runtime_estimate(data: Any, *, n_steps: int, hours: float) -> dict[str, Any]:
    """Return a transparent, conservative local-runtime estimate in seconds."""
    n_nodes = int(getattr(data, "num_nodes", 0) or len(data.x))
    edge_index = getattr(data, "edge_index", None)
    n_edges = int(edge_index.shape[1]) if edge_index is not None else 0
    wound = getattr(data, "mask_wound", None)
    n_wound = int(wound.reshape(-1).bool().sum().item()) if wound is not None else 0
    # Calibrated to the deployed sequential rollout, with a deliberately broad range
    # for GPU model load, mesh density, and the local flow-correction work per step.
    rollout_s = n_steps * (0.08 + 0.00008 * n_nodes + 0.000006 * n_edges)
    return {
        "kind": "estimate",
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_wound": n_wound,
        "geometry": dict(getattr(data, "customer_parametric_spec", {}) or {}),
        "n_steps": n_steps,
        "hours": hours,
        "estimate_low_s": round(max(2.0, 0.70 * rollout_s), 1),
        "estimate_mid_s": round(max(3.0, rollout_s), 1),
        "estimate_high_s": round(max(5.0, 1.80 * rollout_s), 1),
        "note": "Estimate for this geometry and simulated horizon; the first GPU run can take longer while models load.",
    }


# Progress stages the page's bar shows, keyed by the message prefixes the request path and
# CustomerDeployPipeline.run() already log. A message that matches none keeps the current stage.
PROGRESS_STAGES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("Preparing prediction", "Building parametric geometry", "Loading ")),
    (1, ("Running customer deploy model", "[i] Solving local FEM")),
    (2, ("[i] Rolling out",)),
    (3, ("[i] Using t=0 flow field", "[OK] clot_ml_0 rollout done")),
)


def _progress_stage(message: str) -> int | None:
    for stage, prefixes in PROGRESS_STAGES:
        if message.startswith(prefixes):
            return stage
    return None


def _report(progress: Any, message: str, **extra: Any) -> None:
    """Send a progress message with structured fields when the callback accepts them."""
    try:
        progress(message, **extra)
    except TypeError:
        progress(message)


def _execute_request(job_id: str, req: dict[str, Any], progress: Any) -> dict[str, Any]:
    if str(req.get("kind")) == "retrain":
        progress("Starting retrain pipeline…")
        data_dir_path = str(req.get("data_dir", ""))
        if not data_dir_path:
            raise ValueError("Data directory is required for retraining.")
        target_dir = Path(data_dir_path)
        if not target_dir.is_dir():
            raise ValueError(f"Directory not found: {data_dir_path}")
        pipeline = CustomerRetrainPipeline(require_cuda=REQUIRE_CUDA)
        ok = pipeline.run(target_dir, progress, progress)
        if not ok:
            raise RuntimeError("Retrain failed. Check logs for details.")
        return {"kind": "retrain", "message": "Candidate model saved — see log for scores. Not live until a researcher promotes it."}

    re_target, hours, t_final_s, n_steps, mode = _request_settings(req)
    try:
        data = _load_request_data(
            req, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps, progress=progress
        )
    except BoundaryNeedsInput as exc:
        if str(req.get("kind")) != "preview":
            raise
        # Still show the vessel so the user can click its inlet and outlet.
        n = len(exc.pos)
        return _clean({
            "kind": "preview",
            "needs_boundary": True,
            "message": str(exc),
            "upload_id": req.get("upload_id"),
            "pos": np.round(exc.pos.astype(np.float32), 7).tolist(),
            "pos_scale_m": 1.0,
            "masks": {"wall": exc.boundary_mask.tolist(), "wound": [False] * n,
                      "inlet": [False] * n, "outlet": [False] * n},
            "boundary": {"unit": exc.unit},
        })
    if str(req.get("kind", "predict")) == "estimate":
        return _runtime_estimate(data, n_steps=n_steps, hours=hours)
    if str(req.get("kind", "predict")) == "preview":
        # Geometry only -- no FEM solve, no model rollout -- so picking a file (or dragging
        # its sliders, for a parametric vessel) shows the actual mesh immediately instead of
        # a blank canvas until the first full run.
        n = int(getattr(data, "num_nodes", 0) or len(data.x))
        pos = np.round(data.x[:, :2].detach().cpu().numpy().astype(np.float32), 5).tolist()

        def _mask_list(name: str) -> list[bool]:
            m = getattr(data, name, None)
            if m is None:
                return [False] * n
            return np.asarray(m, dtype=bool).reshape(-1).tolist()

        return _clean({
            "kind": "preview",
            "pos": pos,
            "masks": {
                "wall": _mask_list("mask_wall"),
                "wound": _mask_list("mask_wound"),
                "inlet": _mask_list("mask_inlet"),
                "outlet": _mask_list("mask_outlet"),
            },
            "meta": {"geometry": dict(getattr(data, "customer_parametric_spec", {}) or {})},
            "upload_id": req.get("upload_id"),
            "boundary": dict(getattr(data, "customer_boundary", {}) or {}),
            # Preview positions are non-dimensional (units of d_bar); the page multiplies a picked
            # point by this to send an inlet/outlet hint in metres.
            "pos_scale_m": float(data.d_bar.reshape(-1)[0]) if hasattr(data, "d_bar") else 1.0,
        })

    estimate = _runtime_estimate(data, n_steps=n_steps, hours=hours)
    _report(progress, "Running customer deploy model…", expected_rollout_s=estimate["estimate_mid_s"])
    # Flow Simulator wants the velocity field for its own display; Clot mode's scientific
    # metrics (mean_vel_open_lumen, vel_open_lumen_drop_pct) need it too, and it is always
    # computed now that scientific figures ship with every clot prediction.
    traj = _get_pipeline().run(
        data,
        t_final_s=t_final_s,
        progress=progress,
        include_velocity=True,
    )
    csv_url = None
    if str(req.get("app_mode")) != "flow":
        csv_path = _metrics_csv_path(job_id)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_scientific_csv(csv_path, trajectory_scientific_table(traj))
        csv_url = f"/api/job/{job_id}/csv"
    return _trajectory_payload(traj, run_mode=mode, csv_url=csv_url)


def _worker_main(requests: Any, events: Any, require_cuda: bool) -> None:
    """Run model work in a process whose main thread owns signal registration."""
    global PIPELINE, REQUIRE_CUDA
    REQUIRE_CUDA = require_cuda
    PIPELINE = None
    while True:
        item = requests.get()
        if item is None:
            return
        job_id, req = item
        started = time.perf_counter()
        events.put({"type": "running", "job_id": job_id, "message": "Preparing prediction…"})
        try:
            def progress(msg: str, **extra: Any) -> None:
                events.put({"type": "progress", "job_id": job_id, "message": str(msg), **extra})

            result = _execute_request(job_id, req, progress)
            elapsed_s = round(time.perf_counter() - started, 2)
            done_message = (result.get("message") if isinstance(result, dict) else None) or "Prediction complete."
            events.put({
                "type": "done", "job_id": job_id, "message": done_message,
                "result": result, "elapsed_s": elapsed_s,
            })
        except Exception as exc:
            traceback.print_exc()
            events.put({
                "type": "error", "job_id": job_id, "message": "Prediction failed.", "error": str(exc),
            })


def _drain_worker_events(events: Any, process: Any) -> None:
    # Bound to one specific (events, process) pair at thread-start rather than reading the
    # WORKER_EVENTS/WORKER_PROCESS globals each iteration: cancelling a job restarts the worker,
    # which reassigns those globals out from under a still-running drain thread -- it would start
    # reading the *new* queue on top of the fresh thread `_start_inference_worker` also spawns for
    # it, two consumers racing on one queue, and never see its own (now-replaced) process die.
    while True:
        try:
            event = events.get(timeout=0.25)
        except queue.Empty:
            if process is not None and not process.is_alive():
                return
            continue
        if event is None:
            return
        event_type = event.get("type")
        job_id = event.get("job_id")
        if not job_id:
            continue
        if event_type in ("running", "progress"):
            message = event.get("message", "Running…")
            updates: dict[str, Any] = {"status": "running", "message": message}
            stage = _progress_stage(str(message))
            with JOBS_LOCK:
                current = JOBS.get(job_id, {}).get("stage")
            if stage is not None and (current is None or stage > current):
                updates.update(stage=stage, stage_started_at=time.time())
            if event.get("expected_rollout_s") is not None:
                updates["expected_rollout_s"] = event["expected_rollout_s"]
            _set_job(job_id, **updates)
        elif event_type == "done":
            _set_job(
                job_id, status="done", message=event.get("message", "Prediction complete."),
                result=event.get("result"), elapsed_s=event.get("elapsed_s"), finished_at=time.time(),
            )
        elif event_type == "error":
            _set_job(
                job_id, status="error", message=event.get("message", "Prediction failed."),
                error=event.get("error", "Unknown worker error."), finished_at=time.time(),
            )


def _start_inference_worker(require_cuda: bool) -> None:
    global WORKER_EVENTS, WORKER_PROCESS, WORKER_REQUESTS
    context = mp.get_context("spawn")
    WORKER_REQUESTS = context.Queue()
    WORKER_EVENTS = context.Queue()
    WORKER_PROCESS = context.Process(
        target=_worker_main, args=(WORKER_REQUESTS, WORKER_EVENTS, require_cuda), daemon=True
    )
    WORKER_PROCESS.start()
    threading.Thread(
        target=_drain_worker_events, args=(WORKER_EVENTS, WORKER_PROCESS),
        name="hemorgp-web-events", daemon=True,
    ).start()


def _stop_inference_worker(*, force: bool = False) -> None:
    if not force and WORKER_REQUESTS is not None:
        try:
            WORKER_REQUESTS.put(None)
        except Exception:
            pass
    if WORKER_PROCESS is not None:
        # A cancelled job is very likely mid rollout, blocked on GPU work with no chance of ever
        # noticing the sentinel above -- skip the graceful wait and terminate immediately so
        # Cancel actually feels immediate instead of stalling for the full grace period.
        if not force:
            WORKER_PROCESS.join(timeout=8)
        if WORKER_PROCESS.is_alive():
            WORKER_PROCESS.terminate()
            WORKER_PROCESS.join(timeout=5)


def _cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel the job's own worker process and put a fresh one in its place.

    The worker is a single long-lived process draining requests one at a time with no
    cancellation hook inside a running rollout, so "cancel the current job" and "cancel whatever
    the worker happens to be doing right now" are the same operation here -- there is only ever
    one job in flight from this UI at a time. Terminating is what actually stops in-flight GPU
    work; anything still sitting in the (now-discarded) request queue is dropped along with it.
    """
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))
    if not job:
        return {"error": "Unknown job."}
    if job.get("status") in FINISHED_STATUSES:
        return job
    with WORKER_LOCK:
        _stop_inference_worker(force=True)
        _set_job(job_id, status="cancelled", message="Cancelled.", finished_at=time.time())
        _start_inference_worker(REQUIRE_CUDA)
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


class Handler(BaseHTTPRequestHandler):
    server_version = "ClotMLWeb/1.0"
    # Default is HTTP/1.0, which closes the TCP connection after every single response --
    # the status-polling loop alone fires every 700ms, so that is a fresh connect/accept/close
    # cycle per poll. On Windows that churn occasionally loses the race (a request lands while
    # the prior connection is still tearing down) and the browser reports it as "Failed to
    # fetch". HTTP/1.1 keep-alive lets the browser reuse one connection for the whole polling
    # sequence; every response already sets Content-Length, which is what keep-alive requires.
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _trusted_request(self, *, post: bool) -> bool:
        """Refuse requests that did not come from this app's own page.

        Binding to loopback does not stop a web page in the user's browser from reaching the
        server: a cross-site ``text/plain`` POST needs no CORS preflight, and DNS rebinding
        points a foreign hostname at 127.0.0.1.  Either would let any site start jobs here.
        So the Host header must name this server, a present Origin must be this server, and
        a POST must be ``application/json`` -- which a cross-site page cannot send without a
        preflight this server never answers.
        """
        allowed = getattr(self.server, "allowed_hosts", set())
        if self.headers.get("Host", "").lower() not in allowed:
            self._json({"error": "Forbidden host."}, HTTPStatus.FORBIDDEN)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower().removeprefix("http://") not in allowed:
            self._json({"error": "Forbidden origin."}, HTTPStatus.FORBIDDEN)
            return False
        if post:
            ctype = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                self._json({"error": "Expected application/json."},
                           HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return False
        return True

    def _json(self, value: Any, status: int = 200) -> None:
        self._send(json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if not self._trusted_request(post=False):
            return
        if path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/geometries":
            self._json([p.name for p in list_inbox()])
            return
        match = re.fullmatch(r"/api/job/([0-9a-f-]+)(/csv)?", path)
        if match:
            job_id = match.group(1)
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job:
                self._json({"error": "Unknown job."}, HTTPStatus.NOT_FOUND)
                return
            if match.group(2):
                try:
                    csv_path = _metrics_csv_path(job_id)
                except ValueError:
                    # Not a UUID, so no job of this run owns it -- same answer as an unknown job.
                    self._json({"error": "Unknown job."}, HTTPStatus.NOT_FOUND)
                    return
                if not csv_path.is_file():
                    self._json({"error": "CSV is not available for this job."}, HTTPStatus.NOT_FOUND)
                else:
                    self._send(csv_path.read_bytes(), "text/csv; charset=utf-8")
                return
            self._json(job)
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if not self._trusted_request(post=True):
            return
        # Every branch below must drain the request body off the socket before responding,
        # even branches that ignore it: on a keep-alive connection, unread bytes stay buffered
        # and get parsed as the start of the *next* request line, corrupting it (e.g. a stray
        # "{}" here turned the following GET into an unparseable "{}GET" method).
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        if path == "/api/open-folder":
            import os
            import subprocess

            folder = ensure_inbox()
            try:
                if os.name == "nt":
                    os.startfile(str(folder))  # type: ignore[attr-defined]
                else:
                    import sys

                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.Popen([opener, str(folder)])
            except Exception:
                pass
            self._json({"ok": True})
            return
        cancel_match = re.fullmatch(r"/api/job/([0-9a-f-]+)/cancel", path)
        if cancel_match:
            self._json(_cancel_job(cancel_match.group(1)))
            return
        if path != "/api/run":
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        try:
            if content_length <= 0 or content_length > MAX_UPLOAD_BYTES * 2:
                raise ValueError("Request is empty or too large.")
            req = json.loads(raw_body.decode("utf-8"), parse_constant=_reject_json_constant)
            if not isinstance(req, dict):
                raise ValueError("Request must be a JSON object.")
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        job_id = str(uuid.uuid4())
        with WORKER_LOCK:
            if WORKER_PROCESS is None or WORKER_REQUESTS is None or not WORKER_PROCESS.is_alive():
                self._json(
                    {"error": "The inference worker is not available. Restart the customer web launcher."},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            _set_job(job_id, status="queued", message="Queued…", created_at=time.time(), kind=req.get("kind", "predict"))
            WORKER_REQUESTS.put((job_id, req))
        self._json({"job_id": job_id}, HTTPStatus.ACCEPTED)


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def allowed_hosts(host: str, port: int) -> set[str]:
    """``Host`` header values that name this server: loopback aliases plus the bound host."""
    names = set(LOOPBACK_HOSTS) | {host}
    return {(f"[{h}]" if ":" in h else h).lower() + f":{port}" for h in names}


class Server(ThreadingHTTPServer):
    # socketserver.TCPServer's default listen backlog is 5 -- far too small for a UI that
    # polls job status every 700ms (plus canvas/geometry fetches), especially across more than
    # one browser tab. Once 5 connections are pending accept() at once, the OS refuses any
    # further connection outright (ERR_CONNECTION_REFUSED / "Failed to fetch"), even though the
    # server process is alive and healthy. Must be a class attribute: TCPServer.__init__ calls
    # listen() during construction, so setting this on an instance afterward is too late.
    request_queue_size = 128


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ClotML customer browser UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cpu", action="store_true", help="Allow CPU inference (slow; CUDA recommended)")
    args = ap.parse_args(argv)
    global REQUIRE_CUDA
    REQUIRE_CUDA = not args.cpu
    ensure_inbox()
    _remove_orphan_metrics_csvs()
    _start_inference_worker(REQUIRE_CUDA)
    server = Server((args.host, args.port), Handler)
    server.allowed_hosts = allowed_hosts(args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    if args.host not in LOOPBACK_HOSTS:
        print(f"[WARN] Listening on {args.host}: anyone who can reach this address can run jobs "
              "and upload geometries. There is no authentication.", flush=True)
    print(f"[i] ClotML web predict: {url}", flush=True)
    print(f"[i] Geometries folder: {ensure_inbox()}", flush=True)
    if args.cpu:
        print("[WARN] CPU mode (slow). CUDA is recommended.", flush=True)
    # A double-click launcher (the packaged customer bundle's run.bat) has no terminal a
    # non-technical user would think to read a URL from -- open the browser for them. Off the
    # main thread so it can't delay serve_forever() below; harmless no-op if a browser can't
    # be found (e.g. a headless dev box), since webbrowser.open() just returns False then.
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[i] Stopping web predict server.", flush=True)
    finally:
        server.server_close()
        _stop_inference_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
