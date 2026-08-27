"""Local browser UI for customer vessel prediction.

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
import json
import math
import multiprocessing as mp
import queue
import re
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from src.config import VesselConfig
from src.data_gen.lib.customer_geometry_import import (
    SUPPORTED_SUFFIXES,
    apply_customer_mirrored_wound,
    apply_customer_max_pathology,
    build_parametric_customer_graph,
    ensure_inbox,
    list_inbox,
    load_customer_geometry,
)
from src.data_gen.lib.vessel_generator import make_vessel_params
from src.inference.customer_pipeline import CustomerDeployPipeline
from src.tools.customer_predict_metrics import trajectory_scientific_table, write_scientific_csv
from src.utils.paths import get_project_root


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HemoRGP — Customer vessel forecast</title>
<style>
:root{--bg:#f4f0e8;--surface:#fffdf9;--surface2:#f1ece3;--line:#d9d0c2;--ink:#2b261f;--muted:#756c60;--accent:#9a5a13;--accent2:#b2732d;--teal:#8fbdb6;--green:#4f8a66;--red:#c25b4c;--shadow:0 8px 28px rgba(76,58,34,.08);--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;--serif:ui-serif,Georgia,serif;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px}button,input,select{font:inherit;color:inherit}button,select,input{accent-color:var(--accent)}
.topbar{height:68px;margin:18px 22px 0;padding:0 26px;background:var(--surface);border:1px solid var(--line);display:flex;align-items:center;gap:18px;box-shadow:var(--shadow)}
.brand{font-family:var(--serif);font-size:25px;font-weight:600;color:var(--accent)}.brand-sub{font-family:var(--serif);font-size:24px;font-weight:600}.topbar-note{color:var(--muted);font-size:13px}.status{margin-left:auto;color:var(--muted);font-size:12px;text-align:right;max-width:44%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.status.ok{color:var(--green)}.status.err{color:var(--red)}
.shell{display:grid;grid-template-columns:284px minmax(0,1fr);gap:18px;margin:18px 22px 24px}.rail,.workspace-card,.stat{background:var(--surface);border:1px solid var(--line);box-shadow:var(--shadow)}.rail{padding:20px 18px;min-height:calc(100vh - 128px)}
.section{padding:0 0 18px;margin:0 0 18px;border-bottom:1px solid var(--line)}.section:last-child{border-bottom:0;margin-bottom:0}.section h2{margin:0 0 12px;font:600 16px var(--serif);color:var(--accent)}label{display:block;color:var(--muted);font-size:12px;margin:10px 0 5px}.field{width:100%;background:var(--surface2);border:1px solid var(--line);border-radius:5px;padding:9px 10px}.field:focus{outline:2px solid color-mix(in srgb,var(--accent) 35%,transparent);outline-offset:1px}.range-row{display:grid;grid-template-columns:1fr 52px;gap:10px;align-items:center}.range-row output{font-variant-numeric:tabular-nums;color:var(--ink);text-align:right}.hint{font-size:11px;line-height:1.4;color:var(--muted);margin:10px 0 0}.inline{display:flex;gap:8px}.inline>*{flex:1}.btn{border:1px solid var(--line);border-radius:5px;background:var(--surface2);padding:9px 10px;cursor:pointer}.btn:hover{background:#e5dbcb}.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--surface);font-weight:600;width:100%;margin-top:5px}.btn:disabled{opacity:.55;cursor:wait}.preset{margin-top:8px}.hidden{display:none!important}
.workspace{min-width:0;display:grid;gap:14px}.workspace-card{padding:18px}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.stat{padding:13px 15px;box-shadow:none}.stat label{margin:0 0 5px}.stat strong{display:block;font:600 23px var(--serif);font-variant-numeric:tabular-nums}.stat span{display:block;color:var(--muted);font-size:11px;margin-top:3px}.viewer-head{display:flex;align-items:baseline;gap:12px;margin-bottom:12px}.viewer-head h1{margin:0;font:600 22px var(--serif)}.viewer-head span{margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.canvas-wrap{position:relative;background:var(--surface2);border:1px solid var(--line);min-height:430px}.canvas-wrap canvas{display:block;width:100%;height:430px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:var(--muted);font-size:11px}.key{display:inline-flex;align-items:center;gap:5px}.swatch{width:10px;height:10px;display:inline-block}.swatch.square{background:var(--teal)}.swatch.circle{background:var(--accent);border-radius:50%}.swatch.ring{border:2px solid var(--accent);border-radius:50%}
.velocity-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.velocity-grid .canvas-wrap{min-height:360px}.velocity-grid canvas{height:360px}.timeline{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin-top:14px;padding:12px 14px;background:var(--surface2);border:1px solid var(--line)}.timeline label{margin:0;font:600 13px var(--serif);color:var(--accent)}.timeline input{width:100%}.timeline output{min-width:112px;text-align:right;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.science{margin-top:14px}.science h2{margin:0 0 8px;font:600 16px var(--serif);color:var(--accent)}#science-canvas{width:100%;height:210px;display:block;background:var(--surface2);border:1px solid var(--line)}.download{display:inline-block;margin-top:9px;color:var(--accent);font-size:12px}.error{color:var(--red);font-size:12px;line-height:1.4;margin-top:10px}
@media(max-width:900px){.shell{grid-template-columns:1fr}.rail{min-height:0}.topbar{margin-left:12px;margin-right:12px}.shell{margin-left:12px;margin-right:12px}.status{max-width:35%}}@media(max-width:560px){.topbar{padding:0 14px;gap:9px}.brand,.brand-sub{font-size:20px}.topbar-note{display:none}.stats{grid-template-columns:1fr}.velocity-grid{grid-template-columns:1fr}.canvas-wrap canvas{height:320px}.canvas-wrap{min-height:320px}.timeline{grid-template-columns:1fr}.timeline output{text-align:left}}
</style></head>
<body>
<header class="topbar"><span class="brand">HemoRGP</span><span class="brand-sub">Predict</span><span class="topbar-note">clot_ml_0 customer vessel forecast</span><span id="status" class="status" aria-live="polite">Ready — choose a geometry to begin.</span></header>
<main class="shell">
<aside class="rail">
  <section class="section"><h2>Geometry</h2>
    <label for="source">Source</label><select id="source" class="field"><option value="inbox">Inbox geometry</option><option value="upload">Upload geometry</option><option value="parametric">Parametric vessel</option></select>
    <div id="inbox-controls"><label for="inbox">Geometry file</label><select id="inbox" class="field"></select><div class="inline" style="margin-top:8px"><button id="refresh" class="btn" type="button">Refresh</button><button id="open-folder" class="btn" type="button">Open folder</button></div></div>
    <div id="upload-controls" class="hidden"><label for="upload">Geometry file</label><input id="upload" class="field" type="file" accept=".pt,.msh,.nas,.json" multiple><p class="hint">For tagged .msh/.nas, select the matching .json sidecar too.</p></div>
    <div id="param-controls" class="hidden"><label for="width">Width (m)</label><div class="range-row"><input id="width" type="range" min="0.004" max="0.012" step="0.0005" value="0.008"><output id="width-out">0.008</output></div><label for="bend">Bend (deg)</label><div class="range-row"><input id="bend" type="range" min="0" max="90" step="1" value="20"><output id="bend-out">20</output></div><label for="amp">S-amp (m)</label><div class="range-row"><input id="amp" type="range" min="0" max="0.012" step="0.0005" value="0"><output id="amp-out">0.000</output></div><label for="pathology">Pathology preset</label><select id="pathology" class="field"><option value="none">None</option><option value="max_stenosis">Max stenosis</option><option value="max_aneurysm">Max aneurysm</option></select></div>
  </section>
  <section class="section"><h2>Conditions</h2><label for="re">Inlet Re</label><div class="range-row"><input id="re" type="range" min="100" max="900" step="10" value="450"><output id="re-out">450</output></div><label for="hours">Simulation time (hrs)</label><div class="range-row"><input id="hours" type="range" min="1" max="50" step="0.5" value="8"><output id="hours-out">8</output></div></section>
  <section class="section"><h2>Run mode</h2><select id="run-mode" class="field"><option value="clot">Clot</option><option value="clot_velocity">Clot + Velocity</option><option value="scientific">Scientific</option></select><button id="run" class="btn primary" type="button">Run prediction</button><p id="error" class="error hidden" role="alert"></p></section>
  <section class="section"><h2>Wound</h2><label><input id="wound-enabled" type="checkbox"> Add mirrored wound</label><div id="wound-controls" class="hidden"><label for="wound-position">Position along vessel (%)</label><div class="range-row"><input id="wound-position" type="range" min="5" max="95" step="1" value="50"><output id="wound-position-out">50</output></div><label for="wound-width">Mirrored wound width (%)</label><div class="range-row"><input id="wound-width" type="range" min="2" max="60" step="1" value="15"><output id="wound-width-out">15</output></div><p class="hint">The same axial segment is placed on both vessel walls.</p></div></section>
  <p id="estimate-note" class="hint">Runtime depends on mesh size and simulated time.</p><button id="estimate" class="btn" type="button">Estimate runtime</button>
</aside>
<section class="workspace">
  <div class="stats"><div class="stat"><label>Wall coverage</label><strong id="stat-wall">—</strong><span id="stat-wall-sub">selected frame</span></div><div class="stat"><label>Vessel coverage</label><strong id="stat-vessel">—</strong><span id="stat-vessel-sub">selected frame</span></div><div class="stat"><label>Lumen occlusion</label><strong id="stat-occ">—</strong><span id="stat-occ-sub">wall-hop estimate</span></div></div>
  <section class="workspace-card"><div class="viewer-head"><h1 id="viewer-title">Geometry preview</h1><span id="viewer-time">No prediction yet</span></div><div id="field-wrap" class="canvas-wrap"><canvas id="field-canvas" aria-label="Clot prediction field"></canvas></div><div id="velocity-wrap" class="velocity-grid hidden"><div><div class="canvas-wrap"><canvas id="vel0" aria-label="Initial velocity field"></canvas></div></div><div><div class="canvas-wrap"><canvas id="vel1" aria-label="Final velocity field"></canvas></div></div></div><div class="legend"><span class="key"><i class="swatch square"></i>open lumen</span><span class="key"><i class="swatch circle"></i>clot fraction</span><span class="key"><i class="swatch ring"></i>wall boundary</span></div><div class="timeline"><label for="time">Timeline</label><input id="time" type="range" min="0" max="0" step="1" value="0" disabled><output id="time-out">Preview</output></div><div id="science" class="science hidden"><h2>Scientific metrics</h2><canvas id="science-canvas" aria-label="Scientific metrics over time"></canvas><a id="download" class="download hidden" download>Download metrics CSV</a></div></section>
</section></main>
<script>
(() => {
const $=id=>document.getElementById(id), state={data:null,mode:'clot',frame:0,job:null,jobKind:'predict'};
const source=$('source'), inbox=$('inbox'), status=$('status'), error=$('error'), run=$('run'), estimate=$('estimate'), time=$('time');
const setStatus=(msg,tone='')=>{status.textContent=msg;status.className='status '+tone};
const fmt=(x,d=1)=>Number.isFinite(Number(x))?Number(x).toFixed(d):'—';
function resetPreview(){const c=$('field-canvas'),[w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.fillStyle='#f1ece3';ctx.fillRect(0,0,w,h);ctx.fillStyle='#756c60';ctx.font='13px ui-sans-serif';ctx.fillText('Choose a geometry and run prediction to inspect the forecast.',18,24);$('field-wrap').classList.remove('hidden');$('velocity-wrap').classList.add('hidden');$('science').classList.add('hidden');$('viewer-title').textContent='Geometry preview';$('viewer-time').textContent='No prediction yet';time.disabled=true;$('time-out').textContent='Preview';['stat-wall','stat-vessel','stat-occ'].forEach(id=>$(id).textContent='—')}
async function refreshInbox(){try{const r=await fetch('/api/geometries');const names=await r.json();inbox.innerHTML=names.map(n=>`<option value="${esc(n)}">${esc(n)}</option>`).join('');if(!names.length)inbox.innerHTML='<option value="">No geometry files</option>';}catch(e){setStatus('Could not list geometries','err')}}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function markGeometryDirty(){state.data=null;state.frame=0;$('download').classList.add('hidden');if(source.value==='parametric')renderPreview();else resetPreview();setStatus('Geometry changed — ready for a new prediction.','ok')}
function syncWoundControls(){$('wound-controls').classList.toggle('hidden',!$('wound-enabled').checked)}
function toggleSource(){const v=source.value;$('inbox-controls').classList.toggle('hidden',v!=='inbox');$('upload-controls').classList.toggle('hidden',v!=='upload');$('param-controls').classList.toggle('hidden',v!=='parametric');state.data=null;state.frame=0;if(v==='parametric')renderPreview();else resetPreview();}
function bindRange(id,out,digits=0){const el=$(id),o=$(out);const sync=()=>o.value=Number(el.value).toFixed(digits);el.addEventListener('input',sync);sync()}
bindRange('width','width-out',3);bindRange('bend','bend-out');bindRange('amp','amp-out',3);bindRange('re','re-out');bindRange('hours','hours-out',1);bindRange('wound-position','wound-position-out');bindRange('wound-width','wound-width-out');source.addEventListener('change',toggleSource);['width','bend','amp','pathology','wound-position','wound-width'].forEach(id=>{['input','change'].forEach(ev=>$(id).addEventListener(ev,markGeometryDirty))});$('wound-enabled').addEventListener('change',()=>{syncWoundControls();markGeometryDirty()});$('refresh').addEventListener('click',refreshInbox);$('open-folder').addEventListener('click',()=>fetch('/api/open-folder',{method:'POST'}));refreshInbox();syncWoundControls();toggleSource();
function resizeCanvas(c){const r=c.getBoundingClientRect(),d=Math.max(1,window.devicePixelRatio||1);c.width=Math.max(1,Math.round(r.width*d));c.height=Math.max(1,Math.round(r.height*d));return [c.width/d,c.height/d,d]}
function extent(data){let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;for(const p of data.pos){x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1])}return [x0,x1,y0,y1]}
function renderPreview(){if(source.value!=='parametric'||state.data)return;const c=$('field-canvas'),[w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#f1ece3';ctx.fillRect(0,0,w,h);const width=Number($('width').value),bend=Number($('bend').value)*Math.PI/180,amp=Number($('amp').value),arcBend=amp>0?0:bend,pathology=$('pathology').value,woundOn=$('wound-enabled').checked,woundCenter=Number($('wound-position').value)/100,woundWidth=Number($('wound-width').value)/100,pad=38,mid=h/2,base=Math.max(28,Math.min(h*.25,width/.012*h*.62)),pts=[];for(let i=0;i<=96;i++){const q=i/96,x=pad+q*(w-2*pad),center=mid+Math.sin(q*Math.PI)*arcBend*h*.18+(amp/.012)*Math.sin(q*Math.PI*2)*h*.28;let radius=base;if(pathology==='max_stenosis')radius*=1-.48*Math.exp(-Math.pow((q-.52)/.16,2));if(pathology==='max_aneurysm')radius*=1+.48*Math.exp(-Math.pow((q-.52)/.18,2));pts.push([x,center,radius])}ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]-p[2]):ctx.moveTo(p[0],p[1]-p[2]));for(let i=pts.length-1;i>=0;i--){const p=pts[i];ctx.lineTo(p[0],p[1]+p[2])}ctx.closePath();ctx.fillStyle='#8fbdb6';ctx.fill();ctx.strokeStyle='#9a5a13';ctx.lineWidth=2;ctx.stroke();if(woundOn){const selected=pts.filter((_p,i)=>Math.abs(i/96-woundCenter)<=woundWidth/2);for(const sign of [-1,1]){ctx.strokeStyle='#c25b4c';ctx.lineWidth=6;ctx.lineCap='round';ctx.beginPath();selected.forEach((p,i)=>{const y=p[1]+sign*p[2];i?ctx.lineTo(p[0],y):ctx.moveTo(p[0],y)});ctx.stroke()}ctx.lineCap='butt'}ctx.strokeStyle='rgba(255,253,249,.72)';ctx.lineWidth=1;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.fillStyle='#756c60';ctx.font='12px ui-sans-serif';ctx.fillText(woundOn?'Parametric preview — mirrored wound shown in red':'Parametric geometry preview — run prediction to simulate clot',18,22);$('field-wrap').classList.remove('hidden');$('velocity-wrap').classList.add('hidden');$('science').classList.add('hidden');$('viewer-title').textContent='Parametric vessel preview';$('viewer-time').textContent='Not simulated';time.disabled=true;$('time-out').textContent='Preview'}
function drawField(c,idx){const d=state.data;if(!d)return;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--surface2');ctx.fillRect(0,0,w,h);const [x0,x1,y0,y1]=extent(d),pad=28,s=Math.min((w-2*pad)/Math.max(x1-x0,1e-9),(h-2*pad)/Math.max(y1-y0,1e-9)),ox=pad+((w-2*pad)-s*(x1-x0))/2,oy=pad+((h-2*pad)-s*(y1-y0))/2,px=p=>ox+(p[0]-x0)*s,py=p=>oy+(y1-p[1])*s,phi=d.phi[idx]||d.phi[0],wall=d.masks.wall||[],wound=d.masks.wound||[];ctx.fillStyle='#8fbdb6';for(let i=0;i<d.pos.length;i++){if(phi[i]>=0.45)continue;const p=d.pos[i];ctx.fillRect(px(p)-1.5,py(p)-1.5,3,3)}for(let i=0;i<d.pos.length;i++){if(phi[i]<0.45)continue;const p=d.pos[i],q=Math.max(0,Math.min(1,phi[i])),r=2.5+7*Math.pow(q,1.25);ctx.fillStyle=`rgb(${Math.round(237-99*q)},${Math.round(194-128*q)},${Math.round(125-106*q)})`;ctx.beginPath();ctx.arc(px(p),py(p),r,0,Math.PI*2);ctx.fill()}ctx.strokeStyle='#9a5a13';ctx.lineWidth=.7;for(let i=0;i<d.pos.length;i++){if(!wall[i])continue;const p=d.pos[i];ctx.beginPath();ctx.arc(px(p),py(p),4,0,Math.PI*2);ctx.stroke()}ctx.fillStyle='#c25b4c';for(let i=0;i<d.pos.length;i++){if(!wound[i])continue;const p=d.pos[i];ctx.beginPath();ctx.arc(px(p),py(p),4.5,0,Math.PI*2);ctx.fill()}ctx.fillStyle='#756c60';ctx.font='12px ui-sans-serif';ctx.fillText(wound.some(Boolean)?'clot_ml_0 prediction · wound in red':'clot_ml_0 prediction',18,22)}
function drawVelocity(c,idx){const d=state.data;if(!d||!d.velocity)return;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#f1ece3';ctx.fillRect(0,0,w,h);const [x0,x1,y0,y1]=extent(d),pad=28,s=Math.min((w-2*pad)/Math.max(x1-x0,1e-9),(h-2*pad)/Math.max(y1-y0,1e-9)),ox=pad+((w-2*pad)-s*(x1-x0))/2,oy=pad+((h-2*pad)-s*(y1-y0))/2,px=p=>ox+(p[0]-x0)*s,py=p=>oy+(y1-p[1])*s,v=d.velocity[idx]||d.velocity[0],mx=v.reduce((a,b)=>Math.max(a,b),1e-9);for(let i=0;i<d.pos.length;i++){const q=v[i]/mx,p=d.pos[i];ctx.fillStyle=`rgb(${Math.round(220-110*q)},${Math.round(236-75*q)},${Math.round(229-110*q)})`;ctx.fillRect(px(p)-2,py(p)-2,4,4)}ctx.fillStyle='#756c60';ctx.font='12px ui-sans-serif';ctx.fillText(idx===0?'Initial velocity':'Final velocity',18,22)}
function drawScience(){const d=state.data,c=$('science-canvas');if(!d||!d.metrics)return;const [w,h,scale]=resizeCanvas(c),ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#f1ece3';ctx.fillRect(0,0,w,h);const rows=d.metrics,p={l:42,r:14,t:18,b:30},pw=w-p.l-p.r,ph=h-p.t-p.b,px=i=>p.l+(i/Math.max(rows.length-1,1))*pw,py=v=>p.t+ph-(Math.max(0,Math.min(100,v))/100)*ph;ctx.strokeStyle='#d9d0c2';ctx.fillStyle='#756c60';ctx.font='10px ui-sans-serif';for(let k=0;k<=4;k++){const y=py(k*25);ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText(String(k*25),8,y+3)}const line=(key,col)=>{ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>{const x=px(i),y=py(Number(r[key])||0);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()};line('wall_clot_pct','#9a5a13');line('vessel_clot_pct','#8fbdb6');line('max_occlusion_pct','#c25b4c');ctx.fillStyle='#756c60';ctx.fillText('time',w-36,h-8);ctx.fillText('percent',4,12)}
function updateStats(){const d=state.data,m=d.metrics&&d.metrics[state.frame];if(!m)return;$('stat-wall').textContent=fmt(m.wall_clot_pct)+'%';$('stat-vessel').textContent=fmt(m.vessel_clot_pct)+'%';$('stat-occ').textContent=fmt(m.max_occlusion_pct)+'%';$('viewer-time').textContent=`t = ${fmt(d.times[state.frame],0)} s (${fmt(Number(d.times[state.frame])/3750,2)} h)`;$('time-out').textContent=$('viewer-time').textContent}
function geometryLabel(d){const g=d.meta&&d.meta.geometry;if(!g||!g.curve_type)return '';const width=Number(g.width_m);const shape=`${g.curve_type} · ${Number.isFinite(width)?(width*1e3).toFixed(1)+' mm':'custom width'}`;return ` · ${shape}`}
function render(){const d=state.data;if(!d)return;const velocity=state.mode==='clot_velocity';$('field-wrap').classList.toggle('hidden',velocity);$('velocity-wrap').classList.toggle('hidden',!velocity);$('science').classList.toggle('hidden',state.mode!=='scientific');time.disabled=velocity;time.max=Math.max(0,d.times.length-1);time.value=state.frame;const title=velocity?'Velocity bookends':(state.mode==='scientific'?'Clot prediction · Scientific':'Clot prediction');$('viewer-title').textContent=title+geometryLabel(d);if(velocity){drawVelocity($('vel0'),0);drawVelocity($('vel1'),d.velocity.length-1)}else{drawField($('field-canvas'),state.frame);if(state.mode==='scientific')drawScience()}updateStats()}
time.addEventListener('input',()=>{state.frame=Number(time.value);render()});window.addEventListener('resize',()=>{if(state.data)render();else if(source.value==='parametric')renderPreview();else resetPreview()});
function payloadFromForm(){return {source:source.value,filename:inbox.value,run_mode:$('run-mode').value,re:Number($('re').value),hours:Number($('hours').value),width:Number($('width').value),bend:Number($('bend').value),amp:Number($('amp').value),pathology:$('pathology').value,wound_enabled:$('wound-enabled').checked,wound_position:Number($('wound-position').value),wound_width:Number($('wound-width').value)}}
async function startRun(){error.classList.add('hidden');run.disabled=true;setStatus('Preparing prediction…');const p=payloadFromForm();if(p.source==='upload'){const files=Array.from($('upload').files);const f=files.find(x=>/\.(pt|msh|nas)$/i.test(x.name));if(!f){error.textContent='Choose a .pt, .msh, or .nas file first.';error.classList.remove('hidden');run.disabled=false;return}p.filename=f.name;p.data_b64=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(f)});if(/\.(msh|nas)$/i.test(f.name)){const side=files.find(x=>x.name.replace(/\.[^.]+$/,'').toLowerCase()===f.name.replace(/\.[^.]+$/,'').toLowerCase()&&/\.json$/i.test(x.name));if(!side){error.textContent='Tagged mesh uploads need the matching .json sidecar.';error.classList.remove('hidden');run.disabled=false;return}p.sidecar_filename=side.name;p.sidecar_b64=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(side)})}}try{const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const j=await r.json();if(!r.ok)throw new Error(j.error||'Could not start prediction');state.job=j.job_id;poll()}catch(e){error.textContent=e.message;error.classList.remove('hidden');setStatus('Ready','err');run.disabled=false}}
async function readUploadInto(p){if(p.source!=='upload')return true;const files=Array.from($('upload').files),f=files.find(x=>/\.(pt|msh|nas)$/i.test(x.name));if(!f){error.textContent='Choose a .pt, .msh, or .nas file first.';return false}p.filename=f.name;p.data_b64=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(f)});if(/\.(msh|nas)$/i.test(f.name)){const stem=f.name.replace(/\.[^.]+$/,'').toLowerCase(),side=files.find(x=>x.name.replace(/\.[^.]+$/,'').toLowerCase()===stem&&/\.json$/i.test(x.name));if(!side){error.textContent='Tagged .msh/.nas uploads need the matching .json sidecar.';return false}p.sidecar_filename=side.name;p.sidecar_b64=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(side)})}return true}
function duration(s){s=Math.max(0,Number(s)||0);return s>=60?`${Math.floor(s/60)}m ${Math.round(s%60)}s`:`${Math.round(s)} s`}
async function startJob(kind){error.classList.add('hidden');run.disabled=true;estimate.disabled=true;state.jobKind=kind;setStatus(kind==='estimate'?'Inspecting geometry for runtime estimate…':'Preparing prediction…');const p=payloadFromForm();p.kind=kind;try{if(!await readUploadInto(p)){error.classList.remove('hidden');setStatus('Ready','err');return}const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),j=await r.json();if(!r.ok)throw new Error(j.error||'Could not start job');state.job=j.job_id;poll()}catch(e){error.textContent=e.message;error.classList.remove('hidden');setStatus('Ready','err');run.disabled=false;estimate.disabled=false}}
async function poll(){if(!state.job)return;try{const r=await fetch('/api/job/'+state.job),j=await r.json();const elapsed=j.created_at?` · ${duration(Date.now()/1000-j.created_at)} elapsed`:'';setStatus((j.message||j.status)+((j.status==='running'||j.status==='queued')?elapsed:''),j.status==='error'?'err':j.status==='done'?'ok':'');if(j.status==='running'||j.status==='queued'){setTimeout(poll,700);return}if(j.status==='error')throw new Error(j.error||'Prediction failed');if(j.result&&j.result.kind==='estimate'){$('estimate-note').textContent=`Estimated ${duration(j.result.estimate_low_s)}–${duration(j.result.estimate_high_s)} (central ${duration(j.result.estimate_mid_s)}): ${j.result.n_nodes.toLocaleString()} nodes, ${j.result.n_steps} rollout steps. ${j.result.note}`;setStatus('Runtime estimate ready.','ok')}else{state.data=j.result;state.mode=j.result.meta.run_mode||$('run-mode').value;state.frame=0;const dl=$('download');if(j.result.csv_url){dl.href=j.result.csv_url;dl.classList.remove('hidden')}else dl.classList.add('hidden');render();const actual=Number(j.elapsed_s);setStatus(`Prediction complete in ${duration(actual)}.`, 'ok')}run.disabled=false;estimate.disabled=false}catch(e){error.textContent=e.message;error.classList.remove('hidden');run.disabled=false;estimate.disabled=false}}
run.addEventListener('click',()=>startJob('predict'));estimate.addEventListener('click',()=>startJob('estimate'));$('run-mode').addEventListener('change',e=>{const next=e.target.value;if(state.data&&(next==='clot_velocity'||next==='scientific')&&!state.data.velocity){error.textContent='Re-run the prediction in this mode to include velocity fields.';error.classList.remove('hidden');e.target.value=state.mode;return}state.mode=next;if(state.data)render()});$('run-mode').addEventListener('change',()=>{if($('run-mode').value==='clot_velocity')time.value=0});
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
    if include_velocity:
        velocity = [np.round(np.asarray(fr["vel_mag"], dtype=np.float32), 4).tolist() for fr in frames]
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
        "masks": masks,
        "metrics": rows,
        "meta": meta,
        "csv_url": csv_url,
    })


def _parametric_params(req: dict[str, Any]) -> dict[str, Any]:
    """Build a clean, deterministic customer vessel from exactly the visible controls."""
    width = float(req.get("width", 0.008))
    bend = math.radians(float(req.get("bend", 20.0)))
    amp = float(req.get("amp", 0.0))
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
    if pathology in ("max_stenosis", "max_aneurysm"):
        params = apply_customer_max_pathology(params, cfg, pathology)  # type: ignore[arg-type]
    return params


def _parametric_data(req: dict[str, Any], *, re_target: float, t_final_s: float, n_steps: int):
    params = _parametric_params(req)
    try:
        data = build_parametric_customer_graph(
            re_target=re_target, t_final_s=t_final_s, n_steps=n_steps, params_override=params
        )
    except Exception as exc:
        raise ValueError(
            f"{exc} (requested {float(params['width']) * 1e3:.2f} mm "
            f"{params['curve_type']} vessel)"
        ) from exc
    data.customer_parametric_spec = {
        "width_m": float(params["width"]),
        "bend_deg": 0.0 if str(params["curve_type"]) == "s_curve" else float(req.get("bend", 20.0)),
        "s_amplitude_m": float(params["amplitude"]),
        "curve_type": str(params["curve_type"]),
        "pathology": str(params.get("pathology_mode", "none")),
    }
    return _apply_requested_wound(data, req)


def _apply_requested_wound(data: Any, req: dict[str, Any]) -> Any:
    return apply_customer_mirrored_wound(
        data,
        enabled=bool(req.get("wound_enabled", False)),
        position_frac=float(req.get("wound_position", 50.0)) / 100.0,
        width_frac=float(req.get("wound_width", 15.0)) / 100.0,
    )


def _get_pipeline() -> CustomerDeployPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = CustomerDeployPipeline(require_cuda=REQUIRE_CUDA)
    return PIPELINE


def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(updates)


def _request_settings(req: dict[str, Any]) -> tuple[float, float, float, int, str]:
    re_target = float(np.clip(float(req.get("re", 450.0)), 100.0, 900.0))
    hours = float(np.clip(float(req.get("hours", 8.0)), 0.1, 50.0))
    t_final_s = hours * 3750.0
    n_steps = int(max(20, min(120, round(t_final_s / 135.0))))
    mode = str(req.get("run_mode", "clot"))
    if mode not in ("clot", "clot_velocity", "scientific"):
        mode = "clot"
    return re_target, hours, t_final_s, n_steps, mode


def _load_request_data(
    req: dict[str, Any], *, re_target: float, t_final_s: float, n_steps: int, progress: Any
) -> Any:
    source = str(req.get("source", "inbox"))
    if source == "parametric":
        progress("Building parametric geometry…")
        return _parametric_data(req, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps)
    if source == "upload":
        filename = _safe_name(str(req.get("filename", "customer_geometry.pt")))
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("Unsupported geometry type. Use .pt, .msh, or .nas.")
        encoded = str(req.get("data_b64", ""))
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("Geometry file is too large (80 MB limit).")
        upload_dir = ensure_inbox() / "web_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / filename
        path.write_bytes(raw)
        if suffix in (".msh", ".nas"):
            side_b64 = str(req.get("sidecar_b64", ""))
            if not side_b64:
                raise ValueError("Tagged .msh/.nas uploads need their matching .json sidecar.")
            side_raw = base64.b64decode(side_b64, validate=True)
            if len(side_raw) > 2 * 1024 * 1024:
                raise ValueError("Geometry sidecar is too large (2 MB limit).")
            path.with_suffix(".json").write_bytes(side_raw)
        progress(f"Loading {filename}…")
        data = load_customer_geometry(path, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps)
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


def _execute_request(job_id: str, req: dict[str, Any], progress: Any) -> dict[str, Any]:
    re_target, hours, t_final_s, n_steps, mode = _request_settings(req)
    data = _load_request_data(
        req, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps, progress=progress
    )
    if str(req.get("kind", "predict")) == "estimate":
        return _runtime_estimate(data, n_steps=n_steps, hours=hours)

    progress("Running customer deploy model…")
    traj = _get_pipeline().run(
        data,
        t_final_s=t_final_s,
        progress=progress,
        include_velocity=mode in ("clot_velocity", "scientific"),
    )
    csv_url = None
    if mode == "scientific":
        out_dir = ROOT / "outputs" / "customer_predict"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"web_scientific_metrics_{job_id}.csv"
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
            def progress(msg: str) -> None:
                events.put({"type": "progress", "job_id": job_id, "message": str(msg)})

            result = _execute_request(job_id, req, progress)
            elapsed_s = round(time.perf_counter() - started, 2)
            events.put({
                "type": "done", "job_id": job_id, "message": "Prediction complete.",
                "result": result, "elapsed_s": elapsed_s,
            })
        except Exception as exc:
            traceback.print_exc()
            events.put({
                "type": "error", "job_id": job_id, "message": "Prediction failed.", "error": str(exc),
            })


def _drain_worker_events() -> None:
    while True:
        try:
            event = WORKER_EVENTS.get(timeout=0.25)
        except queue.Empty:
            if WORKER_PROCESS is not None and not WORKER_PROCESS.is_alive():
                return
            continue
        if event is None:
            return
        event_type = event.get("type")
        job_id = event.get("job_id")
        if not job_id:
            continue
        if event_type in ("running", "progress"):
            _set_job(job_id, status="running", message=event.get("message", "Running…"))
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
    threading.Thread(target=_drain_worker_events, name="hemorgp-web-events", daemon=True).start()


def _stop_inference_worker() -> None:
    if WORKER_REQUESTS is not None:
        try:
            WORKER_REQUESTS.put(None)
        except Exception:
            pass
    if WORKER_PROCESS is not None:
        WORKER_PROCESS.join(timeout=8)
        if WORKER_PROCESS.is_alive():
            WORKER_PROCESS.terminate()
            WORKER_PROCESS.join(timeout=3)


class Handler(BaseHTTPRequestHandler):
    server_version = "HemoRGPWeb/1.0"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, status: int = 200) -> None:
        self._send(json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
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
                csv_path = ROOT / "outputs" / "customer_predict" / f"web_scientific_metrics_{job_id}.csv"
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
        if path == "/api/open-folder":
            import os
            import subprocess

            folder = ensure_inbox()
            try:
                if os.name == "nt":
                    os.startfile(str(folder))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(folder)])
            except Exception:
                pass
            self._json({"ok": True})
            return
        if path != "/api/run":
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_UPLOAD_BYTES * 2:
                raise ValueError("Request is empty or too large.")
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("Request must be a JSON object.")
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if WORKER_PROCESS is None or WORKER_REQUESTS is None or not WORKER_PROCESS.is_alive():
            self._json(
                {"error": "The inference worker is not available. Restart the customer web launcher."},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        job_id = str(uuid.uuid4())
        _set_job(job_id, status="queued", message="Queued…", created_at=time.time(), kind=req.get("kind", "predict"))
        WORKER_REQUESTS.put((job_id, req))
        self._json({"job_id": job_id}, HTTPStatus.ACCEPTED)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HemoRGP customer browser UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cpu", action="store_true", help="Allow CPU inference (slow; CUDA recommended)")
    args = ap.parse_args(argv)
    global REQUIRE_CUDA
    REQUIRE_CUDA = not args.cpu
    ensure_inbox()
    _start_inference_worker(REQUIRE_CUDA)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[i] HemoRGP web predict: http://{args.host}:{args.port}", flush=True)
    print(f"[i] Geometries folder: {ensure_inbox()}", flush=True)
    if args.cpu:
        print("[WARN] CPU mode (slow). CUDA is recommended.", flush=True)
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
