"""Render the off-wall verdict maps: what the model commits, and what it misses.

Reads `outputs/deployclot/offwall_viz.json` (written by
`scripts/diag_offwall_score_geography.py`) and emits a self-contained page.  The point of
the page is a single comparison: vessels the model never saw (SEALED) beside vessels scored
out-of-fold, drawn the same way, so "does it generalise" is a thing you look at rather than
a number you are told.

    python scripts/build_offwall_viz_page.py --out outputs/deployclot/offwall_viz.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CSS = """
:root{
  --paper:#f6f8f9; --panel:#ffffff; --ink:#14181d; --ink-2:#454d55; --muted:#79838c;
  --rule:#dde3e7; --rule-2:#eef2f4; --field:#e9eef1; --mesh:#c3ccd3;
  --hit:#12705a; --miss:#c33f28; --spur:#b8860f; --wallc:#9aa6ae;
  --shadow:0 1px 2px rgba(20,24,29,.05), 0 10px 28px -20px rgba(20,24,29,.4);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#0f1316; --panel:#161b1f; --ink:#e8ebed; --ink-2:#bcc4ca; --muted:#818c95;
  --rule:#252d33; --rule-2:#1c2226; --field:#1b2226; --mesh:#38434b;
  --hit:#3fb392; --miss:#f0705a; --spur:#e0a92e; --wallc:#5d6a73;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 28px -20px rgba(0,0,0,.9);
}}
:root[data-theme="dark"]{
  --paper:#0f1316; --panel:#161b1f; --ink:#e8ebed; --ink-2:#bcc4ca; --muted:#818c95;
  --rule:#252d33; --rule-2:#1c2226; --field:#1b2226; --mesh:#38434b;
  --hit:#3fb392; --miss:#f0705a; --spur:#e0a92e; --wallc:#5d6a73;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 28px -20px rgba(0,0,0,.9);
}
*{box-sizing:border-box}
body{background:var(--paper); color:var(--ink); margin:0;
  font-family:"Public Sans","Segoe UI",system-ui,sans-serif; font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px; margin:0 auto; padding:0 26px 88px}
code,.n{font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums}
h1{font-family:"Archivo","Public Sans",sans-serif; font-weight:700; letter-spacing:-.022em;
  font-size:clamp(32px,5vw,50px); line-height:1.05; margin:0 0 16px; text-wrap:balance}
h2{font-family:"Archivo",sans-serif; font-weight:600; font-size:24px; letter-spacing:-.01em;
  margin:52px 0 6px}
h3{font-family:"Archivo",sans-serif; font-weight:600; font-size:15px; margin:0}
p{max-width:68ch; margin:0 0 14px; color:var(--ink-2)}
strong{color:var(--ink); font-weight:600}
header{padding:64px 0 26px; border-bottom:1px solid var(--rule); margin-bottom:26px}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); margin:0 0 16px}
.lede{font-size:19px; line-height:1.5; color:var(--ink-2); max-width:64ch}

.tablewrap{overflow-x:auto; border:1px solid var(--rule); border-radius:8px;
  background:var(--panel); box-shadow:var(--shadow); margin:22px 0 8px}
table{border-collapse:collapse; width:100%; font-size:14.5px}
th,td{padding:10px 16px; text-align:left; border-bottom:1px solid var(--rule-2);
  white-space:nowrap}
thead th{font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  font-weight:600; background:var(--rule-2)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums}
td.em{font-weight:600; color:var(--ink)}

.controls{display:flex; gap:18px; align-items:center; flex-wrap:wrap; margin:26px 0 6px;
  font-size:14px; color:var(--ink-2)}
.controls label{display:flex; gap:7px; align-items:center; cursor:pointer}
.legend{display:flex; gap:16px; flex-wrap:wrap; font-size:13.5px; color:var(--ink-2);
  margin:14px 0 4px}
.legend span{display:flex; gap:7px; align-items:center}
.dot{width:11px; height:11px; border-radius:50%; display:inline-block; flex:none}

.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px;
  margin-top:18px}
.card{background:var(--panel); border:1px solid var(--rule); border-radius:8px;
  box-shadow:var(--shadow); padding:16px 16px 14px; display:flex; flex-direction:column}
.card .top{display:flex; justify-content:space-between; align-items:baseline; gap:10px;
  margin-bottom:4px}
.tag{font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; padding:3px 8px; border-radius:99px; border:1px solid var(--rule)}
.tag.sealed{color:var(--miss); border-color:color-mix(in srgb, var(--miss) 45%, transparent)}
.tag.oof{color:var(--muted)}
canvas{width:100%; height:auto; display:block; border-radius:5px; background:var(--field);
  margin:10px 0 8px}
.stats{display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--rule);
  border:1px solid var(--rule); border-radius:6px; overflow:hidden; margin-top:auto}
.stats div{background:var(--panel); padding:7px 9px}
.stats .k{font-size:10px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted)}
.stats .v{font-family:"IBM Plex Mono",monospace; font-size:15px; margin-top:2px}
.note{border:1px solid var(--rule); border-left:3px solid var(--spur); border-radius:8px;
  background:var(--panel); padding:16px 20px; margin:26px 0; box-shadow:var(--shadow)}
.note p:last-child{margin-bottom:0}
footer{margin-top:56px; padding-top:20px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--muted)}
@media (max-width:640px){ .wrap{padding:0 16px 60px} header{padding:40px 0 20px} }
"""

JS = """
const DATA = __DATA__;
const C = () => getComputedStyle(document.documentElement);
function draw(cv, v){
  const st = C();
  const col = {
    bg: st.getPropertyValue('--field').trim(),
    dot: st.getPropertyValue('--mesh').trim(),
    wall: st.getPropertyValue('--wallc').trim(),
    hit: st.getPropertyValue('--hit').trim(),
    miss: st.getPropertyValue('--miss').trim(),
    spur: st.getPropertyValue('--spur').trim(),
  };
  const showWall = document.getElementById('t-wall').checked;
  const showBg = document.getElementById('t-bg').checked;
  const [x0,x1,y0,y1] = v.bounds;
  const w = cv.width, h = cv.height;
  const pad = 10;
  const sx = (w-2*pad)/Math.max(x1-x0,1e-9), sy = (h-2*pad)/Math.max(y1-y0,1e-9);
  const s = Math.min(sx,sy);
  const ox = pad + ((w-2*pad) - (x1-x0)*s)/2, oy = pad + ((h-2*pad) - (y1-y0)*s)/2;
  const X = p => ox + (p[0]-x0)*s, Y = p => h - (oy + (p[1]-y0)*s);
  const g = cv.getContext('2d');
  g.clearRect(0,0,w,h);
  const blob = (pts, c, r) => { g.fillStyle = c;
    for (const p of pts){ g.beginPath(); g.arc(X(p),Y(p),r,0,6.2832); g.fill(); } };
  if (showBg) blob(v.bg, col.dot, 1.1);
  blob(v.wall, col.wall, 1.5);
  if (showWall){ blob(v.w_fp, col.spur, 1.8); blob(v.w_tp, col.hit, 1.8);
                 blob(v.w_fn, col.miss, 1.8); }
  blob(v.fp, col.spur, 2.9);
  blob(v.tp, col.hit, 2.9);
  blob(v.fn, col.miss, 2.9);
}
function sizeAndDraw(){
  for (const v of DATA){
    const cv = document.getElementById('cv-'+v.stem);
    if (!cv) continue;
    const [x0,x1,y0,y1] = v.bounds;
    // Clamp the FRAME, not the drawing: `draw` preserves the true aspect and centres,
    // so a clamp only letterboxes.  Cards then share a height and the grid stops
    // leaving holes under the flat vessels.
    const aspect = Math.max(1.0, Math.min(2.0, (x1-x0)/Math.max(y1-y0,1e-9)));
    const cssW = cv.parentElement.clientWidth - 32;
    const cssH = Math.round(cssW/aspect);
    const dpr = Math.min(window.devicePixelRatio||1, 2);
    cv.style.height = cssH+'px';
    cv.width = Math.round(cssW*dpr); cv.height = Math.round(cssH*dpr);
    draw(cv, v);
  }
}
document.getElementById('t-wall').addEventListener('change', sizeAndDraw);
document.getElementById('t-bg').addEventListener('change', sizeAndDraw);
window.addEventListener('resize', () => { clearTimeout(window._rt);
  window._rt = setTimeout(sizeAndDraw, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', sizeAndDraw);
sizeAndDraw();
"""


def card(v: dict) -> str:
    s = v.get("scores", {}) or {}
    sealed = v["group"] == "SEALED"
    f = lambda k, d=4: ("&mdash;" if s.get(k) is None or s.get(k) != s.get(k)  # noqa: E731
                        else f"{float(s[k]):.{d}f}")
    return f"""
<div class="card">
  <div class="top"><h3><code>{v['stem']}</code></h3>
    <span class="tag {'sealed' if sealed else 'oof'}">{'sealed' if sealed else 'out-of-fold'}</span></div>
  <div style="font-size:13px;color:var(--muted)">
    {len(v['tp'])} found &middot; {len(v['fn'])} missed &middot; {len(v['fp'])} spurious,
    off-wall</div>
  <canvas id="cv-{v['stem']}"></canvas>
  <div class="stats">
    <div><div class="k">off severity</div><div class="v">{f('sev')}</div></div>
    <div><div class="k">off deploy</div><div class="v">{f('dep')}</div></div>
    <div><div class="k">precision</div><div class="v">{f('precision', 2)}</div></div>
    <div><div class="k">recall</div><div class="v">{f('recall', 2)}</div></div>
  </div>
</div>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default="outputs/deployclot/offwall_viz.json")
    ap.add_argument("--geo", default="outputs/deployclot/offwall_score_geography.json")
    ap.add_argument("--out", default="outputs/deployclot/offwall_viz.html")
    args = ap.parse_args()

    viz = json.loads((REPO / args.viz).read_text())
    geo = json.loads((REPO / args.geo).read_text())["per_vessel"]

    def m(grp, key):
        v = [r[key] for r in geo if r["group"] == grp and r.get(key) == r.get(key)]
        return sum(v) / len(v) if v else float("nan")

    order = ([v for v in viz if v["group"] == "SEALED"]
             + [v for v in viz if v["group"] != "SEALED"])
    n_sealed = sum(1 for v in viz if v["group"] == "SEALED")

    rows = "".join(
        f"<tr><td>{lbl}</td><td class='n'>{m('cohort', kc):.4f}</td>"
        f"<td class='n em'>{m('SEALED', kc):.4f}</td>"
        f"<td class='n'>{m('SEALED', kc) - m('cohort', kc):+.4f}</td></tr>"
        for lbl, kc in (("wall &mdash; severity metric", "wall_sev"),
                        ("wall &mdash; deploy metric", "wall_dep"),
                        ("off-wall &mdash; severity metric", "sev"),
                        ("off-wall &mdash; deploy metric", "dep")))

    page = f"""<title>Off-Wall Verdicts</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">DeployClot &middot; off-wall diagnosis &middot; 2026-09-03</p>
  <h1>Where the off-wall model misses</h1>
  <p class="lede">Every node the model committed and every node it should have. Vessels it
  has never seen, drawn beside vessels scored out-of-fold, on the same shipped readout.</p>
</header>

<p>The comparison the maps are for: <strong>sealed vessels do not look different.</strong>
Both sets under both metrics, same masks, same spec.</p>
<div class="tablewrap"><table>
<thead><tr><th>domain and metric</th><th class="n">cohort (n=20)</th>
<th class="n">sealed (n=3)</th><th class="n">difference</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="lede" style="font-size:16px">The off-wall noise floor on this cohort is
&plusmn;0.074, so every row is a tie &mdash; and on the wall the sealed vessels are ahead.
The two metrics are not two measurements of the same thing: the deploy metric
(<code>guiding</code>) is the stricter one, and the severity metric adds an absolute grace
for a handful of missed nodes, which is worth ~0.19 off-wall and ~0.07 on the wall.</p>

<h2>The maps</h2>
<div class="controls">
  <label><input type="checkbox" id="t-wall"> show wall verdicts</label>
  <label><input type="checkbox" id="t-bg" checked> show lumen mesh</label>
</div>
<div class="legend">
  <span><i class="dot" style="background:var(--hit)"></i> committed, correct</span>
  <span><i class="dot" style="background:var(--miss)"></i> clot the model missed</span>
  <span><i class="dot" style="background:var(--spur)"></i> committed, no clot</span>
  <span><i class="dot" style="background:var(--wallc)"></i> vessel wall</span>
  <span><i class="dot" style="background:var(--mesh)"></i> lumen mesh</span>
</div>
<div class="grid">{''.join(card(v) for v in order)}</div>

<div class="note">
<p><strong>What the maps show, and it is the same on both sets.</strong> Neither colour is
scattered. The red sits in contiguous arcs continuing the committed green, and where there is
amber it runs as a ribbon <em>parallel</em> to the green rather than off in the lumen &mdash;
both are a boundary placed a node or two wrong, not a prediction in the wrong place. Precision
runs 0.62 on the cohort and 0.70 on the sealed vessels; recall runs 0.61 and 0.43.
<strong>The dominant failure is recall: an off-wall front that stops too early.</strong></p>
<p>That matters for what to do next: a threshold moves the whole boundary at once, and the
best possible per-vessel threshold is worth only +0.04 here. Extending the front is a model
question, not a readout one.</p>
</div>

<footer><p>Generated by <code>scripts/build_offwall_viz_page.py</code> from
<code>scripts/diag_offwall_score_geography.py</code>. Sealed vessels were read once, on
2026-09-03; these are those predictions re-drawn, not a second read.
Method: <code>docs/DEPLOYCLOT.md</code> &sect;22.</p></footer>
</div>
<script>{JS.replace('__DATA__', json.dumps(viz, separators=(',', ':')))}</script>
"""
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[save] {out}  ({out.stat().st_size // 1024} KB, {len(viz)} vessels, "
          f"{n_sealed} sealed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
