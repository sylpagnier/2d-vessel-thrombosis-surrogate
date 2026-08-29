"""Generate a small L2 batch and measure its shape on the DEPLOY footing.

    python scripts/diag_sampler_shape_vs_deploy.py [N]
    PSTRAIGHT=0.6 ANEU=0.30 STEN=0.28 python scripts/diag_sampler_shape_vs_deploy.py 48

Meshes N vessels, builds graphs (no CFD -- only `x` channels are read) and reports realised
axial lumen max/min and centreline excursion against the FIT deploy packs.  Seconds per run,
so the vessel sampler can be fitted to deployment before any COMSOL budget is committed.

`num_workers=1` deliberately: the pooled path hangs here on Windows, and 48 vessels take
about 25s sequentially anyway.

Fitting against ANALYTIC wall control points instead reads milder than this and pushed
`pro_thrombotic_straight_prob` 0.2 too high, so fit on this footing, not that one.
"""
import shutil, sys, glob
from pathlib import Path
import numpy as np, torch, meshio, json

OUT = Path("outputs/_shape_check/meshes"); OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"): f.unlink()

import os
from src.data_gen.lib.vessel_generator import VesselGenerator
vg = VesselGenerator(phase="kinematics", output_dir=OUT)
if os.environ.get("PSTRAIGHT"):
    vg.cfg.pro_thrombotic_straight_prob = float(os.environ["PSTRAIGHT"])
if os.environ.get("ANEU"):
    vg.cfg.aneurysm_factor_max = float(os.environ["ANEU"])
if os.environ.get("STEN"):
    vg.cfg.stenosis_factor_max = float(os.environ["STEN"])
print("[cfg] p_straight", vg.cfg.pro_thrombotic_straight_prob,
      "aneu_max", vg.cfg.aneurysm_factor_max, "sten_max", vg.cfg.stenosis_factor_max)
vg.run_pipeline(n=int(sys.argv[1]) if len(sys.argv) > 1 else 40, level=2, seed=101, unit="m",
                num_workers=1, chunk_size=8, max_retries=0)

from src.data_gen.lib.mesh_to_graph import MeshToGraph
m = MeshToGraph(phase="kinematics", raw_dir=OUT, proc_dir=Path("outputs/_shape_check/graphs"))
from src.core_physics.wall_cohort_splits import FIT

def measure(d):
    x = d.x.detach().float().numpy(); xn, yn, wn = x[:, 0], x[:, 1], x[:, 15]
    wall = d.mask_wall.reshape(-1).bool().numpy()
    bins = np.linspace(xn[wall].min(), xn[wall].max(), 41)
    idx = np.clip(np.digitize(xn[wall], bins) - 1, 0, 39)
    yw = yn[wall]
    w = np.array([wn[wall][idx == b].mean() if (idx == b).sum() else np.nan for b in range(40)])
    mid = np.array([0.5*(yw[idx==b].min()+yw[idx==b].max()) if (idx==b).sum()>1 else np.nan
                    for b in range(40)])
    w, mid = w[np.isfinite(w)], mid[np.isfinite(mid)]
    return (float(w.max()/(w.min()+1e-9)) if w.size else np.nan,
            float(mid.max()-mid.min()) if mid.size else np.nan)

new = []
for f in sorted(OUT.glob("*.msh")):
    j = OUT / f"{f.stem}.json"
    if not j.is_file(): continue
    d = m.process_mesh(meshio.read(f), json.load(open(j)), stem=f.stem)
    if d is not None: new.append(measure(d))
dep = [measure(torch.load(f"data/processed/graphs_biochem_anchors/{s}.pt",
                          map_location="cpu", weights_only=False))
       for s in sorted(FIT) if Path(f"data/processed/graphs_biochem_anchors/{s}.pt").is_file()]

print(f"\n{'':<22}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}{'<1.15':>8}{'exc p50':>9}")
for tag, rows in (("NEW sampler (meshed)", new), ("DEPLOY FIT", dep)):
    a = np.array([r for r in rows if np.isfinite(r[0])])
    print(f"{tag:<22}" + "".join(f"{np.percentile(a[:,0],p):>8.2f}" for p in (10,25,50,75,90))
          + f"{np.mean(a[:,0]<1.15)*100:>7.0f}%{np.median(a[:,1]):>9.2f}   (n={len(a)})")
shutil.rmtree("outputs/_shape_check", ignore_errors=True)
