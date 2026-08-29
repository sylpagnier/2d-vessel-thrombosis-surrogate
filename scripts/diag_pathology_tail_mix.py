"""How much forced-max pathology restores deployment's severe tail, on the mesh footing.

    python scripts/diag_pathology_tail_mix.py

Meshes one batch per pathology mode and recombines them at candidate `--pathology-mix`
weights, so the tail can be chosen without spending a COMSOL run per candidate.  Reports
`preflight`'s own `sten` statistic (`width.median()/width.min()`), which is the single-node
measure that check uses -- NOT the axial lumen ratio that `diag_sampler_shape_vs_deploy.py`
fits the median on.  The two are separate knobs and this one sets only the tail.

The fitted defaults set the MEDIAN vessel; `--pathology-mix` sets the tail, and the two are
independent knobs.  This meshes a batch per mode and recombines at candidate mix weights, so
the tail can be chosen without another COMSOL run.
"""
import json, shutil
from pathlib import Path
import numpy as np, meshio, torch
from src.data_gen.lib.vessel_generator import VesselGenerator
from src.data_gen.lib.mesh_to_graph import MeshToGraph

N = 40
OUT = Path("outputs/_tail_check"); shutil.rmtree(OUT, ignore_errors=True)

def batch(mode, seed):
    d = OUT / (mode or "random"); d.mkdir(parents=True, exist_ok=True)
    vg = VesselGenerator(phase="kinematics", output_dir=d)
    vg.run_pipeline(n=N, level=2, seed=seed, unit="m", num_workers=1,
                    chunk_size=8, max_retries=0, pathology_mode=mode)
    m = MeshToGraph(phase="kinematics", raw_dir=d, proc_dir=OUT / "g")
    out = []
    for f in sorted(d.glob("*.msh")):
        j = d / f"{f.stem}.json"
        if not j.is_file(): continue
        g = m.process_mesh(meshio.read(f), json.load(open(j)), stem=f.stem)
        if g is not None:
            r = g.x[:, 15]
            out.append(float(r.median() / r.min()))
    return np.array(out)

pools = {m or "random": batch(m, s) for m, s in
         (("random", 301), ("max_stenosis", 302), ("max_aneurysm", 303))}
for k, v in pools.items():
    print(f"  {k:<14} n={v.size:2d}  p50 {np.median(v):5.2f}  p90 {np.percentile(v,90):6.2f}"
          f"  max {v.max():6.2f}   >=2.0 {np.mean(v>=2.0)*100:3.0f}%")

print(f"\n{'mix (rand/sten/aneu)':<24}{'p50':>7}{'p75':>7}{'p90':>7}{'>=2.0':>8}")
rng = np.random.default_rng(0)
for w in ((1.00, 0.00, 0.00), (0.88, 0.08, 0.04), (0.80, 0.13, 0.07), (0.72, 0.18, 0.10)):
    draws = np.concatenate([rng.choice(pools[k], size=int(2000 * p), replace=True)
                            for k, p in zip(("random", "max_stenosis", "max_aneurysm"), w)])
    print(f"{str(w):<24}{np.percentile(draws,50):>7.2f}{np.percentile(draws,75):>7.2f}"
          f"{np.percentile(draws,90):>7.2f}{np.mean(draws>=2.0)*100:>7.0f}%")
print(f"{'DEPLOY all 53':<24}{1.36:>7.2f}{2.41:>7.2f}{4.59:>7.2f}{26:>7.0f}%")
print(f"{'DEPLOY FIT 25':<24}{1.36:>7.2f}{1.49:>7.2f}{1.83:>7.2f}{8:>7.0f}%")
shutil.rmtree(OUT, ignore_errors=True)
