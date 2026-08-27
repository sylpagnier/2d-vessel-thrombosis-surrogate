"""Leave-one-vessel-out evaluation of ``stitch_onset`` (physics_wall_model).

A fifth to nearly half of the wall nodes that clot in GT never cross ``crit`` in the surface
ODE -- all of them have ``gate == 0`` -- and the scoring convention gives every one of them
the same constant.  This measures what replacing that constant with a shear-ordered spread
about a shifted centre is worth, refitting ``(offset, spread)`` on the other 12 vessels for
each held-out vessel.

THE SET IS HELD AT THE GT WALL SET so this isolates timing from the mask.  On a predicted
(over-grown) set the apparent gain is much larger and is mostly a precision effect -- see the
`stitch_onset` docstring.  SEALED is not opened.

    python scripts/eval_stitch_onset.py
"""
import torch, numpy as np, os, warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

# Run directly (`python scripts/eval_stitch_onset.py`) needs the repo root importable.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig
from src.core_physics.physics_wall_model import t0_flow_fields, integrate_mat_trajectory
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p
from src.core_physics.wall_cohort_splits import FIT, DEV, split_of, MIN_T
from src.clot_ml.evaluate import domain_score
D="data/processed/graphs_biochem_anchors"; bio=BiochemConfig(phase="biochem"); CRIT=float(bio.viscosity_mat_crit)
cache={}
for stem in sorted(set(FIT)|set(DEV)):
    p=f"{D}/{stem}.pt"
    if not os.path.exists(p): continue
    d=torch.load(p,map_location="cpu",weights_only=False); T=d.y.shape[0]
    if T<MIN_T: continue
    wall=d.mask_wall.numpy()
    f0=t0_flow_fields(d,bio,hops=3,flow_source="gt"); hook=make_rollout_hook(SHIPPED,bio,f0.sr)
    tr,_=integrate_mat_trajectory(d,bio,f0.gate*wall,da_scale=SHIPPED_DA_SCALE,ap_closure=hook)
    mi=d.y_channel_names.split(",").index("Mat_log1p_nd")
    mg=mat_si_for_gelation_from_log1p(d.y[:,:,mi],bio).reshape(T,-1).numpy()
    og=np.where((mg>=CRIT).any(0),(mg>=CRIT).argmax(0),T).astype(float)
    op=np.where((tr>=CRIT).any(0),(tr>=CRIT).argmax(0),T).astype(float)
    gt_set=wall&(og<T); ign=wall&(op<T); st=gt_set&~ign
    if gt_set.sum()<8 or ign.sum()==0 or st.sum()<3: continue
    times=list(range(0,T,max(1,T//24)))+[T-1]
    cache[stem]=dict(T=T,wall=wall,op=op,og=og,gt_set=gt_set,ign=ign,st=st,sr=f0.sr,
                     ei=torch.tensor(d.edge_index.numpy()),gts={ti:(mg[ti]>=CRIT) for ti in times},times=times)
def mot(c,on):
    v=[]
    for ti in c["times"]:
        pred=c["gt_set"]&(on>=0)&(on<=ti)
        s=domain_score(pred,c["gts"][ti],c["ei"],c["wall"],c["wall"])
        if s==s: v.append(s)
    return float(np.mean(v)) if v else np.nan
def arm(c,off,spread):
    centre=float(np.median(c["op"][c["ign"]]))+off*c["T"]
    on=np.where(c["gt_set"],np.where(c["ign"],c["op"],centre),-1.0)
    st=c["st"]; k=int(st.sum())
    if k and spread>0:
        rank=np.argsort(np.argsort(c["sr"][st]))/max(k-1,1)
        on[st]=centre+spread*(rank-0.5)*c["T"]
    return np.where(c["gt_set"],np.clip(on,0,c["T"]-1),-1.0)
GRID=[(o,s) for o in (0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4) for s in (0.0,0.2,0.4,0.6,0.8,1.0)]
tab={(o,s):{k:mot(c,arm(c,o,s)) for k,c in cache.items()} for o,s in GRID}
base={k:tab[(0.0,0.0)][k] for k in cache}
orc={k:mot(c,np.where(c["gt_set"],np.where(c["ign"],c["op"],c["og"]),-1.0)) for k,c in cache.items()}
print("LEAVE-ONE-VESSEL-OUT: (offset, spread) refitted on the other 12 each time\n")
print(f"{'vessel':13s} {'sp':4s} {'stitch%':>8s} {'base':>8s} {'LOVO':>8s} {'delta':>8s} {'ceiling':>8s} {'chosen':>13s}")
deltas=[]
for held in cache:
    others=[k for k in cache if k!=held]
    o,s=max(GRID,key=lambda g: np.nanmean([tab[g][k] for k in others]))
    new=tab[(o,s)][held]; d0=new-base[held]; deltas.append(d0)
    print(f"{held:13s} {split_of(held):4s} {cache[held]['st'].sum()/cache[held]['gt_set'].sum()*100:7.1f}% "
          f"{base[held]:8.4f} {new:8.4f} {d0:+8.4f} {orc[held]:8.4f} {f'({o},{s})':>13s}")
deltas=np.array(deltas)
rng=np.random.default_rng(0)
bs=np.array([rng.choice(deltas,len(deltas),replace=True).mean() for _ in range(20000)])
print(f"\nLOVO mean delta = {deltas.mean():+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]"
      f"  P(delta<=0) = {(bs<=0).mean():.4f}   positive on {int((deltas>0).sum())}/{len(deltas)} vessels")
print(f"base mean {np.mean(list(base.values())):.4f} -> LOVO {np.mean(list(base.values()))+deltas.mean():.4f}"
      f"   (GT-on-stitch ceiling {np.mean(list(orc.values())):.4f})")
print(f"wall noise floor is +-0.024 (PHASE10 2)")
