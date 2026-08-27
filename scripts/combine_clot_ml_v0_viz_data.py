"""Combine the non-wound OOF and on-demand wound payloads for one toggleable page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nonwound", default="outputs/clot_ml_v0_oof_temporal_data.json")
    ap.add_argument("--wound", default="outputs/clot_ml_v0_wound_temporal_data.json")
    ap.add_argument("--out", default="outputs/clot_ml_v0_temporal_data.json")
    args = ap.parse_args()

    nw = json.loads(Path(args.nonwound).read_text(encoding="utf-8"))
    wd = json.loads(Path(args.wound).read_text(encoding="utf-8"))
    nw_vessels = [v for v in nw if v != "_meta"]
    wd_vessels = [v for v in wd if v != "_meta"]
    overlap = set(nw_vessels) & set(wd_vessels)
    if overlap:
        raise ValueError(f"payloads overlap vessel ids: {sorted(overlap)}")
    sealed = set(nw.get("_meta", {}).get("final_half_excluded", []))
    sealed |= set(wd.get("_meta", {}).get("final_half_excluded", []))
    out = {"_meta": {
        "schema_version": 2,
        "model": "clot_ml_v0",
        "flow": nw.get("_meta", {}).get("flow", "gt"),
        "protocol": "non-wound strict outer-fold OOF + wound leave-one-vessel-out",
        "final_half_excluded": sorted(sealed),
        "mode_vessels": {"nonwound": nw_vessels, "wound": wd_vessels},
        "mode_labels": {"nonwound": "Non-wound · OOF", "wound": "Wound · LOVO"},
        "default_mode": "nonwound",
        "note": "Use the mode selector to switch between independent generalization cohorts.",
    }}
    out.update({v: dict(nw[v], mode="nonwound") for v in nw_vessels})
    out.update({v: dict(wd[v], mode="wound") for v in wd_vessels})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path} ({len(nw_vessels)} non-wound + {len(wd_vessels)} wound vessels)")


if __name__ == "__main__":
    main()
