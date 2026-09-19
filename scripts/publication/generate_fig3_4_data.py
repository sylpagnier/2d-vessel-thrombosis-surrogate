"""Generate Figure 3 & 4 data (clot_ml_0 strict OOF temporal + final)."""
import torch
import pandas as pd


from scripts.publication.config import CONFIG, DATA_DIR
from scripts.publication.oof_data import (
    build_vessel_figure_data,
    ensure_oof_series,
    load_oof_archive,
    metrics_rows_for_vessel,
)


def main():
    print("[i] Generating Data for Figure 3 & 4: clot_ml_0 strict OOF")
    oof_path = ensure_oof_series(CONFIG)
    archive = load_oof_archive(oof_path)

    vessels = sorted(set(CONFIG.fig3_vessels + CONFIG.fig4_vessels))
    records = []

    for stem in vessels:
        oof = archive.get(stem)
        if oof is None:
            print(f"  [WARN] {stem} not in OOF archive; skip")
            continue
        print(f"  -> {stem} [OOF fold {oof.fold}, flow={oof.flow}] ...")
        payload = build_vessel_figure_data(stem, oof)
        records.extend(metrics_rows_for_vessel(stem, payload))
        # Drop scoring context before save (not picklable cleanly with tensors in nested dict)
        save = {k: v for k, v in payload.items() if k != "_score_ctx"}
        torch.save(save, DATA_DIR / f"fig34_{stem}_biochem.pt")

    if not records:
        raise SystemExit("[ERR] No OOF vessels exported; check fig3/fig4 cohort in config.py")

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "fig34_metrics.csv", index=False)
    print(f"[OK] Saved Fig 3/4 metrics to {DATA_DIR / 'fig34_metrics.csv'}")
    final = df.loc[df.groupby("vessel")["time"].idxmax(), ["vessel", "wall", "off"]]
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
