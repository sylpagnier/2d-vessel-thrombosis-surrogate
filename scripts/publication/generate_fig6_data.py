"""Generate Figure 6 data (known failures, strict OOF)."""
import sys
import torch
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR
from scripts.publication.oof_data import (
    build_vessel_figure_data,
    ensure_oof_series,
    load_oof_archive,
    metrics_rows_for_vessel,
)


def main():
    print("[i] Generating Data for Figure 6: Known Failures (strict OOF)")
    oof_path = ensure_oof_series(CONFIG)
    archive = load_oof_archive(oof_path)

    records = []
    for stem in CONFIG.fig6_vessels:
        oof = archive.get(stem)
        if oof is None:
            print(f"  [WARN] {stem} not in OOF archive; skip")
            continue
        print(f"  -> {stem} [OOF fold {oof.fold}] ...")
        payload = build_vessel_figure_data(stem, oof)
        records.extend(metrics_rows_for_vessel(stem, payload))
        save = {k: v for k, v in payload.items() if k != "_score_ctx"}
        torch.save(save, DATA_DIR / f"fig6_{stem}_failures.pt")
        print(f"     [OK] Saved to fig6_{stem}_failures.pt")

    if records:
        pd.DataFrame(records).to_csv(DATA_DIR / "fig6_metrics.csv", index=False)
        print(f"[OK] Saved Fig 6 metrics to {DATA_DIR / 'fig6_metrics.csv'}")


if __name__ == "__main__":
    main()
