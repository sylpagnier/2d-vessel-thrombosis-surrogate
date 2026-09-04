"""Aggregate research sweep summaries into publication-ready metric tables."""


from scripts.publication.config import CONFIG, RESEARCH_SWEEP_DATA_DIR
from scripts.publication.research_sweep_utils import (
    collect_sweep_metrics,
    load_sweep_summary,
    save_research_sweep_metrics,
    summary_to_dataframe,
)


def main() -> None:
    print("[i] Generating research sweep figure data")
    RESEARCH_SWEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Geometry sweeps (01-15)
    geometry_ids = list(CONFIG.research_geometry_sweeps)
    wound_ids = list(CONFIG.research_wound_sweeps)
    all_ids = geometry_ids + wound_ids

    found = []
    missing = []
    for sid in all_ids:
        try:
            load_sweep_summary(sid)
            found.append(sid)
        except FileNotFoundError:
            missing.append(sid)

    if missing:
        print(f"[WARN] Missing sweep outputs ({len(missing)}); run run_research_sweep.py first:")
        for sid in missing[:8]:
            print(f"       - {sid}")
        if len(missing) > 8:
            print(f"       ... and {len(missing) - 8} more")

    if not found:
        print("[ERR] No sweep summaries found under outputs/research_sweeps/")
        return

    df_all = collect_sweep_metrics(found)
    out_all = save_research_sweep_metrics(df_all, "research_sweep_metrics.csv")
    print(f"[OK] Wrote {out_all}")

    for sid in found:
        summary = load_sweep_summary(sid)
        df = summary_to_dataframe(summary)
        if df.empty:
            continue
        path = RESEARCH_SWEEP_DATA_DIR / f"{sid}_summary.csv"
        df.to_csv(path, index=False)
        print(f"[OK] Wrote {path}")


if __name__ == "__main__":
    main()
