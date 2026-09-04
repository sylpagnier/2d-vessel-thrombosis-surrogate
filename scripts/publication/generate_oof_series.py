"""Export strict nested-CV OOF trajectories for publication clot figures."""
import argparse


from scripts.publication.config import CONFIG
from scripts.publication.oof_data import ensure_oof_series, load_oof_archive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-export even if NPZ exists")
    args = ap.parse_args()
    path = ensure_oof_series(CONFIG, regenerate=args.force)
    archive = load_oof_archive(path)
    print(f"[OK] OOF archive: {path} ({len(archive.vessels)} vessels, flow={archive.flow})")


if __name__ == "__main__":
    main()
