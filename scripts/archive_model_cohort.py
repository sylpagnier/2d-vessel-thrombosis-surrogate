"""Snapshot a cohort of promoted models, with a manifest that git can carry.

**Why this exists.**  `outputs/` is gitignored and lives on one disk, so a promoted generation
is currently identified only by a directory name that the next promotion can overwrite.  Every
number in a paper is attached to a specific set of weights, and "the artifact called
`DeployClotS_0` in September" is not an identification -- it is a hope that nobody re-ran the
promoter.

So this writes two things:

  * a dated **snapshot** under `outputs/archive/<cohort>/` -- the locked artifacts, the flow
    checkpoint they consume, the CV score files their published numbers were computed from, and
    the evaluation JSONs;
  * a **manifest** next to it AND under `docs/model_cohorts/`, which IS tracked, carrying a
    SHA-256 for every file plus the provenance the artifacts record about themselves.

The blobs stay out of git; the manifest makes them verifiable.  A later session can ask "is the
`DeployClotS_0` on this disk the one the paper reports?" and get a yes or no instead of a
shrug.

    python scripts/archive_model_cohort.py --cohort split_2026_09 \\
        --artifacts DeployClotS DeployClotS_w DeployClotS_0 \\
                    DeployClotSP DeployClotSP_w DeployClotSP_0 \\
        --checkpoints outputs/runs/E5_band_gateup/kinematics_best_calibrated.pth \\
        --scores dc_v5_split dc_v5_split_seedB \\
        --evals outputs/deployclot/split_vs_shipped.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import get_project_root

REPO = get_project_root()
ARCHIVE = REPO / "outputs" / "archive"
MANIFESTS = REPO / "docs" / "model_cohorts"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _run_scoped(src: Path, prefix: str) -> str:
    """``prefix/<parent dir>/<file>`` -- the destination for a file whose NAME is not its identity.

    Training runs all write their best weights under the same basename, so six distinct
    checkpoints (`E5_band_gateup` plus the five `E8_xf5_f*` folds) all wanted
    `checkpoints/kinematics_best_calibrated.pth`.  `shutil.copy2` silently overwrote, and
    because each file was hashed straight after ITS copy, the manifest recorded six different
    digests for one path -- every one of them correct when taken, five of them describing a
    file the archive no longer held.  The snapshot preserved 1 checkpoint of 6 while claiming
    all six, which is worse than not archiving them: it reads as a verified backup.

    Discovered 2026-09-06 when a restore was checked against the manifest and five entries
    "mismatched".  The originals were still in `outputs/runs/`, so nothing was lost.
    """
    return f"{prefix}/{src.parent.name}/{src.name}"


def _assert_unique(files: list[dict]) -> None:
    """No two records may claim one path -- that is an overwrite the manifest would hide."""
    seen: dict[str, int] = {}
    for rec in files:
        seen[rec["path"]] = seen.get(rec["path"], 0) + 1
    clashes = {p: n for p, n in seen.items() if n > 1}
    if clashes:
        raise SystemExit(
            "refusing to write a manifest with duplicate paths -- each of these was copied "
            "over by a later source, so the archive holds one file while the manifest "
            f"describes several:\n  " +
            "\n  ".join(f"{p} ({n} sources)" for p, n in sorted(clashes.items())))


def _copy_in(src: Path, dst_root: Path, rel: str) -> list[dict]:
    """Copy a file or directory under ``dst_root/rel`` and return per-file records."""
    out: list[dict] = []
    dst = dst_root / rel
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        files = sorted(p for p in dst.rglob("*") if p.is_file())
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files = [dst]
    for f in files:
        out.append({"path": str(f.relative_to(dst_root)).replace("\\", "/"),
                    "bytes": f.stat().st_size, "sha256": _sha256(f)})
    return out


def _artifact_provenance(name: str) -> dict:
    """What the artifact says about itself -- the fields that decide how it may be used."""
    import torch

    d = REPO / "outputs" / "clot_ml" / "locked" / name
    meta: dict = {"name": name, "exists": d.is_dir()}
    if not d.is_dir():
        return meta
    man = d / "manifest.json"
    if man.is_file():
        try:
            raw = json.loads(man.read_text(encoding="utf-8"))
            meta.update({k: raw.get(k) for k in
                         ("metrics_invalid", "metrics_invalid_reason", "flow", "cache",
                          "scores_strict_cv", "members", "created")
                         if k in raw})
            if isinstance(meta.get("members"), list):
                meta["members"] = len(meta["members"])
        except (ValueError, OSError) as exc:
            meta["manifest_error"] = f"{type(exc).__name__}: {exc}"
    for pth in sorted(d.glob("*.pth"))[:1]:
        try:
            raw = torch.load(pth, map_location="cpu", weights_only=False)
            if isinstance(raw, dict):
                meta["checkpoint_keys"] = sorted(k for k in raw if not k.endswith("state_dict"))[:12]
        except Exception as exc:  # noqa: BLE001
            meta["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, help="archive name, e.g. split_2026_09")
    ap.add_argument("--artifacts", nargs="*", default=[],
                    help="locked artifact names under outputs/clot_ml/locked/")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="flow checkpoints to include")
    ap.add_argument("--scores", nargs="*", default=[],
                    help="CV tags; their outputs/phase9_scores/<tag>.npz are archived")
    ap.add_argument("--evals", nargs="*", default=[], help="evaluation JSONs to include")
    ap.add_argument("--note", default="", help="one line on what this cohort is")
    ap.add_argument("--force", action="store_true", help="overwrite an existing cohort")
    args = ap.parse_args()

    root = ARCHIVE / args.cohort
    if root.exists() and not args.force:
        raise SystemExit(f"{root} already exists; pass --force to overwrite. Refusing rather "
                         f"than silently merging two cohorts into one directory.")
    root.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    missing: list[str] = []

    for name in args.artifacts:
        src = REPO / "outputs" / "clot_ml" / "locked" / name
        if not src.is_dir():
            missing.append(f"artifact {name}")
            continue
        files += _copy_in(src, root, f"artifacts/{name}")

    for spec in args.checkpoints:
        src = Path(spec) if Path(spec).is_absolute() else REPO / spec
        if not src.is_file():
            missing.append(f"checkpoint {spec}")
            continue
        files += _copy_in(src, root, _run_scoped(src, "checkpoints"))

    for tag in args.scores:
        src = REPO / "outputs" / "phase9_scores" / f"{tag}.npz"
        if not src.is_file():
            missing.append(f"scores {tag}")
            continue
        files += _copy_in(src, root, f"scores/{tag}.npz")

    for spec in args.evals:
        src = Path(spec) if Path(spec).is_absolute() else REPO / spec
        if not src.is_file():
            missing.append(f"eval {spec}")
            continue
        files += _copy_in(src, root, _run_scoped(src, "evals"))

    _assert_unique(files)
    manifest = {
        "cohort": args.cohort,
        "note": args.note,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_head(),
        "snapshot_dir": str(root.relative_to(REPO)).replace("\\", "/"),
        "artifacts": [_artifact_provenance(n) for n in args.artifacts],
        "files": files,
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "missing": missing,
    }

    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    tracked = MANIFESTS / f"{args.cohort}.json"
    tracked.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] archived {len(files)} files "
          f"({manifest['total_bytes'] / 1e6:.1f} MB) -> {root}")
    print(f"[OK] manifest (tracked) -> {tracked.relative_to(REPO)}")
    for a in manifest["artifacts"]:
        flag = " metrics_invalid" if a.get("metrics_invalid") else ""
        print(f"     {a['name']:<16} members={a.get('members', '?')} "
              f"flow={a.get('flow', '?')}{flag}")
    if missing:
        print(f"[!] {len(missing)} requested item(s) not found:")
        for m in missing:
            print(f"    - {m}")
        return 2
    return 0


def _git_head() -> str:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else ""
    except OSError:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
