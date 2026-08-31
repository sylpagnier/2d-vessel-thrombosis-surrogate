"""Interactive biochem COMSOL -> PyG graph extraction.

**Default:** COMSOL auto-pull is on. Save solves as either
``comsol_models/phase2_nowound_XXX.mph`` (stem ``patientXXX``) or
``comsol_models/phase2_wound_XXX.mph`` (stem ``wound_patientXXX``).

    python -m src.tools.extract_biochem_comsol
    python -m src.tools.extract_biochem_comsol --variant nowound
    python -m src.tools.extract_biochem_comsol --stem patient007 --variant wound

Both physics families extract; they never share a stem, so wound solves cannot
overwrite the canonical nowound cohort graphs.

Runs those exports (mesh from comp1/mesh1), writes ``cfd_results_biochem/*.txt``, builds
``graphs_biochem_anchors/*.pt``. Creates data folders automatically if missing.

After each successful extract, a *lite* pack (graph + mesh, no ``.mph``) is written
to ``data/extract_transfer/<stem>/``. For Google Drive, zip once and upload that::

    python -m src.tools.extract_biochem_comsol --pack-transfer --zip-transfer --stem wound_patient001,wound_patient002

On this laptop leave the Drive download in Downloads (the ``extract_transfer``
folder or ``extract_transfer.zip``), then::

    python -m src.tools.extract_biochem_comsol --install-bundles

**Manual path:** export domain + boundary ``.txt`` yourself, then run without ``--from-comsol``.

PyCharm: **Run** module ``src.tools.extract_biochem_comsol`` (working directory = repo root).

CLI::

    python -m src.data_gen.lib.extract_biochem_comsol_data --stem patient007
    python -m src.data_gen.lib.extract_biochem_comsol_data --stem wound_patient007
    python -m src.tools.extract_biochem_comsol --list-only
    python -m src.tools.extract_biochem_comsol --stem patient048 --force
    python -m src.tools.extract_biochem_comsol --verbose   # mph/JVM + per-file export logs
    python -m src.tools.extract_biochem_comsol --pack-transfer --zip-transfer --only-new
    python -m src.tools.extract_biochem_comsol --install-bundles --only-new
    python -m src.data_gen.lib.extract_biochem_comsol_data --no-from-comsol
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.data_gen.lib.biochem_comsol_auto_export import (
    collect_biochem_extract_stems,
    parse_biochem_extract_stem,
    resolve_biochem_comsol_model_path,
    resolve_stem_selection,
)
from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor
from src.data_gen.pipeline_biochem import _auto_scaffold_anchor_sidecars
from src.tools.prepare_biochem_anchors import enrich_anchor_meshes, stems_in_dir
from src.utils.paths import data_root

_BOUNDARY_SUFFIXES = ("_inlet", "_outlet", "_wall", "_wound")


@dataclass(frozen=True)
class AnchorExtractStatus:
    stem: str
    has_mesh: bool
    has_domain_txt: bool
    has_inlet_txt: bool
    has_outlet_txt: bool
    has_wall_txt: bool
    has_wound_txt: bool
    has_biochem_graph: bool
    has_kine_graph: bool
    biochem_graph_mtime: float | None
    has_comsol_model: bool
    comsol_model_path: Path | None

    @property
    def variant(self) -> str:
        ref = parse_biochem_extract_stem(self.stem)
        return ref.variant if ref is not None else "-"

    @property
    def export_count(self) -> int:
        return sum(
            (
                self.has_domain_txt,
                self.has_inlet_txt,
                self.has_outlet_txt,
                self.has_wall_txt,
            )
        )

    @property
    def exports_ready(self) -> bool:
        return self.export_count == 4

    @property
    def can_extract(self) -> bool:
        return self.has_mesh and self.has_domain_txt

    @property
    def can_pull_from_comsol(self) -> bool:
        return self.has_comsol_model

    @property
    def already_extracted(self) -> bool:
        return self.has_biochem_graph


def _domain_export_stems(label_dir: Path) -> list[str]:
    if not label_dir.is_dir():
        return []
    stems: list[str] = []
    for p in sorted(label_dir.glob("*.txt")):
        stem = p.stem
        if any(stem.endswith(suf) for suf in _BOUNDARY_SUFFIXES):
            continue
        stems.append(stem)
    return stems


def _collect_stems(raw_dir: Path, label_dir: Path) -> list[str]:
    return collect_biochem_extract_stems(raw_dir, label_dir)


def _status_for_stem(
    stem: str,
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
    kine_dir: Path,
) -> AnchorExtractStatus:
    ref = parse_biochem_extract_stem(stem)
    if ref is not None:
        stem = ref.stem
    mesh = (raw_dir / f"{stem}.nas").exists() or (raw_dir / f"{stem}.msh").exists()
    biochem_pt = proc_dir / f"{stem}.pt"
    kine_pt = kine_dir / f"{stem}.pt"
    mtime = biochem_pt.stat().st_mtime if biochem_pt.is_file() else None
    mph_path = resolve_biochem_comsol_model_path(stem)
    return AnchorExtractStatus(
        stem=stem,
        has_mesh=mesh,
        has_domain_txt=(label_dir / f"{stem}.txt").is_file(),
        has_inlet_txt=(label_dir / f"{stem}_inlet.txt").is_file(),
        has_outlet_txt=(label_dir / f"{stem}_outlet.txt").is_file(),
        has_wall_txt=(label_dir / f"{stem}_wall.txt").is_file(),
        has_wound_txt=(label_dir / f"{stem}_wound.txt").is_file(),
        has_biochem_graph=biochem_pt.is_file(),
        has_kine_graph=kine_pt.is_file(),
        biochem_graph_mtime=mtime,
        comsol_model_path=mph_path,
        has_comsol_model=mph_path is not None,
    )


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _exports_label(s: AnchorExtractStatus) -> str:
    if s.exports_ready:
        return "OK (4/4)"
    if s.export_count == 0:
        return "missing"
    return f"partial ({s.export_count}/4)"


def _row_tag(s: AnchorExtractStatus) -> str:
    if s.already_extracted and s.can_extract:
        return "[extracted]"
    if s.can_extract and s.exports_ready:
        return "[ready]"
    if s.can_extract:
        return "[ready*]"
    if s.has_comsol_model and not s.has_mesh:
        return "[mph->mesh]"
    if s.has_domain_txt and not s.has_mesh:
        return "[no mesh]"
    if s.has_mesh and not s.has_domain_txt and s.has_comsol_model:
        return "[mph ready]"
    if s.has_mesh and not s.has_domain_txt:
        return "[no export]"
    return "[incomplete]"


def print_status_table(
    statuses: list[AnchorExtractStatus],
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
) -> None:
    print(f"\n[i] Mesh dir:    {raw_dir}")
    print(f"[i] COMSOL txt:  {label_dir}")
    print(f"[i] Graph out:   {proc_dir}")
    mph_only = [s.stem for s in statuses if s.has_comsol_model and not s.has_mesh]
    if mph_only:
        print(f"[i] COMSOL .mph without mesh yet: {', '.join(mph_only)}")
    print()
    if not statuses:
        print("[WARN] No anchor stems found (need .msh/.nas and/or domain .txt exports).")
        return
    print(
        f"{'#':>3}  {'stem':<22}  {'variant':<8}  {'tag':<14}  {'mesh':<5}  {'exports':<14}  "
        f"{'biochem .pt':<20}  {'kine':<6}  {'COMSOL .mph':<28}"
    )
    print("-" * 128)
    for i, s in enumerate(statuses, start=1):
        mesh = "yes" if s.has_mesh else "no"
        graph = "yes" if s.has_biochem_graph else "no"
        if s.has_biochem_graph:
            graph = f"yes {_fmt_mtime(s.biochem_graph_mtime)}"
        kine = "yes" if s.has_kine_graph else "no"
        mph = s.comsol_model_path.name if s.comsol_model_path else "-"
        print(
            f"{i:>3}  {s.stem:<22}  {s.variant:<8}  {_row_tag(s):<14}  {mesh:<5}  "
            f"{_exports_label(s):<14}  {graph:<20}  {kine:<6}  {mph:<28}"
        )
    print(
        "[i] [ready]=mesh+txt  [extracted]=graph  "
        "phase2_nowound_XXX.mph -> patientXXX;  phase2_wound_XXX.mph -> wound_patientXXX"
    )


def _prompt_yes_no(label: str, *, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y or n.")


def _resolve_choices(
    raw: str,
    statuses: list[AnchorExtractStatus],
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
    kine_dir: Path,
    variant: str | None = None,
) -> list[AnchorExtractStatus]:
    """Parse ``5,8,9``, ``5-9``, ``patient005``, or ``wound_patient007`` into status rows."""
    try:
        stems = resolve_stem_selection(raw, statuses, variant=variant)
    except ValueError as exc:
        print(f"  {exc}")
        return []
    if not stems:
        return []
    by_stem = {s.stem: s for s in statuses}
    picked: list[AnchorExtractStatus] = []
    for stem in stems:
        if stem in by_stem:
            picked.append(by_stem[stem])
        else:
            print(f"  [i] Adding {stem} (not in table; will try if .mph exists).")
            picked.append(
                _status_for_stem(
                    stem,
                    raw_dir=raw_dir,
                    label_dir=label_dir,
                    proc_dir=proc_dir,
                    kine_dir=kine_dir,
                )
            )
    return picked


def _can_run_status(s: AnchorExtractStatus, *, from_comsol: bool) -> bool:
    return s.can_extract or (from_comsol and s.can_pull_from_comsol)


def _maybe_pull_comsol(
    stem: str,
    extractor: PatientDataExtractor,
    *,
    from_comsol: bool,
    model_path: Path | None,
    force: bool,
) -> bool:
    domain_txt = extractor.label_dir / f"{stem}.txt"
    if domain_txt.is_file() and not force:
        return True
    if not from_comsol:
        if not domain_txt.is_file():
            print(
                f"[ERR] {stem}: missing {domain_txt.name}. "
                "Use --from-comsol after saving <stem>.mph, or export txt manually."
            )
        return domain_txt.is_file()

    resolved = resolve_biochem_comsol_model_path(stem, model_path)
    if resolved is None:
        ref = parse_biochem_extract_stem(stem)
        mph_hint = ref.mph_name if ref is not None else f"{stem}.mph"
        print(
            f"[ERR] {stem}: no .mph found. Expected comsol_models/{mph_hint} "
            f"(or {extractor.raw_dir / f'{stem}.mph'} / BIOCHEM_COMSOL_MODEL)."
        )
        return False

    try:
        extractor.pull_comsol_exports(stem, model_path=resolved, force=force)
    except Exception as exc:
        print(f"[ERR] COMSOL pull failed for {stem}: {exc}")
        return False
    return domain_txt.is_file() or (extractor.label_dir / f"{stem}.txt").is_file()


def _run_extract(
    stem: str,
    extractor: PatientDataExtractor,
    *,
    force: bool,
    skip_enrich: bool,
    raw_dir: Path,
    from_comsol: bool,
    model_path: Path | None,
) -> bool:
    ref = parse_biochem_extract_stem(stem)
    if ref is not None:
        stem = ref.stem
    print(f"[NEW] {stem}")
    if not _maybe_pull_comsol(
        stem, extractor, from_comsol=from_comsol, model_path=model_path, force=force
    ):
        return False

    biochem_pt = extractor.proc_dir / f"{stem}.pt"
    if biochem_pt.is_file() and not force:
        print(
            f"[WARN] {stem}: graph already exists ({biochem_pt}). "
            "Use --force or answer y to overwrite."
        )
        if not _prompt_yes_no(f"Overwrite {stem}.pt?", default=False):
            print("[skip] extraction cancelled.")
            return False

    if not skip_enrich:
        enrich_anchor_meshes(
            raw_dir, overwrite=False, dry_run=False, stems=[stem], quiet=True
        )

    try:
        extractor.process_patient(stem)
    except Exception as exc:
        print(f"[ERR] Extraction failed for {stem}: {exc}")
        return False
    if biochem_pt.is_file():
        return True
    print(f"[ERR] {stem}: no graph written.")
    return False


def _run_extract_batch(
    picked: list[AnchorExtractStatus],
    extractor: PatientDataExtractor,
    *,
    force: bool,
    skip_enrich: bool,
    raw_dir: Path,
    from_comsol: bool,
    model_path: Path | None,
) -> tuple[int, int]:
    """Extract multiple stems; returns ``(ok_count, total)``."""
    runnable = [s for s in picked if _can_run_status(s, from_comsol=from_comsol)]
    skipped = [s.stem for s in picked if s not in runnable]
    if skipped:
        print(f"[WARN] Skipping (no .mph / no source): {', '.join(skipped)}")

    partial = [s for s in runnable if not s.exports_ready]
    if partial:
        names = ", ".join(s.stem for s in partial)
        print(f"[WARN] Partial exports for: {names}")
        if not _prompt_yes_no("Continue batch anyway?", default=True):
            return 0, len(picked)

    needs_overwrite = [
        s
        for s in runnable
        if (extractor.proc_dir / f"{s.stem}.pt").is_file() and not force
    ]
    if needs_overwrite:
        names = ", ".join(s.stem for s in needs_overwrite)
        print(f"[WARN] Graphs already exist: {names}")
        if not _prompt_yes_no("Overwrite existing .pt files in this batch?", default=False):
            runnable = [s for s in runnable if s not in needs_overwrite]

    if not runnable:
        return 0, len(picked)

    print(f"\n[i] Batch extract {len(runnable)} stem(s): {', '.join(s.stem for s in runnable)}\n")
    ok = 0
    for s in runnable:
        if _run_extract(
            s.stem,
            extractor,
            force=force,
            skip_enrich=skip_enrich,
            raw_dir=raw_dir,
            from_comsol=from_comsol,
            model_path=model_path,
        ):
            ok += 1
    print(f"\n[i] Batch done: {ok}/{len(runnable)} succeeded.")
    return ok, len(picked)


def _interactive_loop(
    statuses: list[AnchorExtractStatus],
    extractor: PatientDataExtractor,
    *,
    force: bool,
    skip_enrich: bool,
    raw_dir: Path,
    from_comsol: bool,
    model_path: Path | None,
    variant: str | None = None,
) -> None:
    ready = [
        s
        for s in statuses
        if not s.already_extracted
        and (s.can_extract or (from_comsol and s.can_pull_from_comsol))
    ]
    print(f"\n[i] {len(ready)} stem(s) ready to extract (not yet graphed).")
    print(
        "[i] Enter index or stem, or several: 5 | 5,8,9 | 5-9 | patient008 | "
        "wound_patient007 | patient007_wound | 'l' relist, 'q' quit.\n"
    )

    while True:
        raw = input("Extract which anchor(s)? ").strip()
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            break
        if raw.lower() in ("l", "list"):
            print_status_table(
                statuses,
                raw_dir=raw_dir,
                label_dir=extractor.label_dir,
                proc_dir=extractor.proc_dir,
            )
            continue

        picked_list = _resolve_choices(
            raw,
            statuses,
            raw_dir=raw_dir,
            label_dir=extractor.label_dir,
            proc_dir=extractor.proc_dir,
            kine_dir=extractor.kine_anchor_dir,
            variant=variant,
        )
        if not picked_list:
            continue

        if len(picked_list) == 1:
            picked = picked_list[0]
            if not _can_run_status(picked, from_comsol=from_comsol):
                print(
                    f"[ERR] {picked.stem}: need domain .txt or matching "
                    f"phase2_nowound_XXX.mph / phase2_wound_XXX.mph "
                    f"(--from-comsol exports mesh + fields)."
                )
                continue
            if not picked.exports_ready:
                print(
                    f"[WARN] {picked.stem}: only {picked.export_count}/4 COMSOL txt files present; "
                    "continuing anyway (domain txt is required)."
                )
                if not _prompt_yes_no("Continue?", default=False):
                    continue
            ok = _run_extract(
                picked.stem,
                extractor,
                force=force,
                skip_enrich=skip_enrich,
                raw_dir=raw_dir,
                from_comsol=from_comsol,
                model_path=model_path,
            )
            if ok and not _prompt_yes_no("Extract another?", default=True):
                break
        else:
            _run_extract_batch(
                picked_list,
                extractor,
                force=force,
                skip_enrich=skip_enrich,
                raw_dir=raw_dir,
                from_comsol=from_comsol,
                model_path=model_path,
            )
            if not _prompt_yes_no("Extract more?", default=False):
                break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stem",
        type=str,
        default="",
        help="One or more stems: patient007 | wound_patient007 | 7 | 5,8,9 | 5-9 "
        "(non-interactive). Combine with --variant wound to restamp patientXXX.",
    )
    parser.add_argument(
        "--variant",
        choices=("nowound", "wound", "all"),
        default="all",
        help="Restrict listing/extraction to nowound (patientXXX) or wound (wound_patientXXX).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the anchor inventory table and exit (no extract).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull domain txt / overwrite .pt. Keeps mesh and boundary txt unless "
        "BIOCHEM_COMSOL_FORCE_MESH=1 or BIOCHEM_COMSOL_FORCE_BOUNDARY=1.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep COMSOL/mph session and per-file export logs (or set BIOCHEM_EXTRACT_VERBOSE=1).",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Anchor meshes (default: data/raw/biochem_anchors).")
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="COMSOL exports (default: data/processed/cfd_results_biochem).",
    )
    parser.add_argument(
        "--skip-sidecars",
        action="store_true",
        help="Skip automatic cm sidecar scaffold on raw meshes.",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip Gmsh sidecar enrichment before extract.",
    )
    parser.add_argument(
        "--from-comsol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pull domain/boundary txt from solved .mph via mph (default: on).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit path to solved .mph (default: <stem>.mph in biochem_anchors or BIOCHEM_COMSOL_MODEL).",
    )
    parser.add_argument(
        "--install-bundles",
        action="store_true",
        help="Install a transfer folder/zip from Downloads into canonical data/ paths and exit.",
    )
    parser.add_argument(
        "--pack-transfer",
        action="store_true",
        help="Rebuild transfer bundles for --stem (or all extracted stems) without re-pulling COMSOL. "
        "Default pack is lite (graph + mesh only).",
    )
    parser.add_argument(
        "--full-transfer",
        action="store_true",
        help="With --pack-transfer: include domain txt and kine.pt (not needed for graph work).",
    )
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="With --pack-transfer: pack only anchor meshes (.nas/.msh + sidecar, no graph.pt).",
    )
    parser.add_argument(
        "--zip-transfer",
        action="store_true",
        help="Zip data/extract_transfer to data/extract_transfer.zip (one Drive upload).",
    )
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=None,
        help="Override the incoming folder or zip (default: Downloads/extract_transfer).",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Pack/zip only stems newer than the last bundle or zip; install only stems "
        "missing locally (use --force to overwrite).",
    )
    args = parser.parse_args(argv)
    from src.data_gen.lib.extract_logging import quiet_comsol_extract_logs

    quiet_comsol_extract_logs(verbose=True if args.verbose else None)

    dr = data_root()
    raw_dir = args.raw_dir or (dr / "raw" / "biochem_anchors")
    label_dir = args.label_dir or (dr / "processed" / "cfd_results_biochem")

    from src.data_gen.lib.biochem_comsol_mph_export import ensure_biochem_extract_dirs

    extractor = PatientDataExtractor(phase="biochem_anchors", raw_dir=raw_dir, label_dir=label_dir)
    ensure_biochem_extract_dirs(raw_dir, label_dir, extractor.proc_dir)

    from src.data_gen.lib.biochem_extract_transfer import (
        default_downloads_dir,
        extract_transfer_dir,
        filter_stems_for_pack,
        install_incoming_extract_transfer,
        select_bundle_names_for_zip,
        stage_extract_transfer_bundle,
        stems_from_raw_meshes,
        zip_extract_transfer_dir,
    )

    transfer_base = args.transfer_dir or extract_transfer_dir()

    if args.install_bundles:
        incoming, results = install_incoming_extract_transfer(
            transfer_dir=args.transfer_dir,
            downloads_dir=default_downloads_dir(),
            data_transfer_dir=extract_transfer_dir(),
            stems=[s.strip() for s in args.stem.split(",") if s.strip()] or None,
            force=args.force,
            only_new=args.only_new,
        )
        if incoming is None or not results:
            downloads = default_downloads_dir()
            raise SystemExit(
                "[ERR] No transfer folder or extract_transfer.zip found. "
                f"Looked in {downloads} and data/extract_transfer/."
            )
        print(f"[i] installing from {incoming.label}")
        for name, written in results:
            skip_vals = [v for v in written.values() if str(v).startswith("skip")]
            if skip_vals:
                print(f"[skip] {name}: already installed (use --force to overwrite)")
                continue
            print(f"[OK] installed {name}")
            for key, dest in written.items():
                if dest.startswith("skip"):
                    continue
                print(f"      {key} -> {dest}")
        raise SystemExit(0)

    if args.pack_transfer:
        if args.stem.strip():
            pack_stems = [s.strip() for s in args.stem.split(",") if s.strip()]
            resolved = []
            for token in pack_stems:
                ref = parse_biochem_extract_stem(token)
                resolved.append(ref.stem if ref is not None else token)
            pack_stems = resolved
        elif args.mesh_only:
            pack_stems = stems_from_raw_meshes(raw_dir)
        else:
            pack_stems = sorted(p.stem for p in extractor.proc_dir.glob("*.pt") if p.is_file())
        if not pack_stems:
            raise SystemExit("[ERR] No stems to pack.")
        requested_stems = list(pack_stems)
        pack_stems = filter_stems_for_pack(
            pack_stems,
            proc_dir=extractor.proc_dir,
            transfer_dir=transfer_base,
            only_new=args.only_new,
            mesh_only=args.mesh_only,
            raw_dir=raw_dir,
        )
        if args.only_new:
            for stem in requested_stems:
                if stem not in pack_stems:
                    print(f"[skip] {stem}: transfer bundle up to date")
        if not pack_stems:
            print("[i] No new stems to pack.")
            if args.zip_transfer:
                zip_names = select_bundle_names_for_zip(
                    transfer_dir=transfer_base,
                    only_new=True,
                    stems=[s.strip() for s in args.stem.split(",") if s.strip()] or None,
                )
                if not zip_names:
                    raise SystemExit("[ERR] No new bundles to zip.")
                print(f"[i] zipping {len(zip_names)} bundle(s): {', '.join(zip_names)}")
                archive = zip_extract_transfer_dir(
                    transfer_dir=transfer_base,
                    bundle_names=zip_names,
                )
                print(f"[save] {archive}")
            raise SystemExit(0)
        n = 0
        packed: list[str] = []
        for stem in pack_stems:
            bundle = stage_extract_transfer_bundle(
                stem,
                raw_dir=raw_dir,
                label_dir=label_dir,
                proc_dir=extractor.proc_dir,
                kine_dir=extractor.kine_anchor_dir,
                lite=not args.full_transfer and not args.mesh_only,
                mesh_only=args.mesh_only,
            )
            if bundle is None:
                if args.mesh_only:
                    print(f"[WARN] {stem}: no .nas/.msh, skip pack")
                else:
                    print(f"[WARN] {stem}: no graph.pt, skip pack")
                continue
            print(f"[save] {bundle}")
            packed.append(stem)
            n += 1
        if args.zip_transfer:
            zip_names = select_bundle_names_for_zip(
                transfer_dir=transfer_base,
                only_new=args.only_new,
                stems=packed or None,
            )
            if args.only_new and not zip_names:
                raise SystemExit("[ERR] No new bundles to zip.")
            if args.only_new:
                print(f"[i] zipping {len(zip_names)} bundle(s): {', '.join(zip_names)}")
            archive = zip_extract_transfer_dir(
                transfer_dir=transfer_base,
                bundle_names=zip_names if args.only_new else None,
            )
            print(f"[save] {archive}")
        raise SystemExit(0 if n else 1)

    if args.zip_transfer:
        zip_names = select_bundle_names_for_zip(
            transfer_dir=transfer_base,
            only_new=args.only_new,
            stems=[s.strip() for s in args.stem.split(",") if s.strip()] or None,
        )
        if args.only_new:
            if not zip_names:
                raise SystemExit("[ERR] No new bundles to zip.")
            print(f"[i] zipping {len(zip_names)} bundle(s): {', '.join(zip_names)}")
        archive = zip_extract_transfer_dir(
            transfer_dir=transfer_base,
            bundle_names=zip_names if args.only_new else None,
        )
        print(f"[save] {archive}")
        raise SystemExit(0)

    if not args.skip_sidecars:
        _auto_scaffold_anchor_sidecars(raw_dir)
    stems = _collect_stems(raw_dir, label_dir)
    statuses = [
        _status_for_stem(
            stem,
            raw_dir=raw_dir,
            label_dir=label_dir,
            proc_dir=extractor.proc_dir,
            kine_dir=extractor.kine_anchor_dir,
        )
        for stem in stems
    ]
    if args.variant != "all":
        statuses = [s for s in statuses if s.variant == args.variant]

    stem_arg = args.stem.strip()
    # Full inventory is for --list-only and the interactive picker, not --stem extract.
    if args.list_only or not stem_arg:
        print_status_table(statuses, raw_dir=raw_dir, label_dir=label_dir, proc_dir=extractor.proc_dir)

    if args.list_only:
        return

    if stem_arg:
        picked_list = _resolve_choices(
            stem_arg,
            statuses,
            raw_dir=raw_dir,
            label_dir=label_dir,
            proc_dir=extractor.proc_dir,
            kine_dir=extractor.kine_anchor_dir,
            variant=None if args.variant == "all" else args.variant,
        )
        if not picked_list:
            raise SystemExit("[ERR] No stems matched --stem selection.")
        if len(picked_list) == 1:
            s = picked_list[0]
            if not _can_run_status(s, from_comsol=args.from_comsol):
                raise SystemExit(
                    f"[ERR] {s.stem}: missing mesh and/or COMSOL source "
                    "(need .msh/.nas and a domain .txt or matching .mph)."
                )
            ok = _run_extract(
                s.stem,
                extractor,
                force=args.force,
                skip_enrich=args.skip_enrich,
                raw_dir=raw_dir,
                from_comsol=args.from_comsol,
                model_path=args.model_path,
            )
            raise SystemExit(0 if ok else 1)
        ok_count, _ = _run_extract_batch(
            picked_list,
            extractor,
            force=args.force,
            skip_enrich=args.skip_enrich,
            raw_dir=raw_dir,
            from_comsol=args.from_comsol,
            model_path=args.model_path,
        )
        raise SystemExit(0 if ok_count > 0 else 1)

    _interactive_loop(
        statuses,
        extractor,
        force=args.force,
        skip_enrich=args.skip_enrich,
        raw_dir=raw_dir,
        from_comsol=args.from_comsol,
        model_path=args.model_path,
        variant=None if args.variant == "all" else args.variant,
    )


if __name__ == "__main__":
    main()
