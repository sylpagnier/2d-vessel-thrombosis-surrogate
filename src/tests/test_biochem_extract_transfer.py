"""Tests for one-folder biochem extract transfer packs."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_gen.lib.biochem_extract_transfer import (
    bundle_needs_install,
    extract_transfer_zip_path,
    filter_stems_for_pack,
    install_all_extract_transfer_bundles,
    install_extract_transfer_bundle,
    install_incoming_extract_transfer,
    resolve_incoming_transfer,
    select_bundle_names_for_zip,
    stage_extract_transfer_bundle,
    stem_needs_pack,
    unpack_extract_transfer_zip,
    zip_extract_transfer_dir,
)


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stage_and_install_transfer_bundle_roundtrip(tmp_path):
    stem = "wound_patient001"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    kine = tmp_path / "kine"
    _touch(proc / f"{stem}.pt", "graph")
    _touch(proc / f"{stem}_metadata.json", "{}")
    _touch(raw / f"{stem}.msh", "mesh")
    _touch(raw / f"{stem}.nas", "nas")
    _touch(raw / f"{stem}.json", "{\"d_bar\": 1}")
    _touch(label / f"{stem}.txt", "domain")
    _touch(label / f"{stem}_wall.txt", "wall")
    _touch(label / f"{stem}_wound.txt", "wound")
    _touch(kine / f"{stem}.pt", "kine")

    bundle = stage_extract_transfer_bundle(
        stem,
        raw_dir=raw,
        label_dir=label,
        proc_dir=proc,
        kine_dir=kine,
        root=tmp_path,
        lite=False,
    )
    assert bundle is not None
    assert (bundle / "graph.pt").is_file()
    assert (bundle / "wound.txt").is_file()
    assert (bundle / "mesh.nas").is_file()
    assert (bundle / "manifest.json").is_file()

    dest_root = tmp_path / "laptop"
    written = install_extract_transfer_bundle(bundle, root=dest_root, force=True)
    assert written["graph.pt"].endswith(f"{stem}.pt")
    assert (dest_root / "data/processed/graphs_biochem_anchors" / f"{stem}.pt").read_text(
        encoding="utf-8"
    ) == "graph"
    assert (dest_root / "data/raw/biochem_anchors" / f"{stem}.msh").is_file()
    assert (dest_root / "data/processed/cfd_results_biochem" / f"{stem}_wound.txt").is_file()


def test_install_skips_existing_graph_unless_force(tmp_path):
    stem = "patient007"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / f"{stem}.pt", "new-graph")
    bundle = stage_extract_transfer_bundle(
        stem, raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path
    )
    dest = tmp_path / "laptop"
    graph_dest = dest / "data/processed/graphs_biochem_anchors" / f"{stem}.pt"
    _touch(graph_dest, "old-graph")
    written = install_extract_transfer_bundle(bundle, root=dest, force=False)
    assert written["graph.pt"].startswith("skip")
    assert graph_dest.read_text(encoding="utf-8") == "old-graph"
    install_extract_transfer_bundle(bundle, root=dest, force=True)
    assert graph_dest.read_text(encoding="utf-8") == "new-graph"


def test_install_all_filters_by_stem(tmp_path):
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / "wound_patient001.pt")
    _touch(proc / "wound_patient002.pt")
    stage_extract_transfer_bundle(
        "wound_patient001", raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path
    )
    stage_extract_transfer_bundle(
        "wound_patient002", raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path
    )
    dest = tmp_path / "laptop"
    results = install_all_extract_transfer_bundles(
        root=dest,
        transfer_dir=tmp_path / "data" / "extract_transfer",
        stems=["wound_patient001"],
        force=True,
    )
    assert [name for name, _ in results] == ["wound_patient001"]
    assert (dest / "data/processed/graphs_biochem_anchors/wound_patient001.pt").is_file()
    assert not (dest / "data/processed/graphs_biochem_anchors/wound_patient002.pt").is_file()


def test_lite_pack_includes_nas_when_present(tmp_path):
    stem = "wound_patient003"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    kine = tmp_path / "kine"
    _touch(proc / f"{stem}.pt", "graph")
    _touch(proc / f"{stem}_metadata.json", "{}")
    _touch(raw / f"{stem}.msh", "mesh")
    _touch(raw / f"{stem}.nas", "nas")
    _touch(raw / f"{stem}.json", "{}")
    _touch(label / f"{stem}.txt", "domain")
    _touch(label / f"{stem}_wound.txt", "wound")
    _touch(kine / f"{stem}.pt", "kine")

    bundle = stage_extract_transfer_bundle(
        stem,
        raw_dir=raw,
        label_dir=label,
        proc_dir=proc,
        kine_dir=kine,
        root=tmp_path,
        lite=True,
    )
    assert bundle is not None
    assert (bundle / "graph.pt").is_file()
    assert (bundle / "mesh.msh").is_file()
    assert (bundle / "mesh.nas").is_file()
    assert (bundle / "mesh.json").is_file()
    assert (bundle / "wound.txt").is_file()
    assert not (bundle / "kine.pt").exists()
    dest = tmp_path / "laptop"
    written = install_extract_transfer_bundle(bundle, root=dest, force=True)
    assert "graph.pt" in written
    assert "wound.txt" in written
    assert (dest / "data/processed/cfd_results_biochem" / f"{stem}_wound.txt").is_file()


def test_mesh_only_pack_roundtrip(tmp_path):
    stem = "patient041"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(raw / f"{stem}.nas", "nas")
    _touch(raw / f"{stem}.msh", "mesh")
    _touch(raw / f"{stem}.json", '{"unit":"cm"}')
    bundle = stage_extract_transfer_bundle(
        stem,
        raw_dir=raw,
        label_dir=label,
        proc_dir=proc,
        root=tmp_path,
        mesh_only=True,
    )
    assert bundle is not None
    assert (bundle / "mesh.nas").is_file()
    assert not (bundle / "graph.pt").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mesh_only"] is True
    dest = tmp_path / "laptop"
    written = install_extract_transfer_bundle(bundle, root=dest, force=True)
    assert "mesh.nas" in written
    assert (dest / "data/raw/biochem_anchors" / f"{stem}.nas").read_text(encoding="utf-8") == "nas"


def test_lite_repack_drops_full_pack_leftovers(tmp_path):
    stem = "patient007"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / f"{stem}.pt", "graph")
    _touch(raw / f"{stem}.msh", "mesh")
    _touch(label / f"{stem}_wound.txt", "wound")
    stage_extract_transfer_bundle(
        stem, raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path, lite=False
    )
    bundle = stage_extract_transfer_bundle(
        stem, raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path, lite=True
    )
    assert bundle is not None
    assert (bundle / "wound.txt").is_file()


def test_zip_and_unpack_transfer_dir(tmp_path):
    stem = "wound_patient001"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / f"{stem}.pt", "graph")
    _touch(raw / f"{stem}.msh", "mesh")
    _touch(raw / f"{stem}.json", "{}")
    stage_extract_transfer_bundle(
        stem, raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path, lite=True
    )
    archive = zip_extract_transfer_dir(root=tmp_path)
    assert archive == extract_transfer_zip_path(root=tmp_path)
    assert archive.is_file()

    other = tmp_path / "other"
    unpacked = unpack_extract_transfer_zip(archive, root=other)
    assert (unpacked / stem / "graph.pt").is_file()
    dest = tmp_path / "laptop"
    results = install_all_extract_transfer_bundles(
        root=dest, transfer_dir=unpacked, force=True
    )
    assert results[0][0] == stem
    assert (dest / "data/processed/graphs_biochem_anchors" / f"{stem}.pt").read_text(
        encoding="utf-8"
    ) == "graph"


def _stage_lite(tmp_path, stem: str) -> None:
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / f"{stem}.pt", f"graph-{stem}")
    _touch(raw / f"{stem}.msh", "mesh")
    _touch(raw / f"{stem}.json", "{}")
    stage_extract_transfer_bundle(
        stem, raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path, lite=True
    )


def test_install_from_downloads_extract_transfer_folder(tmp_path):
    stem = "wound_patient001"
    _stage_lite(tmp_path, stem)
    downloads = tmp_path / "Downloads"
    incoming_dir = downloads / "extract_transfer"
    incoming_dir.mkdir(parents=True)
    src = tmp_path / "data" / "extract_transfer" / stem
    (incoming_dir / stem).mkdir()
    for name in ("graph.pt", "mesh.msh", "mesh.json", "manifest.json"):
        (incoming_dir / stem / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")

    dest = tmp_path / "laptop"
    incoming, results = install_incoming_extract_transfer(
        root=dest,
        downloads_dir=downloads,
        data_transfer_dir=tmp_path / "empty_data",
        force=True,
    )
    assert incoming is not None
    assert incoming.transfer_dir == incoming_dir
    assert results[0][0] == stem
    assert (dest / "data/processed/graphs_biochem_anchors" / f"{stem}.pt").read_text(
        encoding="utf-8"
    ) == f"graph-{stem}"


def test_install_from_unnamed_downloads_folder(tmp_path):
    stem = "wound_patient002"
    _stage_lite(tmp_path, stem)
    downloads = tmp_path / "Downloads"
    pack = downloads / "wound_graphs"
    src = tmp_path / "data" / "extract_transfer" / stem
    (pack / stem).mkdir(parents=True)
    for name in ("graph.pt", "manifest.json"):
        (pack / stem / name).write_bytes((src / name).read_bytes())

    dest = tmp_path / "laptop"
    incoming, results = install_incoming_extract_transfer(
        root=dest,
        downloads_dir=downloads,
        data_transfer_dir=tmp_path / "empty_data",
        force=True,
    )
    assert incoming is not None
    assert incoming.transfer_dir == pack
    assert [name for name, _ in results] == [stem]


def test_install_from_downloads_zip(tmp_path):
    stem = "wound_patient001"
    _stage_lite(tmp_path, stem)
    archive = zip_extract_transfer_dir(root=tmp_path)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    dest_zip = downloads / "extract_transfer.zip"
    dest_zip.write_bytes(archive.read_bytes())

    dest = tmp_path / "laptop"
    incoming, results = install_incoming_extract_transfer(
        root=dest,
        downloads_dir=downloads,
        data_transfer_dir=tmp_path / "empty_data",
        force=True,
    )
    assert incoming is not None
    assert incoming.zip_path == dest_zip
    assert results[0][0] == stem
    assert (dest / "data/processed/graphs_biochem_anchors" / f"{stem}.pt").is_file()


def test_downloads_preferred_over_data_extract_transfer(tmp_path):
    _stage_lite(tmp_path, "wound_patient001")
    downloads = tmp_path / "Downloads"
    pack = downloads / "extract_transfer" / "wound_patient003"
    pack.mkdir(parents=True)
    _touch(pack / "graph.pt", "from-downloads")
    _touch(pack / "manifest.json", '{"stem": "wound_patient003", "files": {"graph.pt": "data/processed/graphs_biochem_anchors/wound_patient003.pt"}}\n')

    found = resolve_incoming_transfer(
        downloads_dir=downloads,
        data_transfer_dir=tmp_path / "data" / "extract_transfer",
    )
    assert found is not None
    assert found.transfer_dir == downloads / "extract_transfer"


def test_filter_stems_for_pack_only_new(tmp_path):
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    transfer = tmp_path / "data" / "extract_transfer"
    _touch(proc / "wound_patient005.pt", "old")
    _touch(proc / "wound_patient006.pt", "new")
    stage_extract_transfer_bundle(
        "wound_patient005",
        raw_dir=raw,
        label_dir=label,
        proc_dir=proc,
        root=tmp_path,
        lite=True,
    )
    assert not stem_needs_pack(
        "wound_patient005",
        proc_dir=proc,
        transfer_dir=transfer,
    )
    assert stem_needs_pack(
        "wound_patient006",
        proc_dir=proc,
        transfer_dir=transfer,
    )
    packed = filter_stems_for_pack(
        ["wound_patient005", "wound_patient006"],
        proc_dir=proc,
        transfer_dir=transfer,
        only_new=True,
    )
    assert packed == ["wound_patient006"]


def test_select_bundle_names_for_zip_only_new(tmp_path, monkeypatch):
    import time

    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / "wound_patient005.pt", "old")
    stage_extract_transfer_bundle(
        "wound_patient005",
        raw_dir=raw,
        label_dir=label,
        proc_dir=proc,
        root=tmp_path,
        lite=True,
    )
    archive = zip_extract_transfer_dir(root=tmp_path)
    time.sleep(0.05)
    _touch(proc / "wound_patient006.pt", "new")
    stage_extract_transfer_bundle(
        "wound_patient006",
        raw_dir=raw,
        label_dir=label,
        proc_dir=proc,
        root=tmp_path,
        lite=True,
    )
    transfer = tmp_path / "data" / "extract_transfer"
    names = select_bundle_names_for_zip(
        transfer_dir=transfer,
        dest_zip=archive,
        only_new=True,
    )
    assert names == ["wound_patient006"]


def test_install_only_new_skips_existing_graph(tmp_path):
    stem = "wound_patient006"
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    proc = tmp_path / "proc"
    _touch(proc / f"{stem}.pt", "incoming")
    bundle = stage_extract_transfer_bundle(
        stem, raw_dir=raw, label_dir=label, proc_dir=proc, root=tmp_path, lite=True
    )
    assert bundle is not None
    dest = tmp_path / "laptop"
    graph_dest = dest / "data/processed/graphs_biochem_anchors" / f"{stem}.pt"
    _touch(graph_dest, "local")
    assert not bundle_needs_install(bundle, root=dest, only_new=True)
    results = install_all_extract_transfer_bundles(
        root=dest,
        transfer_dir=tmp_path / "data" / "extract_transfer",
        only_new=True,
    )
    assert results[0][1]["graph.pt"].startswith("skip")
    assert graph_dest.read_text(encoding="utf-8") == "local"
