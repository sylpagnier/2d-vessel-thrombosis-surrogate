"""One-folder transfer packs for biochem COMSOL extracts.

Canonical training paths stay split (``raw/biochem_anchors``, ``cfd_results_biochem``,
``graphs_biochem_anchors``). After a successful extract we also mirror the artifacts
into ``data/extract_transfer/<stem>/``.

Default pack is *lite* (graph + mesh + ``.nas`` + wound.txt). Do not copy ``.mph`` files for
graph work — they dominate Drive time and are unused on the analysis laptop.

On the COMSOL PC::

    python -m src.tools.extract_biochem_comsol --pack-transfer --zip-transfer --only-new
    python -m src.tools.extract_biochem_comsol --pack-transfer --zip-transfer --mesh-only --stem comsol009,comsol041

``--mesh-only`` builds a small FEM mesh drop (``.nas``/``.msh`` + sidecar, no ``graph.pt``).

Upload ``data/extract_transfer.zip`` (or the ``extract_transfer`` folder). On this
laptop leave the Drive download in ``Downloads``, then::

    python -m src.tools.extract_biochem_comsol --install-bundles --only-new

``--only-new`` on pack skips stems whose transfer bundle already matches the graph;
on zip it includes only bundles newer than the last ``extract_transfer.zip``;
on install it skips stems already present locally (use ``--force`` to overwrite).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data_gen.lib.biochem_comsol_auto_export import parse_biochem_extract_stem
from src.utils.paths import data_root, get_project_root

MANIFEST_NAME = "manifest.json"
# Drive/laptop copies: graph + mesh sidecar + .nas + wound identity. Skip .mph / domain txt / kine.
_LITE_NAMES = frozenset(
    {
        "graph.pt",
        "graph_metadata.json",
        "mesh.msh",
        "mesh.nas",
        "mesh.json",
        "wound.txt",
    }
)
_MESH_ONLY_NAMES = frozenset({"mesh.msh", "mesh.nas", "mesh.json"})

_BUNDLE_KEYS: tuple[tuple[str, str, bool], ...] = (
    # (bundle filename, dest relative to repo root with {stem}, required)
    ("graph.pt", "data/processed/graphs_biochem_anchors/{stem}.pt", True),
    ("graph_metadata.json", "data/processed/graphs_biochem_anchors/{stem}_metadata.json", False),
    ("mesh.msh", "data/raw/biochem_anchors/{stem}.msh", False),
    ("mesh.nas", "data/raw/biochem_anchors/{stem}.nas", False),
    ("mesh.json", "data/raw/biochem_anchors/{stem}.json", False),
    ("domain.txt", "data/processed/cfd_results_biochem/{stem}.txt", False),
    ("inlet.txt", "data/processed/cfd_results_biochem/{stem}_inlet.txt", False),
    ("outlet.txt", "data/processed/cfd_results_biochem/{stem}_outlet.txt", False),
    ("wall.txt", "data/processed/cfd_results_biochem/{stem}_wall.txt", False),
    ("wound.txt", "data/processed/cfd_results_biochem/{stem}_wound.txt", False),
    ("kine.pt", "data/processed/graphs_kinematics_anchors/carreau/{stem}.pt", False),
)


def canonical_graph_dest(stem: str, *, root: Path | None = None) -> Path:
    """Installed biochem graph path for ``stem``."""
    repo = Path(root) if root is not None else get_project_root()
    return repo / "data/processed/graphs_biochem_anchors" / f"{stem}.pt"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def stem_needs_pack(
    stem: str,
    *,
    proc_dir: Path,
    transfer_dir: Path,
) -> bool:
    """True when the canonical graph is missing a bundle or is newer than the bundle copy."""
    graph_src = Path(proc_dir) / f"{stem}.pt"
    if not graph_src.is_file():
        return False
    bundle_graph = Path(transfer_dir) / stem / "graph.pt"
    if not bundle_graph.is_file():
        return True
    return _mtime(graph_src) > _mtime(bundle_graph)


def stem_needs_mesh_pack(
    stem: str,
    *,
    raw_dir: Path,
    transfer_dir: Path,
) -> bool:
    """True when a mesh bundle is missing or older than the anchor ``.nas``/``.msh``."""
    raw_dir = Path(raw_dir)
    nas = raw_dir / f"{stem}.nas"
    msh = raw_dir / f"{stem}.msh"
    if not nas.is_file() and not msh.is_file():
        return False
    bundle = Path(transfer_dir) / stem
    bundle_nas = bundle / "mesh.nas"
    bundle_msh = bundle / "mesh.msh"
    if nas.is_file():
        if not bundle_nas.is_file():
            return True
        return _mtime(nas) > _mtime(bundle_nas)
    if not bundle_msh.is_file():
        return True
    return _mtime(msh) > _mtime(bundle_msh)


def filter_stems_for_pack(
    stems: Iterable[str],
    *,
    proc_dir: Path,
    transfer_dir: Path,
    only_new: bool,
    mesh_only: bool = False,
    raw_dir: Path | None = None,
) -> list[str]:
    ordered = list(stems)
    if not only_new:
        return ordered
    if mesh_only:
        if raw_dir is None:
            raise ValueError("raw_dir is required for mesh-only --only-new filtering")
        return [
            s
            for s in ordered
            if stem_needs_mesh_pack(s, raw_dir=raw_dir, transfer_dir=transfer_dir)
        ]
    return [s for s in ordered if stem_needs_pack(s, proc_dir=proc_dir, transfer_dir=transfer_dir)]


def bundle_needs_install(
    bundle_dir: Path,
    *,
    root: Path | None = None,
    force: bool = False,
    only_new: bool = False,
) -> bool:
    """Whether an incoming bundle should be installed on this machine."""
    if force or not only_new:
        return True
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stem = str(manifest.get("stem") or bundle_dir.name)
    if bool(manifest.get("mesh_only")):
        local_mesh = (Path(root) if root is not None else get_project_root()) / (
            f"data/raw/biochem_anchors/{stem}.nas"
        )
        incoming_mesh = bundle_dir / "mesh.nas"
        if not incoming_mesh.is_file():
            incoming_mesh = bundle_dir / "mesh.msh"
        if not local_mesh.is_file():
            local_mesh = (Path(root) if root is not None else get_project_root()) / (
                f"data/raw/biochem_anchors/{stem}.msh"
            )
        if not local_mesh.is_file():
            return True
        if not incoming_mesh.is_file():
            return False
        return _mtime(incoming_mesh) > _mtime(local_mesh)
    local_graph = canonical_graph_dest(stem, root=root)
    incoming_graph = bundle_dir / "graph.pt"
    if not local_graph.is_file():
        return True
    if not incoming_graph.is_file():
        return False
    return _mtime(incoming_graph) > _mtime(local_graph)


def select_bundle_names_for_zip(
    *,
    transfer_dir: Path,
    dest_zip: Path | None = None,
    only_new: bool = False,
    stems: Iterable[str] | None = None,
) -> list[str]:
    """Bundle folder names to include in a Drive zip."""
    base = Path(transfer_dir)
    bundles = list_transfer_bundles(transfer_dir=base)
    if stems is not None:
        want = set(stems)
        bundles = [b for b in bundles if b.name in want]
    if not only_new:
        return [b.name for b in bundles]
    archive = Path(dest_zip) if dest_zip is not None else base.with_suffix(".zip")
    if not archive.is_file():
        return [b.name for b in bundles]
    zip_mtime = _mtime(archive)
    return [
        b.name
        for b in bundles
        if (b / "graph.pt").is_file() and _mtime(b / "graph.pt") > zip_mtime
    ]


def extract_transfer_dir(*, root: Path | None = None) -> Path:
    """``data/extract_transfer`` — the single folder to copy between machines."""
    if root is None:
        return data_root() / "extract_transfer"
    return Path(root) / "data" / "extract_transfer"


def extract_transfer_zip_path(*, root: Path | None = None, transfer_dir: Path | None = None) -> Path:
    """``data/extract_transfer.zip`` next to the transfer folder."""
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    return base.with_suffix(".zip")


def default_downloads_dir() -> Path:
    """User Downloads folder (Windows ``~/Downloads``, with OneDrive fallback)."""
    candidates: list[Path] = [Path.home() / "Downloads"]
    userprofile = None
    if userprofile:
        candidates.append(Path(userprofile) / "Downloads")
    onedrive = os.environ.get("OneDrive")
    if onedrive:
        candidates.append(Path(onedrive) / "Downloads")
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


@dataclass(frozen=True)
class IncomingTransfer:
    """Resolved install source: a folder of stem bundles, or a zip to unpack."""

    transfer_dir: Path | None
    zip_path: Path | None
    label: str


def _is_bundle_dir(path: Path) -> bool:
    return path.is_dir() and (path / MANIFEST_NAME).is_file()


def normalize_bundle_root(path: Path) -> Path:
    """Unwrap ``extract_transfer/`` or a single nested folder down to the stem parent."""
    path = Path(path)
    if not path.is_dir():
        return path
    if list_transfer_bundles(transfer_dir=path):
        return path
    nested = path / "extract_transfer"
    if nested.is_dir() and list_transfer_bundles(transfer_dir=nested):
        return nested
    subdirs = [p for p in sorted(path.iterdir()) if p.is_dir() and p.name not in {"__MACOSX"}]
    hits = [p for p in subdirs if list_transfer_bundles(transfer_dir=p)]
    if len(hits) == 1:
        return hits[0]
    if len(subdirs) == 1:
        return normalize_bundle_root(subdirs[0])
    return path


def _newest(paths: Iterable[Path]) -> Path:
    ranked = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return ranked[0]


def discover_downloads_transfer(downloads_dir: Path) -> IncomingTransfer | None:
    """Find one transfer folder or ``extract_transfer*.zip`` in Downloads."""
    downloads_dir = Path(downloads_dir)
    if not downloads_dir.is_dir():
        return None

    named = downloads_dir / "extract_transfer"
    if named.is_dir():
        root = normalize_bundle_root(named)
        if list_transfer_bundles(transfer_dir=root):
            return IncomingTransfer(root, None, str(root))

    wrappers: list[Path] = []
    direct_bundles: list[Path] = []
    for child in downloads_dir.iterdir():
        if not child.is_dir() or child.name in {"__MACOSX"}:
            continue
        if list_transfer_bundles(transfer_dir=child) or list_transfer_bundles(
            transfer_dir=normalize_bundle_root(child)
        ):
            wrappers.append(normalize_bundle_root(child))
        elif _is_bundle_dir(child):
            direct_bundles.append(child)

    if wrappers:
        chosen = _newest(wrappers)
        return IncomingTransfer(chosen, None, str(chosen))
    if direct_bundles:
        return IncomingTransfer(downloads_dir, None, str(downloads_dir))

    zips = [
        p
        for p in downloads_dir.glob("extract_transfer*.zip")
        if p.is_file()
    ]
    if not zips:
        zips = [p for p in downloads_dir.glob("*extract_transfer*.zip") if p.is_file()]
    if zips:
        archive = _newest(zips)
        return IncomingTransfer(None, archive, str(archive))
    return None


def resolve_incoming_transfer(
    *,
    transfer_dir: Path | None = None,
    downloads_dir: Path | None = None,
    data_transfer_dir: Path | None = None,
) -> IncomingTransfer | None:
    """Prefer ``--transfer-dir``, then Downloads, then ``data/extract_transfer``."""
    if transfer_dir is not None:
        path = Path(transfer_dir)
        if path.is_file() and path.suffix.lower() == ".zip":
            return IncomingTransfer(None, path, str(path))
        if path.is_dir():
            root = normalize_bundle_root(path)
            return IncomingTransfer(root, None, str(root))
        return None

    found = discover_downloads_transfer(downloads_dir or default_downloads_dir())
    if found is not None:
        return found

    data_dir = Path(data_transfer_dir) if data_transfer_dir is not None else extract_transfer_dir()
    if list_transfer_bundles(transfer_dir=data_dir):
        return IncomingTransfer(data_dir, None, str(data_dir))
    archive = data_dir.with_suffix(".zip")
    if archive.is_file():
        return IncomingTransfer(None, archive, str(archive))
    return None


def unpack_incoming_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Unpack a Drive zip and return the folder that contains stem bundles."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), dest_dir)
    return normalize_bundle_root(dest_dir)


def bundle_dir_for_stem(stem: str, *, root: Path | None = None) -> Path:
    ref = parse_biochem_extract_stem(stem)
    canonical = ref.stem if ref is not None else stem
    return extract_transfer_dir(root=root) / canonical


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def stage_extract_transfer_bundle(
    stem: str,
    *,
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path,
    kine_dir: Path | None = None,
    root: Path | None = None,
    lite: bool = True,
    mesh_only: bool = False,
) -> Path | None:
    """Mirror one extracted stem into ``data/extract_transfer/<stem>/``. Returns bundle dir or None.

    ``lite=True`` (default) is the Drive pack: graph + mesh sidecar + ``.nas`` + wound.txt,
    no ``.mph`` / domain txt / kine.

    ``mesh_only=True`` packs only anchor meshes (``.nas``/``.msh`` + sidecar) for FEM eval.
    """
    ref = parse_biochem_extract_stem(stem)
    canonical = ref.stem if ref is not None else stem
    graph_src = Path(proc_dir) / f"{canonical}.pt"
    nas_src = Path(raw_dir) / f"{canonical}.nas"
    msh_src = Path(raw_dir) / f"{canonical}.msh"
    if mesh_only:
        if not nas_src.is_file() and not msh_src.is_file():
            return None
    elif not graph_src.is_file():
        return None

    bundle = bundle_dir_for_stem(canonical, root=root)
    bundle.mkdir(parents=True, exist_ok=True)

    sources = {
        "graph.pt": graph_src,
        "graph_metadata.json": Path(proc_dir) / f"{canonical}_metadata.json",
        "mesh.msh": msh_src,
        "mesh.nas": nas_src,
        "mesh.json": Path(raw_dir) / f"{canonical}.json",
        "domain.txt": Path(label_dir) / f"{canonical}.txt",
        "inlet.txt": Path(label_dir) / f"{canonical}_inlet.txt",
        "outlet.txt": Path(label_dir) / f"{canonical}_outlet.txt",
        "wall.txt": Path(label_dir) / f"{canonical}_wall.txt",
        "wound.txt": Path(label_dir) / f"{canonical}_wound.txt",
        "kine.pt": (Path(kine_dir) / f"{canonical}.pt") if kine_dir is not None else None,
    }
    if mesh_only:
        want = set(_MESH_ONLY_NAMES)
    elif lite:
        want = set(_LITE_NAMES)
    else:
        want = set(sources)
    packed: list[str] = []
    for name, src in sources.items():
        if src is None or name not in want:
            continue
        if _copy_if_exists(src, bundle / name):
            packed.append(name)

    keep = set(packed) | {MANIFEST_NAME}
    for leftover in bundle.iterdir():
        if leftover.is_file() and leftover.name not in keep:
            leftover.unlink()

    files = {
        name: rel.format(stem=canonical)
        for name, rel, _required in _BUNDLE_KEYS
        if name in packed
    }
    manifest = {
        "stem": canonical,
        "variant": ref.variant if ref is not None else "unknown",
        "lite": lite and not mesh_only,
        "mesh_only": mesh_only,
        "files": files,
    }
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return bundle


def stems_from_raw_meshes(raw_dir: Path) -> list[str]:
    """Anchor stems with a ``.nas`` or ``.msh`` under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    stems: set[str] = set()
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in {".nas", ".msh"}:
            stems.add(path.stem)
    return sorted(stems)


def list_transfer_bundles(*, root: Path | None = None, transfer_dir: Path | None = None) -> list[Path]:
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / MANIFEST_NAME).is_file():
            out.append(child)
    return out


def install_extract_transfer_bundle(
    bundle_dir: Path,
    *,
    root: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Copy one transfer folder into canonical ``data/`` paths. Returns ``{bundle_name: dest}``."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {bundle_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stem = str(manifest.get("stem") or bundle_dir.name)
    mesh_only = bool(manifest.get("mesh_only"))
    repo = Path(root) if root is not None else get_project_root()
    written: dict[str, str] = {}
    files = manifest.get("files") or {}
    for name, rel, required in _BUNDLE_KEYS:
        src = bundle_dir / name
        dest_rel = files.get(name) or rel.format(stem=stem)
        dest = repo / dest_rel
        if not src.is_file():
            if required and not (mesh_only and name == "graph.pt"):
                raise FileNotFoundError(f"{bundle_dir.name}: missing required {name}")
            continue
        if dest.is_file() and not force:
            written[name] = f"skip {dest_rel}"
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written[name] = dest_rel
    return written


def install_all_extract_transfer_bundles(
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
    stems: Iterable[str] | None = None,
    force: bool = False,
    only_new: bool = False,
) -> list[tuple[str, dict[str, str]]]:
    """Install every (or selected) bundle under the transfer folder."""
    want: set[str] | None = None
    if stems is not None:
        want = set()
        for raw in stems:
            ref = parse_biochem_extract_stem(raw)
            want.add(ref.stem if ref is not None else raw)
    results: list[tuple[str, dict[str, str]]] = []
    for bundle in list_transfer_bundles(root=root, transfer_dir=transfer_dir):
        if want is not None and bundle.name not in want:
            continue
        if not bundle_needs_install(bundle, root=root, force=force, only_new=only_new):
            manifest_path = bundle / MANIFEST_NAME
            mesh_only = False
            if manifest_path.is_file():
                mesh_only = bool(json.loads(manifest_path.read_text(encoding="utf-8")).get("mesh_only"))
            skip_key = "mesh.nas" if mesh_only else "graph.pt"
            results.append((bundle.name, {skip_key: "skip already installed"}))
            continue
        results.append(
            (
                bundle.name,
                install_extract_transfer_bundle(bundle, root=root, force=force),
            )
        )
    return results


def install_incoming_extract_transfer(
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
    downloads_dir: Path | None = None,
    data_transfer_dir: Path | None = None,
    stems: Iterable[str] | None = None,
    force: bool = False,
    only_new: bool = False,
) -> tuple[IncomingTransfer | None, list[tuple[str, dict[str, str]]]]:
    """Discover a Downloads folder/zip (or explicit path), then install stem bundles."""
    incoming = resolve_incoming_transfer(
        transfer_dir=transfer_dir,
        downloads_dir=downloads_dir,
        data_transfer_dir=data_transfer_dir,
    )
    if incoming is None:
        return None, []
    if incoming.zip_path is not None:
        with tempfile.TemporaryDirectory(prefix="extract_transfer_") as tmp:
            bundle_root = unpack_incoming_zip(incoming.zip_path, Path(tmp))
            results = install_all_extract_transfer_bundles(
                root=root,
                transfer_dir=bundle_root,
                stems=stems,
                force=force,
                only_new=only_new,
            )
            return incoming, results
    results = install_all_extract_transfer_bundles(
        root=root,
        transfer_dir=incoming.transfer_dir,
        stems=stems,
        force=force,
        only_new=only_new,
    )
    return incoming, results


def zip_extract_transfer_dir(
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
    dest_zip: Path | None = None,
    bundle_names: Iterable[str] | None = None,
) -> Path:
    """Write one zip of ``data/extract_transfer`` (or a subset) for a single Drive upload."""
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    if not base.is_dir():
        raise FileNotFoundError(f"No transfer folder at {base}")
    if dest_zip is None:
        dest_zip = base.with_suffix(".zip")
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists():
        dest_zip.unlink()
    names = sorted(set(bundle_names)) if bundle_names is not None else None
    if names is None:
        archive = shutil.make_archive(
            str(dest_zip.with_suffix("")),
            "zip",
            root_dir=base.parent,
            base_dir=base.name,
        )
        return Path(archive)
    if not names:
        raise FileNotFoundError(f"No transfer bundles selected under {base}")
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            bundle = base / name
            if not bundle.is_dir():
                raise FileNotFoundError(f"Missing transfer bundle folder {bundle}")
            for path in bundle.rglob("*"):
                if path.is_file():
                    arcname = f"{base.name}/{path.relative_to(base).as_posix()}"
                    zf.write(path, arcname)
    return dest_zip


def unpack_extract_transfer_zip(
    zip_path: Path | None = None,
    *,
    root: Path | None = None,
    transfer_dir: Path | None = None,
) -> Path:
    """Unpack ``extract_transfer.zip`` so ``data/extract_transfer/<stem>/`` exists."""
    base = Path(transfer_dir) if transfer_dir is not None else extract_transfer_dir(root=root)
    archive = Path(zip_path) if zip_path is not None else base.with_suffix(".zip")
    if not archive.is_file():
        raise FileNotFoundError(f"No transfer zip at {archive}")
    shutil.unpack_archive(archive, base.parent)
    return base
