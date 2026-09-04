"""Paths and rheology conventions for Stage-A kinematics graphs."""

from __future__ import annotations

from pathlib import Path

import torch

from src.utils.paths import data_root, get_project_root

# Biochem COMSOL anchors always use Carreau physics; steady kine sidecars match.
BIOCHEM_ANCHOR_KINE_RHEOLOGY = "carreau"

#: Where the REPAIRED copy of each COMSOL anchor mesh lives (see `sync_geometry_from_deploy_pack`).
BIOCHEM_ANCHOR_GRAPH_DIR = "data/processed/graphs_biochem_anchors"


def kinematics_anchor_graph_dir(
    *,
    rheology: str | None = None,
    root: Path | None = None,
) -> Path:
    """Directory for steady ``KINE_*`` graphs extracted from biochem anchors."""
    r = (rheology or BIOCHEM_ANCHOR_KINE_RHEOLOGY).strip().lower()
    base = root if root is not None else get_project_root()
    return base / "data/processed/graphs_kinematics_anchors" / r


def resolve_kinematics_anchor_graph(stem: str, *, rheology: str | None = None) -> Path | None:
    """Return existing anchor kine graph path (prefers Carreau, falls back to legacy newtonian)."""
    stem = str(stem).strip()
    primary = kinematics_anchor_graph_dir(rheology=rheology) / f"{stem}.pt"
    if primary.is_file():
        return primary
    legacy = kinematics_anchor_graph_dir(rheology="newtonian") / f"{stem}.pt"
    if legacy.is_file():
        return legacy
    return None


def kinematics_training_graph_dir(*, rheology: str = "carreau", root: Path | None = None) -> Path:
    dr = data_root() if root is None else root
    return dr / "processed/graphs_kinematics" / rheology.strip().lower()


def kinematics_graph_rheology_dir(rheology: str, *, root: Path | None = None) -> Path:
    """Alias used by trainer / viz (``graphs_kinematics/<rheology>/``)."""
    return kinematics_training_graph_dir(rheology=rheology, root=root)


def iter_comsol_kine_anchor_paths(*, rheology: str | None = None) -> list[Path]:
    """Sorted ``comsol*.pt`` steady kine sidecars (Carreau by default)."""
    anchor_dir = kinematics_anchor_graph_dir(rheology=rheology)
    if not anchor_dir.is_dir():
        return []
    return sorted(anchor_dir.glob("comsol*.pt"))


def load_comsol_kine_anchor_graphs(
    *,
    rheology: str | None = None,
    attach_geometry: bool = True,
    sync_geometry: bool = True,
) -> list:
    """Load comsol COMSOL steady kine graphs for Stage-A finetune / eval."""
    from src.utils.channel_schema import assert_graph_schema, infer_missing_schema
    from src.utils.kinematics_geometry import attach_geometry_metadata
    from src.config import VesselConfig
    from src.utils.channel_schema import KINE_Y_SCHEMA

    paths = iter_comsol_kine_anchor_paths(rheology=rheology)
    if not paths:
        return []
    cfg = VesselConfig(phase="biochem_anchors")
    out = []
    for f in paths:
        data = torch.load(f, map_location="cpu", weights_only=False)
        data = infer_missing_schema(data, phase_hint="kinematics")
        assert_graph_schema(data, expected_y_schema=(KINE_Y_SCHEMA,))
        data.graph_stem = f.stem
        data.is_comsol_anchor = True
        if attach_geometry:
            attach_geometry_metadata(data, mesh_input_dir=cfg.mesh_input_dir, stem=f.stem)
        if sync_geometry:
            sync_geometry_from_deploy_pack(data)
        out.append(data)
    return out


#: Channels that describe the MESH, and must therefore be identical in training and deployment.
#: The prior block (11-14) is deliberately excluded: it is rewritten wholesale by
#: ``legal_priors.apply_prior_source`` and its stored values are the s17 Z2 leak.
GEOMETRY_SYNC_CHANNELS = (4, 5, 6, 7, 8, 9, 15, 16, 17)


def sync_geometry_from_deploy_pack(data, *, deploy_dir: str | Path | None = None) -> bool:
    """Overwrite a training anchor's mesh channels with the deploy pack's repaired values.

    RGP_DEQ_REPAIR_PLAN.md B14.  ``graphs_kinematics_anchors/carreau`` and
    ``graphs_biochem_anchors`` hold the SAME mesh for the same COMSOL anchor -- identical
    ``edge_index``, ``mask_wall``, node positions and ``sdf`` -- but only the biochem copy ever
    received ``repair_pack_wall_normals`` and the width fix.  Measured over all 43 shared
    COMSOL anchors, per-channel rel-L2 between the two copies:

    ```
    [4,5]  wall_normal    0.178 / 0.199        [15]    width_nd   0.149
    [6-9]  node_type_0..3 1.000 (all four)     [16,17] width_d1/2 8.68 / 9.03
    [14]   wss_prior      1.000
    ```

    A rel-L2 of exactly 1.000 means the training copy is **identically zero** where deployment
    is not: Stage-A has never seen a non-zero ``node_type``, and the encoder consumes all four
    channels.  ``wall_normal`` is worse than it looks -- ``mod_adv``/``mod_rheo``/``mod_curve``,
    the GAT's three attention biases, are built entirely from it, so the model trains with one
    attention geometry and deploys with another.

    Only mesh channels are copied.  ``pack_repair``'s own warning applies and is respected: a
    wholesale rebuild does NOT reproduce these packs (the cohort was written by more than one
    extractor revision and they disagree about the prior block), so this writes the affected
    columns and nothing else, and never touches disk.

    Returns ``True`` when a sync happened.  A COMSOL anchor with no deploy pack, or a node-count
    mismatch, is left alone -- silently using a mismatched mesh would be the original bug.
    """
    import torch as _t

    stem = str(getattr(data, "graph_stem", "") or "")
    if not stem:
        return False
    base = Path(deploy_dir) if deploy_dir is not None else get_project_root() / BIOCHEM_ANCHOR_GRAPH_DIR
    src = base / f"{stem}.pt"
    if not src.is_file():
        return False
    ref = _t.load(src, map_location="cpu", weights_only=False)
    if int(ref.x.shape[0]) != int(data.x.shape[0]) or int(ref.x.shape[1]) < 18:
        return False
    x = data.x.clone()
    for c in GEOMETRY_SYNC_CHANNELS:
        x[:, c] = ref.x[:, c].to(x.dtype)
    data.x = x
    data.geometry_synced = True
    return True
