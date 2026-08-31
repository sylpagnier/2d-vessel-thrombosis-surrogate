"""
Interactive Kinematics/2 data pipeline: vessel meshes, optional COMSOL anchors, PyG graphs.

Runs the same logical steps as ``vessel_generator``, ``anchor_generator``, and ``mesh_to_graph``.
Meshes are generated in **SI** (``unit=m``) via the same ``VesselGenerator`` geometry sampler
biochem anchors use (they use ``unit=cm`` for phase2 CGS).  COMSOL import is Gmsh NAS only --
no geometry scaling or remeshing in the template.

**Interactive mode asks every question first** (per phase), then runs Gmsh / COMSOL / mesh-to-graph
with **no further prompts** so you can leave the machine unattended after the planning phase.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    default_level_mix,
    normalize_pathology_mode,
    parse_level_mix,
    prompt_pathology_mode,
    summarize_vessel_mesh_inventory,
    _prompt_int_choice as _vg_prompt_int_choice,
    _prompt_write_mode_vessel as _vg_prompt_write_mode_vessel,
)


def _safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def _purge_anchor_npz_outputs(target_output_dir: Path) -> int:
    """Delete existing anchor ``vessel_*.npz`` files before a full cohort refresh."""
    removed = 0
    if not target_output_dir.exists():
        return removed
    for npz_path in target_output_dir.glob("vessel_*.npz"):
        try:
            npz_path.unlink()
            removed += 1
        except OSError as exc:
            print(f"Warning: could not remove stale anchor file {npz_path}: {exc}")
    return removed


def _prompt_anchor_write_mode() -> bool:
    """Return True if overwriting existing .npz, False for add-only."""
    while True:
        raw = input("Anchor .npz write mode [1=add new only / 2=overwrite existing] [1]: ").strip()
        if raw in ("", "1"):
            return False
        if raw == "2":
            return True
        print("  Enter 1 or 2.")


def _prompt_nonnegative_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return int(default)
        try:
            v = int(raw)
            if v < 0:
                print("Enter a non-negative integer.")
                continue
            return v
        except ValueError:
            print("Invalid input. Enter an integer value.")


def _rheology_from_n(choice_n: int) -> str:
    if choice_n == 1:
        return "newtonian"
    if choice_n == 2:
        return "carreau"
    raise ValueError(f"Unsupported rheology choice: {choice_n}")


def _final_subdir_for_rheology(rheology: str) -> str:
    return str(rheology).strip().lower()


@dataclass
class PhaseInteractivePlan:
    """All interactive choices for one rheology pass (collected before any long-running step)."""

    anchor_target: int
    run_vessel: bool
    level: Optional[int]
    level_mix: Optional[dict[int, int]]
    pathology_mode: Optional[str]
    overwrite: Optional[bool]
    n_vessels: Optional[int]
    seed: Optional[int]
    num_workers: Optional[int]
    chunk_size: Optional[int]
    run_anchors: bool
    allow_overwrite_anchor: bool
    anchor_max_json_scan: Optional[int]
    anchor_shuffle: bool
    anchor_shuffle_seed: Optional[int]
    # When anchor_target == 0 and run_anchors: how many new CFD samples to aim for.
    anchor_manual_max_new: Optional[int]
    run_mesh: bool


def run_interactive_pipeline() -> None:
    print("\n=== Kinematics/2 data generation pipeline (vessel, anchors, graphs) ===\n")

    rheology_scope = _vg_prompt_int_choice(
        "Run Kinematics datagen for (1 = Newtonian Primer, 2 = Carreau Target, 3 = Both sequentially)",
        (1, 2, 3),
    )
    rheology_sequence = (1, 2) if rheology_scope == 3 else (rheology_scope,)

    if len(rheology_sequence) == 2:
        print(
            "\nBoth rheology passes run one after the other (newtonian then carreau), "
            "writing to separate CFD and graph subfolders.\n"
        )

    print(
        "\nEach rheology plan asks vessel generation -> COMSOL anchors (optional) first; "
        "mesh-to-graph runs automatically after. No prompts during execution.\n"
    )

    print(
        "\n--- Planning: answer all questions now; the run after this has **no further prompts** ---\n"
    )
    plans: dict[int, PhaseInteractivePlan] = {}
    for rheology_n in rheology_sequence:
        plans[rheology_n] = _prompt_phase_interactive_plan(rheology_n)

    print(
        f"\n{'=' * 60}\n"
        "  All prompts complete — unattended run starting.\n"
        f"{'=' * 60}\n"
    )

    for rheology_n in rheology_sequence:
        _execute_phase_interactive_plan(rheology_n, plans[rheology_n].anchor_target, plans[rheology_n])

    print("\n=== Pipeline finished ===\n")


def _prompt_phase_interactive_plan(rheology_n: int) -> PhaseInteractivePlan:
    rheology = _rheology_from_n(rheology_n)
    print(f"\n{'=' * 60}\n  PLAN — {rheology.upper()} (independent cohort)\n{'=' * 60}\n")

    # ==========================================================
    # 1. VESSELS
    # ==========================================================
    vg = VesselGenerator(phase="kinematics")
    inv = summarize_vessel_mesh_inventory(vg.output_dir)
    n_on_disk = int(inv["count"])

    print("\n--- Vessel mesh inventory ---")
    print(f"  Meshes currently on disk: {n_on_disk}\n")

    default_n = 50 if n_on_disk > 0 else 500
    n_vessels = _prompt_nonnegative_int("How many vessels to generate? (0 = skip)", default=default_n)
    run_vessel = n_vessels > 0

    level: Optional[int] = None
    level_mix: Optional[dict[int, int]] = None
    pathology_mode: Optional[str] = None
    overwrite: Optional[bool] = None
    seed: Optional[int] = None
    num_workers: Optional[int] = None
    chunk_size: Optional[int] = None

    if run_vessel:
        cohort_mode = _vg_prompt_int_choice(
            "Vessel cohort [1=single level / 2=mixed, `default_level_mix` (now all L2)]",
            (1, 2),
        )
        if cohort_mode == 2:
            level_mix = default_level_mix(n_vessels)
            print(
                f"  Mixed cohort: L0={level_mix[0]}, L1={level_mix[1]}, L2={level_mix[2]} "
                f"(high-thrombus geometries for COMSOL + kinematics training).\n"
            )
        else:
            level = _vg_prompt_int_choice(
                "Geometry level [0=straight / 1=curved / 2=high-thrombus]",
                (0, 1, 2),
            )
            if level == 2:
                print(
                    "  Level 2: pro-thrombotic shapes (sharp turns, aneurysm/stenosis) "
                    "for high-μ biochem-style CFD anchors.\n"
                )
        pathology_mode = prompt_pathology_mode()
        if n_on_disk == 0:
            overwrite = True
            print("  No meshes on disk — starting indices at 0 (overwrite).\n")
        else:
            overwrite = _vg_prompt_write_mode_vessel()

    # ==========================================================
    # 2. ANCHORS
    # ==========================================================
    from src.data_gen.lib.anchor_generator import (
        AnchorGenerator,
        summarize_anchor_inventory,
    )

    print(f"\n--- {rheology.upper()} COMSOL anchors ---")

    # Match anchor write mode to vessel overwrite (new cohort replaces old meshes + anchors).
    if run_vessel and overwrite is True:
        allow_overwrite_anchor = True
        print(
            "  Vessel generation overwrites mesh indices — using anchor overwrite (existing .npz can be replaced).\n"
        )
    else:
        allow_overwrite_anchor = _prompt_anchor_write_mode()

    anchor_output_dir = vg.cfg.output_dir / _final_subdir_for_rheology(rheology)
    gen = AnchorGenerator(phase="kinematics", output_dir=anchor_output_dir, rheology=rheology)
    anchor_inv = summarize_anchor_inventory(gen.mesh_dir, gen.target_output_dir())
    have_npz = int(anchor_inv["existing_npz"])
    ready_add = int(anchor_inv["candidate_pool_ready"])
    ready_all = int(anchor_inv["candidate_pool_including_npz"])

    if not allow_overwrite_anchor:
        print(f"  Anchors already generated: {have_npz}\n")
        pool = ready_add
    else:
        print("  Overwriting existing anchors.\n")
        pool = ready_all

    if run_vessel and overwrite is True:
        print("  Mesh overwrite selected -> anchor overwrite is locked ON.\n")
    default_anchors = (pool // 2) if pool > 0 else 0
    anchor_manual_max_new = _prompt_nonnegative_int(
        "How many anchors to generate? (0 = skip)", default=default_anchors
    )
    run_anchors = anchor_manual_max_new > 0

    # No JSON scan cap; shuffle candidates with a random seed (interactive kinematics defaults).
    anchor_max_json_scan: Optional[int] = None
    anchor_shuffle = True
    anchor_shuffle_seed: Optional[int] = None
    # ==========================================================
    # 3. MESH TO GRAPH (Automatic)
    # ==========================================================
    run_mesh = True

    return PhaseInteractivePlan(
        anchor_target=0,  # Hardcoded to 0 to trigger the manual count in execution
        run_vessel=run_vessel,
        level=level,
        level_mix=level_mix,
        pathology_mode=pathology_mode,
        overwrite=overwrite,
        n_vessels=n_vessels if run_vessel else None,
        seed=seed,
        num_workers=num_workers,
        chunk_size=chunk_size,
        run_anchors=run_anchors,
        allow_overwrite_anchor=allow_overwrite_anchor,
        anchor_max_json_scan=anchor_max_json_scan,
        anchor_shuffle=anchor_shuffle,
        anchor_shuffle_seed=anchor_shuffle_seed,
        anchor_manual_max_new=anchor_manual_max_new,
        run_mesh=run_mesh,
    )


def _execute_phase_interactive_plan(
    rheology_n: int, anchor_target: int, plan: PhaseInteractivePlan
) -> None:
    rheology = _rheology_from_n(rheology_n)
    print(f"\n{'=' * 60}\n  RUN — {rheology.upper()}\n{'=' * 60}\n")

    if plan.run_vessel:
        assert plan.overwrite is not None and plan.n_vessels is not None
        assert plan.level is not None or plan.level_mix is not None
        vg = VesselGenerator(phase="kinematics")
        start_idx = 0 if plan.overwrite else None
        print("\n--- Running vessel generator ---\n")
        vg.run_pipeline(
            n=plan.n_vessels,
            level=0 if plan.level is None else plan.level,
            level_mix=plan.level_mix,
            seed=plan.seed,
            num_workers=plan.num_workers,
            chunk_size=plan.chunk_size,
            start_idx=start_idx,
            pathology_mode=plan.pathology_mode,
        )

    if plan.run_anchors:
        from src.data_gen.lib.anchor_generator import (
            AnchorGenerator,
            summarize_anchor_inventory,
        )

        anchor_output_dir = VesselGenerator(phase="kinematics").cfg.output_dir / _final_subdir_for_rheology(
            rheology
        )
        gen = AnchorGenerator(phase="kinematics", output_dir=anchor_output_dir, rheology=rheology)
        force_full_anchor_refresh = bool(plan.run_vessel and (plan.overwrite is True))
        if force_full_anchor_refresh:
            removed = _purge_anchor_npz_outputs(gen.target_output_dir())
            print(
                f"  Mesh overwrite detected: removed {removed} existing anchor .npz "
                f"from {gen.target_output_dir()} before regeneration.\n"
            )
        inv = summarize_anchor_inventory(gen.mesh_dir, gen.target_output_dir())
        ready_add = int(inv["candidate_pool_ready"])
        ready_all = int(inv["candidate_pool_including_npz"])
        remaining = int(inv["pending_missing_npz"])
        total_v = int(inv["mesh_json_with_valid_nas"])

        pool = ready_all if (plan.allow_overwrite_anchor and not force_full_anchor_refresh) else ready_add
        _safe_print("\n--- Anchor CFD inventory (at run time) ---")
        _safe_print(f"  CFD-ready pool: {pool} (add-only pool: {ready_add})\n")

        if pool == 0:
            if plan.allow_overwrite_anchor:
                _safe_print("No meshes are CFD-ready — skipping anchor batch.\n")
            else:
                msg = "Nothing to add (need .json + non-empty .nas + .msh, and no .npz yet)."
                if remaining > 0:
                    msg += " Some meshes lack .msh — export meshes for those vessels first."
                elif total_v == 0:
                    msg = "No vessel meshes found in the mesh directory."
                _safe_print(msg + "\nSkipping anchor batch.\n")
            max_new = 0
        elif anchor_target > 0:
            asked = min(anchor_target, pool)
            if anchor_target > pool:
                _safe_print(
                    f"  Only {pool} CFD-ready mesh(es); running at most {asked} anchors "
                    f"(target was {anchor_target}).\n"
                )
            else:
                _safe_print(
                    f"  Running up to {asked} anchor CFD sample(s) toward target {anchor_target}.\n"
                )
            max_new = asked
        else:
            assert plan.anchor_manual_max_new is not None
            asked = plan.anchor_manual_max_new
            if asked == 0:
                _safe_print("Skipping anchor batch (0 requested).\n")
                max_new = 0
            else:
                max_new = min(asked, pool)
                if asked > pool:
                    _safe_print(f"Requested {asked} but only {pool} mesh(es) match; running {max_new}.\n")

        if pool > 0 and max_new > 0:
            _safe_print("\n--- Running anchor CFD ---\n")
            with gen:
                gen.run_batch(
                    max_new=max_new,
                    max_json_to_scan=plan.anchor_max_json_scan,
                    shuffle_candidates=plan.anchor_shuffle,
                    shuffle_seed=plan.anchor_shuffle_seed,
                    allow_overwrite=plan.allow_overwrite_anchor,
                    continuation_steps=None,
                )
                pending = _repair_unsolved_anchors(
                    gen,
                    rounds=len(ANCHOR_REPAIR_SCHEDULE),
                    allow_overwrite=plan.allow_overwrite_anchor,
                    max_json_to_scan=plan.anchor_max_json_scan,
                )
            _report_cohort_health(gen, pending)

    if plan.run_mesh:
        _safe_print(f"\n--- Mesh to graph: meshes -> target graphs ({rheology}) ---")
        try:
            from src.data_gen.lib.mesh_to_graph import MeshToGraph

            # Process target graphs only; no intermediate continuation sweeps.
            final_subdir = _final_subdir_for_rheology(rheology)
            processor = MeshToGraph(phase="kinematics", n_subdir=final_subdir, rheology=rheology)
            processor.run()
        except OSError as exc:
            _safe_print(f"\nMesh-to-graph aborted (console I/O): {exc}\n")
            raise
        except Exception as exc:
            _safe_print(f"\nMesh-to-graph failed: {exc}\n")
            raise


def _parse_batch_args(argv: list[str]) -> Optional[argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="Kinematics/2 data pipeline: vessel meshes, optional COMSOL anchors, PyG graphs.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive mode (requires --rheology or --both-rheologies; optional anchor flags).",
    )
    p.add_argument(
        "--both-rheologies",
        action="store_true",
        help="Run newtonian then carreau sequentially (independent cohorts; use --seed-newtonian/--seed-carreau).",
    )
    p.add_argument("--rheology", choices=("newtonian", "carreau"), help="Rheology target; omit when using --both-rheologies")
    p.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Single geometry level (0=straight, 1=curved, 2=high-thrombus). Omit with --mixed-levels.",
    )
    p.add_argument(
        "--mixed-levels",
        action="store_true",
        help="Mixed cohort via `default_level_mix`, which is now ALL L2 because every "
             "biochem deploy patient is level 2; pass --level-mix for the old L0/L1 spread. "
             "Overrides --level.",
    )
    p.add_argument(
        "--level-mix",
        type=str,
        default=None,
        metavar="N0,N1,N2",
        help="Explicit per-level counts (must sum to -n); implies mixed cohort.",
    )
    p.add_argument(
        "--pathology-mix", default=None,
        help="Per-vessel mix, e.g. 'random:0.72,max_stenosis:0.18,max_aneurysm:0.10'. "
             "Weights are fractions (or exact counts summing to -n). One command instead of "
             "one run per mode; the assignment is shuffled so it does not correlate with the "
             "geometry-level schedule. Overrides --pathology-mode.")
    p.add_argument(
        "--pathology-mode",
        choices=("random", "max_stenosis", "max_aneurysm", "straight_max"),
        default="random",
        help=(
            "Pathology sampling: random (default), max_stenosis (~80%% occlusion), "
            "max_aneurysm (up to 3x inlet width), or straight_max (straight + max pathology)."
        ),
    )
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        metavar="N",
        help="Number of vessels to generate (both passes use this if per-rheology flags are omitted)",
    )
    p.add_argument(
        "--num-vessels-newtonian",
        type=int,
        default=None,
        metavar="N",
        help="With --both-rheologies: vessel count for newtonian pass (falls back to -n).",
    )
    p.add_argument(
        "--num-vessels-carreau",
        type=int,
        default=None,
        metavar="N",
        help="With --both-rheologies: vessel count for carreau pass (falls back to -n).",
    )
    p.add_argument("--overwrite", action="store_true",
                   help="Replace the cohort: vessel indices restart at 0. Required "
                        "(or --append) when the target is already populated.")
    p.add_argument("--append", action="store_true",
                   help="Add to an existing cohort: indices continue from the highest "
                        "on disk. Mutually exclusive with --overwrite.")
    p.add_argument(
        "--repair-rounds", type=int, default=4, metavar="N",
        help="Stages tried on a vessel COMSOL could not solve, in order (0 disables): one "
             "re-mesh of the SAME geometry, then re-draws of the same class at 0.70x / 0.50x / "
             "0.35x the original severity until one solves. Refinement alone recovered 2 of 39, "
             "and same-severity re-draws recovered 2 of 38. Default 4 (all stages).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (counts, mix, target dirs) and exit without writing.")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Gmsh RNG seed for vessel generation (single-rheology batch only; empty default = random).",
    )
    p.add_argument(
        "--seed-newtonian",
        type=int,
        default=None,
        metavar="INT",
        help="With --both-rheologies: Gmsh seed for newtonian pass (omit for random).",
    )
    p.add_argument(
        "--seed-carreau",
        type=int,
        default=None,
        metavar="INT",
        help="With --both-rheologies: Gmsh seed for carreau pass (omit for random).",
    )
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib mesh preview after vessel generation (default: skip; avoids blocking on plot windows).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--skip-vessel", action="store_true", help="Skip Gmsh vessel generation")
    p.add_argument("--skip-anchor", action="store_true", help="Skip COMSOL anchor step")
    p.add_argument("--skip-mesh", action="store_true", help="Skip mesh-to-graph conversion")
    p.add_argument(
        "--anchor-max-new",
        type=int,
        default=None,
        metavar="K",
        help="COMSOL: target new .npz per phase if phase-specific flags omitted (omit with --skip-anchor)",
    )
    p.add_argument(
        "--anchor-max-new-newtonian",
        type=int,
        default=None,
        metavar="K",
        help="With --both-rheologies: anchor target for newtonian pass (falls back to --anchor-max-new).",
    )
    p.add_argument(
        "--anchor-max-new-carreau",
        type=int,
        default=None,
        metavar="K",
        help="With --both-rheologies: anchor target for carreau pass (falls back to --anchor-max-new).",
    )
    p.add_argument(
        "--anchor-overwrite",
        action="store_true",
        help="COMSOL: allow replacing existing .npz",
    )
    p.add_argument("--anchor-max-json-scan", type=int, default=None)
    p.add_argument("--anchor-shuffle", action="store_true")
    p.add_argument("--anchor-shuffle-seed", type=int, default=None)
    p.add_argument(
        "--bend-sign-mode",
        choices=("down_only", "bidirectional"),
        default=None,
        help="Arc/hook bend: down_only=historical +x bend; bidirectional=random mirror (L1/L2).",
    )

    args = p.parse_args(argv)
    if not args.batch:
        return None
    if args.both_rheologies and args.rheology is not None:
        p.error("Do not pass --rheology with --both-rheologies")
    if args.both_rheologies and args.seed is not None:
        p.error("With --both-rheologies use --seed-newtonian and --seed-carreau (not --seed)")
    missing = []
    if not args.skip_vessel:
        if not args.both_rheologies and args.rheology is None:
            missing.append("--rheology or --both-rheologies")
        if args.level is None and not args.mixed_levels and args.level_mix is None:
            missing.append("--level, --mixed-levels, or --level-mix")
        if args.both_rheologies:
            ok_nv = args.num_vessels is not None or (
                args.num_vessels_newtonian is not None and args.num_vessels_carreau is not None
            )
            if not ok_nv:
                missing.append(
                    "-n / --num-vessels, or both --num-vessels-newtonian and --num-vessels-carreau"
                )
        elif args.num_vessels is None:
            missing.append("-n / --num-vessels")
    else:
        if not args.both_rheologies and args.rheology is None:
            missing.append("--rheology or --both-rheologies (needed for mesh step paths)")
    if missing:
        p.error(f"--batch mode missing: {', '.join(missing)}")
    if not args.skip_anchor:
        if not getattr(args, "both_rheologies", False):
            if args.anchor_max_new is None:
                p.error("--batch: specify --anchor-max-new or --skip-anchor")
        else:
            for key in ("newtonian", "carreau"):
                av = getattr(args, f"anchor_max_new_{key}", None)
                if av is None and args.anchor_max_new is None:
                    p.error(
                        "--both-rheologies: set --anchor-max-new, or both "
                        "--anchor-max-new-newtonian and --anchor-max-new-carreau"
                    )
    if getattr(args, "both_rheologies", False) and not args.skip_vessel:
        for key in ("newtonian", "carreau"):
            nv = getattr(args, f"num_vessels_{key}", None)
            if nv is None and args.num_vessels is None:
                p.error(
                    "--both-rheologies: set -n / --num-vessels, or both "
                    "--num-vessels-newtonian and --num-vessels-carreau"
                )
    return args


def _batch_num_vessels_for_rheology(rheology: str, args: argparse.Namespace) -> int:
    v = getattr(args, f"num_vessels_{rheology}", None)
    if v is not None:
        return int(v)
    assert args.num_vessels is not None
    return int(args.num_vessels)


def _batch_anchor_max_for_rheology(rheology: str, args: argparse.Namespace) -> int:
    v = getattr(args, f"anchor_max_new_{rheology}", None)
    if v is not None:
        return int(v)
    assert args.anchor_max_new is not None
    return int(args.anchor_max_new)


def _assert_write_intent_declared(args, rheology: str) -> None:
    """Refuse to generate into a populated cohort unless the caller said which they meant.

    `--overwrite` restarts vessel indices at 0; without it `start_idx` is None and the
    generator picks up from whatever is on disk.  Neither is wrong, but *neither is safe as a
    default*: a run intended as a 12-vessel smoke test replaced a 370-graph corpus and its
    meshes, and because `data/` is gitignored there was nothing to restore from.

    So the caller must say `--overwrite` (replace) or `--append` (add to), and only when the
    target is already populated.  A fresh directory needs neither.
    """
    from src.config import VesselConfig

    cfg = VesselConfig(phase="kinematics")
    graph_dir = cfg.graph_output_dir / rheology
    mesh_dir = cfg.mesh_input_dir
    n_graphs = len(list(graph_dir.glob("*.pt"))) if graph_dir.is_dir() else 0
    n_meshes = len(list(mesh_dir.glob("vessel_*.msh"))) if mesh_dir.is_dir() else 0
    if n_graphs == 0 and n_meshes == 0:
        return
    if args.overwrite or getattr(args, "append", False):
        mode = "OVERWRITE (indices restart at 0)" if args.overwrite else "APPEND"
        print(f"[i] target already holds {n_graphs} graphs / {n_meshes} meshes -> {mode}")
        return
    msg = [
        "",
        "REFUSING TO GENERATE: the target cohort is not empty.",
        f"  graphs : {n_graphs:>5}  in {graph_dir}",
        f"  meshes : {n_meshes:>5}  in {mesh_dir}",
        "",
        "Say which you mean:",
        "  --overwrite   replace the cohort (vessel indices restart at 0)",
        "  --append      add to it (indices continue from the highest on disk)",
        "",
        "`data/` is gitignored: an overwrite here is not recoverable.  If you want a",
        "scratch cohort, generate into a copy of the tree rather than the live one.",
        "",
    ]
    raise SystemExit(chr(10).join(msg))


def _run_batch_for_phase(
    rheology: str,
    args: argparse.Namespace,
    *,
    vessel_seed: Optional[int],
    num_vessels: Optional[int] = None,
    anchor_max_new: Optional[int] = None,
) -> None:
    force_full_anchor_refresh = bool(not args.skip_vessel and bool(args.overwrite))

    if not args.skip_vessel:
        _assert_write_intent_declared(args, rheology)
        assert num_vessels is not None
        level_mix = None
        level = int(args.level) if args.level is not None else 0
        if args.level_mix is not None:
            level_mix = parse_level_mix(args.level_mix, num_vessels)
        elif args.mixed_levels:
            level_mix = default_level_mix(num_vessels)
        level_label = (
            ", ".join(f"L{k}={v}" for k, v in sorted(level_mix.items()))
            if level_mix is not None
            else str(level)
        )
        vg = VesselGenerator(phase="kinematics")
        start_idx = 0 if args.overwrite else None
        if args.bend_sign_mode:
            os.environ["KINEMATICS_BEND_SIGN_MODE"] = str(args.bend_sign_mode)
        bend_label = os.environ.get("KINEMATICS_BEND_SIGN_MODE", "bidirectional")
        # A mix spec is passed through verbatim; `run_pipeline` expands it per vessel.
        pathology_mode = (args.pathology_mix.strip() if args.pathology_mix
                          else normalize_pathology_mode(args.pathology_mode))
        print(
            f"--- Vessel generation: rheology={rheology} levels={level_label} n={num_vessels} "
            f"seed={vessel_seed!r} bend_sign_mode={bend_label} "
            f"pathology={pathology_mode} ---\n"
        )
        if getattr(args, "dry_run", False):
            # Print the plan and stop. The cheapest way to confirm flags parse the way you
            # think before committing Gmsh + COMSOL time to 250 vessels.
            from src.config import VesselConfig as _VC

            _cfg = _VC(phase="kinematics")
            _mix = (args.pathology_mix.strip() if args.pathology_mix
                    else (args.pathology_mode or "random"))
            print("--- DRY RUN: nothing will be written ---")
            print(f"  rheology      {rheology}")
            print(f"  vessels       {num_vessels}")
            print(f"  levels        {level_label}")
            print(f"  pathology     {_mix}")
            print(f"  seed          {vessel_seed!r}")
            print(f"  meshes  ->    {_cfg.mesh_input_dir}")
            print(f"  graphs  ->    {_cfg.graph_output_dir / rheology}")
            print(f"  mode          {'OVERWRITE' if args.overwrite else ('APPEND' if getattr(args, 'append', False) else 'fresh')}")
            _rr = int(getattr(args, "repair_rounds", 2))
            print(f"  mesh          lc={_cfg.mesh_lc * 1000:.2f}mm x{_cfg.mesh_size_factor}, "
                  f">={_cfg.mesh_min_elems_across} elements across the throat")
            if _rr <= 0:
                print("  repair        DISABLED")
            else:
                _stages = ", ".join(
                    (f"remesh@{f}" if k == "mesh" else f"redraw@{sv:.2f}x severity")
                    for k, f, _, sv in ANCHOR_REPAIR_SCHEDULE[:_rr])
                print(f"  repair        {_rr} stage(s): {_stages}")
            if args.pathology_mix:
                import numpy as _np
                from collections import Counter as _C
                from src.data_gen.lib.vessel_generator import parse_pathology_mix as _ppm
                _modes = _ppm(args.pathology_mix, num_vessels, _np.random.default_rng(vessel_seed or 0))
                print(f"  mix expands   {dict(_C(_modes))}")
            return
        vg.run_pipeline(
            n=num_vessels,
            level=level,
            level_mix=level_mix,
            seed=vessel_seed,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            start_idx=start_idx,
            pathology_mode=pathology_mode,
        )
        if args.show_vessel_plot:
            saved_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in vg.output_dir.glob("vessel_*.msh")
            )[:9]
            if saved_indices:
                vg.visualize_saved(saved_indices)

    if not args.skip_anchor:
        assert anchor_max_new is not None
        from src.data_gen.lib.anchor_generator import AnchorGenerator, summarize_anchor_inventory

        anchor_output_dir = VesselGenerator(phase="kinematics").cfg.output_dir / _final_subdir_for_rheology(
            rheology
        )
        gen = AnchorGenerator(phase="kinematics", output_dir=anchor_output_dir, rheology=rheology)
        if force_full_anchor_refresh:
            removed = _purge_anchor_npz_outputs(gen.target_output_dir())
            inv = summarize_anchor_inventory(gen.mesh_dir, gen.target_output_dir())
            forced_pool = int(inv["candidate_pool_including_npz"])
            print(
                f"--- Anchor CFD: rheology={rheology} FORCE full overwrite after mesh overwrite "
                f"(removed={removed}, max_new={forced_pool}) ---\n"
            )
            effective_anchor_max_new = forced_pool
            effective_allow_overwrite = True
        else:
            print(f"--- Anchor CFD: rheology={rheology} max_new={anchor_max_new} ---\n")
            effective_anchor_max_new = anchor_max_new
            effective_allow_overwrite = bool(args.anchor_overwrite)
        with gen:
            gen.run_batch(
                max_new=effective_anchor_max_new,
                max_json_to_scan=args.anchor_max_json_scan,
                shuffle_candidates=bool(args.anchor_shuffle),
                shuffle_seed=args.anchor_shuffle_seed,
                allow_overwrite=effective_allow_overwrite,
                continuation_steps=None,
            )
            pending = _repair_unsolved_anchors(
                gen,
                rounds=int(getattr(args, "repair_rounds", 2)),
                allow_overwrite=effective_allow_overwrite,
                max_json_to_scan=args.anchor_max_json_scan,
            )
        _report_cohort_health(gen, pending)

    if not args.skip_mesh:
        _safe_print(f"--- Mesh to graph: meshes -> target graphs ({rheology}) ---")
        try:
            from src.data_gen.lib.mesh_to_graph import MeshToGraph

            # Process target graphs only; no intermediate continuation sweeps.
            final_subdir = _final_subdir_for_rheology(rheology)
            processor = MeshToGraph(phase="kinematics", n_subdir=final_subdir, rheology=rheology)
            processor.run()
        except OSError as exc:
            _safe_print(f"\nMesh-to-graph aborted (console I/O): {exc}\n")
            raise
        except Exception as exc:
            _safe_print(f"\nMesh-to-graph failed: {exc}\n")
            raise


#: What is tried, in order, on a vessel COMSOL could not solve:
#: ``(kind, mesh_refine, min_elems_across, severity_target)``.  ``"mesh"`` re-meshes the SAME
#: geometry finer; ``"reshape"`` re-draws a different vessel of the same class at
#: ``severity_target`` times the original's stenosis / aneurysm ratio.
#:
#: **One refine round, then get easier fast.**  Two runs measured the shape of this:
#:
#: * Two refine rounds (0.6x then 0.4x, the second reaching 25.7k nodes) recovered **2 of 39**.
#:   The second round is where the cost is -- a 25.7k-node mesh is slow to build and slower to
#:   solve -- and not where the recoveries are.  One cheap global refine is kept, because that
#:   does work on vessels failing purely for want of resolution (`vessel_15` solved under a
#:   general refine in COMSOL).
#: * Re-drawing at the SAME severity -- a 0.85x floor, added to protect the severe tail --
#:   recovered almost nothing: **38 re-drawn, 36 still unsolved**.  An equally extreme vessel
#:   fails for the same reason the original did.  A replacement has to be easier than the thing
#:   it replaces or it is not a replacement.
#:
#: So the ladder descends: 0.70x, then 0.50x, then 0.35x of the original severity, at the DEFAULT
#: mesh (an easier vessel does not also need a finer mesh, and pairing both doubled the cost for
#: nothing).  The cohort gives up some of its extreme tail, which is a real cost -- but refusing
#: to soften does not preserve that tail, it deletes it: an unsolved vessel contributes no labels
#: at all and the losses fall entirely on the severe end (RGP_DEQ_REPAIR_PLAN.md B27).  Every
#: substitution is stamped `reshaped_from` with the severity it was and became.
ANCHOR_REPAIR_SCHEDULE: tuple[tuple[str, float, int, "float | None"], ...] = (
    ("mesh", 0.6, 12, None),
    ("reshape", 1.0, 8, 0.70),
    ("reshape", 1.0, 8, 0.50),
    ("reshape", 1.0, 8, 0.35),
)


def _unsolved_stems(gen) -> list[str]:
    """Vessels that are CFD-ready but have no ``.npz`` -- i.e. the solve did not produce one."""
    out_dir = gen.target_output_dir()
    stems = []
    for jf in sorted(gen.mesh_dir.glob("vessel_*.json"),
                     key=lambda q: int(q.stem.split("_")[-1])):
        nas = jf.with_suffix(".nas")
        if not (nas.exists() and nas.stat().st_size > 0):
            continue
        if not (out_dir / f"{jf.stem}.npz").exists():
            stems.append(jf.stem)
    return stems


def _repair_unsolved_anchors(gen, *, rounds: int, allow_overwrite: bool,
                             max_json_to_scan=None) -> list[str]:
    """Rescue vessels COMSOL could not solve, and return what is still unsolved.

    Two stages, in order (see :data:`ANCHOR_REPAIR_SCHEDULE`):

    1. **Re-mesh the same geometry finer.**  Exact -- the wall polylines come from the vessel's
       own ``.json`` -- so the cohort's designed pathology mix is untouched.
    2. **Re-draw an EASIER vessel of the same class**, at a severity target that descends with
       each attempt (0.70x, 0.50x, 0.35x).  Same ``level``, same ``pathology_mode``, different
       RNG stream, and a ``reshaped_from`` stamp recording the severity given up.

    Stage 2 biases the corpus toward solvable realisations within a class.  That is a real cost,
    accepted only because both alternatives measured worse: leaving them unsolved dropped 15% of
    the cohort entirely on the severe-stenosis tail (RGP_DEQ_REPAIR_PLAN.md B27), and re-drawing
    at the SAME severity recovered 2 of 38 while spending a full COMSOL pass per attempt.
    """
    from src.config import VesselConfig
    from src.data_gen.lib.vessel_generator import (
        VesselGenerator, remesh_vessels_from_meta, reshape_vessels_from_meta)

    vg = VesselGenerator(phase="kinematics")
    cfg_dict = vg._cfg_dict()
    cfg = VesselConfig(phase="kinematics")
    pending = _unsolved_stems(gen)
    if not pending:
        return []

    n_reshaped = 0
    stages = ANCHOR_REPAIR_SCHEDULE[: max(0, int(rounds))]
    for r, (kind, refine, elems, sev_target) in enumerate(stages, start=1):
        if kind == "mesh":
            _safe_print(
                f"\n--- Anchor repair {r}/{len(stages)}: {len(pending)} unsolved, re-meshing "
                f"the same geometry at mesh_refine={refine} min_elems_across={elems} ---\n")
            results = remesh_vessels_from_meta(
                pending, gen.mesh_dir, cfg_dict, mesh_refine=refine, min_elems_across=elems)
        else:
            _safe_print(
                f"\n--- Anchor repair {r}/{len(stages)}: {len(pending)} unsolved, RE-DRAWING "
                f"easier vessels of the same class at {sev_target:.2f}x severity ---\n")
            results = reshape_vessels_from_meta(
                pending, gen.mesh_dir, cfg, cfg_dict, attempt=r,
                mesh_refine=refine, min_elems_across=elems,
                severity_target=float(sev_target))

        rebuilt = [stem for stem, ok, _ in results if ok]
        for stem, ok, err in results:
            if not ok:
                _safe_print(f"    {kind} failed: {stem}: {err[:90]}")
        if not rebuilt:
            _safe_print("    nothing could be rebuilt; stopping repair.\n")
            break

        # `only_stems` is not optional here.  These rounds run with `allow_overwrite=True`, which
        # puts every already-solved vessel back in the candidate pool; the batch would then spend
        # its `max_new` budget re-solving healthy geometries and stop before reaching a single
        # repaired one.  That is what made the repair look slow AND useless.
        gen.run_batch(
            max_new=len(rebuilt),
            max_json_to_scan=max_json_to_scan,
            shuffle_candidates=False,
            shuffle_seed=None,
            allow_overwrite=True,
            continuation_steps=None,
            only_stems=rebuilt,
        )
        still = _unsolved_stems(gen)
        recovered = len(pending) - len(still)
        if kind == "reshape":
            n_reshaped += recovered
        pending = still
        _safe_print(f"    {kind} round {r}: recovered {recovered}, still unsolved "
                    f"{len(pending)}\n")
        if not pending:
            break

    if n_reshaped:
        _safe_print(f"  {n_reshaped} vessel(s) were replaced by a new draw of the same class "
                    f"(`reshaped_from` in their .json).\n")
    return pending


def _report_cohort_health(gen, pending: list[str]) -> None:
    """Say plainly how many vessels the cohort actually has labels for."""
    total = sum(1 for jf in gen.mesh_dir.glob("vessel_*.json")
                if jf.with_suffix(".nas").exists())
    solved = total - len(pending)
    _safe_print("\n" + "=" * 62)
    _safe_print(f"  COHORT HEALTH   {solved}/{total} vessels solved")
    if pending:
        _safe_print(f"  {len(pending)} STILL UNSOLVED after repair -- these ship with an "
                    f"all-zero `y` and can only contribute PDE terms:")
        for stem in pending[:20]:
            _safe_print(f"    {stem}")
        if len(pending) > 20:
            _safe_print(f"    ... and {len(pending) - 20} more")
        _safe_print("  Open these in COMSOL, or raise --repair-rounds.")
    else:
        _safe_print("  Every CFD-ready vessel solved.")
    _safe_print("=" * 62 + "\n")


def run_batch_pipeline(args: argparse.Namespace) -> None:
    if getattr(args, "both_rheologies", False):
        rheologies = ("newtonian", "carreau")
        seeds = (args.seed_newtonian, args.seed_carreau)
    else:
        rheologies = (str(args.rheology),)
        seeds = (args.seed,)

    for i, rheology in enumerate(rheologies):
        if len(rheologies) > 1:
            print(f"\n========== Batch rheology {rheology} ==========\n")
        nv = _batch_num_vessels_for_rheology(rheology, args) if not args.skip_vessel else None
        am = _batch_anchor_max_for_rheology(rheology, args) if not args.skip_anchor else None
        _run_batch_for_phase(
            rheology,
            args,
            vessel_seed=seeds[i],
            num_vessels=nv,
            anchor_max_new=am,
        )

    print("\n=== Pipeline finished ===\n")


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mp.freeze_support()

    batch_args = _parse_batch_args(argv)
    if batch_args is not None:
        run_batch_pipeline(batch_args)
        return

    if argv:
        print(
            "Unknown arguments (use --batch for non-interactive). "
            "Re-run without arguments for interactive mode.",
            file=sys.stderr,
        )
        sys.exit(2)

    run_interactive_pipeline()


if __name__ == "__main__":
    main()
