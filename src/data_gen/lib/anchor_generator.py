import logging
import time
import json
import random
import numpy as np
import mph
import meshio
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from src.config import VesselConfig, PhysicsConfig
from src.utils.paths import (
    get_project_root,
)
from src.utils.console_progress import (
    logs_above_bar,
    progress,
    quiet_pipeline_logs,
)
from src.utils.units import MESH_UNIT_CM, d_bar_si_from_sidecar
from scipy.interpolate import NearestNDInterpolator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_MAX_COMSOL_RECONNECTS_PER_BATCH = 20
_CONSECUTIVE_FAST_FAIL_THRESHOLD = 3  # Errno 22 bursts -> reconnect immediately


def _safe_log(level: str, msg: str, *args, **kwargs) -> None:
    """Log without aborting the anchor batch if the console pipe is broken (Windows/PyCharm)."""
    try:
        getattr(logger, level)(msg, *args, **kwargs)
    except OSError:
        pass


def list_anchor_candidate_json_paths(
    mesh_dir: Path,
    output_dir: Path,
    *,
    include_existing_npz: bool = False,
) -> List[Path]:
    """Meshes eligible for anchor CFD: ``vessel_*.json`` with non-empty ``.nas`` and ``.msh``.

    By default skips stems that already have ``.npz``. With ``include_existing_npz=True``,
    those stems are included so runs can overwrite outputs.
    """
    mesh_dir = Path(mesh_dir)
    output_dir = Path(output_dir)
    candidates: List[Path] = []
    if not mesh_dir.exists():
        return candidates
    for json_file in sorted(mesh_dir.glob("vessel_*.json")):
        stem = json_file.stem
        try:
            int(stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        nas_file = mesh_dir / f"{stem}.nas"
        msh_file = mesh_dir / f"{stem}.msh"
        if not nas_file.exists() or nas_file.stat().st_size == 0:
            continue
        if not msh_file.exists():
            continue
        if (output_dir / f"{stem}.npz").exists() and not include_existing_npz:
            continue
        candidates.append(json_file)
    return candidates


def summarize_anchor_inventory(mesh_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Count existing CFD outputs and meshes still missing ``.npz`` (compatible with incremental runs).

    By default ``run_batch`` skips existing ``.npz``; use ``allow_overwrite=True`` to replace them.

    ``candidate_pool_ready`` counts meshes with ``.msh`` beside ``.nas`` and no ``.npz`` yet.
    ``candidate_pool_including_npz`` is the same but includes stems that already have ``.npz``.
    """
    mesh_dir = Path(mesh_dir)
    output_dir = Path(output_dir)
    existing_npz = 0
    if output_dir.exists():
        existing_npz = len(list(output_dir.glob("vessel_*.npz")))
    json_files = sorted(mesh_dir.glob("vessel_*.json")) if mesh_dir.exists() else []
    mesh_with_nas = 0
    pending_missing_npz = 0
    for json_file in json_files:
        stem = json_file.stem
        nas_file = mesh_dir / f"{stem}.nas"
        out_file = output_dir / f"{stem}.npz"
        if not nas_file.exists() or nas_file.stat().st_size == 0:
            continue
        mesh_with_nas += 1
        if not out_file.exists():
            pending_missing_npz += 1
    return {
        "existing_npz": existing_npz,
        "mesh_json_with_valid_nas": mesh_with_nas,
        "pending_missing_npz": pending_missing_npz,
        "candidate_pool_ready": len(
            list_anchor_candidate_json_paths(mesh_dir, output_dir, include_existing_npz=False)
        ),
        "candidate_pool_including_npz": len(
            list_anchor_candidate_json_paths(mesh_dir, output_dir, include_existing_npz=True)
        ),
    }


def _get_import_feature_tag(mesh_j) -> str:
    all_tags = mesh_j.feature().tags()
    for tag in all_tags:
        if mesh_j.feature(tag).getType() == 'Import':
            return tag
    if 'imp1' in all_tags:
        return 'imp1'
    raise RuntimeError("No 'Import' feature found in the COMSOL model mesh sequence.")



def select_anchor_candidates(
    mesh_dir: "Path",
    target_output_dir: "Path",
    *,
    allow_overwrite: bool = False,
    only_stems: Optional[Iterable[str]] = None,
) -> Tuple[List["Path"], List[str]]:
    """The pool of geometries a batch will attempt, and any requested stems that are not ready.

    ``allow_overwrite`` widens the pool to stems that ALREADY have a ``.npz``.  That is right for
    a full cohort refresh and wrong for a repair round, which must touch only the vessels it
    rebuilt -- otherwise the batch spends its ``max_new`` budget re-solving healthy geometries
    and stops before reaching a single repaired one.  ``only_stems`` is how a repair says so.
    """
    cands = list_anchor_candidate_json_paths(
        mesh_dir, target_output_dir, include_existing_npz=allow_overwrite
    )
    if only_stems is None:
        return list(cands), []
    want = {str(x) for x in only_stems}
    keep = [p for p in cands if p.stem in want]
    return keep, sorted(want - {p.stem for p in keep})


class AnchorGenerator:
    """
    Automates COMSOL CFD simulations based on synthetic vessel meshes.
    """

    def __init__(
        self,
        phase="kinematics",
        mesh_dir=None,
        output_dir=None,
        template_path=None,
        rheology: Optional[str] = None,
    ):
        self.vessel_config = VesselConfig(phase=phase)
        # Why the most recent single-anchor attempt failed; collected per vessel by `run_batch`.
        self._last_fail_reason: str = ""
        inferred_rheology = None
        if rheology is None and output_dir is not None:
            leaf = Path(output_dir).name.strip().lower()
            if leaf in {"newtonian", "carreau"}:
                inferred_rheology = leaf
        self.rheology = (
            rheology.strip().lower()
            if isinstance(rheology, str)
            else inferred_rheology
        )
        self.phys_cfg = PhysicsConfig(phase=phase, rheology=self.rheology)

        self.root_dir = get_project_root()

        # --- 1. Resolve Template Path ---
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = self.vessel_config.template_path

        # --- 2. Resolve Input/Output Paths ---
        # Handle Output Directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.vessel_config.output_dir

        # Handle Mesh Directory
        if mesh_dir:
            self.mesh_dir = Path(mesh_dir)
        else:
            self.mesh_dir = self.vessel_config.mesh_input_dir

        self.client: Optional[mph.Client] = None
        self.model: Optional[mph.Model] = None

        if not self.template_path.exists():
            raise FileNotFoundError(f"COMSOL template not found at: {self.template_path}")
        if not self.mesh_dir.exists():
            logger.warning(f"Mesh input directory does not exist: {self.mesh_dir}")

    def _final_target_output_dir(self) -> Path:
        """Directory for the final target n outputs.

        Kinematics final anchors live directly in regime folders
        (``.../<newtonian|carreau>``). Continuation sweeps still write intermediate
        ``n_*`` folders during solving.
        """
        if self.vessel_config.phase == "kinematics":
            return self.output_dir
        return self.output_dir

    def target_output_dir(self) -> Path:
        """Public accessor for current final-target output directory."""
        return self._final_target_output_dir()

    def __enter__(self):
        quiet_pipeline_logs()
        logger.info(f"Connecting to COMSOL... Loading: {self.template_path.name}")
        self.client = mph.start()
        self.model = self.client.load(str(self.template_path))
        self._set_global_physics_parameters()
        self._sync_t0_fluid_study()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._final_target_output_dir().mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            _safe_log("info", "Disconnecting from COMSOL...")
            try:
                self.client.clear()
            except Exception as exc:
                _safe_log("warning", "mph client.clear() failed on exit: %s", exc)
        return False

    def _clear_all_solution_data(self) -> None:
        if not self.model:
            return
        try:
            for tag in self.model.java.sol().tags():
                self.model.java.sol(tag).clearSolutionData()
        except Exception as exc:
            _safe_log("warning", "clearSolutionData failed: %s", exc)

    def _ensure_mesh_handles(self) -> Tuple[Any, str]:
        mesh_j = self.model.java.component("comp1").mesh("mesh1")
        import_tag = _get_import_feature_tag(mesh_j)
        return mesh_j, import_tag

    def _reconnect_comsol_session(self, reason: str) -> Tuple[Any, str]:
        """Drop broken MPh/Java state and reload the template."""
        _safe_log("warning", "COMSOL session recovery: %s", reason)
        try:
            if self.client:
                self.client.clear()
        except Exception:
            pass
        self.client = mph.start(cores=1)
        self.model = self.client.load(str(self.template_path))
        self._set_global_physics_parameters()
        self._sync_t0_fluid_study()
        return self._ensure_mesh_handles()

    @staticmethod
    def _is_comsol_solver_failure(exc: BaseException) -> bool:
        text = repr(exc)
        needles = (
            "FlException",
            "Failed to find a solution",
            "not converged",
            "Maximum number of Newton",
        )
        return any(n in text for n in needles)

    @staticmethod
    def _is_likely_session_broken(exc: BaseException) -> bool:
        if isinstance(exc, OSError):
            return True
        return AnchorGenerator._is_comsol_solver_failure(exc)

    def _set_element_order(self, order: int | None = None) -> str:
        """Set the Laminar Flow node's ``order_fluid`` (1 = P1+P1, 2 = P2+P1).

        The kinematics template ships P1+P1 while every deployment vessel is solved P2+P1, and
        that difference lands on the near-wall shear the clot gate reads -- see
        ``PhysicsConfig.comsol_order_fluid`` for the measurement.  Returns the property group
        that accepted the setting, or "" when the node does not expose one (older COMSOL
        versions name it differently; a failure here must not abort a cohort).
        """
        want = int(self.phys_cfg.comsol_order_fluid if order is None else order)
        ph = self.model.java.physics("spf")
        for grp in ("ShapeProperty", "PhysicsShapeProperty"):
            try:
                ph.prop(grp).set("order_fluid", str(want))
                logger.debug("COMSOL discretization: order_fluid=%s (via %s).", want, grp)
                return grp
            except Exception:
                continue
        logger.warning(
            "Could not set order_fluid=%s: no known property group on physics 'spf'. "
            "The template's own setting stands.", want
        )
        return ""

    def _set_global_physics_parameters(self):
        grp = self._set_element_order()
        logger.info(
            "Global physics: %s, order_fluid=%d%s",
            self.phys_cfg.viscosity_model,
            int(self.phys_cfg.comsol_order_fluid),
            f" (via {grp})" if grp else " (template default)",
        )

        # Update Parameters (Global)
        self.model.parameter('rho_fluid', f'{self.phys_cfg.rho} [kg/m^3]')
        self.model.parameter('Re_target', str(self.phys_cfg.re_target))
        self.model.parameter('mu_ref', f'{self.phys_cfg.mu_ref} [Pa*s]')
        self.model.parameter('mu_inf', f'{self.phys_cfg.mu_inf} [Pa*s]')
        self.model.parameter('mu_0', f'{self.phys_cfg.mu_0} [Pa*s]')
        self.model.parameter('lambda_cy', f'{self.phys_cfg.lam} [s]')
        self.model.parameter('n_index', str(self.phys_cfg.n))
        self.model.parameter('a_yasuda', str(self.phys_cfg.a))

        # Update Variables (Component Level)
        var_node = self.model.java.component('comp1').variable('var1')

        if self.phys_cfg.viscosity_model == "carreau":
            carreau_expr = 'mu_inf + (mu_0 - mu_inf) * (1 + (lambda_cy * spf.sr)^a_yasuda)^((n_index - 1) / a_yasuda)'
            var_node.set('mu_final', carreau_expr)
        else:
            # Newtonian fallback
            var_node.set('mu_final', 'mu_ref')

        if str(getattr(self.phys_cfg, "comsol_fluid_study", "") or "").strip():
            from src.data_gen.lib.comsol_t0_fluid import apply_t0_fluid_physics
            apply_t0_fluid_physics(self.model, self.phys_cfg)

    def _sync_t0_fluid_study(self) -> None:
        tag = str(getattr(self.phys_cfg, "comsol_fluid_study", "") or "").strip()
        if not tag:
            return
        from src.data_gen.lib.comsol_t0_fluid import ensure_fluid_only_study
        tlist = str(getattr(self.phys_cfg, "comsol_fluid_tlist", "") or "") or None
        ensure_fluid_only_study(self.model, tlist=tlist)

    def _run_comsol_solve(self) -> None:
        tag = str(getattr(self.phys_cfg, "comsol_fluid_study", "") or "").strip()
        if tag:
            from src.data_gen.lib.comsol_t0_fluid import solve_fluid_only
            solve_fluid_only(self.model, tag)
            return
        self.model.solve()

    def _interp_data_tag(self) -> str:
        tag = str(getattr(self.phys_cfg, "comsol_fluid_study", "") or "").strip()
        if not tag:
            return "dset1"
        from src.data_gen.lib.comsol_t0_fluid import interpolation_dataset_tag
        return interpolation_dataset_tag(self.model.java)

    def _evaluate_exprs(self, exprs, coords: np.ndarray):
        """Evaluate arbitrary COMSOL expressions at arbitrary coordinates.

        ``_evaluate_at_coords`` asks for ``u, v, p, spf.mu`` (phase2 fluid-only Carreau).
        can also be asked for the quantities the clot stack actually consumes.  Audited on the
        generation box: `spf.sr`, `d(spf.sr,x/y)`, `ux/uy/vx/vy`, `spf.U`, `spf.cellRe` and
        `spf.mu` all evaluate; `spf.divU` does not.
        """
        results = []
        model_j = self.model.java
        tag = "py_expr_interp"
        try:
            try:
                model_j.result().numerical().remove(tag)
            except Exception:
                pass
            model_j.result().numerical().create(tag, "Interp")
            it = model_j.result().numerical(tag)
            it.set("data", self._interp_data_tag())
            it.set("expr", list(exprs))
            it.setInterpolationCoordinates(coords.T.tolist())
            from src.data_gen.lib.comsol_t0_fluid import last_time_slice, pin_interp_last_time
            pin_interp_last_time(it)
            data = it.getData()
            n = coords.shape[0]
            for k in range(len(exprs)):
                results.append(last_time_slice(data[k], n))
        finally:
            try:
                model_j.result().numerical().remove(tag)
            except Exception:
                pass
        return results

    def _evaluate_at_coords(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        High-performance evaluation using COMSOL Java API Interp feature.
        """
        coords_T = coords.T
        model_j = self.model.java
        results = model_j.result()
        interp_name = "py_interp_temp"

        try:
            interp_tag = results.numerical().create(interp_name, "Interp").tag()
            interp = results.numerical(interp_tag)

            interp.set("data", self._interp_data_tag())
            interp.set("expr", ["u", "v", "p", "spf.mu"])
            interp.setInterpolationCoordinates(coords_T.tolist())
            from src.data_gen.lib.comsol_t0_fluid import last_time_slice, pin_interp_last_time
            pin_interp_last_time(interp)
            data = interp.getData()

            if len(data) < 4:
                raise ValueError(f"COMSOL returned incomplete data. Shape: {len(data)}")

            n = coords.shape[0]
            u = last_time_slice(data[0], n)
            v = last_time_slice(data[1], n)
            p = last_time_slice(data[2], n)
            mu = last_time_slice(data[3], n)

            return u, v, p, mu

        except Exception as e:
            logger.error(f"COMSOL Evaluation failed: {e}")
            nan_arr = np.full(coords.shape[0], np.nan)
            return nan_arr, nan_arr, nan_arr, nan_arr

        finally:
            try:
                results.numerical().remove(interp_name)
            except Exception:
                pass

    def _process_single_anchor(
        self, json_file: Path, mesh_j, import_tag, *, allow_overwrite: bool = False, continuation_steps: Optional[List[float]] = None
    ) -> bool:
        """Run COMSOL for one vessel; write ``.npz`` if field checks pass. Returns True if saved."""
        file_stem = json_file.stem
        try:
            i = int(file_stem.split("_")[1])
        except (ValueError, TypeError, IndexError):
            return False

        nas_file = self.mesh_dir / f"{file_stem}.nas"
        msh_file = self.mesh_dir / f"{file_stem}.msh"

        # Build sequence of n_values to solve (continuation steps + target)
        n_sequence = continuation_steps.copy() if continuation_steps else []
        if self.phys_cfg.n not in n_sequence:
            n_sequence.append(self.phys_cfg.n)

        if not nas_file.exists() or nas_file.stat().st_size == 0:
            return False

        try:
            _safe_log("debug", "[%s] Purging old solution data from COMSOL memory...", i)
            self._clear_all_solution_data()

            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            d_bar_si, mesh_unit = d_bar_si_from_sidecar(
                meta, stem=file_stem, builder="AnchorGenerator"
            )
            if mesh_unit == MESH_UNIT_CM and self.vessel_config.phase == "kinematics":
                raise ValueError(
                    f"[{i}] kinematics corpus requires SI meshes (sidecar unit=m). "
                    f"Biochem anchor meshes use unit=cm and PatientDataExtractor, not AnchorGenerator."
                )

            from src.data_gen.lib.comsol_t0_fluid import (
                FLUID_ONLY_STUDY,
                FLUID_ONLY_T_END,
                apply_phase2_inlet_parameters,
                import_gmsh_nas_mesh,
                mesh_vertex_xy_metres,
                validate_mesh_import,
            )
            u_ref = self.phys_cfg.get_u_ref(d_bar_si)
            apply_phase2_inlet_parameters(self.model, d_bar_si, u_ref, self.phys_cfg)

            n_verts = import_gmsh_nas_mesh(mesh_j, import_tag, nas_file)
            logger.debug("[%s] imported mesh with %d COMSOL vertices.", i, n_verts)
            if n_verts < 10:
                raise RuntimeError(f"Mesh {i} is empty/corrupt (Vertices: {n_verts})")

            mesh = meshio.read(msh_file)
            target_nodes = mesh.points[:, :2]
            n_nodes = target_nodes.shape[0]
            self._last_n_nodes = n_nodes
            if n_nodes != n_verts:
                logger.warning(
                    "[%s] Gmsh node count (%d) != COMSOL vertex count (%d); "
                    "continuing with Gmsh evaluation grid.",
                    i, n_nodes, n_verts,
                )
            try:
                if self.phys_cfg.comsol_mesh_import_validate:
                    import_stats = validate_mesh_import(
                        target_nodes, mesh_vertex_xy_metres(self.model.java)
                    )
                    logger.debug(
                        "[%s] mesh import OK: nn_p90=%.2e m exact=%d/%d",
                        i,
                        import_stats["nn_p90_m"],
                        int(import_stats["exact_matches"]),
                        n_nodes,
                    )
            except ValueError as exc:
                self._last_fail_reason = str(exc)
                logger.error("[%s] %s", i, exc)
                return False

            # Build boundary masks once per mesh so NaN repair can preserve boundary classes.
            wall_tags = {self.vessel_config.TAGS["Walls"]}
            outlet_tags = {
                tag_id for name, tag_id in self.vessel_config.TAGS.items() if "Outlet" in name
            }
            boundary_node_mask = np.zeros(n_nodes, dtype=bool)
            wall_node_mask = np.zeros(n_nodes, dtype=bool)
            outlet_node_indices = []
            try:
                line_cells = mesh.get_cells_type("line")
                line_tags = mesh.get_cell_data("gmsh:physical", "line")
                has_lines = len(line_cells) > 0
            except Exception:
                line_cells = []
                line_tags = []
                has_lines = False

            if has_lines:
                for j, tag in enumerate(line_tags):
                    if j >= len(line_cells):
                        continue
                    nodes_j = np.asarray(line_cells[j], dtype=np.int64)
                    nodes_j = nodes_j[(nodes_j >= 0) & (nodes_j < n_nodes)]
                    if nodes_j.size == 0:
                        continue
                    boundary_node_mask[nodes_j] = True
                    if int(tag) in wall_tags:
                        wall_node_mask[nodes_j] = True
                    if int(tag) in outlet_tags:
                        outlet_node_indices.extend(nodes_j.tolist())
            interior_node_mask = ~boundary_node_mask

            # --- THE CONTINUATION LOOP ---
            for step_idx, n_val in enumerate(n_sequence):
                is_target = (n_val == self.phys_cfg.n and step_idx == len(n_sequence) - 1)

                # Setup output paths
                if is_target:
                    out_file = self._final_target_output_dir() / f"{file_stem}.npz"
                else:
                    step_dir = self.output_dir / f"n_{n_val:.3f}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    out_file = step_dir / f"{file_stem}.npz"

                if out_file.exists() and not allow_overwrite:
                    logger.debug(f"[{i}] Skipping step n={n_val}, already exists.")
                    continue

                _safe_log("debug", "[%s] solving n_index=%s", i, n_val)
                from src.data_gen.lib.comsol_t0_fluid import set_carreau_n
                set_carreau_n(self.model, n_val)
                try:
                    self._run_comsol_solve()
                except Exception as solve_exc:
                    _safe_log(
                        "warning",
                        "[%s] COMSOL model.solve() failed at n_index=%s: %s: %s",
                        i,
                        n_val,
                        type(solve_exc).__name__,
                        getattr(solve_exc, "msg", solve_exc),
                    )
                    raise

                u, v, p, mu = self._evaluate_at_coords(target_nodes)
                u, v, p, mu = u.flatten(), v.flatten(), p.flatten(), mu.flatten()

                def fix_boundary_nans(field, coords):
                    """Fill NaNs without leaking interior values onto wall nodes."""
                    out = np.asarray(field, dtype=np.float64).copy()
                    nan_mask = np.isnan(out)
                    if not nan_mask.any():
                        return out
                    if nan_mask.all():
                        return out

                    def _fill_group(target_mask, source_mask):
                        missing = nan_mask & target_mask
                        if not missing.any():
                            return
                        available = source_mask & (~np.isnan(out))
                        if not available.any():
                            return
                        interpolator = NearestNDInterpolator(coords[available], out[available])
                        out[missing] = interpolator(coords[missing])

                    # Prefer class-consistent fills first.
                    _fill_group(interior_node_mask, interior_node_mask)
                    _fill_group(wall_node_mask, wall_node_mask)
                    boundary_nonwall = boundary_node_mask & (~wall_node_mask)
                    _fill_group(boundary_nonwall, boundary_nonwall)

                    # Fallback for any residual NaNs.
                    residual = np.isnan(out)
                    if residual.any():
                        available = ~residual
                        if available.any():
                            interpolator = NearestNDInterpolator(coords[available], out[available])
                            out[residual] = interpolator(coords[residual])
                    return out

                u = fix_boundary_nans(u, target_nodes)
                v = fix_boundary_nans(v, target_nodes)
                p = fix_boundary_nans(p, target_nodes)
                mu = fix_boundary_nans(mu, target_nodes)

                if has_lines and outlet_node_indices:
                    unique_indices = np.unique(np.asarray(outlet_node_indices, dtype=np.int64))
                    valid_indices = unique_indices[(unique_indices >= 0) & (unique_indices < len(p))]
                    if len(valid_indices) > 0:
                        p_offset = np.mean(p[valid_indices])
                        p = p - p_offset

                nan_u = np.isnan(u).sum()
                nan_v = np.isnan(v).sum()
                nan_p = np.isnan(p).sum()
                nan_mu = np.isnan(mu).sum()
                total_nodes = len(u)

                if nan_u > 0 or nan_v > 0 or nan_p > 0 or nan_mu > 0:
                    logger.warning(
                        f"NaNs detected in {nas_file.name} at n={n_val} | Total Nodes: {total_nodes} | "
                        f"NaN counts -> u: {nan_u}, v: {nan_v}, p: {nan_p}, mu: {nan_mu}"
                    )
                    self._last_fail_reason = (
                        f"NaNs at n={n_val} (u={nan_u}, v={nan_v}, p={nan_p}, mu={nan_mu} "
                        f"of {total_nodes})"
                    )
                    return False

                p_std = np.std(p)
                u_max = np.max(np.abs(u))
                if p_std < 1e-9 or u_max < 1e-7:
                    logger.warning(f"[{i}] Skipping: Trivial solution detected at n={n_val}")
                    self._last_fail_reason = (
                        f"trivial solution at n={n_val} (p_std={p_std:.3e}, u_max={u_max:.3e})"
                    )
                    return False

                # TRUE P2 MID-SIDE LABELS.  `elevate_to_p2` sets every mid-side label to the
                # mean of its two corners, which makes the field piecewise-linear along each
                # half-edge BY CONSTRUCTION -- and `dsrx`, the gate branch that decides
                # deployment, is a second derivative of it.  Measured on this box
                # (`scripts/exp_comsol_element_order.py --p2-nodes`), asking COMSOL for the
                # mid-side values instead carries **2.2-4.4x** the wall `dsrx` spread.
                #
                # It only works at `order_fluid=2`: with linear velocity elements COMSOL's
                # interpolant at an edge midpoint IS the corner mean (measured ratio 1.00,
                # |du|max 0.001).  Both halves or neither -- see PILOT_COHORT_RUNBOOK.md §7.3.
                # COMSOL's OWN shear rate and its along-flow derivative.  Every `sr` / `dsrx`
                # in this project is otherwise reconstructed by finite-differencing sampled
                # velocity -- twice for `dsrx` -- and measured on the generation box that costs
                # ~18% of the amplitude (`ours/its` 0.82-0.84 at order 2, corr 0.94-0.99).
                #
                # At order_fluid=1 `d(spf.sr,x)` is IDENTICALLY ZERO: with linear velocity
                # elements the strain rate is elementwise constant, so its spatial derivative
                # does not exist.  The P1 solve cannot represent the gate's dominant argument
                # at all, which is the strongest reason order 2 is mandatory.
                shear_arrays = {}
                try:
                    sr_t, dsrx_t, dsry_t, _ = self._evaluate_exprs(
                        ["spf.sr", "d(spf.sr,x)", "d(spf.sr,y)", "spf.U"], target_nodes)
                    shear_arrays = dict(sr_comsol=sr_t, dsrx_comsol=dsrx_t, dsry_comsol=dsry_t)
                except Exception as exc:
                    logger.warning("[%s] COMSOL shear export unavailable (%s: %s); the packs "
                                   "fall back to reconstructing it from u,v.",
                                   i, type(exc).__name__, exc)

                mid_arrays = {}
                if int(self.phys_cfg.comsol_order_fluid) >= 2:
                    try:
                        from src.data_gen.lib.mesh_triangle6_edges import (
                            mesh_undirected_edge_pairs,
                        )

                        pairs = mesh_undirected_edge_pairs(mesh)
                        mid_xy = 0.5 * (target_nodes[pairs[:, 0]] + target_nodes[pairs[:, 1]])
                        mu_, mv_, mp_, mmu_ = self._evaluate_at_coords(mid_xy)
                        flat = lambda z: np.asarray(z, dtype=float).reshape(-1)
                        mid_arrays = dict(
                            mid_x=mid_xy[:, 0], mid_y=mid_xy[:, 1],
                            mid_u=flat(mu_), mid_v=flat(mv_),
                            mid_p=flat(mp_), mid_mu=flat(mmu_),
                        )
                        if any(a.shape[0] != pairs.shape[0] for a in mid_arrays.values()):
                            logger.warning("[%s] mid-side evaluation returned the wrong count; "
                                           "falling back to interpolated labels.", i)
                            mid_arrays = {}
                    except Exception as exc:
                        logger.warning("[%s] mid-side evaluation failed (%s: %s); falling back "
                                       "to interpolated labels.", i, type(exc).__name__, exc)
                        mid_arrays = {}

                np.savez(
                    out_file,
                    x=target_nodes[:, 0],
                    y=target_nodes[:, 1],
                    **shear_arrays,
                    u=u,
                    v=v,
                    p=p,
                    mu=mu,
                    **mid_arrays,
                    d_bar=d_bar_si,
                    mesh_unit=mesh_unit,
                    config_id=i,
                    carreau_n=n_val,
                    fluid_study=FLUID_ONLY_STUDY,
                    fluid_t_end=FLUID_ONLY_T_END,
                    order_fluid=int(self.phys_cfg.comsol_order_fluid),
                    comsol_template=self.template_path.name,
                    gmsh_n_nodes=int(n_nodes),
                    comsol_n_vertices=int(n_verts),
                    u_ref=float(u_ref),
                )
            return True

        except OSError:
            self._clear_all_solution_data()
            raise
        except Exception as e:
            if self._is_comsol_solver_failure(e):
                _safe_log(
                    "warning",
                    "Sample %s: COMSOL solve failed (n_index=%s): %s: %s",
                    i,
                    self.phys_cfg.n,
                    type(e).__name__,
                    e,
                )
                self._last_fail_reason = f"solver did not converge: {type(e).__name__}: {e}"
            else:
                _safe_log("error", "Error on %s: %s", i, e)
                self._last_fail_reason = f"{type(e).__name__}: {e}"
            self._clear_all_solution_data()
            return False

    def run_batch(
        self,
        max_new: int = 500,
        max_json_to_scan: Optional[int] = None,
        shuffle_candidates: bool = False,
        shuffle_seed: Optional[int] = None,
        allow_overwrite: bool = False,
        continuation_steps: Optional[List[float]] = None,
        only_stems: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Write up to ``max_new`` new healthy ``vessel_*.npz`` files.

        Builds the pool of geometries (valid ``.nas`` + ``.msh``). By default only stems without
        ``.npz`` are eligible; with ``allow_overwrite=True``, existing ``.npz`` files may be replaced.
        Then walks candidates **in order** until enough saves succeed or the list is exhausted.
        Failed solves (NaNs, trivial flow, exceptions) **do not** count toward ``max_new``; each
        failure skips to the **next** CFD-ready geometry so the batch still aims for ``max_new``
        successes when enough candidates exist.

        Parameters
        ----------
        max_new
            Target number of new ``.npz`` files to write this run.
        max_json_to_scan
            Optional cap on how many **candidates** to attempt (after building the pool). ``None`` = try
            every candidate. Prefer a large pool of meshes or leave unset so failures can be offset by
            later indices (the old behavior truncated the global sorted list *before* filtering, which
            could hide viable vessels).
        shuffle_candidates
            If True, randomize candidate order (e.g. spread load across geometry types).
        shuffle_seed
            Seed for shuffling; only used when ``shuffle_candidates`` is True.
        allow_overwrite
            If True, include stems that already have ``.npz`` and replace files after a successful solve.
        only_stems
            Restrict the pool to exactly these vessel stems.  Required by the repair rounds: they
            run with ``allow_overwrite=True`` (inherited from a ``--overwrite`` cohort), which
            puts every ALREADY-SOLVED vessel back in the pool.  Combined with a small ``max_new``
            that silently broke the repair -- with 43 solved and 7 to fix, the batch re-solved
            healthy vessels, hit its 7 successes on those, and stopped without attempting a
            single repaired geometry.  It looked like "7 unsolved" followed by a screen of
            "Finished solving study".
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_output_dir = self._final_target_output_dir()
        target_output_dir.mkdir(parents=True, exist_ok=True)
        existing_npz = len(list(target_output_dir.glob("vessel_*.npz")))
        candidates, missing = select_anchor_candidates(
            self.mesh_dir, target_output_dir,
            allow_overwrite=allow_overwrite, only_stems=only_stems,
        )
        if missing:
            _safe_log("warning", "only_stems: %d requested stem(s) are not CFD-ready: %s",
                      len(missing), ", ".join(sorted(missing)[:10]))
        pool_full = len(candidates)

        if shuffle_candidates:
            rng = random.Random(shuffle_seed)
            rng.shuffle(candidates)

        if max_json_to_scan is not None:
            candidates = candidates[: int(max_json_to_scan)]

        scope = ("restricted to %d named stem(s)" % len(candidates)) if only_stems is not None \
            else ("no .npz, has .nas+.msh" if not allow_overwrite else "including existing .npz")
        logger.info(
            f"Anchor batch: existing .npz={existing_npz}, target new successes={max_new}, "
            f"candidate pool ({scope})={pool_full}, "
            f"will attempt min(len(pool), cap)={len(candidates)} geometries."
        )
        logger.info(
            "Failed solves (exceptions, NaNs, trivial flow) do not count toward the target; "
            "the batch continues with the next CFD-ready geometry until the target is met or "
            "the candidate list is exhausted."
        )

        try:
            mesh_j, import_tag = self._ensure_mesh_handles()
        except Exception as e:
            _safe_log("critical", "Setup failed: %s", e)
            return {
                "existing_before": existing_npz,
                "requested_new": max_new,
                "new_written": 0,
                "attempted": 0,
                "failed_or_discarded": 0,
                "failed_stems": [],
                "failures": [],
                "pool_full": pool_full,
                "pool_attempted": len(candidates),
                "pool_exhausted": True,
                "setup_failed": True,
            }

        reconnect_count = 0
        consecutive_fast_fails = 0
        new_written = 0
        attempted = 0
        n_failed = 0
        # Which vessels failed, and why.  Without this a failed solve is invisible downstream:
        # `mesh_to_graph` still writes a pack (all-zero `y`, `is_anchor=False`) and the cohort
        # ships short with nothing saying so -- 39/250 on 2026-08-28, all stenosis geometries
        # (RGP_DEQ_REPAIR_PLAN.md B27).
        failures: List[Tuple[str, str]] = []
        quiet_pipeline_logs()
        bar = progress(candidates, desc="Anchors", unit="vessel", total=min(len(candidates), max_new))
        with logs_above_bar():
            for json_file in bar:
                if new_written >= max_new:
                    break
                attempted += 1
                ok = False
                self._last_fail_reason = ""
                self._last_n_nodes = 0
                t_vessel = time.perf_counter()
                try:
                    ok = self._process_single_anchor(
                        json_file,
                        mesh_j,
                        import_tag,
                        allow_overwrite=allow_overwrite,
                        continuation_steps=continuation_steps,
                    )
                except Exception as exc:
                    consecutive_fast_fails += 1
                    if (
                        reconnect_count < _MAX_COMSOL_RECONNECTS_PER_BATCH
                        and consecutive_fast_fails >= _CONSECUTIVE_FAST_FAIL_THRESHOLD
                        and (
                            isinstance(exc, OSError)
                            or self._is_comsol_solver_failure(exc)
                        )
                    ):
                        reason = (
                            f"{consecutive_fast_fails} fast failures "
                            f"(last: {type(exc).__name__}: {exc})"
                        )
                        mesh_j, import_tag = self._reconnect_comsol_session(reason)
                        reconnect_count += 1
                        consecutive_fast_fails = 0
                        attempted -= 1
                        continue
                    consecutive_fast_fails = 0
                    _safe_log("error", "Unhandled error on %s: %s", json_file.stem, exc)
                    self._last_fail_reason = f"{type(exc).__name__}: {exc}"
                    ok = False

                if ok:
                    new_written += 1
                    consecutive_fast_fails = 0
                else:
                    n_failed += 1
                    failures.append((json_file.stem, self._last_fail_reason or "unknown"))
                dt = time.perf_counter() - t_vessel
                nodes = int(getattr(self, "_last_n_nodes", 0) or 0)
                if ok:
                    logger.info("[%s] solved  %6d nodes  %5.0fs", json_file.stem.split("_")[-1], nodes, dt)
                else:
                    logger.warning(
                        "[%s] FAILED  %6d nodes  %5.0fs  %s",
                        json_file.stem.split("_")[-1], nodes, dt,
                        (self._last_fail_reason or "unknown")[:80],
                    )
                bar.set_postfix_str(f"ok={new_written} fail={n_failed}")

        pool_exhausted = new_written < max_new and attempted >= len(candidates)
        if new_written < max_new:
            logger.warning(
                f"Anchor batch finished short: {new_written}/{max_new} new .npz after "
                f"{attempted} attempt(s) ({n_failed} failed or discarded, {new_written} saved). "
                + (
                    "All CFD-ready candidates in this pass were tried — no more geometries left to "
                    "reach the target; generate more vessel meshes for this phase or raise "
                    "max_json_to_scan / remove the scan cap."
                    if pool_exhausted
                    else "Raise max_json_to_scan (or leave it unset) to try more existing geometries."
                )
            )
        else:
            logger.info(f"Anchor batch: wrote {new_written} new .npz (target was {max_new}).")

        if failures:
            _safe_log("warning", "%d vessel(s) did not solve:", len(failures))
            for stem, why in failures[:20]:
                _safe_log("warning", "    %-16s %s", stem, why[:96])
            if len(failures) > 20:
                _safe_log("warning", "    ... and %d more", len(failures) - 20)

        return {
            "existing_before": existing_npz,
            "requested_new": max_new,
            "new_written": new_written,
            "attempted": attempted,
            "failed_or_discarded": n_failed,
            "failed_stems": [stem for stem, _ in failures],
            "failures": failures,
            "pool_full": pool_full,
            "pool_attempted": len(candidates),
            "pool_exhausted": pool_exhausted,
            "setup_failed": False,
        }


def _prompt_int_choice(label: str, allowed: Tuple[int, ...]) -> int:
    """Read an integer from stdin until it is one of ``allowed``."""
    allowed_str = "/".join(str(x) for x in allowed)
    while True:
        raw = input(f"{label} ({allowed_str}): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  Enter an integer: {allowed_str}")
            continue
        if v in allowed:
            return v
        print(f"  Must be one of: {allowed_str}")


if __name__ == "__main__":
    try:
        def _prompt_int(label, default):
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

        def _prompt_write_mode() -> bool:
            """Return True if overwrite mode, False for add-only."""
            while True:
                raw = input("Write mode [1=add new files only / 2=overwrite existing .npz] [1]: ").strip()
                if raw in ("", "1"):
                    return False
                if raw == "2":
                    return True
                print("  Enter 1 or 2.")

        phase_n = _prompt_int_choice("Phase", (1, 2))
        phase = f"phase{phase_n}"
        gen = AnchorGenerator(phase=phase)
        target_dir = gen.target_output_dir()
        inv = summarize_anchor_inventory(gen.mesh_dir, target_dir)
        total_v = int(inv["mesh_json_with_valid_nas"])
        have_npz = int(inv["existing_npz"])
        remaining = int(inv["pending_missing_npz"])
        ready_add = int(inv["candidate_pool_ready"])
        ready_all = int(inv["candidate_pool_including_npz"])
        print("\n--- Anchor CFD inventory ---")
        print(f"  Output: {target_dir}")
        print(f"  Mesh:   {gen.mesh_dir}")
        print(f"  Total number of phase vessels: {total_v}")
        print(f"  Number of anchors already generated: {have_npz}")
        print(f"  Number of non-anchors remaining: {remaining}")
        if remaining > ready_add:
            print(
                f"  ({remaining - ready_add} of those still need a .msh export before CFD.)"
            )
        print()
        allow_overwrite = _prompt_write_mode()
        pool = ready_all if allow_overwrite else ready_add
        default_more = min(pool, 50) if pool > 0 else 0
        if pool == 0:
            if allow_overwrite:
                print("No meshes are CFD-ready (need .json + non-empty .nas + .msh).")
            else:
                msg = "Nothing to add (need .json + non-empty .nas + .msh, and no .npz yet)."
                if remaining > 0:
                    msg += " Some meshes lack .msh — re-run mesh export for those vessels."
                elif total_v == 0:
                    msg = "No vessel meshes found in the mesh directory."
                print(msg)
            raise SystemExit(0)
        mode_note = "CFD runs to attempt" if allow_overwrite else "new CFD samples to generate"
        asked = _prompt_int(f"How many {mode_note}", default_more)
        if asked == 0:
            print("Exiting (0 requested).")
            raise SystemExit(0)
        max_new = min(asked, pool)
        if asked > pool:
            print(f"Requested {asked} but only {pool} mesh(es) match this mode; running {max_new}.")
        with gen:
            gen.run_batch(max_new=max_new, allow_overwrite=allow_overwrite)
    except SystemExit:
        raise
    except Exception as e:
        print(e)