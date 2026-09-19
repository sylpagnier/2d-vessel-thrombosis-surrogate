"""Phase2 Study-2 (fluid only) is the kinematics t=0 problem."""

from src.config import PhysicsConfig
from src.data_gen.lib.comsol_t0_fluid import (
    FLUID_ONLY_STUDY,
    FLUID_ONLY_TLIST,
    gel_identity_carreau_si,
    interpolation_dataset_tag,
)
from src.tests.test_biochem_comsol_datasets import _FakeDataset, _FakeModelJava


def test_gel_identity_matches_physics_config():
    phys = PhysicsConfig(phase="kinematics", rheology="carreau")
    got = gel_identity_carreau_si(phys)
    assert got["mu0"] == "0.056[Pa*s]"
    assert got["mu_inf"] == "0.0035[Pa*s]"
    assert got["lam_car"] == "3.313[s]"
    assert got["n_car"] == "0.3568"


def test_fluid_only_study_matches_phase2_xml():
    assert FLUID_ONLY_STUDY == "std2"
    assert FLUID_ONLY_TLIST == "range(0,0.1,15)"


def test_newtonian_physics_config_uses_std2_study_field():
    phys = PhysicsConfig(phase="kinematics", rheology="newtonian")
    assert phys.viscosity_model == "newtonian"
    assert phys.comsol_fluid_study == "std2"


def test_last_time_slice_picks_final_block():
    import numpy as np
    from src.data_gen.lib.comsol_t0_fluid import last_time_slice

    n, nt = 4, 151
    stacked = np.arange(nt * n, dtype=float)
    got = last_time_slice(stacked, n)
    assert got.tolist() == stacked.reshape(nt, n)[-1].tolist()
    assert last_time_slice(np.arange(n, dtype=float), n).tolist() == [0, 1, 2, 3]


def test_interp_prefers_sol2_study2_dataset():
    model = _FakeModelJava(
        {
            "dset1": _FakeDataset("Study 1 (fluid + biochemistry)/Solution 1 (sol1)", "sol1"),
            "dset2": _FakeDataset("Study 2 (only fluid)/Soluzione 2 (sol2)", "sol2"),
        }
    )
    assert interpolation_dataset_tag(model) == "dset2"
