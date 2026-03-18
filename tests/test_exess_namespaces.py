import types

from rush import (
    exess,
    exess_geo_opt as exess_geo_opt_module,
    exess_qmmm as exess_qmmm_module,
)
from rush.exess import exess_energy, exess_interaction_energy
from rush.exess_geo_opt import exess_geo_opt
from rush.exess_qmmm import exess_qmmm


def test_exess_modules_expose_module_functions():
    assert isinstance(exess, types.ModuleType)
    assert isinstance(exess_geo_opt_module, types.ModuleType)
    assert isinstance(exess_qmmm_module, types.ModuleType)

    assert getattr(exess, "exess_energy") is exess_energy
    assert getattr(exess, "exess_interaction_energy") is exess_interaction_energy
    assert exess_geo_opt_module.exess_geo_opt is exess_geo_opt
    assert exess_qmmm_module.exess_qmmm is exess_qmmm


def test_exess_modules_expose_module_specific_types():
    frag_keywords = exess.FragKeywords()
    assert frag_keywords.level == "Dimer"
    assert exess_qmmm_module.Trajectory is exess_qmmm_module.Trajectory
    assert (
        exess_geo_opt_module.OptimizationKeywords
        is exess_geo_opt_module.OptimizationKeywords
    )
