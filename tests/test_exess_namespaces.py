import types

from rush import exess
from rush.exess import (
    FragKeywords,
    OptimizationKeywords,
    Trajectory,
    energy,
    interaction_energy,
    optimization,
    qmmm,
)


def test_exess_is_a_module():
    assert isinstance(exess, types.ModuleType)


def test_exess_exposes_computation_functions():
    assert getattr(exess, "energy") is energy
    assert getattr(exess, "interaction_energy") is interaction_energy
    assert getattr(exess, "optimization") is optimization
    assert getattr(exess, "qmmm") is qmmm


def test_exess_exposes_module_specific_types():
    frag_keywords = FragKeywords()
    assert frag_keywords.level == "Dimer"
    assert exess.FragKeywords is FragKeywords
    assert exess.Trajectory is Trajectory
    assert exess.OptimizationKeywords is OptimizationKeywords
