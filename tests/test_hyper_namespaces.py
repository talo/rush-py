import types

from rush import hyper
from rush.hyper import (
    HyperConfig,
    HyperMinimizeConfig,
    HyperRunConfig,
    MinimizeInput,
    RunInput,
    RunOutput,
    hyper_minimize_sumo,
    hyper_run_sumo,
    hyper_solvate_sumo,
)


def test_hyper_is_a_module():
    assert isinstance(hyper, types.ModuleType)


def test_hyper_exposes_entrypoint_wrappers():
    assert getattr(hyper, "hyper_solvate_sumo") is hyper_solvate_sumo
    assert getattr(hyper, "hyper_minimize_sumo") is hyper_minimize_sumo
    assert getattr(hyper, "hyper_run_sumo") is hyper_run_sumo


def test_hyper_exposes_module_specific_types():
    assert hyper.HyperConfig is HyperConfig
    assert hyper.HyperMinimizeConfig is HyperMinimizeConfig
    assert hyper.HyperRunConfig is HyperRunConfig
    assert hyper.MinimizeInput is MinimizeInput
    assert hyper.RunInput is RunInput
    assert hyper.RunOutput is RunOutput
