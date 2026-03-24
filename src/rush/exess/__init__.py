"""
EXESS module for the Rush Python client.

Provides energy calculations, geometry optimization, and QM/MM simulations
via the EXESS quantum chemistry engine.

Usage::

    from rush import exess

    run = exess.energy("mol.json", method="RestrictedHF", basis="cc-pVDZ")
    result = run.fetch()
    print(result.calculation.qmmbe.expanded_hf_energy)
"""

# --- Energy (single-point, interaction energy) ---
from ._energy import (
    # Functions (accessed as exess.energy(), not directly imported)
    calculate,  # noqa: F401
    energy,  # noqa: F401
    interaction_energy,  # noqa: F401
    # Result types
    Calculation,
    ManyBodyExpansion,
    Nmer,
    Result,
    ResultPaths,
    ResultRef,
    # Config types
    Model,
    System,
    SCFKeywords,
    FragKeywords,
    KSDFTKeywords,
    ExportKeywords,
    StandardDescriptorGrid,
    DescriptorGrid,
    CustomDescriptorGrid,
    RegularDescriptorGrid,
    XCGridParameters,
    DefaultGridResolution,
    CustomGridResolution,
    ClosestAtomBatching,
    OctreeBatching,
    SpaceFillingBatching,
    GauXCBatching,
    Octree,
    # Type aliases
    MethodT,
    BasisT,
    AuxBasisT,
    StandardOrientationT,
    TensorLike,
)

# --- Geometry optimization ---
from ._optimization import (
    optimization,  # noqa: F401
    OptimizationKeywords,
    OptimizationConvergenceCriteria,
    TrustRegionKeywords,
    LBFGSKeywords,
    OptimizationResult,
    OptimizationResultPaths,
    OptimizationResultRef,
    OptimizationStep,
)

# --- QM/MM ---
from ._qmmm import (
    qmmm,  # noqa: F401
    Trajectory,
    Restraints,
    QMMMResult,
    QMMMResultPaths,
    QMMMResultRef,
)

__all__ = [
    # Config types
    "Model",
    "System",
    "SCFKeywords",
    "FragKeywords",
    "KSDFTKeywords",
    "ExportKeywords",
    "StandardDescriptorGrid",
    "DescriptorGrid",
    "CustomDescriptorGrid",
    "RegularDescriptorGrid",
    "XCGridParameters",
    "DefaultGridResolution",
    "CustomGridResolution",
    "ClosestAtomBatching",
    "OctreeBatching",
    "SpaceFillingBatching",
    "GauXCBatching",
    "Octree",
    # Result types
    "Calculation",
    "ManyBodyExpansion",
    "Nmer",
    "Result",
    "ResultPaths",
    "ResultRef",
    # Optimization types
    "OptimizationKeywords",
    "OptimizationConvergenceCriteria",
    "TrustRegionKeywords",
    "LBFGSKeywords",
    "OptimizationResult",
    "OptimizationResultPaths",
    "OptimizationResultRef",
    "OptimizationStep",
    # QM/MM types
    "Trajectory",
    "Restraints",
    "QMMMResult",
    "QMMMResultPaths",
    "QMMMResultRef",
    # Type aliases
    "MethodT",
    "BasisT",
    "AuxBasisT",
    "StandardOrientationT",
    "TensorLike",
]
