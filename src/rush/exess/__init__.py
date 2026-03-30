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
    MethodT,
    BasisT,
    AuxBasisT,
    StandardOrientationT,
    TensorLike,
    ConvergenceMetricT,
    FockBuildTypeT,
    FragmentLevelT,
    CutoffTypeT,
    DistanceMetricT,
    RadialQuadT,
    PruningSchemeT,
    XCGridResolutionT,
    XCBatchingSchemeT,
    KSDFTMethodT,
    # Result types
    Calculation,
    ManyBodyExpansion,
    Nmer,
    Result,
    ResultPaths,
    ResultRef,
)

# --- Geometry optimization ---
from ._optimization import (
    # Functions
    optimization,  # noqa: F401
    # Config types
    OptimizationKeywords,
    OptimizationConvergenceCriteria,
    TrustRegionKeywords,
    LBFGSKeywords,
    CoordinateSystemT,
    HessianGuessTypeT,
    OptimizationAlgorithmTypeT,
    LBFGSLinesearchT,
    # Result types
    OptimizationResult,
    OptimizationResultPaths,
    OptimizationResultRef,
    OptimizationStep,
)

# --- QM/MM ---
from ._qmmm import (
    # Functions
    qmmm,  # noqa: F401
    # Result Types
    Trajectory,
    Restraints,
    QMMMResult,
    QMMMResultPaths,
    QMMMResultRef,
)

__all__ = [
    # Generic & Energy Types
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
    "CoordinateSystemT",
    "HessianGuessTypeT",
    "OptimizationAlgorithmTypeT",
    "LBFGSLinesearchT",
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
    "ConvergenceMetricT",
    "FockBuildTypeT",
    "FragmentLevelT",
    "CutoffTypeT",
    "DistanceMetricT",
    "RadialQuadT",
    "PruningSchemeT",
    "XCGridResolutionT",
    "XCBatchingSchemeT",
    "KSDFTMethodT",
]
