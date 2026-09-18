"""L2 is the only calculation package allowed to consume return labels."""

from .config import load_predictivity_config, load_return_decomposition_config
from .contracts import (
    ALLOWED_PREREGISTERED_GROUPS, DailyDesign, FamaMacBethPoint, L2Chain,
    NeutralizationFidelityPoint, PredictivityConfig, PredictivityPoint,
    RegressionInput, RegressionMode, RegressionResult, ResearchEvent,
    ReturnDecompositionConfig,
)
from .predictivity import (
    benjamini_hochberg_passes, descriptive_ic_term_structure,
    estimate_signal_half_life, fdr_denominator, newey_west_mean,
    neutralization_fidelity, normalize_diagonal_icir_weights,
    purged_k_fold_indices, rolling_out_of_sample_ic, spearman_ic,
    weighted_residualize_against_frozen_composite,
    classify_alpha_decay_or_riskification,
    rolling_fama_macbeth,
)
from .design_matrix import (
    build_daily_categorical_constrained_design, build_daily_industry_constrained_design,
)
from .g3_runner import load_current_g3_manifest, run_g3_risk_only
from .g3_history_runner import run_g3_risk_only_history
from .g3_reconciliation import bind_g3_reconciliation_to_current, run_g3_reconciliation
from .run_reconciliation import run_g3_run_reconciliation
from .g3_attestation import (
    bind_g3_attestation_to_current, run_g3_canonicalization_attestation,
)
from .g4_diagnostics import run_g4_attribution_diagnostics
from .g4_spectrum import run_g4_statistical_factor_spectrum
from .g4_alpha_overlap import run_g4_statistical_alpha_overlap
from .g4_style_semantic_probe import run_g4_style_loading_semantic_probe
from .g4_weight_selection import run_g4_wls_weight_selection
from .g4_weight_freeze import freeze_and_publish_g4_wls_weight
from .g4_structural_residual import run_g4_structural_weight_residual
from .g4_3_residual_permutation import run_g4_3
from .g4_3_residual_permutation_v2_runner import run_g4_3_v2
from .pit_contract_e2e import run_pit_contract_e2e
from .research_log import ResearchEventStore, SemanticProbeLedger, configuration_sha
from .return_decomposition import (
    attribution_identity_error, decompose_cross_section, grouped_residual_correlation,
)
from .runner import AuditedPredictivityRunner
from .publishing import L2AtomicPublisher, load_current_l2_product, validate_l2b_products
from .parquet_runner import L2BParquetInputs, run_l2b_from_parquet
from .risk_modeling import (
    BiasTestResult, ResidualCorrelationResult, bias_test, ewma_factor_covariance,
    ewma_specific_variance, risk_set_freeze_decision,
    specific_variance_winsorization_comparison,
    summarize_specific_risk_ratio_by_group,
    stratified_residual_permutation_test,
)
from .validator import cross_chain_magnitude_validation
from .label_policy import (
    ReturnLabelPolicy, apply_l2b_label_policy, load_return_label_policy,
)

__all__ = [
    "ALLOWED_PREREGISTERED_GROUPS", "AuditedPredictivityRunner", "BiasTestResult",
    "DailyDesign", "FamaMacBethPoint",
    "L2AtomicPublisher", "L2BParquetInputs", "L2Chain", "NeutralizationFidelityPoint", "PredictivityConfig",
    "PredictivityPoint", "RegressionInput", "RegressionMode", "RegressionResult",
    "ResearchEvent", "ResearchEventStore", "SemanticProbeLedger", "ResidualCorrelationResult",
    "ReturnDecompositionConfig", "ReturnLabelPolicy",
    "apply_l2b_label_policy", "attribution_identity_error", "benjamini_hochberg_passes",
    "bias_test", "build_daily_industry_constrained_design", "configuration_sha",
    "build_daily_categorical_constrained_design", "run_g3_risk_only",
    "run_g3_risk_only_history",
    "run_g3_reconciliation",
    "bind_g3_reconciliation_to_current",
    "run_g3_run_reconciliation",
    "run_g3_canonicalization_attestation",
    "bind_g3_attestation_to_current",
    "run_g4_attribution_diagnostics",
    "run_g4_statistical_factor_spectrum",
    "run_g4_statistical_alpha_overlap",
    "run_g4_style_loading_semantic_probe",
    "run_g4_wls_weight_selection",
    "freeze_and_publish_g4_wls_weight",
    "run_g4_structural_weight_residual",
    "run_g4_3",
    "run_g4_3_v2",
    "cross_chain_magnitude_validation",
    "classify_alpha_decay_or_riskification",
    "decompose_cross_section", "descriptive_ic_term_structure",
    "estimate_signal_half_life", "fdr_denominator",
    "grouped_residual_correlation",
    "load_current_l2_product", "load_predictivity_config", "load_return_label_policy",
    "load_current_g3_manifest",
    "load_return_decomposition_config", "newey_west_mean", "purged_k_fold_indices",
    "ewma_factor_covariance", "ewma_specific_variance", "risk_set_freeze_decision",
    "specific_variance_winsorization_comparison",
    "summarize_specific_risk_ratio_by_group",
    "neutralization_fidelity", "normalize_diagonal_icir_weights", "rolling_fama_macbeth",
    "rolling_out_of_sample_ic", "run_l2b_from_parquet", "spearman_ic",
    "stratified_residual_permutation_test", "validate_l2b_products",
    "weighted_residualize_against_frozen_composite",
]
