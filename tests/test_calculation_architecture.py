from datetime import date

import polars as pl
import pytest

from factor_matrix.calculation import (
    ArtifactKey,
    ArtifactRef,
    CalculationExecutor,
    CalculationRegistry,
    CalculationRequest,
    CalculationResult,
    OperationSpec,
    ParameterField,
    ParameterSchema,
    ParameterType,
    QualityReport,
    QualityStatus,
    ScopeMode,
    Stage,
    TableOutput,
    architecture_boundary_policy,
)
from factor_matrix.calculation.core.contracts import LoadedArtifact
from factor_matrix.calculation.publishing import LocalArtifactStore
from factor_matrix.calculation.components import (
    AlphaExposurePlan,
    ComponentPlanRegistry,
    ComponentPlanSpec,
)
from factor_matrix.calculation.artifacts import (
    ArtifactSchemaRegistry,
    architecture_artifact_schemas,
)
from factor_matrix.factor_engine import FactorFamily, FactorRegistry
from factor_matrix.storage import DataLake


AS_OF = date(2026, 8, 14)


def reference(artifact_type: str, *, tags: frozenset[str] = frozenset()) -> ArtifactRef:
    return ArtifactRef(
        key=ArtifactKey(artifact_type, "1", f"{artifact_type}_run", AS_OF, "board:MAIN"),
        uri=f"memory://{artifact_type}",
        checksum="fixture",
        quality_status=QualityStatus.PASSED,
        tags=tags,
    )


class ExposureFixture:
    spec = OperationSpec(
        operation_id="fixture_exposure",
        version="1",
        stage=Stage.EXPOSURE,
        input_artifact_types=("base_matrix", "tradable_universe"),
        input_artifact_versions={"base_matrix": "1", "tradable_universe": "1"},
        output_artifact_types=("risk_exposure_matrix",),
        output_artifact_versions={"risk_exposure_matrix": "1"},
        parameters=ParameterSchema(
            fields=(ParameterField(
                name="transformer_id",
                type_id=ParameterType.STRING,
                required=True,
                description="Explicit transform implementation",
            ),),
            allow_extra=False,
        ),
        scope_mode=ScopeMode.PER_BOARD,
        accepts_benchmark_inputs=False,
        description="test fixture",
    )

    def calculate(self, request, inputs):
        assert request.parameters["transformer_id"] == "test_transform"
        assert {item.reference.key.artifact_type for item in inputs} == {
            "base_matrix", "tradable_universe"
        }
        return CalculationResult(
            outputs=(TableOutput(
                artifact_type="risk_exposure_matrix",
                tables={"values": pl.DataFrame({"asset_id": ["A"]})},
            ),),
            quality=QualityReport(status=QualityStatus.PASSED, checks=()),
        )


class AlphaExposureFixture:
    spec = OperationSpec(
        operation_id="fixture_alpha_exposure",
        version="1",
        stage=Stage.EXPOSURE,
        input_artifact_types=(
            "base_matrix", "tradable_universe", "risk_exposure_matrix", "risk_factor_set",
        ),
        input_artifact_versions={
            "base_matrix": "1", "tradable_universe": "1",
            "risk_exposure_matrix": "1", "risk_factor_set": "1",
        },
        output_artifact_types=("exposure_matrix",),
        output_artifact_versions={"exposure_matrix": "1"},
        parameters=ParameterSchema(fields=(), allow_extra=False),
        scope_mode=ScopeMode.PER_BOARD,
        accepts_benchmark_inputs=False,
        description="alpha projection fixture",
    )

    def calculate(self, request, inputs):
        return CalculationResult(
            outputs=(TableOutput(
                artifact_type="exposure_matrix",
                tables={"values": pl.DataFrame({"asset_id": ["A"]})},
            ),),
            quality=QualityReport(status=QualityStatus.PASSED, checks=()),
        )


class Loader:
    def load(self, item):
        return LoadedArtifact(item, {"values": pl.DataFrame()})


class Publisher:
    def publish(self, spec, request, result):
        return (reference(result.outputs[0].artifact_type),)


def test_registry_is_indexable_by_stage_and_version() -> None:
    registry = CalculationRegistry([ExposureFixture()])
    assert registry.get("fixture_exposure", "1").spec.stage is Stage.EXPOSURE
    assert [spec.operation_id for spec in registry.search(stage=Stage.EXPOSURE)] == [
        "fixture_exposure"
    ]
    assert registry.search(stage=Stage.L2_RISK_MODEL) == ()


def test_executor_requires_explicit_parameters_and_board_scope() -> None:
    executor = CalculationExecutor(
        CalculationRegistry([ExposureFixture()]), Loader(), Publisher(),
        architecture_boundary_policy(),
    )
    inputs = (reference("base_matrix"), reference("tradable_universe"))
    with pytest.raises(ValueError, match="PARAMETER_REQUIRED"):
        executor.execute(CalculationRequest(
            "fixture_exposure", "1", AS_OF, "board:MAIN", {}, inputs
        ))
    with pytest.raises(ValueError, match="BOARD_SCOPE_REQUIRED"):
        executor.execute(CalculationRequest(
            "fixture_exposure", "1", AS_OF, "market:ALL",
            {"transformer_id": "test_transform"}, inputs,
        ))
    outputs = executor.execute(CalculationRequest(
        "fixture_exposure", "1", AS_OF, "board:MAIN",
        {"transformer_id": "test_transform"}, inputs,
    ))
    assert outputs[0].key.artifact_type == "risk_exposure_matrix"


def test_benchmark_only_artifact_cannot_enter_unapproved_calculation() -> None:
    executor = CalculationExecutor(
        CalculationRegistry([ExposureFixture()]), Loader(), Publisher(),
        architecture_boundary_policy(),
    )
    with pytest.raises(RuntimeError, match="BENCHMARK_ONLY_INPUT_REJECTED"):
        executor.execute(CalculationRequest(
            "fixture_exposure", "1", AS_OF, "board:MAIN",
            {"transformer_id": "test_transform"},
            (
                reference("base_matrix", tags=frozenset({"benchmark_only"})),
                reference("tradable_universe"),
            ),
        ))


def test_alpha_exposure_requires_a_frozen_risk_factor_set() -> None:
    executor = CalculationExecutor(
        CalculationRegistry([AlphaExposureFixture()]), Loader(), Publisher(),
        architecture_boundary_policy(),
    )
    inputs = (
        reference("base_matrix"), reference("tradable_universe"),
        reference("risk_exposure_matrix"), reference("risk_factor_set"),
    )
    with pytest.raises(RuntimeError, match="FROZEN_RISK_FACTOR_SET_REQUIRED"):
        executor.execute(CalculationRequest(
            "fixture_alpha_exposure", "1", AS_OF, "board:MAIN", {}, inputs
        ))


def test_definition_registry_is_unified_but_separate_from_calculation_registry() -> None:
    definitions = FactorRegistry.discover()
    assert definitions.factor_ids(FactorFamily.ALPHA) == ()
    assert "size" in definitions.factor_ids(FactorFamily.RISK)
    assert CalculationRegistry().search() == ()


def test_documented_artifact_contracts_are_versioned_and_indexable() -> None:
    registry = ArtifactSchemaRegistry(architecture_artifact_schemas())
    assert registry.get("exposure_matrix", "1").dynamic_column_prefixes == (
        "exposure_", "is_imputed_",
    )
    assert registry.get("stock_score", "1").primary_key == (
        "trade_date", "asset_id", "portfolio_size_id",
    )
    assert registry.get("benchmark_factor_returns", "1").description.endswith(
        "tag benchmark_only"
    )
    # Assert the executable chain's contracts rather than freezing the catalog
    # size: adding a versioned artifact must not break unrelated consumers.
    for artifact_type in architecture_boundary_policy().producer_stage_by_artifact_type:
        schema = registry.get(artifact_type, "1")
        assert schema.primary_key
        assert all(
            not column.nullable for column in schema.columns
            if column.name in schema.primary_key
        )
    assert registry.get("probe_signal_panel", "1").primary_key == (
        "trade_date", "asset_id", "signal_id",
    )


def test_local_publisher_round_trips_an_immutable_artifact(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    store = LocalArtifactStore(lake)
    spec = OperationSpec(
        operation_id="fixture_publish",
        version="1",
        stage=Stage.L2_PREDICTIVITY,
        input_artifact_types=(),
        input_artifact_versions={},
        output_artifact_types=("factor_predictivity",),
        output_artifact_versions={"factor_predictivity": "1"},
        parameters=ParameterSchema(fields=(), allow_extra=False),
        scope_mode=ScopeMode.CONFIGURABLE,
        accepts_benchmark_inputs=False,
        description="publisher fixture",
    )
    request = CalculationRequest("fixture_publish", "1", AS_OF, "board:MAIN", {}, ())
    result = CalculationResult(
        outputs=(TableOutput(
            artifact_type="factor_predictivity",
            tables={"values": pl.DataFrame({"value": [1.0]})},
        ),),
        quality=QualityReport(status=QualityStatus.PASSED, checks=()),
    )
    published = store.publish(spec, request, result)
    loaded = store.load(published[0])
    assert loaded.tables["values"].get_column("value").to_list() == [1.0]


def test_component_plans_can_be_added_removed_and_versioned() -> None:
    plan = ComponentPlanSpec(
        plan_id="research_exposure",
        version="draft-1",
        stage=Stage.EXPOSURE,
        configuration=AlphaExposurePlan(
            alpha_factor_ids=("factor_a",),
            frozen_risk_factor_set_artifact_type="risk_factor_set",
            frozen_risk_factor_set_version="1",
            raw_builder_ids=(("factor_a", "builder_a"),),
            imputer_id="explicit_imputer",
            outlier_handler_id="explicit_outlier_handler",
            initial_scaler_id="explicit_scaler",
            weighted_projection_solver_id="explicit_projection",
            weight_model_id="explicit_weight_model",
            residual_scaler_id="explicit_residual_scaler",
            residual_centering_allowed=False,
            board_scope_policy_id="explicit_board_policy",
        ),
        description="No numerical method is selected by the framework",
    )
    registry = ComponentPlanRegistry([plan])
    assert registry.get("research_exposure", "draft-1") == plan
    registry.remove("research_exposure", "draft-1")
    assert registry.search() == ()


def test_alpha_residual_cannot_be_recentered() -> None:
    with pytest.raises(ValueError, match="ALPHA_RESIDUAL_RECENTERING_FORBIDDEN"):
        AlphaExposurePlan(
            alpha_factor_ids=("factor_a",),
            frozen_risk_factor_set_artifact_type="risk_factor_set",
            frozen_risk_factor_set_version="1",
            raw_builder_ids=(("factor_a", "builder_a"),),
            imputer_id="imputer",
            outlier_handler_id="outlier",
            initial_scaler_id="initial_scaler",
            weighted_projection_solver_id="projection",
            weight_model_id="weight",
            residual_scaler_id="scale_only",
            residual_centering_allowed=True,
            board_scope_policy_id="board_policy",
        )


def test_boundary_policy_prevents_a_risk_plugin_from_publishing_alpha() -> None:
    bad_spec = OperationSpec(
        operation_id="bad_risk",
        version="1",
        stage=Stage.L2_RISK_MODEL,
        input_artifact_types=(),
        input_artifact_versions={},
        output_artifact_types=("alpha_forecast",),
        output_artifact_versions={"alpha_forecast": "1"},
        parameters=ParameterSchema(fields=(), allow_extra=False),
        scope_mode=ScopeMode.CONFIGURABLE,
        accepts_benchmark_inputs=False,
        description="invalid cross-layer output",
    )
    with pytest.raises(ValueError, match="ARTIFACT_STAGE_CONFLICT"):
        architecture_boundary_policy().validate_spec(bad_spec)


def test_exposure_contract_cannot_omit_tradable_universe() -> None:
    bad_spec = OperationSpec(
        operation_id="exposure_without_domain",
        version="1",
        stage=Stage.EXPOSURE,
        input_artifact_types=("base_matrix",),
        input_artifact_versions={"base_matrix": "1"},
        output_artifact_types=("risk_exposure_matrix",),
        output_artifact_versions={"risk_exposure_matrix": "1"},
        parameters=ParameterSchema(fields=(), allow_extra=False),
        scope_mode=ScopeMode.PER_BOARD,
        accepts_benchmark_inputs=False,
        description="invalid missing tradable domain",
    )
    with pytest.raises(ValueError, match="ARTIFACT_REQUIRED_INPUT_MISSING"):
        architecture_boundary_policy().validate_spec(bad_spec)
