"""Materialize the registered probe signal from realized returns."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..core.contracts import (
    CalculationRequest,
    CalculationResult,
    LoadedArtifact,
    OperationSpec,
    ParameterField,
    ParameterSchema,
    ParameterType,
    QualityCheck,
    QualityReport,
    QualityStatus,
    ScopeMode,
    Stage,
    TableOutput,
)


def _table(inputs: tuple[LoadedArtifact, ...], artifact_type: str) -> pl.DataFrame:
    matches = [item for item in inputs if item.reference.key.artifact_type == artifact_type]
    if len(matches) != 1:
        raise ValueError(f"PROBE_SIGNAL_INPUT_CARDINALITY type={artifact_type}")
    tables = matches[0].tables
    for name, frame in tables.items():
        if name.startswith("realized_returns"):
            return frame
    raise ValueError("PROBE_SIGNAL_RETURNS_TABLE_MISSING")


@dataclass(frozen=True)
class MaterializeProbeSignal:
    """Compute the frozen reversal_20d_v1 signal once upstream."""

    lake: object
    spec: OperationSpec = field(default_factory=lambda: OperationSpec(
        operation_id="materialize_probe_signal",
        version="1",
        stage=Stage.L2_PREDICTIVITY,
        input_artifact_types=("realized_returns", "tradable_universe"),
        input_artifact_versions={"realized_returns": "1", "tradable_universe": "1"},
        output_artifact_types=("probe_signal_panel",),
        output_artifact_versions={"probe_signal_panel": "1"},
        parameters=ParameterSchema(fields=(
            ParameterField(
                name="start_date", type_id=ParameterType.STRING, required=False,
                description="Inclusive signal output start date.",
            ),
            ParameterField(
                name="end_date", type_id=ParameterType.STRING, required=False,
                description="Inclusive signal output end date.",
            ),
        ), allow_extra=False),
        scope_mode=ScopeMode.UNIFIED_WITH_BOARD,
        accepts_benchmark_inputs=False,
        description=(
            "Materialize the registered reversal_20d_v1 probe signal from "
            "realized returns; formation includes the current trading day."
        ),
    ))

    def calculate(
        self,
        request: CalculationRequest,
        inputs: tuple[LoadedArtifact, ...],
    ) -> CalculationResult:
        source = _table(inputs, "realized_returns")
        required = {"trade_date", "asset_id", "realized_return"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"PROBE_SIGNAL_RETURNS_COLUMNS_MISSING {missing}")

        universe = next(item for item in inputs if item.reference.key.artifact_type == "tradable_universe")
        domain = next(iter(universe.tables.values())).select("trade_date", "asset_id")
        source = source.join(domain, on=["trade_date", "asset_id"], how="inner", validate="1:1")
        start = request.parameters.get("start_date")
        end = request.parameters.get("end_date")
        frame = (
            source.select("trade_date", "asset_id", "realized_return")
            .sort(["asset_id", "trade_date"])
            .with_columns(
                pl.when(
                    pl.col("realized_return").is_finite()
                    & (pl.col("realized_return") > -1.0)
                )
                .then(pl.col("realized_return").log1p())
                .otherwise(None)
                .alias("valid_log_return")
            )
            .with_columns(
                pl.col("valid_log_return")
                .rolling_sum(window_size=20, min_samples=15)
                .over("asset_id")
                .alias("formation_log_return")
            )
            .with_columns(
                (1.0 - pl.col("formation_log_return").exp()).alias("signal_value")
            )
            .select(
                "trade_date", "asset_id",
                pl.lit("reversal_20d_v1").alias("signal_id"),
                "signal_value",
            )
        )
        if start is not None:
            frame = frame.filter(pl.col("trade_date") >= pl.lit(start).str.to_date())
        if end is not None:
            frame = frame.filter(pl.col("trade_date") <= pl.lit(end).str.to_date())

        checks = (
            QualityCheck(
                check_id="rows_present",
                passed=frame.height > 0,
                detail="the signal panel contains at least one row",
                observed=frame.height,
            ),
            QualityCheck(
                check_id="signal_identity_frozen",
                passed=frame.get_column("signal_id").unique().to_list() == ["reversal_20d_v1"],
                detail="signal id is the registered probe definition",
                observed=frame.get_column("signal_id").unique().to_list(),
            ),
        )
        status = QualityStatus.PASSED if all(check.passed for check in checks) else QualityStatus.FAILED
        return CalculationResult(
            outputs=(TableOutput(
                artifact_type="probe_signal_panel",
                tables={"probe_signal_panel_v1": frame},
                metadata={
                    "signal_id": "reversal_20d_v1",
                    "lookback_trading_days": 20,
                    "min_valid_observations": 15,
                    "formation_window": "[t-19,t]",
                    "declared_direction": 1,
                    "parent_run_ids": [item.reference.key.run_id for item in inputs],
                },
            ),),
            quality=QualityReport(status=status, checks=checks),
        )
