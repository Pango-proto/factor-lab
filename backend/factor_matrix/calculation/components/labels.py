"""Forward return labels on the unified market calendar.

The horizon is counted in market trading days, not in an asset's own trading
days. A five day label spans the same five calendar sessions for every asset,
so a suspended stretch inside the window shortens nothing: the position was
locked, and zero is the honest return for those days. Counting only an asset's
own sessions would silently give different assets different holding periods and
compute IC across a mixture of them.

return_label_policy_v1 governs three cases and all three are consumed here:

    label:              next_trading_day_realized_return
    exchange_first_day: exclude
    resumption_day:     exclude
    suspension_day:     retain_if_prior_price_known

"retain" is the operative word for suspensions. Silver already records a halted
session as a zero return once a prior close is known, so the day stays in the
compounding path rather than being skipped.

Resumptions are excluded from the path, not merely from the start. A resumption
gap is unobservable: silver reports no return on the resuming session because
there is no comparable prior close, yet a holder would have absorbed the entire
jump. 002680.SZ resumed on 2019-10-16 and then fell the limit for seven straight
sessions. Suspensions are usually triggered by bad news, so admitting windows
that straddle a resumption would understate losses in one direction only. That
is a survivorship-shaped bias, not noise, and no later diagnostic would separate
it from genuine predictability. Excluding those windows costs a small number of
observations, reported in the manifest, and buys a label with no one-sided leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ...storage import DataLake
from ..l2.label_policy import resolve_return_label_policy_path
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

TRADED_STATE = "TRADED"

DEFAULT_HORIZONS = (1, 3, 5, 10, 20)

# A window more than half suspended produces a label near zero by construction
# rather than by prediction. The threshold is a development-period choice and
# belongs in the sensitivity list, not in the set of settled parameters.
DEFAULT_MAX_SUSPENDED_FRACTION = 0.5


@dataclass(frozen=True)
class BuildForwardLabels:
    """Compound realized returns forward over a fixed market-day horizon."""

    lake: DataLake
    label_policy_path: Path | None = field(default=None, kw_only=True)

    spec: OperationSpec = field(default_factory=lambda: OperationSpec(
        operation_id="build_forward_labels",
        version="1",
        stage=Stage.L2_PREDICTIVITY,
        input_artifact_types=("realized_returns", "tradable_universe"),
        input_artifact_versions={"realized_returns": "1", "tradable_universe": "1"},
        output_artifact_types=("forward_labels",),
        output_artifact_versions={"forward_labels": "1"},
        parameters=ParameterSchema(
            fields=(
                ParameterField(
                    name="horizons", type_id=ParameterType.INTEGER_LIST, required=False,
                    description=(
                        "Forward horizons in market trading days. Defaults to "
                        "1, 3, 5, 10 and 20."
                    ),
                ),
                ParameterField(
                    name="max_suspended_fraction", type_id=ParameterType.NUMBER,
                    required=False, minimum=0.0, maximum=1.0,
                    description=(
                        "Null the label when more than this fraction of the "
                        "window was suspended. Development-period choice."
                    ),
                ),
            ),
            allow_extra=False,
        ),
        scope_mode=ScopeMode.UNIFIED_WITH_BOARD,
        accepts_benchmark_inputs=False,
        description=(
            "Forward realized returns compounded over a fixed number of market "
            "trading days, with suspensions retained at zero and any window "
            "touching a resumption excluded"
        ),
    ))

    def _label_policy(self) -> dict:
        path = resolve_return_label_policy_path(self.lake.root, self.label_policy_path)
        return json.loads(path.read_text(encoding="utf-8"))

    def calculate(
        self,
        request: CalculationRequest,
        inputs: tuple[LoadedArtifact, ...],
    ) -> CalculationResult:
        by_type = {loaded.reference.key.artifact_type: loaded for loaded in inputs}
        missing = {"realized_returns", "tradable_universe"} - set(by_type)
        if missing:
            raise ValueError(
                f"FORWARD_LABELS_INPUT_MISSING types={','.join(sorted(missing))}"
            )
        returns = by_type["realized_returns"].tables["realized_returns_v1"]

        horizons = tuple(
            int(value) for value in (request.parameters.get("horizons") or DEFAULT_HORIZONS)
        )
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError(f"FORWARD_LABEL_HORIZON_INVALID horizons={horizons}")
        max_suspended = float(
            request.parameters.get("max_suspended_fraction")
            if request.parameters.get("max_suspended_fraction") is not None
            else DEFAULT_MAX_SUSPENDED_FRACTION
        )

        # The market calendar is the union of dates present in the panel, so the
        # horizon is the same span for every asset. Reindexing each asset onto it
        # makes a suspended or absent session an explicit row rather than an
        # implicit shortening of the window.
        calendar = returns.select("trade_date").unique().sort("trade_date")
        assets = returns.select("asset_id").unique()
        grid = calendar.join(assets, how="cross")

        panel = (
            grid.join(returns, on=["trade_date", "asset_id"], how="left")
            .with_columns(
                pl.col("observation_state").fill_null("ABSENT"),
                pl.col("is_resumption_day").fill_null(False),
                # A suspended session compounds at zero: the holder earned
                # nothing, which is different from having no observation.
                pl.when(pl.col("observation_state") == TRADED_STATE)
                .then(pl.col("realized_return"))
                .otherwise(0.0)
                .alias("path_return"),
                (pl.col("observation_state") != TRADED_STATE).alias("is_halted"),
            )
            .sort(["asset_id", "trade_date"])
        )

        # Any missing return on a traded session would poison the product, so it
        # is tracked and must be zero before the label is trusted.
        unusable = int(
            panel.filter(
                (pl.col("observation_state") == TRADED_STATE)
                & pl.col("realized_return").is_null()
                & ~pl.col("is_resumption_day")
            ).height
        )

        frames: list[pl.DataFrame] = []
        coverage: dict[str, float] = {}
        for horizon in horizons:
            offsets = range(1, horizon + 1)
            growth = [
                (pl.lit(1.0) + pl.col("path_return").shift(-offset).over("asset_id"))
                for offset in offsets
            ]
            product = growth[0]
            for factor in growth[1:]:
                product = product * factor

            halted_in_window = pl.sum_horizontal([
                pl.col("is_halted").shift(-offset).over("asset_id").cast(pl.Int32)
                for offset in offsets
            ])
            resumption_in_window = pl.max_horizontal([
                pl.col("is_resumption_day").shift(-offset).over("asset_id")
                for offset in offsets
            ])
            window_complete = (
                pl.col("path_return").shift(-horizon).over("asset_id").is_not_null()
            )

            labelled = (
                panel.with_columns(
                    (product - pl.lit(1.0)).alias("raw_label"),
                    halted_in_window.alias("halted_days"),
                    resumption_in_window.alias("window_has_resumption"),
                    window_complete.alias("window_complete"),
                )
                .with_columns(
                    pl.when(
                        # Start must be a genuine traded, non-resuming session.
                        (pl.col("observation_state") == TRADED_STATE)
                        & ~pl.col("is_resumption_day")
                        & pl.col("window_complete")
                        & ~pl.col("window_has_resumption").fill_null(False)
                        & (
                            pl.col("halted_days").cast(pl.Float64) / float(horizon)
                            <= max_suspended
                        )
                    )
                    .then(pl.col("raw_label"))
                    .otherwise(None)
                    .alias("target_return")
                )
                .filter(pl.col("observation_state") == TRADED_STATE)
                .with_columns(
                    pl.lit(f"h{horizon}d").alias("horizon_id"),
                    pl.col("halted_days").cast(pl.Int64),
                )
                .select([
                    "trade_date", "asset_id", "horizon_id", "target_return",
                    "halted_days", "window_has_resumption", "window_complete",
                ])
            )
            frames.append(labelled)
            eligible = labelled.filter(pl.col("window_complete"))
            coverage[f"h{horizon}d"] = (
                float(eligible.filter(pl.col("target_return").is_not_null()).height)
                / eligible.height
                if eligible.height else 0.0
            )

        frame = pl.concat(frames).sort(["trade_date", "asset_id", "horizon_id"])

        duplicates = (
            frame.height
            - frame.select(["trade_date", "asset_id", "horizon_id"]).n_unique()
        )
        resumption_dropped = int(
            frame.filter(
                pl.col("window_complete")
                & pl.col("window_has_resumption").fill_null(False)
            ).height
        )
        extreme = int(
            frame.filter(pl.col("target_return").abs() > 10.0).height
        )

        policy = self._label_policy().get("l2b", {})
        actions = (
            policy.get("resumption_day", {}).get("action"),
            policy.get("suspension_day", {}).get("action"),
        )
        policy_intact = actions == ("exclude", "retain_if_prior_price_known")

        checks = (
            QualityCheck(
                check_id="rows_present",
                passed=frame.height > 0,
                detail="at least one labelled row must survive",
                observed=frame.height,
            ),
            QualityCheck(
                check_id="primary_key_unique",
                passed=duplicates == 0,
                detail="trade_date, asset_id and horizon_id must be unique",
                observed=duplicates,
            ),
            QualityCheck(
                check_id="no_unusable_traded_return",
                passed=unusable == 0,
                detail=(
                    "a traded non-resuming session with a null return would "
                    "silently zero out every window containing it"
                ),
                observed=unusable,
            ),
            QualityCheck(
                check_id="label_policy_intact",
                passed=policy_intact,
                detail=(
                    "return_label_policy_v1 must still exclude resumption days "
                    "and retain suspensions; this operation is built on that "
                    "reading and would be silently wrong if it changed"
                ),
                observed=actions,
            ),
            QualityCheck(
                check_id="labels_not_absurd",
                passed=extreme == 0,
                detail=(
                    "a forward return above 1000 percent points at a compounding "
                    "or alignment error rather than at a real move"
                ),
                observed=extreme,
            ),
        )
        status = (
            QualityStatus.PASSED
            if all(check.passed for check in checks)
            else QualityStatus.FAILED
        )
        return CalculationResult(
            outputs=(
                TableOutput(
                    artifact_type="forward_labels",
                    tables={"forward_labels_v1": frame},
                    metadata={
                        "horizons": list(horizons),
                        "horizon_calendar": "unified_market_trading_days",
                        "suspension_treatment": "retained_at_zero_return",
                        "resumption_treatment": "window_excluded_entirely",
                        "max_suspended_fraction": max_suspended,
                        "development_period_choices": [
                            "max_suspended_fraction",
                            "resumption_excluded_from_path_not_only_from_start",
                        ],
                        "resumption_windows_dropped": resumption_dropped,
                        "policy_actions": {
                            "resumption_day": actions[0],
                            "suspension_day": actions[1],
                        },
                        "upstream_realized_returns_run_id": (
                            by_type["realized_returns"].reference.key.run_id
                        ),
                        "upstream_universe_run_id": (
                            by_type["tradable_universe"].reference.key.run_id
                        ),
                        "known_residual": (
                            "Returns across a resumption gap are unobservable in "
                            "silver, so those windows carry no label. Suspensions "
                            "skew toward bad news, so retaining them would "
                            "understate losses in one direction only"
                        ),
                    },
                ),
            ),
            quality=QualityReport(status=status, checks=checks),
            metrics={
                "row_count": frame.height,
                "asset_count": frame["asset_id"].n_unique(),
                "date_count": frame["trade_date"].n_unique(),
                "labelled_rows": int(frame["target_return"].is_not_null().sum()),
                "resumption_windows_dropped": resumption_dropped,
                "label_coverage_by_horizon": coverage,
            },
        )
