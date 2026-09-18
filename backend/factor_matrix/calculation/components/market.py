"""First executable operations on the calculation core.

These plugins exist to make the registry / executor / policy / publishing chain
real. Nothing had ever been registered against it before, so L1, G3 and G4 each
grew their own runner and the orchestration contract stayed on paper.

    tradable_universe + silver.returns_daily
      -> materialize_realized_returns  -> realized_returns
      -> market_return_dispersion      -> market_return_dispersion

Four decisions are worth stating, because each was reached by being wrong first.

The join runs from the universe outward, not from silver. silver.returns_daily
carries rows the research domain does not contain: 90 assets under 920xxx.BJ
codes have October 2019 returns even though the Beijing exchange did not open
until November 2021, their NEEQ-era history having been relabelled under the
later BSE code. Driving the join from silver imports them silently.

Whether a session traded is read from observation_state, not reassembled from
is_suspended and is_exchange_first_day. The universe already distinguishes
TRADED, SUSPENDED_CONFIRMED, NONTRADING_INFERRED and LISTING_DAY; rebuilding
that judgement from two boolean flags loses NONTRADING_INFERRED entirely.

Resumption days are derived here because the universe does not label them.
return_label_policy_v1 froze "resumption_day: exclude" long ago, but no code
consumed it, so the rule existed without effect. A resumption day is the first
TRADED session after a non-TRADED one: there is no comparable prior close, so
silver reports no return, and 002680.SZ on 2019-10-16 is the visible case. The
seven consecutive limit-down sessions that follow it are the reason the rule
matters rather than being a technicality: a resumption is exactly when price
discovery is most violent, and admitting a fabricated return there would inject
noise at the worst possible moment.

Suspension state is taken from the universe rather than from silver. Both carry
the flag and they disagree on a handful of rows; if downstream code is free to
pick either, two filters that look identical select different data. The
agreement rate is published so divergence is a number rather than a surprise.
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

_MINIMUM_CROSS_SECTION = 30

# The universe state meaning the session actually traded. SUSPENDED_CONFIRMED,
# NONTRADING_INFERRED and LISTING_DAY are all days on which no return exists.
TRADED_STATE = "TRADED"


def _report(checks: tuple[QualityCheck, ...]) -> QualityReport:
    status = QualityStatus.PASSED if all(check.passed for check in checks) else QualityStatus.FAILED
    return QualityReport(status=status, checks=checks)


def _resumption_flag() -> pl.Expr:
    """First TRADED session following a non-TRADED one, per asset.

    Derived from the universe's own state sequence rather than from gaps in the
    silver panel: a missing silver row is evidence of the same thing but weaker,
    since it cannot distinguish a halt from an ingestion gap. The first session
    an asset ever appears in is not a resumption, so a null predecessor yields
    false.
    """
    previous = pl.col("observation_state").shift(1).over("asset_id")
    return (
        (pl.col("observation_state") == TRADED_STATE)
        & previous.is_not_null()
        & (previous != TRADED_STATE)
    )


def _live_session() -> pl.Expr:
    """Rows that must carry a return: traded, and not a resumption."""
    return (pl.col("observation_state") == TRADED_STATE) & ~pl.col(
        "is_resumption_day"
    ).fill_null(False)


@dataclass(frozen=True)
class MaterializeRealizedReturns:
    """Project silver returns onto the research domain as a gold artifact.

    Observation state, resumption status and suspension travel with the return.
    A forward label built without them would treat a halted, unpriced or newly
    resumed day as an ordinary tradable one, so dropping any of these flags
    makes every downstream horizon subtly wrong in a way no later check catches.
    """

    lake: DataLake
    label_policy_path: Path | None = field(default=None, kw_only=True)

    spec: OperationSpec = field(default_factory=lambda: OperationSpec(
        operation_id="materialize_realized_returns",
        version="1",
        stage=Stage.L2_RETURN_DECOMPOSITION,
        input_artifact_types=("tradable_universe",),
        input_artifact_versions={"tradable_universe": "1"},
        output_artifact_types=("realized_returns",),
        output_artifact_versions={"realized_returns": "1"},
        parameters=ParameterSchema(
            fields=(
                ParameterField(
                    name="start_date", type_id=ParameterType.STRING, required=True,
                    description="Inclusive first trade date, ISO format.",
                ),
                ParameterField(
                    name="end_date", type_id=ParameterType.STRING, required=True,
                    description="Inclusive last trade date, ISO format.",
                ),
                ParameterField(
                    name="resumption_lookback_days", type_id=ParameterType.INTEGER,
                    required=False, minimum=0,
                    description=(
                        "Calendar days of universe history to read before "
                        "start_date so the first in-window session can be "
                        "classified as a resumption or not. Without it the "
                        "window's opening day has no predecessor and would be "
                        "misread as an ordinary traded session."
                    ),
                ),
            ),
            allow_extra=False,
        ),
        scope_mode=ScopeMode.UNIFIED_WITH_BOARD,
        accepts_benchmark_inputs=False,
        description=(
            "Project silver returns onto the tradable universe, carrying "
            "observation state, derived resumption status, suspension and board "
            "membership, so downstream operations consume a versioned "
            "checksummed artifact rather than reaching into silver themselves"
        ),
    ))

    def _silver_version(self) -> str:
        from ...revisioned_silver import SilverVersionLedger
        current = SilverVersionLedger(self.lake).current()
        if current:
            return str(current["version_id"])
        pointer = self.lake.root / "metadata" / "silver_versions" / "_CURRENT.json"
        if not pointer.exists():
            return "unrecorded"
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        return str(
            payload.get("silver_version_id")
            or payload.get("version_id")
            or payload.get("run_id")
            or "unrecorded"
        )

    def _label_policy(self) -> dict:
        path = resolve_return_label_policy_path(self.lake.root, self.label_policy_path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_silver(self, start: str, end: str) -> pl.DataFrame:
        path = self.lake.silver / "returns_daily" / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"SILVER_RETURNS_MISSING path={path}")
        return (
            pl.scan_parquet(path)
            .filter(
                (pl.col("trade_date") >= pl.lit(start).str.to_date())
                & (pl.col("trade_date") <= pl.lit(end).str.to_date())
            )
            .select([
                "trade_date", "asset_id",
                pl.col("total_return").alias("realized_return"),
                "return_source", "missing_return_policy",
                pl.col("is_suspended").alias("silver_is_suspended"),
            ])
            .collect()
        )

    def calculate(
        self,
        request: CalculationRequest,
        inputs: tuple[LoadedArtifact, ...],
    ) -> CalculationResult:
        if len(inputs) != 1:
            raise ValueError("REALIZED_RETURNS_SINGLE_INPUT_REQUIRED")
        tables = inputs[0].tables
        if "tradable_universe_v1" not in tables:
            raise ValueError(
                f"REALIZED_RETURNS_UNIVERSE_TABLE_MISSING observed={sorted(tables)}"
            )

        start = str(request.parameters["start_date"])
        end = str(request.parameters["end_date"])
        lookback = int(request.parameters.get("resumption_lookback_days") or 0)

        # Resumption is a property of the state sequence, so it is derived over
        # the full available history and only then cut to the requested window.
        # Deriving it inside the window would classify every asset's opening row
        # by a predecessor that was filtered away.
        full = (
            tables["tradable_universe_v1"]
            .select([
                "trade_date", "asset_id", "board_id", "observation_state",
                "is_suspended", "is_exchange_first_day",
            ])
            .sort(["asset_id", "trade_date"])
            .with_columns(_resumption_flag().alias("is_resumption_day"))
        )
        window_start = pl.lit(start).str.to_date()
        if lookback:
            window_start = window_start - pl.duration(days=lookback)
        universe = full.filter(
            (pl.col("trade_date") >= pl.lit(start).str.to_date())
            & (pl.col("trade_date") <= pl.lit(end).str.to_date())
        )
        returns = self._read_silver(start, end)

        frame = (
            universe.join(returns, on=["trade_date", "asset_id"], how="left")
            .select([
                "trade_date", "asset_id", "board_id", "realized_return",
                "observation_state", "is_resumption_day", "is_suspended",
                "is_exchange_first_day", "return_source", "missing_return_policy",
            ])
            .sort(["trade_date", "asset_id"])
        )

        duplicates = frame.height - frame.select(["trade_date", "asset_id"]).n_unique()
        expected = frame.filter(_live_session())
        unexplained_nulls = int(expected["realized_return"].null_count())

        traded = frame.filter(pl.col("observation_state") == TRADED_STATE)
        resumption_rows = int(frame["is_resumption_day"].fill_null(False).sum())
        resumption_nulls = int(
            frame.filter(pl.col("is_resumption_day").fill_null(False))[
                "realized_return"
            ].null_count()
        )
        nontrading_rows = frame.height - traded.height

        matched = universe.join(returns, on=["trade_date", "asset_id"], how="inner")
        silver_only = returns.join(
            universe.select(["trade_date", "asset_id"]),
            on=["trade_date", "asset_id"], how="anti",
        )
        suspension_agreement = None
        if matched.height:
            agree = matched.filter(
                pl.col("is_suspended").fill_null(False)
                == pl.col("silver_is_suspended").fill_null(False)
            ).height
            suspension_agreement = agree / matched.height

        policy = self._label_policy()
        resumption_action = (
            policy.get("l2b", {}).get("resumption_day", {}).get("action")
        )

        checks = (
            QualityCheck(
                check_id="rows_present",
                passed=frame.height > 0,
                detail="the requested window must contain at least one universe row",
                observed=frame.height,
            ),
            QualityCheck(
                check_id="primary_key_unique",
                passed=duplicates == 0,
                detail="trade_date plus asset_id must be unique",
                observed=duplicates,
            ),
            QualityCheck(
                check_id="live_return_present",
                passed=unexplained_nulls == 0,
                detail=(
                    "a traded session that is not a resumption must carry a "
                    "return; non-trading states and resumption days are expected "
                    "to be null and are counted separately"
                ),
                observed=unexplained_nulls,
            ),
            QualityCheck(
                check_id="resumption_policy_frozen",
                passed=resumption_action == "exclude",
                detail=(
                    "return_label_policy_v1 must still declare resumption days "
                    "excluded; this operation derives the flag on that basis and "
                    "would be silently wrong if the frozen rule changed"
                ),
                observed=resumption_action,
            ),
            QualityCheck(
                check_id="suspension_flags_agree",
                passed=suspension_agreement is None or suspension_agreement >= 0.999,
                detail=(
                    "the universe and silver suspension flags must describe the "
                    "same halts; a gap here means two filters that look "
                    "identical would select different rows"
                ),
                observed=suspension_agreement,
            ),
        )
        return CalculationResult(
            outputs=(
                TableOutput(
                    artifact_type="realized_returns",
                    tables={"realized_returns_v1": frame},
                    metadata={
                        "silver_version_id": self._silver_version(),
                        "silver_table": "returns_daily",
                        "start_date": start,
                        "end_date": end,
                        "resumption_lookback_days": lookback,
                        "domain_source": "tradable_universe",
                        "domain_source_run_id": inputs[0].reference.key.run_id,
                        "trading_state_authority": "tradable_universe.observation_state",
                        "traded_state_value": TRADED_STATE,
                        "resumption_rule": "return_label_policy_v1.l2b.resumption_day",
                        "resumption_action": resumption_action,
                        "resumption_derivation": (
                            "first TRADED session after a non-TRADED one, from the "
                            "universe state sequence; the universe publishes no "
                            "resumption state of its own"
                        ),
                        "unaddressed_silver_read": True,
                        "lineage_note": (
                            "Return values are read from silver without an "
                            "ArtifactRef. The silver version above is the only "
                            "record of what was read"
                        ),
                    },
                ),
            ),
            quality=_report(checks),
            metrics={
                "row_count": frame.height,
                "asset_count": frame["asset_id"].n_unique(),
                "date_count": frame["trade_date"].n_unique(),
                "traded_rows": traded.height,
                "nontrading_rows": nontrading_rows,
                "resumption_rows": resumption_rows,
                "resumption_null_returns": resumption_nulls,
                "suspended_rows": int(frame["is_suspended"].fill_null(False).sum()),
                "first_day_rows": int(
                    frame["is_exchange_first_day"].fill_null(False).sum()
                ),
                "silver_rows_outside_universe": silver_only.height,
                "silver_assets_outside_universe": silver_only["asset_id"].n_unique(),
                "suspension_flag_agreement": suspension_agreement,
            },
        )


@dataclass(frozen=True)
class MarketReturnDispersion:
    """Daily all-A-share cross sectional return dispersion.

    No factor model is fitted. This is the denominator many later diagnostics
    quietly assume, and computing it here first means the executor chain is
    proven on a quantity that can be recomputed by hand.

    Only traded, non-resumption sessions enter the standard deviation, using the
    same predicate as the artifact's own quality gate so the two cannot drift
    apart.
    """

    minimum_cross_section: int = _MINIMUM_CROSS_SECTION

    spec: OperationSpec = field(default_factory=lambda: OperationSpec(
        operation_id="market_return_dispersion",
        version="1",
        stage=Stage.L2_RETURN_DECOMPOSITION,
        input_artifact_types=("realized_returns",),
        input_artifact_versions={"realized_returns": "1"},
        output_artifact_types=("market_return_dispersion",),
        output_artifact_versions={"market_return_dispersion": "1"},
        parameters=ParameterSchema(fields=(), allow_extra=False),
        scope_mode=ScopeMode.UNIFIED_WITH_BOARD,
        accepts_benchmark_inputs=False,
        description=(
            "Daily all-A-share return dispersion without fitting a factor model; "
            "sigma_r is the sample standard deviation over traded sessions"
        ),
    ))

    def calculate(
        self,
        request: CalculationRequest,
        inputs: tuple[LoadedArtifact, ...],
    ) -> CalculationResult:
        if len(inputs) != 1:
            raise ValueError("MARKET_DISPERSION_SINGLE_INPUT_REQUIRED")
        tables = inputs[0].tables
        if "realized_returns_v1" not in tables:
            raise ValueError(
                f"MARKET_DISPERSION_INPUT_TABLE_MISSING observed={sorted(tables)}"
            )
        frame = (
            tables["realized_returns_v1"]
            .filter(_live_session())
            .drop_nulls("realized_return")
            .group_by("trade_date")
            .agg(
                pl.len().alias("n_valid"),
                pl.col("realized_return").std(ddof=1).alias("sigma_r"),
            )
            .sort("trade_date")
        )
        thin = frame.filter(pl.col("n_valid") < self.minimum_cross_section)
        null_sigma = int(frame["sigma_r"].null_count())
        nonpositive = int(frame.filter(pl.col("sigma_r") <= 0.0).height)
        checks = (
            QualityCheck(
                check_id="dates_present",
                passed=frame.height > 0,
                detail="at least one trade date must survive",
                observed=frame.height,
            ),
            QualityCheck(
                check_id="cross_section_sufficient",
                passed=thin.height == 0,
                detail=(
                    "every date carries at least "
                    f"{self.minimum_cross_section} valid observations"
                ),
                observed=thin.height,
            ),
            QualityCheck(
                check_id="sigma_finite",
                passed=null_sigma == 0,
                detail="sigma_r is defined on every retained date",
                observed=null_sigma,
            ),
            QualityCheck(
                check_id="sigma_positive",
                passed=nonpositive == 0,
                detail="a zero or negative dispersion means the cross section collapsed",
                observed=nonpositive,
            ),
        )
        return CalculationResult(
            outputs=(
                TableOutput(
                    artifact_type="market_return_dispersion",
                    tables={"market_return_dispersion_v1": frame},
                    metadata={
                        "estimator": "sample_standard_deviation_ddof_1",
                        "minimum_cross_section": self.minimum_cross_section,
                        "included_state": TRADED_STATE,
                        "excluded": ["non_traded_observation_state", "resumption_day"],
                    },
                ),
            ),
            quality=_report(checks),
            metrics={
                "date_count": frame.height,
                "median_n_valid": float(frame["n_valid"].median()) if frame.height else None,
                "median_sigma_r": float(frame["sigma_r"].median()) if frame.height else None,
            },
        )
