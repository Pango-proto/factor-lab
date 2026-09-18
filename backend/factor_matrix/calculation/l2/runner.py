from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import (
    NeutralizationFidelityPoint, PredictivityConfig, PredictivityPoint, ResearchEvent,
)
from .predictivity import neutralization_fidelity, rolling_out_of_sample_ic
from .research_log import ResearchEventStore


class AuditedPredictivityRunner:
    """The only supported L2a entry point; audit insertion precedes label access."""

    def __init__(
        self, event_store: ResearchEventStore, config: PredictivityConfig,
        frozen_horizons_by_factor: Mapping[str, int],
        risk_set_versions_by_factor: Mapping[str, int] | None = None,
        feature_versions_by_factor: Mapping[str, int] | None = None,
    ) -> None:
        self._event_store = event_store
        self._config = config
        self._frozen_horizons = dict(frozen_horizons_by_factor)
        self._risk_set_versions = dict(risk_set_versions_by_factor or {})
        self._feature_versions = dict(feature_versions_by_factor or {})

    def run_single_factor(
        self,
        *,
        event: ResearchEvent,
        factor_id: str,
        attempt_id: str,
        exposures_by_date: Sequence[Sequence[float]],
        forward_returns_by_date: Sequence[Sequence[float]],
    ) -> tuple[PredictivityPoint, ...]:
        if factor_id not in event.factor_ids:
            raise ValueError("L2A_EVENT_FACTOR_SCOPE_MISMATCH")
        try:
            evaluation_horizon_days = self._frozen_horizons[factor_id]
        except KeyError as exc:
            raise ValueError("L2A_FROZEN_EVALUATION_HORIZON_MISSING") from exc
        if event.payload.get("evaluation_horizon_days") != evaluation_horizon_days:
            raise ValueError("L2A_EVENT_HORIZON_DOES_NOT_MATCH_FROZEN_REGISTRY")
        risk_set_version = self._risk_set_versions.get(factor_id)
        if risk_set_version is None:
            raise ValueError("L2A_FROZEN_RISK_SET_VERSION_MISSING")
        feature_version = self._feature_versions.get(factor_id, 1)
        self._event_store.validate_attempt(
            attempt_id, feature_id=factor_id, feature_version=feature_version,
            risk_set_version=risk_set_version, horizon_days=evaluation_horizon_days,
        )
        self._event_store.append(
            event, attempt_id=attempt_id,
        )
        return rolling_out_of_sample_ic(
            exposures_by_date,
            forward_returns_by_date,
            horizon_days=evaluation_horizon_days,
            config=self._config,
        )

    def run_single_factor_with_neutralization_fidelity(
        self, *, event: ResearchEvent, factor_id: str, attempt_id: str,
        raw_exposures_by_date: Sequence[Sequence[float]],
        neutralized_exposures_by_date: Sequence[Sequence[float]],
        forward_returns_by_date: Sequence[Sequence[float]],
        minimum_absolute_raw_ic: float, minimum_retention_ratio: float,
    ) -> tuple[
        tuple[PredictivityPoint, ...], tuple[PredictivityPoint, ...],
        tuple[NeutralizationFidelityPoint, ...],
    ]:
        fidelity = event.payload.get("neutralization_fidelity")
        if not isinstance(fidelity, Mapping):
            raise ValueError("L2A_NEUTRALIZATION_FIDELITY_NOT_PREREGISTERED")
        if (
            fidelity.get("minimum_absolute_raw_ic") != minimum_absolute_raw_ic
            or fidelity.get("minimum_retention_ratio") != minimum_retention_ratio
        ):
            raise ValueError("L2A_NEUTRALIZATION_FIDELITY_CONFIG_MISMATCH")
        if factor_id not in event.factor_ids:
            raise ValueError("L2A_EVENT_FACTOR_SCOPE_MISMATCH")
        horizon = self._frozen_horizons[factor_id]
        if event.payload.get("evaluation_horizon_days") != horizon:
            raise ValueError("L2A_EVENT_HORIZON_DOES_NOT_MATCH_FROZEN_REGISTRY")
        risk_set_version = self._risk_set_versions.get(factor_id)
        if risk_set_version is None:
            raise ValueError("L2A_FROZEN_RISK_SET_VERSION_MISSING")
        self._event_store.validate_attempt(
            attempt_id, feature_id=factor_id,
            feature_version=self._feature_versions.get(factor_id, 1),
            risk_set_version=risk_set_version, horizon_days=horizon,
        )
        self._event_store.append(event, attempt_id=attempt_id)
        raw = rolling_out_of_sample_ic(
            raw_exposures_by_date, forward_returns_by_date,
            horizon_days=horizon, config=self._config,
        )
        residual = rolling_out_of_sample_ic(
            neutralized_exposures_by_date, forward_returns_by_date,
            horizon_days=horizon, config=self._config,
        )
        return raw, residual, neutralization_fidelity(
            raw, residual, minimum_absolute_raw_ic=minimum_absolute_raw_ic,
            minimum_retention_ratio=minimum_retention_ratio,
        )

    def run_factor_evaluation_snapshot(
        self, *, event: ResearchEvent, factor_id: str, attempt_id: str,
        raw_exposures_by_date: Sequence[Sequence[float]],
        neutralized_exposures_by_date: Sequence[Sequence[float]],
        incremental_exposures_by_date: Sequence[Sequence[float]],
        forward_returns_by_date: Sequence[Sequence[float]],
        minimum_absolute_raw_ic: float, minimum_retention_ratio: float,
    ) -> tuple[
        tuple[PredictivityPoint, ...], tuple[PredictivityPoint, ...],
        tuple[PredictivityPoint, ...], tuple[NeutralizationFidelityPoint, ...],
    ]:
        """Produce every label-reading admission output under one audit event."""
        snapshot = event.payload.get("factor_evaluation_snapshot")
        if not isinstance(snapshot, Mapping):
            raise ValueError("L2A_FACTOR_EVALUATION_SNAPSHOT_NOT_PREREGISTERED")
        if set(snapshot.get("outputs", ())) != {
            "raw_ic", "neutralized_ic", "neutralization_retention", "incremental_ic",
        }:
            raise ValueError("L2A_FACTOR_EVALUATION_SNAPSHOT_OUTPUTS_INCOMPLETE")
        if snapshot.get("variant_count_charge") != 1:
            raise ValueError("L2A_FACTOR_EVALUATION_SNAPSHOT_VARIANT_CHARGE_INVALID")
        fidelity = event.payload.get("neutralization_fidelity")
        if not isinstance(fidelity, Mapping) or (
            fidelity.get("minimum_absolute_raw_ic") != minimum_absolute_raw_ic
            or fidelity.get("minimum_retention_ratio") != minimum_retention_ratio
        ):
            raise ValueError("L2A_NEUTRALIZATION_FIDELITY_CONFIG_MISMATCH")
        if factor_id not in event.factor_ids:
            raise ValueError("L2A_EVENT_FACTOR_SCOPE_MISMATCH")
        horizon = self._frozen_horizons[factor_id]
        if event.payload.get("evaluation_horizon_days") != horizon:
            raise ValueError("L2A_EVENT_HORIZON_DOES_NOT_MATCH_FROZEN_REGISTRY")
        risk_set_version = self._risk_set_versions.get(factor_id)
        if risk_set_version is None:
            raise ValueError("L2A_FROZEN_RISK_SET_VERSION_MISSING")
        self._event_store.validate_attempt(
            attempt_id, feature_id=factor_id,
            feature_version=self._feature_versions.get(factor_id, 1),
            risk_set_version=risk_set_version, horizon_days=horizon,
        )
        self._event_store.append(event, attempt_id=attempt_id)
        raw = rolling_out_of_sample_ic(
            raw_exposures_by_date, forward_returns_by_date,
            horizon_days=horizon, config=self._config,
        )
        neutralized = rolling_out_of_sample_ic(
            neutralized_exposures_by_date, forward_returns_by_date,
            horizon_days=horizon, config=self._config,
        )
        incremental = rolling_out_of_sample_ic(
            incremental_exposures_by_date, forward_returns_by_date,
            horizon_days=horizon, config=self._config,
        )
        return raw, neutralized, incremental, neutralization_fidelity(
            raw, neutralized, minimum_absolute_raw_ic=minimum_absolute_raw_ic,
            minimum_retention_ratio=minimum_retention_ratio,
        )
