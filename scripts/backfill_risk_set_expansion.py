#!/usr/bin/env python3
"""Materialize the published L1 logical-to-physical risk-column mapping."""

from __future__ import annotations

from pathlib import Path

from factor_matrix.factor_engine import FactorRegistry, FactorRegistryStore
from factor_matrix.risk_model import load_risk_factor_set


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    store = FactorRegistryStore(PROJECT / "data/metadata/factor_registry.sqlite")
    registry = FactorRegistry.discover()
    store.sync_definitions(registry)
    risk_path = PROJECT / "config/risk_factor_set_candidate_v1.json"
    store.sync_risk_factor_set(load_risk_factor_set(risk_path, registry), registry)
    store.sync_risk_set_expansion(PROJECT / "config/risk_set_expansion_v1.json")
    print("risk_set_expansion backfilled")


if __name__ == "__main__":
    main()
