from __future__ import annotations

from typing import Protocol

from .contracts import FactorSpec


class FactorDefinition(Protocol):
    spec: FactorSpec
