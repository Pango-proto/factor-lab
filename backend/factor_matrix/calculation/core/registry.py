from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

from .contracts import Stage
from .plugin import CalculationPlugin


class CalculationRegistry:
    def __init__(self, plugins: Iterable[CalculationPlugin] = ()) -> None:
        self._plugins: dict[tuple[str, str], CalculationPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: CalculationPlugin) -> None:
        key = (plugin.spec.operation_id, plugin.spec.version)
        if key in self._plugins:
            raise ValueError(f"CALCULATION_DUPLICATE id={key[0]} version={key[1]}")
        self._plugins[key] = plugin

    def remove(self, operation_id: str, version: str) -> None:
        key = (operation_id, version)
        if key not in self._plugins:
            raise KeyError(key)
        del self._plugins[key]

    def get(self, operation_id: str, version: str) -> CalculationPlugin:
        try:
            return self._plugins[(operation_id, version)]
        except KeyError as exc:
            raise KeyError(
                f"CALCULATION_UNKNOWN id={operation_id} version={version}"
            ) from exc

    def search(self, *, stage: Stage | None = None, query: str = "") -> tuple[OperationSpec, ...]:
        needle = query.strip().lower()
        specs = (
            plugin.spec
            for plugin in self._plugins.values()
            if stage is None or plugin.spec.stage is stage
        )
        return tuple(sorted(
            (
                spec for spec in specs
                if not needle or needle in " ".join(
                    (spec.operation_id, spec.version, spec.description, spec.stage.value)
                ).lower()
            ),
            key=lambda spec: (spec.stage.value, spec.operation_id, spec.version),
        ))

    @classmethod
    def discover(cls, packages: tuple[str, ...]) -> "CalculationRegistry":
        registry = cls()
        for package in packages:
            module = importlib.import_module(package)
            for info in pkgutil.iter_modules(module.__path__, f"{package}."):
                plugin_module = importlib.import_module(info.name)
                for plugin in getattr(plugin_module, "CALCULATION_PLUGINS", ()):
                    registry.register(plugin)
        return registry
