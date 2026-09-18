from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

from .contracts import FactorFamily, FactorRole, FactorStatus, FeatureRole
from .plugin import FactorDefinition


class FactorRegistry:
    """In-memory feature definitions; role assertions are persisted separately."""

    def __init__(self, plugins: Iterable[FactorDefinition] = ()) -> None:
        self._plugins: dict[tuple[str, int], FactorDefinition] = {}
        self._feature_keys: dict[str, tuple[str, int]] = {}
        for plugin in plugins:
            self.register(plugin)
        self.validate_graph()

    def register(self, plugin: FactorDefinition) -> None:
        spec = plugin.spec
        key = (spec.factor_id, spec.version)
        if key in self._plugins:
            raise ValueError(f"FACTOR_DUPLICATE id={spec.factor_id} version={spec.version}")
        existing = self._feature_keys.get(spec.feature_key)
        if existing is not None:
            raise ValueError(
                f"FACTOR_FEATURE_KEY_DUPLICATE existing={existing[0]}@{existing[1]} "
                f"new={spec.factor_id}@{spec.version}"
            )
        self._plugins[key] = plugin
        self._feature_keys[spec.feature_key] = key

    def remove(self, factor_id: str, version: int) -> None:
        plugin = self.get(factor_id, version)
        if plugin.spec.status not in {
            FactorStatus.DRAFT, FactorStatus.TESTING, FactorStatus.BLOCKED,
        }:
            raise ValueError("FROZEN_FACTOR_IMMUTABLE")
        del self._plugins[(factor_id, version)]
        del self._feature_keys[plugin.spec.feature_key]

    def get(self, factor_id: str, version: int | None = None) -> FactorDefinition:
        versions = sorted(v for candidate_id, v in self._plugins if candidate_id == factor_id)
        if version is None:
            if len(versions) != 1:
                raise KeyError(f"FACTOR_VERSION_REQUIRED id={factor_id}")
            version = versions[0]
        try:
            return self._plugins[(factor_id, version)]
        except KeyError as exc:
            raise KeyError(f"FACTOR_UNKNOWN id={factor_id} version={version}") from exc

    def factor_ids(self, family: FactorFamily | None = None) -> tuple[str, ...]:
        allowed_roles = None
        if family is FactorFamily.RISK:
            allowed_roles = {
                FactorRole.COUNTRY, FactorRole.INDUSTRY, FactorRole.BOARD,
                FactorRole.MEMBERSHIP, FactorRole.STYLE_RISK,
            }
        elif family is FactorFamily.ALPHA:
            allowed_roles = {FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR}
        return tuple(sorted({
            spec.factor_id for spec in (plugin.spec for plugin in self._plugins.values())
            if allowed_roles is None or spec.role in allowed_roles
        }))

    def search(
        self, *, family: FactorFamily | None = None, role: FactorRole | None = None,
        status: FactorStatus | None = None,
    ) -> tuple[FactorDefinition, ...]:
        allowed_roles = None
        if family is FactorFamily.RISK:
            allowed_roles = {
                FactorRole.COUNTRY, FactorRole.INDUSTRY, FactorRole.BOARD,
                FactorRole.MEMBERSHIP, FactorRole.STYLE_RISK,
            }
        elif family is FactorFamily.ALPHA:
            allowed_roles = {FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR}
        return tuple(
            plugin for _, plugin in sorted(self._plugins.items())
            if (allowed_roles is None or plugin.spec.role in allowed_roles)
            and (role is None or plugin.spec.role is role)
            and (status is None or plugin.spec.status is status)
        )

    def search_feature_role(
        self, feature_role: FeatureRole, *, status: FactorStatus | None = None,
    ) -> tuple[FactorDefinition, ...]:
        """Adapter view; persistent role membership lives in separate tables."""
        if feature_role is FeatureRole.BASIS:
            return self.search(family=FactorFamily.RISK, status=status)
        if feature_role is FeatureRole.BETTABLE:
            return self.search(role=FactorRole.ALPHA_CANDIDATE, status=status)
        if feature_role is FeatureRole.PROBE:
            return self.search(role=FactorRole.DESCRIPTOR, status=status)
        return tuple(
            plugin for plugin in self.search(status=status)
            if plugin.spec.role not in {
                FactorRole.COUNTRY, FactorRole.INDUSTRY, FactorRole.BOARD,
                FactorRole.MEMBERSHIP, FactorRole.STYLE_RISK,
                FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR,
            }
        )

    def ordered_risk_factors(self, member_ids: tuple[str, ...]) -> tuple[str, ...]:
        members = set(member_ids)
        self._validate_dependencies(members)
        edges = {
            factor_id: tuple(
                predecessor for predecessor in self.get(factor_id).spec.orthogonalize_after
                if predecessor in members
            )
            for factor_id in member_ids
        }
        ordered: list[str] = []
        remaining = set(member_ids)
        while remaining:
            ready = sorted(node for node in remaining if set(edges[node]) <= set(ordered))
            if not ready:
                raise ValueError("FACTOR_ORTHOGONALIZATION_CYCLE")
            ordered.extend(ready)
            remaining.difference_update(ready)
        return tuple(ordered)

    def _validate_dependencies(self, selected: set[str] | None = None) -> None:
        available = set(self.factor_ids())
        targets = selected or available
        for factor_id in targets:
            spec = self.get(factor_id).spec
            dependencies = set(spec.depends_on) | set(spec.orthogonalize_after)
            missing = sorted(dependencies - available)
            if missing:
                raise ValueError(
                    f"FACTOR_DEPENDENCY_MISSING factor={factor_id} dependencies={','.join(missing)}"
                )
            if spec.role in {
                FactorRole.COUNTRY, FactorRole.INDUSTRY, FactorRole.BOARD,
                FactorRole.MEMBERSHIP, FactorRole.STYLE_RISK,
            }:
                alpha_dependencies = sorted(
                    dependency for dependency in dependencies
                    if self.get(dependency).spec.role in {
                        FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR,
                    }
                )
                if alpha_dependencies:
                    raise ValueError(
                        f"RISK_DEPENDS_ON_ALPHA factor={factor_id} "
                        f"dependencies={','.join(alpha_dependencies)}"
                    )

    def validate_graph(self) -> None:
        self._validate_dependencies()
        risk_ids = tuple(sorted(
            spec.factor_id for spec in (plugin.spec for plugin in self._plugins.values())
            if spec.role in {
                FactorRole.COUNTRY, FactorRole.INDUSTRY, FactorRole.BOARD,
                FactorRole.MEMBERSHIP, FactorRole.STYLE_RISK,
            }
        ))
        if risk_ids:
            self.ordered_risk_factors(risk_ids)

    @classmethod
    def discover(cls, package: str = "factor_matrix.factor_engine.plugins") -> "FactorRegistry":
        module = importlib.import_module(package)
        plugins: list[FactorDefinition] = []
        for info in pkgutil.walk_packages(module.__path__, f"{package}."):
            plugin_module = importlib.import_module(info.name)
            plugins.extend(getattr(plugin_module, "PLUGINS", ()))
            single = getattr(plugin_module, "FACTOR", None)
            if single is not None:
                plugins.append(single)
        return cls(plugins)
