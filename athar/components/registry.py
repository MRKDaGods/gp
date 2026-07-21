"""Component registry: name → factory, per component kind.

Profiles reference components as ``(kind, name)``; plugins register
implementations at import time. Config passed to ``create`` is validated by
the component's own typed config model (each factory takes keyword config).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


class UnknownComponent(KeyError):
    pass


class ComponentRegistry:
    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Callable[..., Any]] = {}

    def register(self, kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            key = (kind, name)
            if key in self._factories:
                raise ValueError(f"component already registered: {kind}/{name}")
            self._factories[key] = factory
            return factory

        return decorator

    def create(self, kind: str, name: str, **config: Any) -> Any:
        try:
            factory = self._factories[(kind, name)]
        except KeyError:
            available = ", ".join(sorted(n for k, n in self._factories if k == kind)) or "<none>"
            raise UnknownComponent(
                f"no {kind} named {name!r}; registered: {available}"
            ) from None
        return factory(**config)

    def names(self, kind: str) -> Iterable[str]:
        return sorted(n for k, n in self._factories if k == kind)


registry = ComponentRegistry()
"""Process-global default registry (plugins may also be given private ones in tests)."""
