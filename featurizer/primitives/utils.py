# coding: utf-8

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

AggregationRegistry = Dict[str, Callable[..., Any]]
TransformationRegistry = Dict[str, Callable[..., Any]]

_AGGREGATIONS: AggregationRegistry = {}
_TRANSFORMATIONS: TransformationRegistry = {}


def register_aggregation(name: str, aggregator: Callable[..., Any]) -> None:
    """Register an aggregation primitive for use in feature generation.

    Args:
        name: Unique name for this aggregation
        aggregator: Callable aggregation function

    Raises:
        ValueError: If name is not a non-empty string
        TypeError: If aggregator is not callable
    """
    if not name:
        raise ValueError("Aggregation name must be a non-empty string.")
    if not callable(aggregator):
        raise TypeError(f"Aggregator '{name}' must be callable.")
    _AGGREGATIONS[name] = aggregator


def register_transformer(name: str, transformer: Callable[..., Any]) -> None:
    """Register a transformation primitive for use in feature generation.

    Args:
        name: Unique name for this transformer
        transformer: Callable transformation function

    Raises:
        ValueError: If name is not a non-empty string
        TypeError: If transformer is not callable
    """
    if not name:
        raise ValueError("Transformer name must be a non-empty string.")
    if not callable(transformer):
        raise TypeError(f"Transformer '{name}' must be callable.")
    _TRANSFORMATIONS[name] = transformer


def get_aggregations(names: Optional[Iterable[str]] = None) -> AggregationRegistry:
    if names is None:
        return dict(_AGGREGATIONS)
    missing = [name for name in names if name not in _AGGREGATIONS]
    if missing:
        raise KeyError(f"Unknown aggregations requested: {', '.join(missing)}")
    return {name: _AGGREGATIONS[name] for name in names}


def get_transformers(names: Optional[Iterable[str]] = None) -> TransformationRegistry:
    if names is None:
        return dict(_TRANSFORMATIONS)
    missing = [name for name in names if name not in _TRANSFORMATIONS]
    if missing:
        raise KeyError(f"Unknown transformers requested: {', '.join(missing)}")
    return {name: _TRANSFORMATIONS[name] for name in names}


def list_aggregations() -> Iterable[str]:
    return sorted(_AGGREGATIONS.keys())


def list_transformations() -> Iterable[str]:
    return sorted(_TRANSFORMATIONS.keys())
