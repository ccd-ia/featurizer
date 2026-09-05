"""The featurizer terminal cockpit, built on the lynkeus shell.

``python -m featurizer tui --config path/to/config.yaml`` opens it. ``status``,
``runs list|show``, ``query`` and ``actions list|run`` print the same data
headlessly, with ``--json`` for agents, so what a person watches and what an
agent reads come from one place.

Importing this package imports Textual and lynkeus, which only install on
Python 3.12 or newer (the ``tui`` extra). ``import featurizer`` never imports
it; the CLI verbs that need it import it inside the function and say what to
install when that fails.

The adapters live in :mod:`featurizer.tui.adapters`, the project screens in
:mod:`featurizer.tui.screens`, the assembly in :mod:`featurizer.tui.app`.
"""

from __future__ import annotations

from typing import Any

from .adapters import (
    SAVED_QUERIES,
    FeaturizerActions,
    FeaturizerRuns,
    FeaturizerStatus,
    Materialization,
    Project,
    find_materializations,
    source_for,
)

__all__ = [
    "SAVED_QUERIES",
    "FeaturizerActions",
    "FeaturizerRuns",
    "FeaturizerStatus",
    "Materialization",
    "Project",
    "build_app",
    "find_materializations",
    "source_for",
]


def build_app(*args: Any, **kwargs: Any) -> Any:
    """See :func:`featurizer.tui.app.build_app`. Imported late — Textual is heavy."""
    from .app import build_app as _build

    return _build(*args, **kwargs)
