"""Tests for the featurizer.viz visualization toolkit.

Plotting backends are optional ([viz] extra), so plot tests importorskip their
dependency. The pandas-only paths (feature_cols, summary table, matrix-contract
normalization, from_featurizer wiring) run without any plotting deps installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from featurizer import Featurizer, FeaturizerViz


@pytest.fixture(autouse=True)
def _set_agg_backend():
    """Use a non-interactive backend when matplotlib is present (never skips)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass
    yield


@pytest.fixture
def matrix_df() -> pd.DataFrame:
    """A small (2 as-of dates x 4 entities) matrix with a deliberate NaN."""
    return pd.DataFrame(
        {
            "as_of_date": ["2024-01-01"] * 4 + ["2024-02-01"] * 4,
            "entity_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "f_a": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, None, 4.5],
            "f_b": [10.0, 9.0, 8.0, 7.0, 11.0, 8.5, 7.5, 6.5],
            "f_c": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def viz(matrix_df) -> FeaturizerViz:
    return FeaturizerViz(matrix_df)


# --------------------------------------------------------------------------- #
# Contract / pandas-only paths (no plotting deps required)
# --------------------------------------------------------------------------- #


def test_feature_cols_excludes_keys(viz):
    assert "as_of_date" not in viz.feature_cols
    assert "entity_id" not in viz.feature_cols
    assert set(viz.feature_cols) == {"f_a", "f_b", "f_c", "target"}


def test_accepts_multiindex_form(matrix_df):
    """A frame indexed by (as_of_date, entity_id) is normalized to columns."""
    indexed = matrix_df.set_index(["as_of_date", "entity_id"])
    v = FeaturizerViz(indexed)
    assert set(v.feature_cols) == {"f_a", "f_b", "f_c", "target"}
    # The caller's frame must not be mutated.
    assert list(indexed.index.names) == ["as_of_date", "entity_id"]


def test_feature_summary_table(viz):
    summary = viz.feature_summary_table()
    assert list(summary.columns) == ["mean", "std", "skewness", "pct_missing"]
    # f_a has 1 missing of 8 rows.
    assert summary.loc["f_a", "pct_missing"] == pytest.approx(12.5)
    assert summary.loc["f_b", "pct_missing"] == 0.0


def test_from_featurizer_resolves_entity_col(matrix_df):
    """from_featurizer must read the entity column from the target's id."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: users
max_depth: 1
intervals: []
entities:
  - alias: users
    table: users
    id: user_id
    variables:
      age: {type: numeric}
""")
        f.flush()
        featurizer = Featurizer(f.name)
        Path(f.name).unlink()

    df = matrix_df.rename(columns={"entity_id": "user_id"})
    v = FeaturizerViz.from_featurizer(featurizer, df=df)
    assert v.entity_col == "user_id"
    assert "user_id" not in v.feature_cols


# --------------------------------------------------------------------------- #
# Plot smoke tests (return a Figure; importorskip the backend)
# --------------------------------------------------------------------------- #


def test_plot_feature_variance(viz):
    pytest.importorskip("matplotlib")
    import matplotlib.figure

    assert isinstance(viz.plot_feature_variance(), matplotlib.figure.Figure)


def test_plot_feature_distributions(viz):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.figure

    assert isinstance(viz.plot_feature_distributions(), matplotlib.figure.Figure)


def test_plot_feature_importance_handles_nan(viz):
    """sklearn rejects NaN; the method must median-impute locally and succeed."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    import matplotlib.figure

    fig = viz.plot_feature_importance(target_col="target")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_entity_embedding_pca(viz):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    import matplotlib.figure

    fig = viz.plot_entity_embedding(method="pca")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_entity_dendrogram(viz):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    import matplotlib.figure

    assert isinstance(viz.plot_entity_dendrogram(), matplotlib.figure.Figure)


def test_plot_feature_timeseries(viz):
    pytest.importorskip("matplotlib")
    import matplotlib.figure

    fig = viz.plot_feature_timeseries(entity_id=1, normalize=True)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_entity_feature_heatmap(viz):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.figure

    assert isinstance(
        viz.plot_entity_feature_heatmap(entity_id=1), matplotlib.figure.Figure
    )
