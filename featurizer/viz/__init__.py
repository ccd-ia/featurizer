from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from featurizer.featurizer import Featurizer


class FeaturizerViz:
    """Visualization toolkit for featurizer output DataFrames.

    The matrix produced by :meth:`Featurizer.to_dataframe` is indexed by
    ``(as_of_date, <entity id column>)`` where the second level is the target
    entity's actual id column name (e.g. ``customer_id``), not the literal
    ``"entity_id"``. This class accepts either that MultiIndex form or a flat
    frame; index levels matching ``as_of_col`` / ``entity_col`` are moved to
    columns automatically. Prefer :meth:`from_featurizer`, which wires the
    correct column names for you.

    Args:
        df: Feature matrix. May be indexed by (as_of_date, entity) or flat.
        as_of_col: Name of the as-of date column / index level.
        entity_col: Name of the entity id column / index level.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        as_of_col: str = "as_of_date",
        entity_col: str = "entity_id",
    ) -> None:
        # Normalize: if as_of/entity live in the index (the to_dataframe shape),
        # move them to columns so every method can treat them uniformly. Work on
        # a copy — never mutate the caller's frame.
        index_names = [n for n in (df.index.names or []) if n is not None]
        if as_of_col in index_names or entity_col in index_names:
            df = df.reset_index()
        self.df = df
        self.as_of_col = as_of_col
        self.entity_col = entity_col
        self._feature_cols: list[str] | None = None

    @classmethod
    def from_featurizer(
        cls,
        featurizer: Featurizer,
        df: pd.DataFrame | None = None,
    ) -> FeaturizerViz:
        """Build a ``FeaturizerViz`` with column names taken from the Featurizer.

        Resolves ``entity_col`` from the target entity's id column (so the
        ``entity_id`` default never silently treats the id as a feature). When
        ``df`` is None, executes ``featurizer.to_dataframe()``.

        Raises:
            ValueError: if the target entity defines no primary id.
        """
        if featurizer.target.id is None:
            raise ValueError(
                f"Target entity '{featurizer.target.alias}' defines no primary "
                f"id; cannot determine the entity column for visualization."
            )
        entity_col = featurizer.target.id.name
        if df is None:
            df = featurizer.to_dataframe()
        return cls(df, as_of_col="as_of_date", entity_col=entity_col)

    @property
    def feature_cols(self) -> list[str]:
        if self._feature_cols is None:
            self._feature_cols = [
                c for c in self.df.columns if c not in {self.as_of_col, self.entity_col}
            ]
        return self._feature_cols

    # Methods are defined as module-level functions (first arg ``self``) and
    # bound here as class attributes, grouped by source file.
    from .correlation import plot_correlation_clustermap, plot_redundancy_graph
    from .distributions import feature_summary_table, plot_feature_distributions
    from .importance import plot_feature_importance, plot_feature_variance
    from .missing import plot_missing_heatmap, plot_missing_over_time
    from .similarity import plot_entity_dendrogram, plot_entity_embedding
    from .temporal import plot_entity_feature_heatmap, plot_feature_timeseries
