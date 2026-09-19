# coding: utf-8

"""SQL rendering utilities."""

from __future__ import annotations

from loguru import logger

from .boundary import as_of_dates_source
from .planner import PlannerResult


class SQLRenderer:
    """Builds the final query from planner artifacts."""

    def render(self, plan: PlannerResult) -> str:
        """Render a complete SQL query from planner artifacts.

        Args:
            plan: PlannerResult containing CTEs, features, and join information

        Returns:
            Complete SQL query string with CTEs wrapped in lateral join
        """
        ctes = ",".join(plan.ctes)
        target_alias = plan.target.alias
        # Both are the bare defaults unless the config pairs each as-of date
        # with its own ids (issue #10), which keeps the default byte-identical.
        spine = as_of_dates_source(paired=plan.cohort_id_column is not None)
        post_filter = f" {plan.cohort_post_filter}" if plan.cohort_post_filter else ""
        query = f"""
        select aod.as_of_date, t.*
        from {spine} as aod
        cross join lateral (

        with

        {ctes}

        select * from {target_alias}_transform{post_filter}
        ) as t

        order by aod.as_of_date
        """
        logger.debug(
            "Rendered SQL for target '{}': {} CTEs, {} chars",
            target_alias,
            len(plan.ctes),
            len(query),
        )
        return query
