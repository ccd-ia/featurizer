# coding: utf-8

"""SQL rendering utilities."""

from __future__ import annotations

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
        ctes = ','.join(plan.ctes)
        target_alias = plan.target.alias
        return f"""
        select aod.as_of_date, t.*
        from as_of_dates as aod
        cross join lateral (

        with

        {ctes}

        select * from {target_alias}_transform
        ) as t

        order by aod.as_of_date
        """
