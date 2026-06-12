"""Schema-sanity tier for the seeded realistic datasets.

These tests validate the *seeder's output*, not featurizer: row-count bands,
referential integrity, and edge-table invariants. They skip (via the dataset
fixtures) when the schema is absent or seeded at the wrong version.
"""

from __future__ import annotations

import pytest

from ._realistic import expect_sql

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestFoodInspectionsSeed:
    def test_row_count_bands(self, food_db):
        """Counts sit in wide bands — catches truncated or runaway downloads."""
        assert (
            10_000
            <= expect_sql(food_db, "select count(*) from food_inspections.facilities")
            <= 40_000
        )
        assert (
            40_000
            <= expect_sql(food_db, "select count(*) from food_inspections.inspections")
            <= 120_000
        )
        assert (
            10_000
            <= expect_sql(food_db, "select count(*) from food_inspections.licenses")
            <= 80_000
        )
        assert (
            1_000
            <= expect_sql(food_db, "select count(*) from food_inspections.chain_edges")
            <= 100_000
        )

    def test_inspections_reference_existing_facilities(self, food_db):
        orphans = expect_sql(
            food_db,
            """
            select count(*) from food_inspections.inspections i
            left join food_inspections.facilities f using (license_no)
            where f.license_no is null
            """,
        )
        assert orphans == 0

    def test_licenses_reference_existing_facilities(self, food_db):
        orphans = expect_sql(
            food_db,
            """
            select count(*) from food_inspections.licenses l
            left join food_inspections.facilities f using (license_no)
            where f.license_no is null
            """,
        )
        assert orphans == 0

    def test_inspections_within_subset_window(self, food_db):
        out_of_window = expect_sql(
            food_db,
            """
            select count(*) from food_inspections.inspections
            where inspection_date < '2014-01-01' or inspection_date > '2016-12-31'
            """,
        )
        assert out_of_window == 0

    def test_chain_edges_invariants(self, food_db):
        """Edges are unordered pairs of existing facilities with a causal stamp."""
        assert (
            expect_sql(
                food_db,
                "select count(*) from food_inspections.chain_edges "
                "where knowable_at is null",
            )
            == 0
        )
        assert (
            expect_sql(
                food_db,
                "select count(*) from food_inspections.chain_edges "
                "where source_license >= target_license",
            )
            == 0
        )
        dangling = expect_sql(
            food_db,
            """
            select count(*) from food_inspections.chain_edges e
            left join food_inspections.facilities s on s.license_no = e.source_license
            left join food_inspections.facilities t on t.license_no = e.target_license
            where s.license_no is null or t.license_no is null
            """,
        )
        assert dangling == 0

    def test_knowable_at_not_before_either_endpoint(self, food_db):
        """The causal stamp is the later of the two facilities' first_seen."""
        violations = expect_sql(
            food_db,
            """
            select count(*) from food_inspections.chain_edges e
            join food_inspections.facilities s on s.license_no = e.source_license
            join food_inspections.facilities t on t.license_no = e.target_license
            where e.knowable_at < greatest(s.first_seen, t.first_seen)
            """,
        )
        assert violations == 0


class TestDonorsChooseSeed:
    def test_row_count_bands(self, donorschoose_db):
        assert 1_000 <= expect_sql(
            donorschoose_db, "select count(*) from donorschoose.projects"
        )
        assert 10_000 <= expect_sql(
            donorschoose_db, "select count(*) from donorschoose.donations"
        )

    def test_donations_reference_existing_projects(self, donorschoose_db):
        orphans = expect_sql(
            donorschoose_db,
            """
            select count(*) from donorschoose.donations d
            left join donorschoose.projects p using (projectid)
            where p.projectid is null
            """,
        )
        assert orphans == 0

    def test_project_edges_invariants(self, donorschoose_db):
        assert (
            expect_sql(
                donorschoose_db,
                "select count(*) from donorschoose.project_edges "
                "where knowable_at is null",
            )
            == 0
        )
        assert (
            expect_sql(
                donorschoose_db,
                "select count(*) from donorschoose.project_edges "
                "where source_project >= target_project",
            )
            == 0
        )
