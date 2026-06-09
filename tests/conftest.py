from pathlib import Path

import pytest


@pytest.fixture
def sample_config_path() -> Path:
    """Provide an isolated config for featurizer integration tests."""
    return Path(__file__).parent / "fixtures" / "sample_config.yaml"


@pytest.fixture
def snapshot_config_path() -> Path:
    """Small, executable config used for the rendered-SQL snapshot test."""
    return Path(__file__).parent / "fixtures" / "sample_config_snapshot.yaml"
