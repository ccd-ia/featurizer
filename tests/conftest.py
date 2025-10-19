from pathlib import Path

import pytest


@pytest.fixture
def sample_config_path() -> Path:
    """Provide an isolated config for featurizer integration tests."""
    return Path(__file__).parent / "fixtures" / "sample_config.yaml"
