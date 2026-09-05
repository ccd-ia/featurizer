from importlib.util import find_spec
from pathlib import Path

import pytest

# The cockpit's snapshot fixture (``shell_snapshot``) comes from lynkeus, which
# the ``tui`` extra only installs on Python 3.12+. Registering it here, without
# importing it, keeps pytest's assertion rewriting for the plugin and lets the
# ``tests/test_tui*.py`` modules skip themselves with ``importorskip`` on 3.10
# and 3.11 rather than fail.
if find_spec("lynkeus") is not None:
    pytest_plugins = ["lynkeus.testing"]


@pytest.fixture
def sample_config_path() -> Path:
    """Provide an isolated config for featurizer integration tests."""
    return Path(__file__).parent / "fixtures" / "sample_config.yaml"


@pytest.fixture
def snapshot_config_path() -> Path:
    """Small, executable config used for the rendered-SQL snapshot test."""
    return Path(__file__).parent / "fixtures" / "sample_config_snapshot.yaml"
