"""
Test configurations.
"""

from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir():
    """Fixture to return the path to the data directory."""
    return TEST_DIR / "test_data"
