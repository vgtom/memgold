"""Pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memgold.api.deps import reset_container
from memgold.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    reset_container()
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_container()
