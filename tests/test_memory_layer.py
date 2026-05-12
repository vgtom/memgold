"""Example tests for ingestion, search, and consolidation."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient


def test_ingestion_and_search_isolation() -> None:
    from memgold.api.deps import build_container
    from memgold.retrieval.types import RetrievalRequest

    async def _run() -> None:
        c = build_container()
        u1 = "user-one"
        u2 = "user-two"
        await c.ingestion.ingest_text(user_id=u1, text="I prefer dark mode for all my coding tools.")
        await c.ingestion.ingest_text(user_id=u2, text="I love hiking in the Alps every summer.")
        hits = await c.retrieval_service.search(
            RetrievalRequest(
                user_id=u1,
                query="coding theme",
                top_k=5,
            ),
        )
        assert hits
        assert all(m.user_id == u1 for m in hits)

    asyncio.run(_run())


def test_api_memories_flow(client: TestClient) -> None:
    uid = "api-user"
    r = client.post("/memories", json={"user_id": uid, "text": "Remember: my SSH key lives in ~/.ssh/id_ed25519"})
    assert r.status_code == 200
    data = r.json()
    assert "memories" in data
    assert len(data["memories"]) >= 1
    mid = data["memories"][0]["id"]

    s = client.get("/memories/search", params={"user_id": uid, "q": "ssh", "top_k": 5})
    assert s.status_code == 200
    ids = {m["id"] for m in s.json()["memories"]}
    assert mid in ids

    g = client.get(f"/memories/{mid}")
    assert g.status_code == 200
    assert g.json()["memory"]["user_id"] == uid


def test_consolidation_endpoint(client: TestClient) -> None:
    uid = "heavy-user"
    for i in range(5):
        resp = client.post("/memories", json={"user_id": uid, "text": f"episodic note number {i} about project X."})
        assert resp.status_code == 200
    r = client.post("/memories/consolidate", json={"user_id": uid})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["memories"], list)
