# memgold

Async-first AI memory layer scaffolding (extraction, vector + KV storage, hybrid retrieval, ranking, context).

## Run the API

```bash
uv sync
uv run memgold-api
```

Or:

```bash
uv run uvicorn memgold.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API docs.
