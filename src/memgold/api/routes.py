"""Thin route handlers delegating to application services."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from memgold.api.deps import get_context_builder, get_read_service, get_write_service
from memgold.api.schemas import (
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryListResponse,
    MemoryReadRequest,
    MemoryWriteRequest,
)
from memgold.services.context import ContextBuilder
from memgold.services.read import MemoryReadService
from memgold.services.write import MemoryWriteService

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/write", response_model=MemoryListResponse)
async def write_memory(
    body: MemoryWriteRequest,
    service: MemoryWriteService = Depends(get_write_service),
) -> MemoryListResponse:
    """Extract memories from text and persist them to vector and KV stores."""
    memories = await service.process_input(body.user_id, body.text)
    return MemoryListResponse(memories=memories)


@router.post("/read", response_model=MemoryListResponse)
async def read_memory(
    body: MemoryReadRequest,
    service: MemoryReadService = Depends(get_read_service),
) -> MemoryListResponse:
    """Retrieve ranked memories for a natural-language query."""
    memories = await service.retrieve(body.user_id, body.query)
    return MemoryListResponse(memories=memories)


@router.post("/context", response_model=MemoryContextResponse)
async def build_context(
    body: MemoryContextRequest,
    builder: ContextBuilder = Depends(get_context_builder),
) -> MemoryContextResponse:
    """Return structured memories plus a concatenated context string for prompting."""
    bundle = await builder.build_context(body.user_id, body.query)
    return MemoryContextResponse(
        memories=bundle["memories"],
        context_string=bundle["context_string"],
    )
