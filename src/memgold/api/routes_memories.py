"""REST endpoints for the production memory API surface."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from memgold.api.deps import (
    get_consolidation_service,
    get_ingestion_service,
    get_reflection_service,
    get_retrieval_service,
)
from memgold.api.schemas import (
    ConsolidateMemoriesRequest,
    CreateMemoryRequest,
    JobMemoriesResponse,
    MemoryListResponse,
    ReflectMemoriesRequest,
    SingleMemoryResponse,
    SummarizeMemoriesRequest,
)
from memgold.models.enums import MemoryType
from memgold.retrieval.types import RetrievalRequest
from memgold.services.consolidation import ConsolidationService
from memgold.services.ingestion import MemoryIngestionService
from memgold.services.reflection import ReflectionService
from memgold.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", response_model=MemoryListResponse)
async def create_memories(
    body: CreateMemoryRequest,
    svc: MemoryIngestionService = Depends(get_ingestion_service),
) -> MemoryListResponse:
    """Ingest text, extract structured memories, embed, and index."""
    memories = await svc.ingest_text(
        user_id=body.user_id,
        text=body.text,
        session_id=body.session_id,
    )
    return MemoryListResponse(memories=memories)


@router.get("/search", response_model=MemoryListResponse)
async def search_memories(
    user_id: str = Query(..., min_length=1),
    q: str = Query(..., min_length=1),
    top_k: int = Query(12, ge=1, le=100),
    hierarchy_prefix: str | None = None,
    session_id: str | None = None,
    expand_graph: bool = True,
    memory_types: list[MemoryType] | None = Query(None),
    svc: RetrievalService = Depends(get_retrieval_service),
) -> MemoryListResponse:
    """Hybrid retrieval with optional hierarchy and type filters."""
    types = frozenset(memory_types) if memory_types else None
    req = RetrievalRequest(
        user_id=user_id,
        query=q,
        top_k=top_k,
        hierarchy_prefix=hierarchy_prefix,
        memory_types=types,
        session_id=session_id,
        expand_graph=expand_graph,
    )
    memories = await svc.search(req)
    return MemoryListResponse(memories=memories)


@router.get("/{memory_id}", response_model=SingleMemoryResponse)
async def get_memory(
    memory_id: UUID,
    svc: RetrievalService = Depends(get_retrieval_service),
) -> SingleMemoryResponse:
    """Fetch a single memory by id and record an access touch."""
    mem = await svc.get(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return SingleMemoryResponse(memory=mem)


@router.post("/consolidate", response_model=JobMemoriesResponse)
async def consolidate_memories(
    body: ConsolidateMemoriesRequest,
    svc: ConsolidationService = Depends(get_consolidation_service),
) -> JobMemoriesResponse:
    """Run consolidation to emit room-level summaries."""
    created = await svc.consolidate_user(user_id=body.user_id)
    return JobMemoriesResponse(memories=created)


@router.post("/summarize", response_model=SingleMemoryResponse)
async def summarize_memories(
    body: SummarizeMemoriesRequest,
    svc: ConsolidationService = Depends(get_consolidation_service),
) -> SingleMemoryResponse:
    """Summarize an explicit list of memory ids into a bundle memory."""
    if not body.memory_ids:
        raise HTTPException(status_code=400, detail="memory_ids must not be empty")
    mem = await svc.summarize_selection(user_id=body.user_id, memory_ids=body.memory_ids)
    if mem is None:
        raise HTTPException(status_code=404, detail="No matching memories for summarization")
    return SingleMemoryResponse(memory=mem)


@router.post("/reflection", response_model=SingleMemoryResponse)
async def reflect_memories(
    body: ReflectMemoriesRequest,
    svc: ReflectionService = Depends(get_reflection_service),
) -> SingleMemoryResponse:
    """Synthesize a reflective memory from recent activity."""
    mem = await svc.reflect(user_id=body.user_id, lookback=body.lookback)
    if mem is None:
        raise HTTPException(status_code=404, detail="Nothing to reflect on yet")
    return SingleMemoryResponse(memory=mem)
