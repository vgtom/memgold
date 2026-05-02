"""Turn unstructured text into structured ``Memory`` records."""

from __future__ import annotations

from memgold.models.memory import Memory, MemoryType


class MemoryExtractor:
    """Pluggable extraction pipeline; this build uses heuristic stubs."""

    async def extract(self, text: str) -> list[Memory]:
        """Parse *text* into candidate memories.

        Returns:
            One or two placeholder memories so downstream components can be exercised
            without an LLM. Production systems would call an LLM or rules engine here.
        """
        snippet = text.strip() or "(empty input)"
        preview = snippet if len(snippet) <= 120 else snippet[:117] + "..."
        return [
            Memory(
                content=f"Stub fact extracted from user message: {preview}",
                type=MemoryType.FACT,
                confidence_score=0.72,
                tags=["stub", "fact"],
                metadata={"source": "MemoryExtractor.stub"},
            ),
            Memory(
                content=f"Stub semantic summary of input: {preview}",
                type=MemoryType.SEMANTIC,
                confidence_score=0.55,
                tags=["stub", "semantic"],
                metadata={"source": "MemoryExtractor.stub"},
            ),
        ]
