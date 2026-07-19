"""
Takes the query plus the final set of documents (after retrieval, reranking, and optional compression) and produces a grounded,
source-cited natural language answer using the 70B NVIDIA NIM model.
"""
from __future__ import annotations
from openai import AsyncOpenAI
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

GENERATION_SYSTEM_PROMPT = """You are a precise research assistant answering questions about NLP, RAG, and retrieval techniques based on academic sources. Use ONLY the provided sources to answer — do not add information the sources do not support. If the sources do not contain enough information to answer fully, say so explicitly rather than guessing. Cite sources using [Source N] notation immediately after each claim that depends on that source."""

GENERATION_USER_PROMPT = """Sources:
{context}

Question: {query}

Answer the question using the sources above, citing [Source N] for every claim that comes from a specific source."""

# Used only for the NO_RETRIEVAL path — no sources to cite, so the assistant relies on its own general knowledge, with an explicit instruction to be transparent about that.
NO_SOURCES_SYSTEM_PROMPT = """You are a helpful research assistant for a system that answers questions about NLP and RAG techniques. Answer naturally and concisely. Do not fabricate citations since no sources were retrieved for this query."""

class AnswerGenerator:
    def __init__(self, model: str | None = None):
        self.client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )
        self.model = model or settings.generator_model

    def _format_context(self, documents: list[Document]) -> str:
        """
        Numbers each document as [Source N] so the model has a
        consistent reference scheme to cite against. This numbering
        is REGENERATED per call based on final reranked order — Source 1
        is always the top-ranked document for THIS query, not a
        permanent ID tied to the document itself.
        """
        return "\n\n".join(
            f"[Source {i + 1}] {doc.title}\n{doc.content}"
            for i, doc in enumerate(documents)
        )

    async def generate(self, query: str, documents: list[Document]) -> str:
        """
        WHY WE BRANCH ON EMPTY documents EXPLICITLY:
        The NO_RETRIEVAL path (pipeline.py) calls this with an empty list. Using the same "cite your sources" prompt with zero sources would either confuse the model or cause it to fabricate fake citations. We use a completely different, simpler prompt for this case instead.
        """
        if not documents:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NO_SOURCES_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return response.choices[0].message.content

        context = self._format_context(documents)
        user_prompt = GENERATION_USER_PROMPT.format(context=context, query=query)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # WHY temperature=0.2 HERE, NOT 0.0 LIKE THE RERANKER:
            # pure temperature=0.0 can produce oddly repetitive or robotic phrasing across sentences. 0.2 keeps output grounded and consistent while avoiding that stiffness.
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content