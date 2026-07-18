"""
LLM-based reranking using NVIDIA NIM 
"""
from __future__ import annotations
import asyncio
import json
import re
from openai import AsyncOpenAI
from rag_reranker.reranking.base import BaseReranker
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key
    )
    
POINTWISE_PROMPT = """ You are a relevance judge. Rate how relevant the DOCUMENT is to the QUERY.

QUERY: {query}

DOCUMENT:
TITLE: {title}
CONTENT: {content}

Respond with only a single decimal number between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
No explaination, no other text, just the number.
""" 

class PointwiseLLMReranker(BaseReranker):
    """
    Scores each document independently via seperate API calls, fired concurrently using asyncio.gather
    """
    
    def __init__(self, model: str | None = None, max_concurrency: int = 5):
        """
        max_concurrency set to 5 as NVIDIA NIM free tier has rate limits so 5 is safely under typical free tier limits.
        """
        self.client = _get_client()
        self.model = model or settings.reranker_llm_model
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
    async def _score_one(self, query: str, doc: Document) -> float:
        """
        Score a single document. Wrapped in try/except because one failed API call should not crash the entire reranking. 
        """
        prompt = POINTWISE_PROMPT.format(query=query, title=doc.title, content=doc.content,)
        
        # Semaphore acquired here - this call waits if 5 other calls are already in flight, then proceeds once one finishes.
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model = self.model,
                    messages = [{"role": "user", "content": prompt}],
                    temperature = 0.0,
                    max_tokens=10,
                )
                text = response.choices[0].message.content.strip()
                
                match = re.search(r"[\d.]+",text)
                score = float(match.group()) if match else 0.0
                
                # Return between [0,1]
                return max(0.0, min(1.0, score))
            except Exception as e:
                print(f"[warn] scoring failed for '{doc.title}': {e} ")
                return 0.0
    async def rerank(self, query: str, documents: list[Document], top_k: int |None = None,) -> list[Document]:
        if not documents:
            return []
        
        top_k = top_k or settings.rerank_top_k
        
        # All N scoring calls are scheduled at once. 
        # asyncio.gather waits for all of them to complete, running up to 5 (semaphore limit) simultaneously at any given moment.
        
        scores = await asyncio.gather(*[self._score_one(query,doc) for doc in documents])
        
        scored_documents=[]
        for doc, score in zip(documents, scores):
            scored_doc = doc.model_copy()
            scored_doc.score = score
            scored_documents.append(scored_doc)
            
        scored_documents.sort(key=lambda d: d.score, reverse=True)
        return scored_documents[:top_k]
    


LISTWISE_PROMPT = """You are a relevance ranking system. Given a QUERY and a list of DOCUMENTS, 
rank all DOCUMENTS from most relevant to least relevant.

QUERY: {query}

DOCUMENTS:
{doc_list}

Respond with only a JSON array of document numbers in ranked order, most relevant first. Example: [3, 1, 4, 2]
No explanation, just the JSON array."""


class ListwiseLLMReranker(BaseReranker):
    """
    RankGPT-style reranking: One API call ranks the entire candidate list.

    OFTEN OUTPERFORMS POINTWISE:
    Pointwise asks "is document A relevant? is document B relevant?"
    as two independent questions — the model never directly compares
    A and B against each other.

    Listwise asks "given all these documents, which is most relevant?"
    in one shot. The model can directly compare candidates, catching
    subtle relative differences that independent scoring misses.
    Research shows this produces measurably better rankings.

    WHY WE DO NOT IMPLEMENT SLIDING WINDOW HERE:
    Production RankGPT handles 100+ candidates by processing overlapping
    windows of ~10-20 documents and merging results. Our candidate pool
    (capped at 15 by the pipeline) fits in one context window easily,
    so sliding window would add complexity without benefit at our scale.
    """

    def __init__(self, model: str | None = None):
        self.client = _get_client()
        self.model = model or settings.reranker_llm_model

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k

        # Build a numbered list of all documents for the prompt
        doc_list_str = "\n".join(
            f"[{i + 1}] Title: {doc.title}\nContent: {doc.content}\n"
            for i, doc in enumerate(documents)
        )

        prompt = LISTWISE_PROMPT.format(query=query, doc_list=doc_list_str)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()

            # Defensive parsing: strip markdown code fences if the model wraps its JSON despite instructions not to
            text = re.sub(r"```(?:json)?", "", text).strip()
            order = json.loads(text)

        except Exception as e:
            print(f"  [warn] listwise ranking failed: {e}")
            # Fail-safe: fall back to original order rather than crashing
            order = list(range(1, len(documents) + 1))

        reranked: list[Document] = []
        seen_indices: set[int] = set()

        for rank_position, doc_number in enumerate(order):
            index = doc_number - 1  # convert 1-indexed to 0-indexed

            # Defensive check: the model might hallucinate an invalid document number (e.g. [3, 1, 15] when only 10 docs exist)
            if 0 <= index < len(documents) and index not in seen_indices:
                scored_doc = documents[index].model_copy()

                # Synthetic descending score derived from position,
                # not a numeric judgment — the LLM gave us an ORDER,
                # not scores. We convert position to a score so that
                # downstream code (which expects .score) still works
                # consistently across all reranker types.
                scored_doc.score = round(
                    1.0 - (rank_position / max(len(order), 1)), 4
                )
                reranked.append(scored_doc)
                seen_indices.add(index)

        # Safety net: if the LLM forgot to include some documents in
        # its ranking, append them at the end with score 0.0 rather
        # than silently dropping them
        for i, doc in enumerate(documents):
            if i not in seen_indices:
                scored_doc = doc.model_copy()
                scored_doc.score = 0.0
                reranked.append(scored_doc)

        return reranked[:top_k]
        
        
    
