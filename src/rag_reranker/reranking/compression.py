"""
Contextual compression: extracts relevant sentences from the documents after reranking
has already selected which document to use.

This is the final quality-refinement stage before generation.
"""

from __future__ import annotations
import re
import asyncio
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

def _split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter using punctuation boundaries.
    
    Not using spacy or nltk as those library handle edge cases which is not required here in this type of document.
    Regex is enough to filter out.
    """
    
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]

class EmbeddingsFilterCompressor:
    """
    Filters sentences by embedding similarity to the query.
    
    Using same embedding model as dense_retriever.py to get the sentence level similarity same way.
    """
    def __init__(self, model_name: str | None = None,keep_fraction: float = 0.5,):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        self.model_name = model_name or settings.local_embedding_model
        self.model = SentenceTransformer(
            self.model_name,
            device="cuda" if self._is_cuda_available() else "cpu",
        )
        self.keep_fraction = keep_fraction

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def compress(self, query: str, doc: Document) -> Document:
        """
        Compress a single document to only its query-relevant sentences.
        If sentence has <=1 sentence, then return the same sentence without any computation.
        """
        
        sentences = _split_sentences(doc.content)
        
        if len(sentences) <=2:
            return doc
        
        # Embed query once, all sentences in one batch call
        query_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        
        sentence_vecs = self.model.encode(sentences,normalize_embeddings=True, convert_to_numpy=True)
        
        # Cosine similarity via dot product. Vectors normalized to unit length
        similarities = sentence_vecs @ query_vec
        
        num_to_keep = max(1, int(len(sentences) * self.keep_fraction))

        # argsort ascending, take the last num_to_keep (highest scores),
        # then sort THOSE indices back into original document order so
        # the compressed text still reads naturally top-to-bottom rather
        # than being reordered by relevance score.
        top_indices = self.np.argsort(similarities)[-num_to_keep:]
        top_indices_in_order = sorted(top_indices)

        kept_sentences = [sentences[i] for i in top_indices_in_order]

        compressed_doc = doc.model_copy()
        compressed_doc.content = " ".join(kept_sentences)
        return compressed_doc
    
    def compress_all(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Compress a batch of documents. Syncronously - no API calls involved,
        just local GPU embedding computation.
        """
        return [self.compress(query,doc) for doc in documents]
    
EXTRACTION_PROMPT = """Extract only the sentences from the DOCUMENT below that are directly relevant to answering the QUERY. 
Copy them verbatim — do not paraphrase, summarize, or add anything of your own. If nothing is directly relevant, return
the single most related sentence as-is.

QUERY: {query}

DOCUMENT:
{content}

Relevant sentences (verbatim from the document, no extra commentary):"""

class LLMExtractorCompressor:
    """
    Uses the LLM reranker model to extract relevant sentences via prompting.
    """

    def __init__(self, model: str | None = None, max_concurrency: int = 5):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )
        self.model = model or settings.reranker_llm_model
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _compress_one(self, query: str, doc: Document) -> Document:
        prompt = EXTRACTION_PROMPT.format(query=query, content=doc.content)

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )
                extracted_text = response.choices[0].message.content.strip()

                compressed_doc = doc.model_copy()
                
                # SAFETY CHECK: if the LLM returns something suspiciously short (e.g. it misunderstood and returned "None" or
                # an empty string), fall back to the original content rather than losing information entirely.
                if len(extracted_text) > 10:
                    compressed_doc.content = extracted_text
                else:
                    compressed_doc.content = doc.content

                return compressed_doc
            
            except Exception as e:
                print(f"  [warn] compression failed for '{doc.title}': {e}")
                # Fail-safe: return original document uncompressed rather than crashing the pipeline over one bad call
                return doc
    
    async def compress_all(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Concurrent compression across all documents — same asyncio.gather
        pattern as PointwiseLLMReranker in llm_reranker.py. If you are
        compressing 5 documents, all 5 API calls fire together instead
        of sequentially.
        """
        return await asyncio.gather(
            *[self._compress_one(query, doc) for doc in documents]
        )