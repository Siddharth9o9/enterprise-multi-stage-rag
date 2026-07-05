from __future__ import annotations
from pydantic import BaseModel, Field
import uuid

class Document(BaseModel):
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    topic: str = "general"
    source: str = "unknown"
    score: float | None = None
    retrieved_by: list[str] = Field(default_factory=list)
    
    def __hash__(self):
        # Makes Document usable in sets and as dict keys.
        # Needed in fusion.py where we deduplicate across retrievers.
        return hash(self.id)
    
RAW_KNOWLEDGE_BASE: list[dict] = [
    {
        "title": "Vector Embeddings in NLP",
        "content": (
            "Vector embeddings transform text into dense numerical representations "
            "in high-dimensional space. Models like BERT and sentence-transformers "
            "create embeddings where semantically similar texts are close together. "
            "Cosine similarity measures the angle between vectors to find related "
            "content. These embeddings capture semantic meaning beyond keyword "
            "matching. The dimensionality typically ranges from 384 to 1536 "
            "depending on the model architecture and training objective."
        ),
        "topic": "embeddings",
        "source": "internal-docs",
    },
    {
        "title": "Transformer Architecture",
        "content": (
            "Transformers use self-attention mechanisms to process sequences in "
            "parallel. The attention formula scores relationships between all token "
            "pairs simultaneously using query, key, and value projections. "
            "Multi-head attention allows the model to attend to different "
            "representation subspaces at once. BERT uses bidirectional transformers "
            "for deep contextual understanding, while GPT uses causal masking "
            "for autoregressive text generation."
        ),
        "topic": "architecture",
        "source": "internal-docs",
    },
    {
        "title": "RAG System Overview",
        "content": (
            "Retrieval-Augmented Generation combines a retriever with a language "
            "model generator. The retriever fetches relevant documents from a "
            "knowledge base using vector similarity or keyword search. The generator "
            "then conditions its output on both the query and retrieved context. "
            "RAG reduces hallucinations and enables knowledge-grounded responses "
            "without retraining the underlying language model on new data."
        ),
        "topic": "rag",
        "source": "internal-docs",
    },
    {
        "title": "Cross-Encoder Reranking",
        "content": (
            "Cross-encoders jointly encode query and document pairs to produce "
            "relevance scores. Unlike bi-encoders which encode independently, "
            "cross-encoders use full attention across both inputs enabling precise "
            "relevance assessment through deep token-level interaction. They are "
            "slower but significantly more accurate than embedding-based similarity. "
            "Models like ms-marco-MiniLM and bge-reranker are popular for passage "
            "reranking tasks in production search systems."
        ),
        "topic": "reranking",
        "source": "internal-docs",
    },
    {
        "title": "LLM-based Relevance Scoring",
        "content": (
            "Large language models can assess document relevance through zero-shot "
            "prompting without any fine-tuning. By asking the LLM to rate "
            "query-document relevance on a numeric scale, we get nuanced scores "
            "that leverage the LLM world knowledge and reasoning capability. "
            "Listwise approaches like RankGPT ask the model to reorder an entire "
            "list at once, which captures relative ordering better than scoring "
            "documents independently one by one."
        ),
        "topic": "reranking",
        "source": "internal-docs",
    },
    {
        "title": "Reciprocal Rank Fusion",
        "content": (
            "Reciprocal Rank Fusion combines rankings from multiple retrieval "
            "systems without needing to normalize their raw scores. Each document "
            "gets a score of 1 divided by k plus its rank, where k is typically "
            "60. Scores from all rankers are summed to produce a final fused "
            "ranking. RRF is robust to scale differences between systems like "
            "BM25 and dense vector search, and consistently outperforms any "
            "single ranker alone in empirical evaluations."
        ),
        "topic": "reranking",
        "source": "internal-docs",
    },
    {
        "title": "Contextual Compression in RAG",
        "content": (
            "Contextual compression extracts only the relevant portions from "
            "retrieved documents before passing them to the generator. Instead "
            "of passing full documents, a compressor identifies pertinent "
            "sentences or paragraphs using either an LLM extractor or an "
            "embeddings-based filter. This reduces token usage significantly "
            "and focuses the LLM on truly relevant content, mitigating the "
            "lost-in-the-middle problem where LLMs underweight information "
            "appearing in the middle of long contexts."
        ),
        "topic": "optimization",
        "source": "internal-docs",
    },
    {
        "title": "Hybrid Search Strategies",
        "content": (
            "Hybrid search combines dense vector search with sparse BM25 keyword "
            "search to get the best of both worlds. Dense retrieval excels at "
            "semantic similarity and paraphrase matching, while BM25 handles "
            "exact term matching for rare words, product codes, and technical "
            "jargon. Combining both approaches via fusion techniques improves "
            "recall across diverse query types compared to either method alone."
        ),
        "topic": "retrieval",
        "source": "internal-docs",
    },
    {
        "title": "FAISS Vector Store",
        "content": (
            "FAISS, developed by Facebook AI, enables efficient similarity search "
            "at scale across millions of vectors. It supports multiple index types "
            "including Flat for exact search, IVF for approximate search with "
            "clustering, and HNSW graph-based indexes for fast high-recall lookups. "
            "The choice of index type trades off search speed against recall "
            "accuracy. Quantization techniques like PQ reduce memory usage while "
            "preserving most of the search quality at scale."
        ),
        "topic": "infrastructure",
        "source": "internal-docs",
    },
    {
        "title": "Query Decomposition Techniques",
        "content": (
            "Complex queries can be decomposed into simpler sub-queries for better "
            "retrieval coverage. Multi-query retrieval generates several query "
            "variations using an LLM to improve recall across different phrasings. "
            "Step-back prompting creates a more abstract version of the query to "
            "retrieve broader supporting context first. Query decomposition is "
            "especially useful for multi-hop reasoning tasks that require "
            "synthesizing information from several distinct document sources."
        ),
        "topic": "optimization",
        "source": "internal-docs",
    },
    {
        "title": "Adaptive RAG and Self-RAG",
        "content": (
            "Adaptive RAG dynamically routes queries based on their complexity "
            "instead of always running the same fixed expensive pipeline. Simple "
            "factual questions may not need any retrieval at all if the LLM "
            "already knows the answer from training. Complex analytical questions "
            "benefit from iterative multi-step retrieval and reasoning. Self-RAG "
            "trains the model to emit special reflection tokens that decide when "
            "to retrieve, whether retrieved passages are relevant, and whether "
            "the generated answer is supported by the evidence."
        ),
        "topic": "advanced",
        "source": "internal-docs",
    },
    {
        "title": "BM25 Sparse Retrieval",
        "content": (
            "BM25 is a probabilistic ranking function based on term frequency and "
            "inverse document frequency. It improves upon plain TF-IDF by "
            "normalizing for document length and saturating term frequency so that "
            "repeating a word many times yields diminishing returns. BM25 remains "
            "highly competitive for keyword-heavy queries and technical content, "
            "and requires no GPU making it extremely cheap to run at scale."
        ),
        "topic": "retrieval",
        "source": "internal-docs",
    },
    {
        "title": "ColBERT and Late Interaction Models",
        "content": (
            "ColBERT introduces late interaction retrieval sitting between fast "
            "bi-encoders and accurate but slow cross-encoders. Instead of one "
            "vector per document, ColBERT stores one vector per token. At query "
            "time each query token finds its best matching document token using "
            "a MaxSim operation, and these are summed into a final relevance "
            "score. This token-level matching achieves much of the accuracy of "
            "cross-encoders while keeping an index that can be searched "
            "efficiently at scale without scoring every document fully."
        ),
        "topic": "retrieval",
        "source": "internal-docs",
    },
    {
        "title": "Evaluation Metrics for Retrieval",
        "content": (
            "Retrieval and reranking quality is measured using ranking-aware "
            "metrics. NDCG, normalized discounted cumulative gain, rewards "
            "relevant documents appearing near the top of the ranking with a "
            "logarithmic discount penalizing lower positions. Mean Reciprocal "
            "Rank focuses on the position of the first relevant result. Hit "
            "Rate at K checks whether any relevant document appears within the "
            "top K results, which is useful for quick sanity checks during "
            "development and debugging of retrieval pipelines."
        ),
        "topic": "evaluation",
        "source": "internal-docs",
    },
]


def load_documents() -> list[Document]:
    return [Document(**raw) for raw in RAW_KNOWLEDGE_BASE]