"""
Standard information-retrieval evaluation metrics: MRR, Hit@K, NDCG.
"""
from __future__ import annotations
import math
from rag_reranker.ingestion.loader import Document

def _doc_identity(doc: Document) -> str:
    """
    Identifying documents by title for evaluation purposes.
    """
    return doc.title

def reciprocal_rank(ranked_docs: list[Document], relevant_titles: set[str]) -> float:
    """
    RR = 1 / (rank of first relevant document), or 0.0 if none found.
    """
    for position, doc in enumerate(ranked_docs):
        if _doc_identity(doc) in relevant_titles:
            return 1.0 / (position + 1)
        
    return 0.0

def hit_rate_at_k(ranked_docs: list[Document], relevant_titles: set[str], k: int) -> float:
    """
    1.0 if ANY relevant document appears in the top k, else 0.0.
    
    Returns a float to average this value across many queries to get an overall hit rate like 0.80 or 0.85 etc. 
    """
    top_k_titles = {_doc_identity(d) for d in ranked_docs[:k]}
    return 1.0 if top_k_titles & relevant_titles else 0.0

def ndcg_at_k(ranked_docs: list[Document], relevant_titles: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain — the most complete ranking metric.

    DCG@K = sum over top K positions of: relevance / log2(position + 2)
    IDCG@K = DCG of the IDEAL ordering (all relevant docs pushed to front)
    NDCG@K = DCG@K / IDCG@K  → normalizes to a [0, 1] scale where
             1.0 means "this ranking IS the best possible ranking"
    """
    def dcg(titles_in_order: list[str]) -> float:
        score = 0.0
        for position, title in enumerate(titles_in_order[:k]):
            relevance = 1.0 if title in relevant_titles else 0.0
            score += relevance / math.log2(position + 2)
        return score

    actual_titles = [_doc_identity(d) for d in ranked_docs]
    actual_dcg = dcg(actual_titles)

    # Ideal ranking: every relevant doc pushed to the very front,
    # followed by whatever else was in the actual ranking
    ideal_titles = list(relevant_titles) + [
        t for t in actual_titles if t not in relevant_titles
    ]
    ideal_dcg = dcg(ideal_titles)

    # Guard against division by zero — if there are somehow zero
    # relevant documents defined for this query, ideal_dcg is 0
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

class RetrievalEvaluator:
    """
    Aggregates all three metrics across a full evaluation set.

    WHY THIS CLASS EXISTS INSTEAD OF JUST CALLING THE THREE FUNCTIONS
    DIRECTLY IN evaluate.py:
    This is the exact discipline you would apply as a CI check in a
    real team — "does my new reranker beat the old one on these
    metrics before I merge this change?" Wrapping evaluation logic in
    one reusable class means evaluate.py stays focused on defining
    WHAT to test (the ground truth set), not HOW to compute averages.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate_single(
        self, ranked_docs: list[Document], relevant_titles: set[str]
    ) -> dict:
        return {
            "mrr": reciprocal_rank(ranked_docs, relevant_titles),
            f"hit_rate@{self.k}": hit_rate_at_k(ranked_docs, relevant_titles, self.k),
            f"ndcg@{self.k}": ndcg_at_k(ranked_docs, relevant_titles, self.k),
        }

    def evaluate_batch(
        self, results: list[tuple[list[Document], set[str]]]
    ) -> dict:
        """
        results: list of (ranked_docs, relevant_titles) pairs, one
        entry per evaluation query.

        Returns the MEAN of each metric across the whole set — this
        is what "MRR = 0.83" means when reported for a whole system:
        the average reciprocal rank across every query you tested,
        not just one lucky or unlucky example.
        """
        all_scores = [
            self.evaluate_single(docs, relevant)
            for docs, relevant in results
        ]
        metric_names = all_scores[0].keys()
        return {
            name: sum(scores[name] for scores in all_scores) / len(all_scores)
            for name in metric_names
        }