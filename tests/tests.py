from rag_reranker.ingestion.loader import Document
from rag_reranker.evaluation.metrics import reciprocal_rank, hit_rate_at_k, ndcg_at_k

def make_docs(titles):
    return [Document(title=t, content='x') for t in titles]

# Test 1: relevant doc at rank 1 -> MRR should be exactly 1.0
docs = make_docs(['Correct', 'Wrong1', 'Wrong2'])
mrr = reciprocal_rank(docs, {'Correct'})
print(f'MRR (relevant at rank 1): {mrr} (expected: 1.0)')

# Test 2: relevant doc at rank 3 -> MRR should be exactly 1/3
docs = make_docs(['Wrong1', 'Wrong2', 'Correct'])
mrr = reciprocal_rank(docs, {'Correct'})
print(f'MRR (relevant at rank 3): {mrr:.4f} (expected: {1/3:.4f})')

# Test 3: relevant doc not found -> MRR should be exactly 0.0
docs = make_docs(['Wrong1', 'Wrong2'])
mrr = reciprocal_rank(docs, {'Correct'})
print(f'MRR (not found): {mrr} (expected: 0.0)')

print()

# Test 4: Hit@3 when relevant doc is within top 3 -> 1.0
docs = make_docs(['A', 'B', 'Correct', 'D'])
hit = hit_rate_at_k(docs, {'Correct'}, k=3)
print(f'Hit@3 (correct at position 3): {hit} (expected: 1.0)')

# Test 5: Hit@3 when relevant doc is OUTSIDE top 3 -> 0.0
docs = make_docs(['A', 'B', 'C', 'Correct'])
hit = hit_rate_at_k(docs, {'Correct'}, k=3)
print(f'Hit@3 (correct at position 4): {hit} (expected: 0.0)')

print()

# Test 6: perfect ranking -> NDCG should be exactly 1.0
docs = make_docs(['Correct', 'A', 'B'])
ndcg = ndcg_at_k(docs, {'Correct'}, k=3)
print(f'NDCG (perfect ranking): {ndcg:.4f} (expected: 1.0000)')

# Test 7: worse ranking -> NDCG should be strictly LESS than 1.0
docs = make_docs(['A', 'B', 'Correct'])
ndcg = ndcg_at_k(docs, {'Correct'}, k=3)
print(f'NDCG (correct buried at rank 3): {ndcg:.4f} (expected: less than 1.0)')