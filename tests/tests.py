from rag_reranker.routing.adaptive_router import HeuristicQueryClassifier

classifier = HeuristicQueryClassifier()

test_queries = [
    'Hello there',
    'Thanks for the help',
    'Who are you?',
    'What can you do?',
    'What is BM25?',
    'Explain FAISS',
    'What is a state space model?',
    'Difference between cross-encoder and bi-encoder reranking',
    'Why is RRF robust to score scale differences?',
    'Compare the reranking approaches across multiple papers',
]

for q in test_queries:
    strategy = classifier.classify(q)
    print(f'{strategy.value:20} | {q}')