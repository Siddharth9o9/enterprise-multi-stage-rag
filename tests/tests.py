from rag_reranker.ingestion.loader import load_documents
from rag_reranker.ingestion.chunker import chunk_documents
from rag_reranker.retrieval.dense_retriever import DenseRetriever
from rag_reranker.retrieval.sparse_retriever import SparseRetriever

docs = load_documents()
chunks = chunk_documents(docs)

dense = DenseRetriever()
dense.build_index(chunks)

sparse = SparseRetriever()
sparse.build_index(chunks)

# Query 1: semantic — dense should win
q1 = 'how to improve the quality of retrieved documents'
print('=== Query 1 (semantic) ===')
print(f'Query: {q1}')
print()
print('Dense top 3:')
for d in dense.retrieve(q1, top_k=3):
    print(f'  {d.score:.4f} | {d.title}')
print('Sparse top 3:')
for d in sparse.retrieve(q1, top_k=3):
    print(f'  {d.score:.4f} | {d.title}')

print()

# Query 2: keyword — sparse should win
q2 = 'BM25 Okapi term frequency inverse document frequency'
print('=== Query 2 (keyword exact match) ===')
print(f'Query: {q2}')
print()
print('Dense top 3:')
for d in dense.retrieve(q2, top_k=3):
    print(f'  {d.score:.4f} | {d.title}')
print('Sparse top 3:')
for d in sparse.retrieve(q2, top_k=3):
    print(f'  {d.score:.4f} | {d.title}')