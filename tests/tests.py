import asyncio
from rag_reranker.pipeline import RAGPipeline

async def main():
    pipeline = RAGPipeline(reranker_strategy='cross_encoder')
    pipeline.index()

    complex_queries = [
        'Why is RRF robust to score scale differences?',
        'Compare cross-encoder and bi-encoder reranking approaches',
        'What is the overview of adaptive RAG techniques used in modern systems?',
    ]

    for q in complex_queries:
        strategy = pipeline.classifier.classify(q)
        dense = pipeline.dense_retriever.retrieve(q, top_k=20)
        sparse = pipeline.sparse_retriever.retrieve(q, top_k=20)
        from rag_reranker.retrieval.fusion import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion([dense, sparse])
        reranked = await pipeline.reranker.rerank(q, fused[:15], top_k=5)

        before_total = sum(len(d.content) for d in reranked)
        keep_fraction = pipeline._select_keep_fraction(q)
        pipeline.compressor.keep_fraction = keep_fraction
        compressed = pipeline.compressor.compress_all(q, reranked)
        after_total = sum(len(d.content) for d in compressed)

        reduction = 100 * (1 - after_total / before_total) if before_total else 0
        print(f'keep_fraction={keep_fraction} | reduction: {reduction:.1f}% | {q}')

asyncio.run(main())