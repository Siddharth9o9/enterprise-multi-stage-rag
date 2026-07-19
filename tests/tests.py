import asyncio
from rag_reranker.pipeline import RAGPipeline

async def main():
    pipeline = RAGPipeline(reranker_strategy='cross_encoder')
    pipeline.index()

    query = 'Why is RRF robust to score scale differences between BM25 and dense retrieval?'
    result = await pipeline.run(query)

    print('Reranked docs:')
    for d in result.reranked_docs:
        print(f'  - {d.title}')
    print()
    print('Answer:')
    print(result.answer)

asyncio.run(main())