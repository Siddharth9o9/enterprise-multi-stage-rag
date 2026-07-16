from rag_reranker.reranking.base import BaseReranker
from rag_reranker.reranking.cross_encoder_reranker import LocalCrossEncoderReranker

reranker = LocalCrossEncoderReranker()

print(f'Is a BaseReranker: {isinstance(reranker, BaseReranker)}')
print(f'Reranker name: {reranker.name}')
print(f'Has rerank method: {hasattr(reranker, "rerank")}')

import inspect
print(f'rerank is coroutine function: {inspect.iscoroutinefunction(reranker.rerank)}')