"""
FastAPI server exposing RAGPipelines over HTTP.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_reranker.pipeline import RAGPipeline
from rag_reranker.config import settings

class AppState:
    
    pipeline: RAGPipeline | None = None
    
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server process starts, before any request is accepted - and again when the server shuts down.
    """
    print("Starting up: building RAG pipeline...")
    app_state.pipeline = RAGPipeline(reranker_strategy="cross_encoder")
    app_state.pipeline.index()
    print("Stratup complete: pipeline indexed and ready.")
    
    # Server runs and accept request here
    yield 
    
    print("Shutting down.")
    
app = FastAPI(
    title= "RAG Reranking System",
    description=(
        "Production RAG pipeline with hybrid retrieval (dense + BM25), "
        "RRF fusion, multi-strategy reranking (cross-encoder / LLM "
        "pointwise / LLM listwise), contextual compression, and "
        "adaptive query routing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

# Request / Response Schemas

class QueryRequest(BaseModel):
    """
    min_length=1 & max_length=1000:
    
    Pydantic validates before route handler code ever runs. Capping the min & max length would protect against both wasted compute and unexpectedly huge prompts
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["Why is RRF robust to score scale differences?"]
    )
    reranker: str = Field(
        default="cross_encoder",
        description="One of: cross_encoder | llm_pointwise | llm_listwise",
    )
    
class DocumentResponse(BaseModel):
    """
    The response model exposes only what a caller actually needs to see — title, content, score, and which retriever(s) found it — keeping the public API contract intentionally narrower than our internal data model.
    """
    title: str
    content: str
    score: float | None
    retrieved_by: list[str]
    
class QueryResponse(BaseModel):
    query: str
    strategy_used: str
    answer: str
    sources: list[DocumentResponse]
    
class HealthResponse(BaseModel):
    status: str
    indexed: bool
    
# Routes
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Setting up an Health Endpoint for load balancers, container orchestrators (Docker, Kubernetes) & uptime monitors to check server status
    """
    return HealthResponse(
        status="ok",
        indexed=app_state.pipeline is not None and app_state.pipeline._indexed,
    )
    
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if app_state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Reranking Switch as per user
    if request.reranker != "cross_encoder":
        app_state.pipeline.reranker = app_state.pipeline._build_reranker(
            request.reranker
        )
        
    try:
        result = await app_state.pipeline.run(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    
    sources = [
        DocumentResponse(
            title=doc.title,
            content=doc.content,
            score=doc.score,
            retrieved_by=doc.retrieved_by,
        )
        for doc in result.reranked_docs
    ]

    return QueryResponse(
        query=result.query,
        strategy_used=result.strategy_used,
        answer=result.answer,
        sources=sources,
    )


if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host=settings.api_host, port=settings.api_port)
