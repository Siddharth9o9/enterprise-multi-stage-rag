from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    
    ## NVIDIA NIM
    nvidia_api_key: str = Field(
        default="",
        description="API key from build.nvidia.com"        
    )
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # Generator: Model for final answer writing
    generator_model: str = "meta/llama-3.1-70b-instruct"
    
    # Reranker LLM: smaller/faster model (score 0-1)
    reranker_llm_model: str = "meta/llama-3.1-8b-instruct"
    
    ## Embedding Model
    use_nvidia_embeddings: bool = False
    nvidia_embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    ## Cross Encoder Reranker
    use_nvidia_reranker: bool = False
    nvidia_reranker_model: str = "nvidia/nv-rerankqa-mistral-4b-v3"
    local_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    ## Retrieval Hyperparameters
    chunk_size: int = 512
    chunk_overlap: int = 64
    
    dense_top_k: int = 20
    sparse_top_k: int = 20
    
    rrf_k_constant: int = 60
    
    rerank_top_k: int = 8
    
    simple_query_word_limit: int = 6
    
    ## API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
settings=Settings()