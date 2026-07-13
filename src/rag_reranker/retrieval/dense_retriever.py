from __future__ import annotations
import numpy as np
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

class DenseRetriever:
    
    """
    Bi-encoder retriever using sentence-transformers + FAISS
    
    Flow:
        build_index (documents) - called once at startup
        retrieve(query,top_k) - called on every user query
        
    FAISS: approximate nearest search neighbour
        -sub -linear time: searches without computing all distances
        -uses index structure to prune the search space
    
    """
    
    def __init__(self, model_name: str | None = None):
        
        # Importing libraries inside __init__ to reduce the import cost (1~2 seconds) when importing DenseRetriever
        from sentence_transformers import SentenceTransformer
        import faiss
        
        # Setting up similarity search library and model selection for embedding 
        self._faiss = faiss
        self.model_name = model_name or (
            settings.nvidia_embedding_model
            if settings.use_nvidia_embeddings
            else settings.local_embedding_model
        )
        
        self.model = SentenceTransformer(
            self.model_name, device = "cuda" if self._is_cuda_available() else "cpu",
        )
        
        self.index = None
        self.documents: list[Document] = []
        
    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        
    def build_index(self, documents: list[Document]) ->None:
        """
        Offline Phase: embed all documents and build FAISS index
        Called only during startup and not as per-query
        """
        
        # Assigning instance variable and priortizing doc title as it carries semantic signals about the document
        # Including it in the embedded text biases the vector toward's the doc primary subject, improving  retriever precision 
        self.documents = documents
        texts=[
            f"{doc.title}. {doc.content}"
            for doc in documents
        ]
        
        print(f"Embedding {len(texts)} documents on "
              f"{'GPU' if self._is_cuda_available() else 'CPU'}")
        
        # Initiating embeddings to convert text to vectors.
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True, # unit vector, cosine via dot product
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Converting the no. format of the vectors to 32-bit floating point as required format for FAISS database entry 
        embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]
        
        # IndexFlatIP = exact inner product search (no approximation)
        # For production with millions of vectors, swap to:
        # IndexHNSWFlat → graph-based ANN, faster but approximate
        # IndexIVFFlat  → cluster-based ANN, configurable speed/recall
        # Same API for all index types — that is FAISS's design strength
        self.index = self._faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index built:{self.index.ntotal} vectors,"
              f"{dimension} dimensions")
        
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        Online Phase: embed query, search index and return ranked documents
        Returns documents sorted in descending similarity score
        Each returned is a copy. 
        """
        
        if self.index is None:
            raise RuntimeError(
                "build_index() must be called before retrieve()."
                "The pipeline calls this at startup automatically."
            )
         
        # Sets the number of results to return. If user provided top_k else defauls to global system setting    
        top_k = top_k or settings.dense_top_k
        
        # Encoding query to vectors for index searching
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        
        # Output scores/similarity score and indices/document database row number 
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Initializes empty list to hold document result
        results: list[Document] = []
        
        # Loops through top matched items by pairing up each document score with its corresponding database index
        for score, idx in zip(scores[0], indices[0]):
            # Skips invalid matches - If fewer docs then requested top_k
            if idx == -1:
                continue
            # Creating a clean independent copy
            doc = self.documents[idx].model_copy()
            doc.score = float(score)
            doc.retrieved_by = ["dense"]
            results.append(doc)
            
        return results
            
            
    