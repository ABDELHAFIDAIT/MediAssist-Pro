import time
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings



class VectorService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        api_key = settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
        
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
            timeout=60 
        )
        
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        self._connect_with_retry(max_retries=10, delay=5)

        self._vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )


    def _connect_with_retry(self, max_retries: int = 10, delay: int = 2):
        retries = 0
        while retries < max_retries:
            try:
                
                self.client.get_collections()
                self._ensure_collection_exists()
                return
            except (ResponseHandlingException, Exception) as e:
                retries += 1
                time.sleep(delay)
        
        raise ConnectionError("Impossible de se connecter à Qdrant après plusieurs tentatives.")


    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024, 
                    distance=models.Distance.COSINE
                )
            )


    def add_documents(self, documents):
        self._vector_store.add_documents(documents)


    def list_chunks(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "id": p.id,
                "content": p.payload.get("page_content"),
                "metadata": p.payload.get("metadata")
            } for p in points
        ]


    def get_hybrid_retriever(self, k: int = 5):
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


vector_store_service = VectorService()