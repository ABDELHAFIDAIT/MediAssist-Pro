import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings

class VectorService :
    def __init__(self) :
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )
        self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
    
    def get_vector_store(self) :
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="mediassist_docs"
        )
    
    
    def add_documents(self, documents) :
        vector_store = self.get_vector_store()
        vector_store.add_documents(documents)
        
    
    def as_agent_retriever(self, search_type="mmr", k=5) :
        vector_store = self.get_vector_store()
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k" : k,
                "fetch_k" : 20,
                "lambda_mult" : 0.5
            }
        )



vector_store_service = VectorService()