import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
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
        
    
    def search_relevant_context(self, query: str, k: int) :
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(query=query, k=k)



vector_store_service = VectorService()