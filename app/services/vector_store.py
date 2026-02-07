from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from typing import List

class VectorService:
    def __init__(self):
        # 1. Initialisation du modèle d'embeddings BGE-M3
        # Ce modèle est chargé localement pour une performance maximale
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} # Changez en 'cuda' si vous avez un GPU
        )

        # 2. Connexion au client Qdrant
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # 3. Création de la collection si elle n'existe pas
        self._ensure_collection_exists()

        # 4. Initialisation du Vector Store LangChain
        self._vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def _ensure_collection_exists(self):
        """Vérifie et crée la collection avec les bons paramètres de dimension."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            # BGE-M3 produit des vecteurs de 1024 dimensions
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024, 
                    distance=models.Distance.COSINE
                )
            )

    def get_vector_store(self) -> QdrantVectorStore:
        return self._vector_store

    def add_documents(self, documents):
        """Ajoute des documents à la base avec gestion des métadonnées."""
        self._vector_store.add_documents(documents)

    def get_hybrid_retriever(self, k: int = 5):
        """
        Retourne un retriever configuré pour la recherche hybride.
        Qdrant gère nativement le filtrage par métadonnées.
        """
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k
            }
        )

# Instance unique pour l'application
vector_store_service = VectorService()