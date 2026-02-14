from unittest.mock import MagicMock, patch
from qdrant_client import models
from app.services.vector_store import VectorService

def test_ensure_collection_exists_creates_if_missing():
    with patch("app.services.vector_store.QdrantClient") as MockClient:
        mock_instance = MockClient.return_value
        
        # 1. Simuler l'absence de collections initiales
        mock_instance.get_collections.return_value = MagicMock(collections=[])
        
        # 2. Configurer une réponse réaliste pour get_collection afin de passer les tests LangChain
        # On définit manuellement la structure attendue par le validateur de LangChain
        mock_vectors_config = MagicMock()
        mock_vectors_config.size = 1024
        mock_vectors_config.distance = models.Distance.COSINE # Utilise l'énumération réelle
        
        mock_coll_info = MagicMock()
        mock_coll_info.config.params.vectors = mock_vectors_config
        mock_instance.get_collection.return_value = mock_coll_info
        
        # Initialisation du service (déclenche _ensure_collection_exists)
        service = VectorService()
        
        # Vérifier que la création a été appelée car la liste était vide
        assert mock_instance.create_collection.called
        
        # Vérification des paramètres de création
        _, kwargs = mock_instance.create_collection.call_args
        assert kwargs["vectors_config"].size == 1024
        assert kwargs["vectors_config"].distance == models.Distance.COSINE