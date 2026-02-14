import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.rag_service import RAGService

@pytest.mark.asyncio
async def test_format_docs():
    service = RAGService()
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "doc.pdf", "chapter": "1", "section": "Intro"}
    mock_doc.page_content = "Contenu de test"
    
    result = service._format_docs([mock_doc])
    assert "[SOURCE: doc.pdf | CHAPITRE: 1 | SECTION: Intro]" in result
    assert "Contenu de test" in result



@pytest.mark.asyncio
async def test_answer_question_error_handling():
    service = RAGService()
    # On force une exception lors de la récupération
    with patch("app.services.vector_store.vector_store_service.get_hybrid_retriever", side_effect=Exception("Qdrant Down")):
        response = await service.answer_question("test")
        assert response == "Une erreur de connexion au moteur d'IA a eu lieu."