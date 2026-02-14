import pytest
from unittest.mock import AsyncMock, patch



@pytest.mark.asyncio
async def test_chat_endpoint_success(client, mock_current_user):
    mock_response = "Vérifiez la batterie du moniteur."
    
    with patch("app.services.rag_service.rag_service.answer_question", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = mock_response
        
        response = client.post(
            "/api/v1/chat/",
            json={"message": "Comment réparer le moniteur ?"}
        )
        
    assert response.status_code == 200
    assert response.json()["response"] == mock_response
    assert response.json()["status"] == "success"