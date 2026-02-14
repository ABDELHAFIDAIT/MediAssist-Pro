from unittest.mock import patch
import uuid



def test_register_user(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "youness@gmail.com", "username": "youness", "password": "youness", "role": "technicien"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "youness@gmail.com"



def test_login_success(client, db, test_user):
    with patch("app.core.security.verify_password", return_value=True):
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "hafid", "password": "hafid"}
        )
    assert response.status_code == 200
    assert "access_token" in response.json()