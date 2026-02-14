from unittest.mock import MagicMock
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_mock_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine
from app.main import app
from app.db.session import Base
from app.api.v1.deps import get_db, get_current_user
from app.db.models.user import User


SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



@pytest.fixture(scope="function")
def db() -> Generator:
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)



@pytest.fixture(scope="function")
def client(db) -> Generator:
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()



@pytest.fixture
def test_user(db):
    user = User(
        email="hafid@gmail.com",
        username="hafid",
        hashed_password="hafid",
        role="technicien"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user



@pytest.fixture
def mock_current_user(test_user):
    app.dependency_overrides[get_current_user] = lambda: test_user
    yield test_user
    app.dependency_overrides.pop(get_current_user, None)




@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Empêche MLflow de tenter de se connecter pendant les tests."""
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda x: None)
    monkeypatch.setattr("mlflow.set_experiment", lambda x: None)
    monkeypatch.setattr("mlflow.start_run", lambda **kwargs: MagicMock())
    monkeypatch.setattr("mlflow.log_params", lambda x: None)
    monkeypatch.setattr("mlflow.log_metrics", lambda x: None)