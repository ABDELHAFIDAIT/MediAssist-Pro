from fastapi import FastAPI
from app.core.config import settings
from app.db.session import engine, Base
from app.db.models.user import User
from app.db.models.query import Query

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

@app.get("/")
async def root() :
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION
    }