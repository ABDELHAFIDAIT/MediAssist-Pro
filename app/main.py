from fastapi import FastAPI
from app.core.config import settings
from app.db.session import engine, Base
from app.db.models.user import User
from app.db.models.query import Query
from app.api.v1.api import api_router


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)


app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root() :
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION
    }