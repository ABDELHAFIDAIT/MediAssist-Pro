from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings) :
    PROJECT_NAME : str
    VERSION : str
    API_V1_STR : str
    
    POSTGRES_SERVER : str
    POSTGRES_USER : str
    POSTGRES_PASSWORD : str
    POSTGRES_DB : str
    POSTGRES_PORT : int
    
    DATABASE_URL : str
    
    JWT_SECRET : str
    ALGORITHM : str
    ACCESS_TOKEN_EXPIRE_MINUTES : int
    
    OLLAMA_BASE_URL : str
    LLM_MODEL : str
    EMBEDDING_MODEL : str
    
    CHROMA_PERSIST_DIRECTORY : str
    UPLOADS_DIRECTORY : str
    
    QDRANT_URL: str 
    QDRANT_API_KEY: str | None 
    QDRANT_COLLECTION_NAME: str 

    CHUNK_SIZE: int 
    CHUNK_OVERLAP: int
    
    MLFLOW_TRACKING_URI : str
    MLFLOW_EXPERIMENT_NAME : str
    
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )
    

settings = Settings()