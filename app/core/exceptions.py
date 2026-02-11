from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class MediAssistException(Exception):
    """Exception de base pour le projet MediAssist Pro."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code

class VectorStoreError(MediAssistException):
    def __init__(self, message: str = "Erreur de base de données vectorielle"):
        super().__init__(message, status_code=503)

class LLMServiceError(MediAssistException):
    def __init__(self, message: str = "Le service d'IA est indisponible"):
        super().__init__(message, status_code=502)

async def mediassist_exception_handler(request: Request, exc: MediAssistException):
    logger.error(f"Erreur interceptée : {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.message,
            "path": request.url.path
        },
    )