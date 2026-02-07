from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.services.rag_service import rag_service
from app.db.session import SessionLocal
from app.db.models.query import Query
from app.db.models.user import User
from app.api.v1.deps import get_current_user

router = APIRouter()


class ChatRequest(BaseModel):
    message: str




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()





@router.post("/")
async def chat_with_expert(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        ai_response = await rag_service.answer_question(
            question=request.message, 
            user_id=current_user.id
        )

        new_query = Query(
            user_id=current_user.id,
            query=request.message,
            response=ai_response
        )
        
        db.add(new_query)
        db.commit()
        db.refresh(new_query)

        return {
            "query_id": new_query.id,
            "response": ai_response,
            "status": "success"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du traitement du message : {str(e)}"
        )