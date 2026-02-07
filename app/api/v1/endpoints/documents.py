import os
import shutil
import re
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.services.vector_store import vector_store_service
from app.core.config import settings
from app.api.v1.deps import get_current_user
from app.db.models.user import User

router = APIRouter()


def convert_to_markdown_table(html_table: str) -> str:
    if not html_table:
        return ""
    table = re.sub(r"<tr>", "\n| ", html_table)
    table = re.sub(r"</td>|</th>", " | ", table)
    table = re.sub(r"<[^>]+>", "", table)
    return table



@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    file_path = os.path.join(settings.UPLOADS_DIRECTORY, file.filename)
    os.makedirs(settings.UPLOADS_DIRECTORY, exist_ok=True)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            chunking_strategy="by_title",
            max_characters=1000,
            combine_under_n_chars=200
        )
        
        raw_elements = loader.load()
        final_docs = []
        current_context = {"chapter": "", "section": ""}

        for el in raw_elements:
            content = el.page_content
            category = el.metadata.get("category")
            
            if category == "Title":
                if "Chapitre" in content:
                    current_context["chapter"] = content
                else:
                    current_context["section"] = content

            base_metadata = {
                "source": file.filename,
                "user_id": current_user.id,
                "chapter": current_context["chapter"],
                "section": current_context["section"],
                "doc_id": str(uuid4())
            }

            if category == "Table":
                html_table = el.metadata.get("text_as_html")
                markdown_table = convert_to_markdown_table(html_table)
                final_docs.append(Document(
                    page_content=f"### TABLEAU ({current_context['section']})\n{markdown_table}",
                    metadata={**base_metadata, "type": "technical_table"}
                ))
            else:
                final_docs.append(Document(
                    page_content=content,
                    metadata={**base_metadata, "type": "narrative"}
                ))

        if final_docs:
            clean_chunks = filter_complex_metadata(final_docs)
            vector_store_service.add_documents(clean_chunks)

        return {"status": "success", "chunks_indexed": len(final_docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)




@router.get("/chunks")
async def get_indexed_chunks(
    limit: int = Query(10, description="Nombre de chunks à visualiser"),
    current_user: User = Depends(get_current_user)
):
    """Visualiser les chunks stockés dans Qdrant pour l'utilisateur actuel."""
    try:
        chunks = vector_store_service.list_chunks(user_id=current_user.id, limit=limit)
        return {
            "total_requested": limit,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération : {str(e)}")