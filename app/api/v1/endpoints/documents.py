import os
import shutil
import re
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from langchain_unstructured import UnstructuredLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from app.services.vector_store import vector_store_service
from app.core.config import settings
from app.api.v1.deps import get_current_user
from app.db.models.user import User


router = APIRouter()


def clean_text(text: str) -> str:
    text = re.sub(r'\.{2,}|_{2,}', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def is_dense_technical_content(text: str) -> bool:
    units_pattern = r'\d+\s?(mm|nm|v|hz|kg|°c|%|ma|ms|µm|bar|psi)'
    has_measures = re.search(units_pattern, text, re.IGNORECASE)
    
    id_pattern = r'\b[A-Z]{2,}\b\s?Code|ID|Ref|S/N|#|\b\d{4,}\b'
    has_ids = re.search(id_pattern, text, re.IGNORECASE)
    
    digit_count = sum(c.isdigit() for c in text)
    is_numeric_dense = (digit_count / len(text)) > 0.05 if len(text) > 5 else False
    
    return bool(has_measures or has_ids or is_numeric_dense)



@router.post('/upload')
async def upload_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)) :
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés !")

    file_path = os.path.join(settings.UPLOADS_DIRECTORY, file.filename)
    os.makedirs(settings.UPLOADS_DIRECTORY, exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try :
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            infer_table_structure=True
        )
        
        raw_elements = loader.load()
        
        tech_chunks = []
        narrative_docs = []
        
        for el in raw_elements :
            content = clean_text(el.page_content)
            
            if not content :
                continue
            
            metadata = {
                **el.metadata,
                "source_file" : file.filename,
                "technicien_upload" : current_user.username
            }
            
            if el.metadata.get("category") == "Table" or is_dense_technical_content(content) :
                metadata['is_technical'] = True
                tagged_content = f"### [DONNEES TECHNIQUES / FICHE]\n{content}"
                tech_chunks.append(
                    Document(
                        page_content=tagged_content, 
                        metadata=metadata
                    )
                )
            else :
                metadata["is_technical"] = False
                narrative_docs.append(
                    Document(
                        page_content=content, 
                        metadata=metadata
                    )
                )
            
        embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )
        
        semantic_splitter = SemanticChunker(embeddings)
        final_chunks = []
        
        if narrative_docs :
            final_chunks.extend(semantic_splitter.split_documents(narrative_docs))
            
        final_chunks.extend(tech_chunks)
        
        clean_chunks = filter_complex_metadata(final_chunks)
        vector_store_service.add_documents(clean_chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks": len(clean_chunks),
            "tech_blocks": len(tech_chunks)
        }
    
    except Exception as e:
        print(f"INGESTION_ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Échec de l'indexation : {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



@router.get("/debug-chunks")
async def get_chunks(limit: int = 100, current_user: User = Depends(get_current_user)):
    vector_store = vector_store_service.get_vector_store()
    results = vector_store.get(limit=limit)
    return results