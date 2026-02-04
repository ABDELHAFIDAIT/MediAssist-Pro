import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from langchain_unstructured import UnstructuredLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from app.services.vector_store import vector_store_service
from app.core.config import settings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from app.api.v1.deps import get_current_user
from app.db.models.user import User

router = APIRouter()


@router.post("/upload")
async def uplaod_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)) :
    if not file.filename.endswith(".pdf") :
        raise HTTPException(status_code=404, detail="Seuls les fichiers PDF sont accéptés !")
    
    file_path = os.path.join(settings.UPLOADS_DIRECTORY, file.filename)
    
    os.makedirs(settings.UPLOADS_DIRECTORY, exist_ok=True)
    
    with open(file_path, "wb") as buffer :
        shutil.copyfileobj(file.file, buffer)
    
    try : 
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        
        raw_elements = loader.load()
        processed_docs = []

        for element in raw_elements:
            if element.metadata.get("category") == "Table":
                table_content = element.metadata.get("text_as_html", element.page_content)
                content = f"### TABLE DE MAINTENANCE DETECTEE\n\n{table_content}"
                processed_docs.append(Document(page_content=content, metadata=element.metadata))
            else:
                processed_docs.append(element)
        
        embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )
        semantic_splitter = SemanticChunker(embeddings)
        final_chunks = semantic_splitter.split_documents(processed_docs)

        clean_chunks = filter_complex_metadata(final_chunks)

        vector_store_service.add_documents(clean_chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(clean_chunks)
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur d'indexation : {str(e)}")


@router.get("/debug-chunks")
async def get_chunks(limit: int = 10, current_user: User = Depends(get_current_user)):
    vector_store = vector_store_service.get_vector_store()
    results = vector_store.get(limit=limit)
    
    debug_list = []
    for i in range(len(results["documents"])):
        debug_list.append({
            "content": results["documents"][i],
            "metadata": results["metadatas"][i]
        })
    return debug_list