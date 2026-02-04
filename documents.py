import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_unstructured import UnstructuredLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from app.services.vector_store import vector_store_service
from app.core.config import settings

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    file_path = os.path.join(settings.UPLOADS_DIRECTORY, file.filename)
    os.makedirs(settings.UPLOADS_DIRECTORY, exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy=None 
        )
        
        raw_elements = loader.load()
        processed_docs = []

        
        for element in raw_elements:
            
            if element.metadata.get("category") == "Table":
                
                
                table_content = element.metadata.get("text_as_html", element.page_content)
                
                
                content = f"### TABLE DE MAINTENANCE\n\n{table_content}\n"
                
                processed_docs.append(Document(
                    page_content=content, 
                    metadata=element.metadata
                ))
            else:
                processed_docs.append(element)

        embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )

        
        
        
        semantic_splitter = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="percentile"
        )
        
        final_chunks = semantic_splitter.split_documents(processed_docs)

        
        vector_store_service.add_documents(final_chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(final_chunks),
            "table_processing": "Strict Markdown/HTML Hybrid"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du parsing hybride : {str(e)}")