import logging
from typing import List, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.services.vector_store import vector_store_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=8192
        )

        self.prompt = ChatPromptTemplate.from_template("""
            Tu es MediAssist-Pro, un expert technique en maintenance de matériel biomédical.
            Réponds à la question en utilisant UNIQUEMENT le contexte fourni.
            
            RÈGLES :
            1. Si la réponse n'est pas dans le contexte, dis-le. N'invente rien.
            2. Indique toujours la SECTION et la SOURCE (ex: [Section: Sécurité, Source: documentation.pdf]).
            3. Pour les dépannages, utilise des étapes claires (1, 2, 3...).

            CONTEXTE TECHNIQUE :
            {context}

            QUESTION :
            {question}

            RÉPONSE EXPERTE :
        """)


    def _format_docs(self, docs: List[Any]) -> str:
        formatted = []
        for doc in docs:
            m = doc.metadata
            header = f"[SOURCE: {m.get('source')} | CHAPITRE: {m.get('chapter')} | SECTION: {m.get('section')}]"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(formatted)


    async def answer_question(self, question: str, user_id: int):
        """Pipeline RAG complet."""
        try:
            retriever = vector_store_service.get_hybrid_retriever(k=5)
            
            chain = (
                {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            return await chain.ainvoke(question)

        except Exception as e:
            logger.error(f"Erreur RAG : {str(e)}")
            return "Une erreur de connexion au moteur d'IA a eu lieu."


rag_service = RAGService()