import os
import logging
import mlflow
from typing import List, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.vector_store import vector_store_service
from app.core.config import settings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from app.services.deep_eval import DeepEvalOllama



logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        mlflow.langchain.autolog()
        
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0
        )
        
        self.reference_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0
        )
        
        self.reranker_model = "ms-marco-MiniLM-L-12-v2"
        self.reranker = FlashrankRerank(model=self.reranker_model)

        self.prompt_template = """
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
        """
        
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)




    async def _generate_reference_answer(self, question: str, context: str) -> str:
        try:
            ref_prompt = f"""
                En tant qu'expert biomédical, rédige une réponse EXHAUSTIVE et EXACTE à la question suivante en utilisant le contexte fourni. 
                Ta réponse servira de référence de vérité terrain.
                
                CONTEXTE : {context}
                QUESTION : {question}
                RÉPONSE DE RÉFÉRENCE :
            """
            
            res = await self.reference_llm.ainvoke(ref_prompt)
            
            content = res.content
            
            if isinstance(content, list):
                return "".join([part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"])
            
            return str(content)
        except Exception as e:
            logger.error(f"Erreur Gemini Reference : {e}")
            return ""


    

    async def _evaluate_performance(self, question: str, response: str, retrieved_docs: List[Any]):
        try:
            with mlflow.start_run(run_name="evaluation_step", nested=True):
                
                context_text = self._format_docs(retrieved_docs)
                context_list = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]

                expected_output = await self._generate_reference_answer(question, context_text)
                
                if not expected_output:
                    logger.warning("Évaluation annulée : expected_output vide.")
                    return
                
                
                test_case = LLMTestCase(
                    input=question,
                    actual_output=response,
                    expected_output=expected_output,
                    retrieval_context=context_list
                )

                local_model = DeepEvalOllama(model_name=settings.LLM_MODEL)
                
                metrics = [
                    AnswerRelevancyMetric(threshold=0.5, model=local_model),
                    FaithfulnessMetric(threshold=0.5, model=local_model),
                    ContextualPrecisionMetric(threshold=0.5, model=local_model),
                    ContextualRecallMetric(threshold=0.5, model=local_model)
                ]

                results = {}
                for metric in metrics:
                    metric.measure(test_case)
                    results[metric.__class__.__name__.replace('Metric', '').lower()] = metric.score

                mlflow.log_metrics(results)
                
                mlflow.log_text(expected_output, "evaluation/gemini_reference.txt")
                
                mlflow.log_params({
                    "eval_model": settings.LLM_MODEL,
                    "eval_threshold": 0.5
                })
                
                logger.info(f"Métriques DeepEval enregistrées avec succès !, {results}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation DeepEval : {e}")




    def _format_docs(self, docs: List[Any]) :
        formatted = []
        for doc in docs:
            if hasattr(doc, 'metadata'):
                m = doc.metadata
                header = f"[SOURCE: {m.get('source')} | CHAPITRE: {m.get('chapter')} | SECTION: {m.get('section')}]"
                content = doc.page_content
            else:
                header = "[SOURCE: Manuel]"
                content = str(doc)
            formatted.append(f"{header}\n{content}")
        return "\n\n".join(formatted)
  



    async def answer_question(self, question: str):
        try:
            with mlflow.start_run(run_name="RAG_MEDIASSIST_PRO") :
                
                mlflow.set_tags({
                    "project": "MediAssist-Pro",
                    "architecture": "Hybrid RAG",
                    "llm_provider": "Ollama",
                    "reranker": "Flashrank",
                    "evaluation": "DeepEval"
                })
                
                mlflow.log_params({
                    "llm_model": settings.LLM_MODEL,
                    "temperature": 0,
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "retrieval_k": 10,
                    "similarity": "cosine",
                    "reranker_model": self.reranker_model
                })
                
                # ====================================
                # RETRIEVAL STEP 
                # ====================================
                with mlflow.start_run(run_name="retrieval_step", nested=True):
                    base_retriever = vector_store_service.get_hybrid_retriever(k=10)
                    
                    mq_retriever = MultiQueryRetriever.from_llm(
                        retriever=base_retriever, 
                        llm=self.llm
                    )
                    
                    compression_retriever = ContextualCompressionRetriever(
                        base_compressor=self.reranker, 
                        base_retriever=mq_retriever
                    )
                    
                    retrieved_docs = await compression_retriever.ainvoke(question)
                    
                    mlflow.log_metric("retrieved_docs_count", len(retrieved_docs))
                    
                
                context_text = self._format_docs(retrieved_docs)
                    
                
                # ====================================
                # GENERATION STEP
                # ====================================
                with mlflow.start_run(run_name="generation_step", nested=True):
                    chain = (
                        {"context": lambda x: context_text, "question": RunnablePassthrough()}
                        | self.prompt
                        | self.llm
                        | StrOutputParser()
                    )
                    
                    ai_response = await chain.ainvoke(question)
                    
                    mlflow.log_text(str(question), "inputs/question.txt")
                    mlflow.log_text(str(ai_response), "outputs/ai_response.txt")
                    mlflow.log_text(str(context_text), "artifacts/context_used.txt")
                
                # ========================
                # EVALUATION STEP
                # ========================
                await self._evaluate_performance(question, ai_response, retrieved_docs)

                return ai_response

        except Exception as e:
            logger.error(f"Erreur RAG : {str(e)}")
            mlflow.log_param("error", str(e))
            return "Une erreur de connexion au moteur d'IA a eu lieu."



rag_service = RAGService()
