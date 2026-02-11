# import logging
# import mlflow
# from typing import List, Any
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI

# from app.services.vector_store import vector_store_service
# from app.core.config import settings

# from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
# from deepeval.test_case import LLMTestCase

# from app.services.deep_eval import DeepEvalOllama


# logger = logging.getLogger(__name__)

# class RAGService:
#     def __init__(self):
        
#         mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
#         mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
#         mlflow.langchain.autolog() # 
        
#         self.llm = ChatOllama(
#             model=settings.LLM_MODEL,
#             base_url=settings.OLLAMA_BASE_URL,
#             temperature=0
#         )
        
#         self.reference_llm = ChatGoogleGenerativeAI(
#             model="gemini-flash-latest",
#             google_api_key=settings.GEMINI_API_KEY,
#             temperature=0
#         )
        
#         self.reranker_model = "ms-marco-MiniLM-L-12-v2"
#         self.reranker = FlashrankRerank(model=self.reranker_model)

#         self.prompt_template = """
#             Tu es MediAssist-Pro, un expert technique en maintenance de matériel biomédical.
#             Réponds à la question en utilisant UNIQUEMENT le contexte fourni.
            
#             RÈGLES :
#             1. Si la réponse n'est pas dans le contexte, dis-le. N'invente rien.
#             2. Indique toujours la SECTION et la SOURCE (ex: [Section: Sécurité, Source: documentation.pdf]).
#             3. Pour les dépannages, utilise des étapes claires (1, 2, 3...).

#             CONTEXTE TECHNIQUE :
#             {context}

#             QUESTION :
#             {question}

#             RÉPONSE EXPERTE :
#         """
        
#         self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
    
    
    
    
#     async def _generate_reference_answer(self, question: str, context: str) -> str:
#         try:
#             ref_prompt = f"""
#                 En tant qu'expert biomédical, rédige une réponse EXHAUSTIVE et EXACTE à la question suivante en utilisant le contexte fourni. 
#                 Ta réponse servira de référence de vérité terrain.
                
#                 CONTEXTE : {context}
#                 QUESTION : {question}
#                 RÉPONSE DE RÉFÉRENCE :
#             """
            
#             res = await self.reference_llm.ainvoke(ref_prompt)
#             return res.content
#         except Exception as e:
#             logger.error(f"Erreur Gemini Reference : {e}")
#             return ""
    
    
    
    
#     def _log_rag_config(self):
#         try:
#             mlflow.log_params({
#                 "llm_model": settings.LLM_MODEL,
#                 "llm_temperature": 0,
#                 "llm_top_p": 0.9,
#                 "llm_top_k": 40,
#                 "prompt_template": self.prompt_template
#             })

#             mlflow.log_params({
#                 "chunk_size": settings.CHUNK_SIZE,
#                 "chunk_overlap": settings.CHUNK_OVERLAP,
#                 "embedding_model": settings.EMBEDDING_MODEL,
#                 "retrieval_k": 10,
#                 "retrieval_similarity_algo": "cosine",
#                 "rerank_model": self.reranker_model,
#             })
            
#             mlflow.log_params({
#                 "embedding_model": settings.EMBEDDING_MODEL,
#                 "embedding_dimension": 1024,
#                 "embedding_normalization": "False" 
#             })
            
#             logger.info("Configuration RAG enregistrée dans MLflow.")
#         except Exception as e:
#             logger.warning(f"Impossible de logger dans MLflow : {e}")




#     async def _evaluate_performance(self, question: str, response: str, retrieved_docs: List[Any]):
#         try:
#             context_text = self._format_docs(retrieved_docs)
#             context_list = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]

#             expected_output = await self._generate_reference_answer(question, context_text)
            
#             if not expected_output:
#                 logger.warning("Évaluation annulée : expected_output vide.")
#                 return
            
            
#             test_case = LLMTestCase(
#                 input=question,
#                 actual_output=response,
#                 expected_output=expected_output,
#                 retrieval_context=context_list
#             )

#             local_model = DeepEvalOllama(model_name=settings.LLM_MODEL)
            
#             relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=local_model)
#             faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=local_model)
#             precision_metric = ContextualPrecisionMetric(threshold=0.5, model=local_model)
#             recall_metric = ContextualRecallMetric(threshold=0.5, model=local_model)

#             relevancy_metric.measure(test_case)
#             faithfulness_metric.measure(test_case)
#             precision_metric.measure(test_case)
#             recall_metric.measure(test_case)
            
#             print("answer_relevance :", relevancy_metric.score)
#             print("faithfulness :", faithfulness_metric.score,)
#             print("contextual_precision :", precision_metric.score,)
#             print("contextual_recall :", recall_metric.score)

#             mlflow.log_metrics({
#                 "answer_relevance": relevancy_metric.score,
#                 "faithfulness": faithfulness_metric.score,
#                 "contextual_precision": precision_metric.score,
#                 "contextual_recall": recall_metric.score
#             })
            
#             mlflow.log_text(expected_output, "evaluation/gemini_reference.txt")
            
#             logger.info("Métriques DeepEval enregistrées.")
            
#         except Exception as e:
#             logger.error(f"Erreur lors de l'évaluation DeepEval : {e}")




#     # def _format_docs(self, docs: List[Any]) -> str:
#     #     formatted = []
#     #     for doc in docs:
#     #         m = doc.metadata
#     #         header = f"[SOURCE: {m.get('source')} | CHAPITRE: {m.get('chapter')} | SECTION: {m.get('section')}]"
#     #         formatted.append(f"{header}\n{doc.page_content}")
#     #     return "\n\n".join(formatted)
    
#     def _format_docs(self, docs: List[Any]) -> str:
#         """Formatage sécurisé des documents pour le prompt."""
#         formatted = []
#         for doc in docs:
#             if hasattr(doc, 'metadata'):
#                 m = doc.metadata
#                 header = f"[SOURCE: {m.get('source', 'Inconnue')} | CHAPITRE: {m.get('chapter', 'N/A')}]"
#                 content = doc.page_content
#             else:
#                 header = "[SOURCE: Contextuelle]"
#                 content = str(doc)
#             formatted.append(f"{header}\n{content}")
#         return "\n\n".join(formatted)





#     async def answer_question(self, question: str, user_id: int):
#         """Pipeline RAG complet."""
#         with mlflow.start_run(run_name="RAG_Configuration") as run :
#             self._log_rag_config()
#             try:
#                 base_retriever = vector_store_service.get_hybrid_retriever(k=10)
                
#                 mq_retriever = MultiQueryRetriever.from_llm(
#                     retriever=base_retriever, 
#                     llm=self.llm
#                 )
                
#                 compression_retriever = ContextualCompressionRetriever(
#                     base_compressor=self.reranker, 
#                     base_retriever=mq_retriever
#                 )
                
#                 retrieved_docs = await compression_retriever.ainvoke(question)
#                 context_text = self._format_docs(retrieved_docs)
                
#                 # context_list = [doc.page_content for doc in retrieved_docs]
                
                
#                 # chain = (
#                 #     {"context": lambda x: context_text, "question": RunnablePassthrough()}
#                 #     | self.prompt
#                 #     | self.llm
#                 #     | StrOutputParser()
#                 # )
                
#                 # mlflow.langchain.log_model(chain, name="rag_pipeline")
                
#                 # ai_response = await chain.ainvoke(question)
                
#                 chain = self.prompt | self.llm | StrOutputParser()
                
#                 ai_response = await chain.ainvoke({"context": context_text, "question": question})

#                 mlflow.log_text(question, "inputs/question.txt")
#                 mlflow.log_text(ai_response, "outputs/ai_response.txt")
#                 mlflow.log_text(context_text, "artifacts/context.txt")

#                 await self._evaluate_performance(question, ai_response, retrieved_docs)

#                 return ai_response

#             except Exception as e:
#                 logger.error(f"Erreur RAG : {str(e)}")
#                 mlflow.log_param("error", str(e))
#                 return "Une erreur de connexion au moteur d'IA a eu lieu."


# rag_service = RAGService()


import logging
import mlflow
import mlflow.langchain
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
        
        # CORRECTIF : Désactiver log_models pour éviter les conflits de sérialisation asynchrone
        mlflow.langchain.autolog()
        
        # Hyperparamètres LLM
        self.llm_params = {
            "model": settings.LLM_MODEL,
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 0.9,
            "top_k": 40
        }

        self.llm = ChatOllama(
            model=self.llm_params["model"],
            base_url=settings.OLLAMA_BASE_URL,
            temperature=self.llm_params["temperature"]
        )
        
        self.reranker_model = "ms-marco-MiniLM-L-12-v2"
        self.reranker = FlashrankRerank(model=self.reranker_model)

        self.prompt = ChatPromptTemplate.from_template("""
            Tu es MediAssist-Pro, un expert technique en maintenance de matériel biomédical.
            Réponds à la question en utilisant UNIQUEMENT le contexte fourni.
            {context}
            QUESTION : {question}
            RÉPONSE EXPERTE :
        """)
        
        self.reference_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0
        )

        self._log_rag_config()

    def _log_rag_config(self):
        """Logger les hyperparamètres du LLM et du RAG."""
        try:
            with mlflow.start_run(run_name="RAG_Configuration_Audit", nested=True):
                mlflow.log_params(self.llm_params)
                mlflow.log_params({
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "reranker_model": self.reranker_model
                })
        except Exception as e:
            logger.warning(f"Logging MLflow échoué : {e}")

    async def _generate_reference_answer(self, question: str, context: str) -> str:
        """Génère la réponse de référence et garantit le retour d'une String."""
        try:
            ref_prompt = f"CONTEXTE :\n{context}\n\nQUESTION :\n{question}\n\nRédige la réponse de référence idéale :"
            res = await self.reference_llm.ainvoke(ref_prompt)
            
            # CORRECTIF : Gestion du format de réponse Gemini (String vs List[Dict])
            content = res.content
            if isinstance(content, list):
                return "".join([part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"])
            return str(content)
        except Exception as e:
            logger.error(f"Erreur Gemini Reference : {e}")
            return ""

    async def _evaluate_performance(self, question: str, response: str, retrieved_docs: List[Any]):
        """Calcule les métriques DeepEval avec sécurité sur les types."""
        try:
            context_text = self._format_docs(retrieved_docs)
            
            # CORRECTIF : Extraction sécurisée du texte des chunks (évite l'erreur d'attribut 'metadata')
            context_list = [d.page_content if hasattr(d, 'page_content') else str(d) for d in retrieved_docs]

            expected_output = await self._generate_reference_answer(question, context_text)
            if not expected_output: 
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
            logger.info(f"📊 Évaluation DeepEval réussie : {results}")

        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation DeepEval : {e}")

    def _format_docs(self, docs: List[Any]) -> str:
        """Formatage supportant les objets Documents et les chaînes de texte."""
        formatted = []
        for doc in docs:
            if hasattr(doc, 'metadata'):
                m = doc.metadata
                header = f"[SOURCE: {m.get('source', 'N/A')} | CHAPITRE: {m.get('chapter', 'N/A')}]"
                content = doc.page_content
            else:
                header = "[SOURCE: Manuel]"
                content = str(doc)
            formatted.append(f"{header}\n{content}")
        return "\n\n".join(formatted)

    async def answer_question(self, question: str, user_id: int):
        """Pipeline RAG avec logging des réponses et contextes."""
        try:
            with mlflow.start_run(run_name=f"Query_{user_id}", nested=True):
                base_retriever = vector_store_service.get_hybrid_retriever(k=10)
                mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)
                compression_retriever = ContextualCompressionRetriever(base_compressor=self.reranker, base_retriever=mq_retriever)
                
                retrieved_docs = await compression_retriever.ainvoke(question)
                context_text = self._format_docs(retrieved_docs)
                
                chain = (
                    {"context": lambda x: context_text, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                ai_response = await chain.ainvoke(question)

                # Logging des entrées/sorties en texte brut
                mlflow.log_text(question, "inputs/question.txt")
                mlflow.log_text(ai_response, "outputs/ai_response.txt")
                mlflow.log_text(context_text, "artifacts/context_used.txt")

                # Évaluation
                await self._evaluate_performance(question, ai_response, retrieved_docs)

                return ai_response

        except Exception as e:
            logger.error(f"Erreur RAG : {str(e)}")
            mlflow.log_param("error", str(e))
            return "Une erreur technique est survenue."

rag_service = RAGService()