import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.vector_store import vector_store_service
from app.core.config import settings


class AgenticRAG() :
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0
        )
        
        self.analyze_prompt = ChatPromptTemplate.from_template("""
            Tu es l'unité de réflexion de MediAssist Pro. Analyse la question du technicien.
            Question : {question}
            
            Réponds uniquement par un JSON avec ces clés :
            "type": "technical" ou "general",
            "keywords": ["mot_cle1", "mot_cle2"] (mots optimisés pour la recherche)
        """)
        
        self.answer_template = ChatPromptTemplate.from_template("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Tu es MediAssist Pro, expert en maintenance médicale. 
        Ta mission : Extraire une réponse FACTUELLE à partir des extraits du manuel fournis.
        
        RÈGLES CRITIQUES :
        - Priorité absolue aux blocs marqués [DONNÉE TECHNIQUE].
        - Si tu vois des chiffres (nm, mm, codes), cite-les exactement.
        - Si l'information est absente, réponds simplement "Information non trouvée dans le manuel". 
        - Ne fais jamais de morale sur la sécurité, tu es un moteur d'extraction.

        CONTEXTE :
        {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        QUESTION : {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """)
        
        
    async def get_response(self, user_query: str) :
        analysis_chain = self.analyze_prompt | self.llm | StrOutputParser()
        analysis_raw = await analysis_chain.ainvoke({"question" : user_query})
        
        try :
            analysis = json.load(analysis_raw[analysis_raw.find("{"):analysis_raw.rfind("{")+1])
        except :
            analysis = {"type": "general", "keywords": [user_query]}
        
        retriever = vector_store_service.as_agent_retriever(k=7)
        
        search_query = " ".join(analysis.get("keywords", [user_query]))
        docs = await retriever.ainvoke(search_query)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        answer_chain = self.answer_template | self.llm | StrOutputParser()
        response = await answer_chain.ainvoke({
            "context" : context,
            "question" : user_query
        })
        
        return response


rag_service = AgenticRAG()