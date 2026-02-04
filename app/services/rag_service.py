from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.services.vector_store import vector_store_service
from app.core.config import settings




class RAGService :
    def __init__(self):
        self.llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL
        )
        
        self.template = """Vous êtes MediAssist Pro, un assistant expert en maintenance de matériel médical.
        Utilisez les extraits de manuels fournis pour répondre à la question du technicien.
        Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'inventez rien.
        
        CONTEXTE : {context}
        
        QUESTION : {question}
        
        RÉPONSE TECHNIQUE :"""
        
        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["context", "question"]
        )
        
        
    async def get_response(self, user_query: str):
        vector_store = vector_store_service.get_vector_store()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        response = qa_chain.invoke(user_query)
        return response["result"]


rag_service = RAGService()