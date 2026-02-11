import re
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from app.core.config import settings

class DeepEvalOllama(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model = ChatOllama(
            model=model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0, # Crucial pour la stabilité du format
            format="json"  # Force Ollama à utiliser le mode JSON si supporté 
        )

    def load_model(self):
        return self.model

    def _clean_json_output(self, output: str) -> str:
        """Extrait uniquement le bloc JSON pour éviter les erreurs de parsing."""
        # Recherche le premier '{' et le dernier '}'
        match = re.search(r'(\{.*\}|\[.*\])', output, re.DOTALL)
        if match:
            return match.group(1)
        return output

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.invoke(prompt)
        return self._clean_json_output(res.content)

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return self._clean_json_output(res.content)

    def get_model_name(self):
        return f"Ollama {settings.LLM_MODEL}"