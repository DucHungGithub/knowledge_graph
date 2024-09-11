from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def load_openai_llm(model: str, temperature: float) -> BaseChatModel:
    return ChatOpenAI(
        model=model,
        temperature=temperature
    )
    