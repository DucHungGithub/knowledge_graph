from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from query_context.system_prompt_builder.builders import GlobalContextBuilder, LocalContextBuilder

class QuestionResult(BaseModel):
    response: List[str]
    context_data: str | Dict[str, Any]
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    

class BaseQuestionGen(ABC):
    def __init__(
        self,
        llm: BaseChatModel,
        context_builder: GlobalContextBuilder | LocalContextBuilder,
        token_encoder: str | None = None,
        context_builder_params: Optional[Dict[str, Any]] = None
    ) -> None:
        self.llm = llm
        self.context_builder = context_builder
        self.token_encoder = token_encoder
        self.context_builder_params = context_builder_params or {}
        
    @abstractmethod
    def generate(
        self,
        question_history: List[str],
        context_data: Optional[str],
        question_count: int,
        **kwargs
    ) -> QuestionResult:
        """Generate questions."""
        
    @abstractmethod
    async def agenerate(
        self,
        question_history: List[str],
        context_data: Optional[str],
        question_count: int,
        **kwargs
    ) -> QuestionResult:
        """Generate questions asynchronously."""