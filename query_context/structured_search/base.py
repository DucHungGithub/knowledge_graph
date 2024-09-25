from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

from langchain_core.language_models import BaseChatModel


from query_context.system_prompt_builder.builders import GlobalContextBuilder, LocalContextBuilder
from query_context.system_prompt_builder.history.conversation_history import ConversationHistory

@dataclass
class SearchResult:
    """A Structured Search Result."""

    response: str | Dict[str, Any] | List[Dict[str, Any]]
    context_data: str | List[pd.DataFrame] | Dict[str, pd.DataFrame]
    # actual text strings that are in the context window, built from context_data
    context_text: str | List[str] | Dict[str, str]
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    
@dataclass
class GlobalSearchResult(SearchResult):
    """A GlobalSearchResult."""
    
    map_responses: List[SearchResult]
    reduce_context_data: str | List[pd.DataFrame] | Dict[str, pd.DataFrame]
    reduce_context_text: str | List[str] | Dict[str, str]


class BaseSearch(ABC):
    def __init__(
        self,
        llm: BaseChatModel,
        context_builder: LocalContextBuilder | GlobalContextBuilder,
        token_encoder: Optional[str] = None,
        context_builder_params: Optional[Dict[str, Any]] = None
    ) -> None:
        self.llm = llm
        self.context_builder = context_builder
        self.token_encoder = token_encoder
        self.context_builder_params = context_builder_params or {}
        
    @abstractmethod
    def search(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> SearchResult:
        """Search for the given query."""
        

    @abstractmethod
    async def asearch(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> SearchResult:
        """Search for the given query asynchronously"""
        
    @abstractmethod
    async def astream_search(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream search for the given query."""