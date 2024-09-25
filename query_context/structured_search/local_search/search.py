from collections.abc import AsyncGenerator
import logging
import time
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
import pydgraph

from query_context.structured_search.base import BaseSearch, SearchResult

from query_context.structured_search.local_search.prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from query_context.system_prompt_builder.builders import LocalContextBuilder
from query_context.system_prompt_builder.history.conversation_history import ConversationHistory
from utils import list_of_token


import colorlog

# Set up basic logging configuration with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# Get the logger and add the handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.addHandler(handler)


class LocalSearch(BaseSearch):
    """Seatch orchestration for local search mode"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        context_builder: LocalContextBuilder,
        token_encoder: Optional[str] = None,
        system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
        response_type: str = "multiple paragraphs",
        context_builder_params: Optional[Dict[str, Any]] = None,        
    ):
        super().__init__(
            llm = llm,
            context_builder = context_builder,
            token_encoder = token_encoder,
            context_builder_params = context_builder_params
        )
        self.system_prompt = system_prompt
        self.response_type = response_type
        
    def search(
        self, 
        client: pydgraph.DgraphClient,
        query: str, 
        conversation_history: ConversationHistory | None = None, 
        **kwargs
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user question."""
        start_time = time.time()
        search_prompt = ""
        context_text, context_records = self.context_builder.build_context(
            client=client,
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params
        )
        logger.info(f"GENERATE ANSWER {start_time}. QUERY: {query}")
        try:
            search_prompt = self.system_prompt.format(
                context_data = context_text, response_type = self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query}
            ]
            
            
            response = self.llm.invoke(search_messages)
            
            return SearchResult(
                response=response.content,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time()-start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(search_prompt, self.token_encoder))
            )
            
        except Exception:
            logger.exception("Exception in _map_response_single_batch", exc_info=True)
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(search_prompt, self.token_encoder)),
            )
    
    
    async def asearch(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""

        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        logger.info(f"GENERATE ANSWER {start_time}. QUERY: {query}")
        try:
            search_prompt = self.system_prompt.format(
                context_data = context_text, response_type = self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query}
            ]
            
            response = await self.llm.ainvoke(search_messages)
            
            return SearchResult(
                response=response.content,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(search_prompt, self.token_encoder)),
            )

        except Exception:
            logger.exception("Exception in _asearch", exc_info=True)
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(search_prompt, self.token_encoder)),
            )
            
            
    async def astream_search(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None
    ) -> AsyncGenerator:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()

        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **self.context_builder_params,
        )
        
        logger.info(f"GENERATE ANSWER {start_time}. QUERY: {query}")
        search_prompt = self.system_prompt.format(
            context_data = context_text, response_type = self.response_type
        )
        search_messages = [
            {"role": "system", "content": search_prompt},
            {"role": "user", "content": query}
        ]
        
        yield context_records
        async for response in self.llm.astream(search_messages):
            yield response