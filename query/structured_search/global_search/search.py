import asyncio
from collections.abc import AsyncGenerator
import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from query.structured_search.base import BaseSearch, GlobalSearchResult, SearchResult

from query.structured_search.global_search.map_system_prompt import MAP_SYSTEM_PROMPT
from query.structured_search.global_search.reduce_system_prompt import GENERAL_KNOWLEDGE_INSTRUCTION, NO_DATA_ANSWER, REDUCE_SYSTEM_PROMPT
from query.system_prompt_builder.builders import GlobalContextBuilder
from query.system_prompt_builder.history.conversation_history import ConversationHistory
from utils import list_of_token, try_parse_json_object



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

DEFAULT_MAP_LLM_PARAMS = {
    "max_tokens": 1000,
    "temperature": 0.0,
}

DEFAULT_REDUCE_LLM_PARAMS = {
    "max_tokens": 2000,
    "temperature": 0.0,
}

class GlobalSearch(BaseSearch):
    """Search orchestration for global search mode."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        context_builder: GlobalContextBuilder,
        token_encoder: Optional[str] = None,
        map_system_prompt: str = MAP_SYSTEM_PROMPT,
        reduce_system_prompt: str = REDUCE_SYSTEM_PROMPT,
        response_type: str = "multiple paragraphs",
        allow_general_knowledge: bool = False,
        general_knowledge_inclusion_prompt: str = GENERAL_KNOWLEDGE_INSTRUCTION,
        json_mode: bool = True,
        max_data_tokens: int = 8000,
        map_llm_params: Dict[str, Any] = DEFAULT_MAP_LLM_PARAMS,
        reduce_llm_params: Dict[str, Any] = DEFAULT_REDUCE_LLM_PARAMS,
        context_builder_params: Optional[Dict[str, Any]] = None,
        concurrent_coroutines: int = 32        
    ) -> None:
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            context_builder_params=context_builder_params
        )
        self.map_system_prompt = map_system_prompt
        self.reduce_system_prompt = reduce_system_prompt
        self.response_type = response_type
        self.allow_general_knowledge = allow_general_knowledge
        self.general_knowledge_inclusion_prompt = general_knowledge_inclusion_prompt
        self.max_data_tokens = max_data_tokens
        
        self.map_llm_params = map_llm_params
        self.reduce_llm_params = reduce_llm_params
        self.json_mode = json_mode
        
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
    
    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """Perform a global search synchronously."""
        return asyncio.run(self.asearch(query, conversation_history))
    
    
    
    async def asearch(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> GlobalSearchResult:
        """
        Peform a global search.
        
        Global search mode includes two steps:
        
        - Step 1: Run parallel LLM calls on communities' short summaries to generate answer for each batch 
        - Step 2: Combine the answers from step 2 to generate the final answer
        """
        
        # Step 1: Generate answers for each batch of community short summaries
        start_time = time.time()
        context_chunks, context_records = self.context_builder.build_context(
            conversation_history=conversation_history, **self.context_builder_params
        )
        
        tasks = None
        
        if isinstance(context_chunks, List):
            tasks = [
                self._map_response_single_batch(
                    context_data=data, query=query, **self.map_llm_params
                )
                for data in context_chunks
            ]
        elif isinstance(context_chunks, str):
            tasks = [
                self._map_response_single_batch(
                    context_data=data, query=query, **self.map_llm_params
                )
                for data in context_chunks.split("\n\n")
            ]
        
        map_responses = await asyncio.gather(*tasks)
        
        map_llm_calls = sum(response.llm_calls for response in map_responses)
        map_prompt_tokens = sum(response.prompt_tokens for response in map_responses)
        
        # Step 2: Combine the intermediate answers from step 2 to generate the final answer
        reduce_response = await self._reduce_response(
            map_responses=map_responses,
            query=query,
            **self.reduce_llm_params,
        )
        
        return GlobalSearchResult(
            response=reduce_response.response,
            context_data=context_records,
            context_text=context_chunks,
            map_responses=map_responses,
            reduce_context_data=reduce_response.context_data,
            reduce_context_text=reduce_response.context_text,
            completion_time=time.time() - start_time,
            llm_calls=map_llm_calls + reduce_response.llm_calls,
            prompt_tokens=map_prompt_tokens + reduce_response.prompt_tokens,
        )
    
    
    
    async def _reduce_response(
        self,
        map_responses: List[SearchResult],
        query: str,
        **kwargs
    ) -> SearchResult:
        """Combine all intermediate responses from single batches into a final answer to the user query."""
        text_data = ""
        search_prompt = ""
        start_time = time.time()
        try:
            # Collect all key points into a single list to prepare for sorting
            key_points = []
            for index, response in enumerate(map_responses):
                if not isinstance(response.response, List):
                    continue
                for element in response.response:
                    if not isinstance(element, Dict):
                        continue
                    if "answer" not in element or "score" not in element:
                        continue
                    key_points.append({
                        "analyst": index,
                        "answer": element["answer"],
                        "score": element["score"]
                    })
                    
            # filter response with score = 0 and rank responses by descending order of score
            filtered_key_points = [
                point
                for point in key_points
                if point["score"] > 0  
            ]
            
            if len(filtered_key_points) == 0 and not self.allow_general_knowledge:
                # return no data answer if no key points are found
                logger.warning(
                    "Warning: All map responses have score 0 (i.e., no relevant information found from the dataset), returning a canned 'I do not know' answer. You can try enabling `allow_general_knowledge` to encourage the LLM to incorporate relevant general knowledge, at the risk of increasing hallucinations."
                )
                return SearchResult(
                    response=NO_DATA_ANSWER,
                    context_data="",
                    context_text="",
                    completion_time=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                )
            
            filtered_key_points = sorted(
                filtered_key_points,
                key=lambda x: x["score"],  
                reverse=True,
            )
            
            data = []
            total_tokens = 0
            for point in filtered_key_points:
                formatted_response_data = []
                formatted_response_data.append(
                    f'----Analyst {point["analyst"] + 1}----'
                )
                formatted_response_data.append(
                    f'Importance Score: {point["score"]}'
                )
                formatted_response_data.append(point["answer"]) 
                formatted_response_text = "\n".join(formatted_response_data)
                if (
                    total_tokens
                    + len(list_of_token(formatted_response_text, self.token_encoder))
                    > self.max_data_tokens
                ):
                    break
                data.append(formatted_response_text)
                total_tokens += len(list_of_token((formatted_response_text, self.token_encoder)))
            
            text_data = "\n\n".join(data)

            search_prompt = self.reduce_system_prompt.format(
                report_data=text_data, response_type=self.response_type
            )

            if self.allow_general_knowledge:
                search_prompt += "\n" + self.general_knowledge_inclusion_prompt
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]
            
            self.llm.max_tokens = self.map_llm_params["max_tokens"]
            
            search_response = await self.llm.ainvoke(search_messages)
            
            return SearchResult(
                response=search_response,
                context_data=text_data,
                context_text=text_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token((search_prompt, self.token_encoder))),
            )
        except Exception:
            logger.exception("Exception in reduce_response", exc_info=True)
            return SearchResult(
                response="",
                context_data=text_data,
                context_text=text_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token((search_prompt, self.token_encoder))),
            )
        
    
    
    async def astream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator:
        """Stream the global search response."""
        context_chunks, context_records = self.context_builder.build_context(
            conversation_history=conversation_history, **self.context_builder_params
        )
        
        
        tasks = None
        
        if isinstance(context_chunks, List):
            tasks = [
                self._map_response_single_batch(
                    context_data=data, query=query, **self.map_llm_params
                )
                for data in context_chunks
            ]
        elif isinstance(context_chunks, str):
            tasks = [
                self._map_response_single_batch(
                    context_data=data, query=query, **self.map_llm_params
                )
                for data in context_chunks.split("\n\n")
            ]
        
        
        map_responses = await asyncio.gather(*tasks)

        # send context records first before sending the reduce response
        yield context_records
        async for response in self._stream_reduce_response(
            map_responses=map_responses,  # type: ignore
            query=query,
            **self.reduce_llm_params,
        ):
            yield response
    
     
        
    async def _map_response_single_batch(
        self,
        context_data: str,
        query: str,
        **kwargs
    ) -> SearchResult:
        """Generate answer for a single chunk of community reports."""
        start_time = time.time()
        search_prompt = ""
        
        try:
            search_prompt = self.map_system_prompt.format(context_data=context_data)
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]
            
            llm_response = self.llm.bind(response_format={"type": "json_object"}) if self.json_mode else self.llm
            llm_response.max_tokens = self.map_llm_params["max_tokens"]
            
            async with self.semaphore:
                search_response = await llm_response.ainvoke(search_messages, **kwargs)
                logger.info(f"Map response: {search_response}")
                
            try:
                processed_response = self.parse_search_response(search_response)
            except ValueError:
                # Clean up and retry parse
                try:
                    # parse search response json
                    processed_response = self.parse_search_response(search_response)
                except ValueError:
                    logger.warning(
                        "Warning: Error parsing search response json - skipping this batch"
                    )
                    processed_response = []
            
            return SearchResult(
                response=processed_response,
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(search_prompt, self.token_encoder)),
            )

        except Exception:
            logger.exception("Exception in _map_response_single_batch", exc_info=True)
            return SearchResult(
                response=[{"answer": "", "score": 0}],
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(search_prompt, self.token_encoder)),
            )
    
    
          
    def parse_search_response(self, search_response: str) -> List[Dict[str, Any]]:
        """Parse the search response json and return a list of key points.

        Parameters
        ----------
        search_response: str
            The search response json string

        Returns
        -------
        list[dict[str, Any]]
            A list of key points, each key point is a dictionary with "answer" and "score" keys
        """
        search_response, _j = try_parse_json_object(search_response)
        if _j == {}:
            return [{"answer": "", "score": 0}]

        parsed_elements = json.loads(search_response).get("points")
        if not parsed_elements or not isinstance(parsed_elements, list):
            return [{"answer": "", "score": 0}]

        return [
            {
                "answer": element["description"],
                "score": int(element["score"]),
            }
            for element in parsed_elements
            if "description" in element and "score" in element
        ]
        
        
    async def _stream_reduce_response(
        self,
        map_responses: list[SearchResult],
        query: str,
        **llm_kwargs,
    ) -> AsyncGenerator[str, None]:
        # collect all key points into a single list to prepare for sorting
        key_points = []
        for index, response in enumerate(map_responses):
            if not isinstance(response.response, list):
                continue
            for element in response.response:
                if not isinstance(element, dict):
                    continue
                if "answer" not in element or "score" not in element:
                    continue
                key_points.append({
                    "analyst": index,
                    "answer": element["answer"],
                    "score": element["score"],
                })
                
        # filter response with score = 0 and rank responses by descending order of score
        filtered_key_points = [
            point
            for point in key_points
            if point["score"] > 0
        ]
        
        if len(filtered_key_points) == 0 and not self.allow_general_knowledge:
            # return no data answer if no key points are found
            logger.warning(
                "Warning: All map responses have score 0 (i.e., no relevant information found from the dataset), returning a canned 'I do not know' answer. You can try enabling `allow_general_knowledge` to encourage the LLM to incorporate relevant general knowledge, at the risk of increasing hallucinations."
            )
            yield NO_DATA_ANSWER
            return

        filtered_key_points = sorted(
            filtered_key_points,
            key=lambda x: x["score"],
            reverse=True,
        )
        
        
        data = []
        total_tokens = 0
        for point in filtered_key_points:
            formatted_response_data = [
                f'----Analyst {point["analyst"] + 1}----',
                f'Importance Score: {point["score"]}',
                point["answer"],
            ]
            formatted_response_text = "\n".join(formatted_response_data)
            if (
                total_tokens + len(list_of_token((formatted_response_text, self.token_encoder)))
                > self.max_data_tokens
            ):
                break
            data.append(formatted_response_text)
            total_tokens += len(list_of_token((formatted_response_text, self.token_encoder)))
        text_data = "\n\n".join(data)
        
        search_prompt = self.reduce_system_prompt.format(
            report_data=text_data, response_type=self.response_type
        )
        if self.allow_general_knowledge:
            search_prompt += "\n" + self.general_knowledge_inclusion_prompt
        search_messages = [
            {"role": "system", "content": search_prompt},
            {"role": "user", "content": query},
        ]
        
        async for resp in self.llm.astream( 
            search_messages,
            **llm_kwargs, 
        ):
            yield resp