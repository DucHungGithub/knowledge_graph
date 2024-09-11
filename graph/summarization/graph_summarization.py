import json
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from graph.summarization.prompts import SUMMARIZE_PROMPT
from utils import list_of_token
from verbs.entities.summarization.typing import SummarizedDescriptionResult


# Max token size for input prompts
DEFAULT_MAX_INPUT_TOKENS = 4_000
# Max token count for LLM answers
DEFAULT_MAX_SUMMARY_LENGTH = 500



class SummarizeExtractor:
    _llm: BaseChatModel
    _entity_types: List[str]
    _llm: BaseChatModel
    _entity_name_key: str | None = None
    _input_description_key: str | None = None
    _summarization_prompt: str | None = None
    _max_summary_length: int | None = None
    _max_input_tokens: int | None = None
    
    
    def __init__(
        self,
        llm: BaseChatModel,
        entity_name_key: str | None = None,
        input_description_key: str | None = None,
        summarization_prompt: str | None = None,
        max_summary_length: int | None = None,
        max_input_tokens: int | None = None
    ) -> None:
        self._llm = llm
        self._entity_name_key = entity_name_key or "entity_name"
        self._input_description_key = input_description_key or "description_list"
        
        self._summarization_prompt = summarization_prompt or SUMMARIZE_PROMPT
        self._max_summary_length = max_summary_length or DEFAULT_MAX_SUMMARY_LENGTH
        self._max_input_tokens = max_input_tokens or DEFAULT_MAX_INPUT_TOKENS
        
        self._llm.max_tokens=self._max_summary_length
        
    async def invoke(
        self,
        items: str| Tuple[str, str],
        descriptions: List[str]
    ) -> SummarizedDescriptionResult:
        result = ""
        if len(descriptions) == 0:
            result = ""
        if len(descriptions) == 1:
            result = descriptions[0]
        else:
            result = await self.summarize_descriptions(items=items, descriptions=descriptions)
        
        return SummarizedDescriptionResult(
            items=items,
            description=result or ""
        )
        
        
    async def summarize_descriptions(
        self,
        items: str | Tuple[str, str],
        descriptions: List[str]
    ) -> str:
        sorted_items = sorted(items) if isinstance(items, List) else items
        
        if not isinstance(descriptions, List):
            descriptions = [descriptions]
            
        usable_tokens = self._max_input_tokens - len(list_of_token(
            self._summarization_prompt
        ))
        
        descriptions_collected = []
        result = ""
        
        for i, description in enumerate(descriptions):
            usable_tokens -= len(list_of_token(description))
            descriptions_collected.append(description)
            
            if (usable_tokens < 0 and len(descriptions_collected) > 1) or (i==len(descriptions) -1):
                result = await self.summarize_descriptions_with_llm(
                    sorted_items, descriptions_collected
                )
                if i!= len(descriptions) -1:
                    descriptions_collected = [result]
                    usable_tokens = (
                        self._max_input_tokens - len(list_of_token(self._summarization_prompt)) - len(list_of_token(result))
                    )
        return result
                
    
    async def summarize_descriptions_with_llm(
        self, items: str | Tuple[str, str] | List[str], descriptions: List[str]
    ):
        
        template = ChatPromptTemplate.from_template([
            ("system", self._summarization_prompt)   
        ]
        )
        
        variables = {
            self._entity_name_key: json.dumps(items),
            self._input_description_key: json.dumps(sorted(descriptions), ensure_ascii=False)
        }

        prompt = template.format(**variables)
        
        

        response = self._llm.invoke(
            prompt
        )
        return response.content
                
                