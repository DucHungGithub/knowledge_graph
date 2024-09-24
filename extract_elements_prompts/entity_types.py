import json
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain.pydantic_v1 import BaseModel, Field

from extract_elements_prompts.defaults import DEFAULT_TASK
from prompts.entity_types import (
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
    ENTITY_TYPE_GENERATION_PROMPT
)


class EntityTypes(BaseModel):
    entity_types: List[str] = Field(description="The list of entity types.")


async def generate_entity_types(
    llm: BaseChatModel,
    domain: str,
    persona: str,
    docs: str | List[str],
    task: str = DEFAULT_TASK,
    json_mode: bool = False
) -> str | List[str]:
    """
    Generate entity type categories from a given set of (small) documents.

    Example Output:
    "entity_types": ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']
    """
    
    formatted_task = task.format(domain=domain)
    
    formatted_docs = "\n".join(docs) if isinstance(docs, List) else docs
    
    entity_types_prompt = (
        ENTITY_TYPE_GENERATION_JSON_PROMPT
        if json_mode
        else ENTITY_TYPE_GENERATION_PROMPT
    ).format(task=formatted_task, input_text=formatted_docs)
    
    messages = [
        {
            "role": "system", "content":  persona,
        },
        {
            "role": "user", "content": entity_types_prompt
        }
    ]
    
    if json_mode:
        structure_llm = llm.with_structured_output(EntityTypes)
        response = structure_llm.invoke(
            input=messages
        )
        return response.entity_types or []
    
    response = llm.invoke(
        input=messages
    )
    
    return str(response.content)

