import asyncio
import json
from typing import List

from langchain_core.language_models import BaseChatModel


from langchain.pydantic_v1 import BaseModel, Field

from prompts.entity_relationship import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT
)

from config import MAX_MESSAGES



class Entity(BaseModel):
    name: str = Field(description="Name of the entity, capitalized")
    type: str = Field(description="One of the following entity types")
    description: str = Field(description="Comprehensive description of the entity's attributes and activities")
    
class Relationship(BaseModel):
    source: str = Field(description="Name of the source entity, as identified in step 1")
    target: str = Field(description="Name of the target entity, as identified in step 1")
    relationship: str = Field(description="Explanation as to why you think the source entity and the target entity are related to each other")
    relationship_strength: int = Field(description="An integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity")


class EntityRelationship(BaseModel):
    entities: List[Entity] = Field(description="The list of entities")
    relationships: List[Relationship] = Field(description="The list of relationships")

async def generate_entity_relationship_examples(
    llm: BaseChatModel,
    persona: str,
    entity_types: str | List[str] | None,
    docs: str | List[str],
    language: str,
    json_mode: bool = False
) -> List[str]:
    """Generate a list of entity/relationships examples for use in generating an entity configuration.

    Will return entity/relationships examples as either JSON or in tuple_delimiter format depending
    on the json_mode parameter.
    """
    
    docs_list = [docs] if isinstance(docs, str) else docs

    
    if entity_types:
        entity_types_str = (
            entity_types
            if isinstance(entity_types, str)
            else ", ".join(map(str, entity_types))
        )
        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(entity_types=entity_types_str, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        messages = [
            UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                input_text=doc, language=language
            )
            for doc in docs_list
        ]
    
    messages = messages[:MAX_MESSAGES]
    
    model_answer = llm.with_structured_output(EntityRelationship) if json_mode else llm
    
    tasks = [
        model_answer.invoke([
                {
                    "role": "system",
                    "content": persona
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        ) for message in messages
    ]
    
    responses = await asyncio.gather(*tasks)
            
    
    return [
        json.dumps(response.dict() or "") if json_mode else str(response.content)
        for response in responses
    ]
    