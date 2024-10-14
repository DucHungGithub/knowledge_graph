from typing import List, Tuple

import pandas as pd

from models.entity import Entity

from external_utils.token import list_of_token

def build_entity_context(
    selected_entities: List[Entity],
    token_encoder: str = None,
    max_tokens: int = 8000,
    include_entity_rank: bool = True,
    rank_description: str = "number of relationships",
    column_delimiter: str = "|",
    context_name: str = "Entities"
) -> Tuple[str, pd.DataFrame]:
    """Prepare the entity data table as context data for system prompt"""
    if len(selected_entities) == 0:
        return "", pd.DataFrame()
    
    current_context_text = f"-----{context_name}-----" + "\n"
    header = ["id", "entity", "description"]
    
    if include_entity_rank:
        header.append(rank_description)
    
    attribute_cols = (
        list(selected_entities[0].attributes.keys())
        if selected_entities[0].attributes
        else []
    )
    header.extend(attribute_cols)
    current_context_text += column_delimiter.join(header) + "\n"
    
    current_tokens = len(list_of_token(current_context_text, token_encoder))
    
    all_context_records = [header]
    
    for entity in selected_entities:
        new_context = [
            entity.short_id if entity.short_id else "",
            entity.title,
            entity.description if entity.description else "", 
        ]
        
        if include_entity_rank:
            new_context.append(str(entity.rank))
        
        for field in attribute_cols:
            field_value = (
                str(entity.attributes.get(field))
                if entity.attributes and entity.attributes.get(field)
                else ""
            )
            new_context.append(field_value)
        
        # Ensure new_context has the same number of elements as header
        if len(new_context) != len(header):
            raise ValueError(f"Context size mismatch: {len(new_context)} != {len(header)}")
        
        new_context_text = column_delimiter.join(new_context) + "\n"
        new_tokens = len(list_of_token(new_context_text, token_encoder))
        
        if current_tokens + new_tokens > max_tokens:
            break
        
        current_context_text += new_context_text
        all_context_records.append(new_context)  # Append new_context, not new_tokens
        current_tokens += new_tokens
    
    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:], columns=all_context_records[0]
        )
    else:
        record_df = pd.DataFrame()
        
    return current_context_text, record_df

    