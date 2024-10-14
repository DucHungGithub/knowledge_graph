import json
import random
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pydgraph

from models.entity import Entity
from models.relationship import Relationship
from models.text_unit import TextUnit
from external_utils.token import list_of_token

"""
Contain util functions to build text unit context for the search's system prompt
"""


def build_text_unit_context(
    text_units: List[TextUnit],
    token_encoder: str = None,
    column_delimiter: str = "|",
    shuffle_data: bool = True,
    max_tokens: int = 8000,
    context_name: str = "Sources",
    random_state: int = 6969
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """Prepare text-unit data table as context data for system prompt."""
    if text_units is None or len(text_units) == 0:
        return ("", {})
    
    if shuffle_data:
        random.seed(random_state)
        random.shuffle(text_units)
        
    current_context_text = f"-----{context_name}-----" + "\n"
    
    header = ["id", "text", "source"]
    attribute_cols = (
        list(text_units[0].attributes.keys()) if text_units[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)
    
    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = len(list_of_token(current_context_text, token_encoder))
    all_context_records = [header]
    
    for unit in text_units:
        new_context = [
            unit.short_id,
            unit.text,
            unit.source,
            *[
                str(unit.attributes.get(field, "")) if unit.attributes else ""
                for field in attribute_cols
            ]
        ]
        
        new_context_text = column_delimiter.join(new_context) + "\n"
        new_tokens = len(list_of_token(new_context_text, token_encoder))
        
        if current_tokens + new_tokens > max_tokens:
            break
        
        current_context_text += new_context_text
        all_context_records.append(new_context)
        current_tokens += new_tokens
    
    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:], columns=cast(Any, all_context_records[0])
        )
    else:
        record_df = pd.DataFrame()
    return current_context_text, {context_name.lower(): record_df}



def count_relationship(
    client: pydgraph.DgraphClient,
    text_unit: TextUnit,
    entity: Entity
) -> int:
    """Count the number of relationships of the selected entity that are associated with the text unit."""
    entity_relationships = []
    txn = client.txn()
    matching_relationships = []
    try:
        if text_unit.relationship_ids is None:
            query = f"""
            {{
                getRelationships(func: type(Entity)) @filter(has(connect)) {{
                    connect @facets(eq(source, "{entity.title}") OR eq(target, "{entity.title}")) @facets{{
                        expand(_all_)
                    }}
                }}
            }}
            """
            
            res = txn.query(query=query)
            ppl = json.loads(res.json)
            
            relationships = ppl.get("getRelationships", [])
            
            for rel in relationships:
                connects = rel["connect"]
                for con in connects:
                    entity_relationships.append(Relationship(
                        id=con["connect|id"],
                        short_id=con["connect|short_id"],
                        source=con["connect|source"],
                        target=con["connect|target"],
                        weight=con["connect|weight"],
                        description=con["connect|description"],
                        text_unit_ids=json.loads(con["connect|text_unit_ids"]) if con.get("connect|text_unit_ids", None) else None
                    ))
                    
            entity_relationships = [
                rel for rel in entity_relationships if rel.text_unit_ids
            ]
            
            matching_relationships = [
                rel 
                for rel in entity_relationships
                if text_unit.id in rel.text_unit_ids
            ]
        else:
            for rel_id in text_unit.relationship_ids:
                query = f"""
                {{
                    getRelationships(func: type(Entity)) @filter(has(connect)) {{
                        connect @facets(eq(id, {rel_id})) @facets{{
                            expand(_all_)
                        }}
                    }}
                }}
                """
                res = txn.query(query=query)
                ppl = json.loads(res.json)
                
                relationships = ppl.get("getRelationships", [])
                
                for rel in relationships:
                    connects = rel["connect"]
                    for con in connects:
                        entity_relationships.append(Relationship(
                            id=con["connect|id"],
                            short_id=con["connect|short_id"],
                            source=con["connect|source"],
                            target=con["connect|target"],
                            weight=con["connect|weight"],
                            description=con["connect|description"],
                            text_unit_ids=json.loads(con["connect|text_unit_ids"]) if con.get("connect|text_unit_ids", None) else None
                        ))
                        
            matching_relationships = [
                rel
                for rel in entity_relationships
                if rel.source == entity.title or rel.target == entity.title
            ]
                    
        
            
    finally:
        txn.discard()


    return len(matching_relationships)




# def count_relationship(
#     text_unit: TextUnit,
#     entity: Entity,
#     relationships: Dict[str, Relationship]
# ) -> int:
#     """Count the number of relationships of the selected entity that are associated with the text unit."""
#     matching_relationships = []
#     if text_unit.relationship_ids is None:
#         entity_relationships = [
#             rel 
#             for rel in relationships.values()
#             if rel.source == entity.title or rel.target == entity.title
#         ]
        
#         entity_relationships = [
#             rel for rel in entity_relationships if rel.text_unit_ids
#         ]
        
#         matching_relationships = [
#             rel 
#             for rel in entity_relationships
#             if text_unit.id in rel.text_unit_ids
#         ]
#     else:
#         text_unit_relationships = [
#             relationships[rel_id]
#             for rel_id in text_unit.relationship_ids
#             if rel_id in relationships
#         ]
#         matching_relationships = [
#             rel
#             for rel in text_unit_relationships
#             if rel.source == entity.title or rel.target == entity.title
#         ]
#     return len(matching_relationships)


