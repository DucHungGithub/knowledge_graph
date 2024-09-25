import json
from typing import Any, Iterable, List, Optional, cast
import uuid

import pandas as pd
import pydgraph

from models.entity import Entity



def get_entity_by_key(
    client: pydgraph.DgraphClient,
    key: str,
    value: str | int
) -> Optional[Entity]:
    """Get entity by key."""

    txn = client.txn()
    entity = None
    
    try:
        query = f"""
        {{
            getEntity(func: type(Entity)) @filter(eq({key},{value})){{
                id
                title
                short_id
                type
                description
                rank
                text_unit_ids
                community_ids
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        entity = ppl.get("getEntity")
        
    finally:
        txn.discard()

    if len(entity) == 0:
        return None
    
    return Entity(**entity[0])


def get_top_rank_entity(
    client: pydgraph.DgraphClient,
    k: int
) -> List[Entity]:
    txn = client.txn()
    top_k_entities = []
    
    entities = []
    
    try:
        query = f"""
        {{
            getEntities(func: type(Entity), orderdesc: rank, first: {k}) {{
                id
                title
                short_id
                type
                description
                rank
                text_unit_ids
                community_ids
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        entities = ppl.get("getEntities")
        
    finally:
        txn.discard()
        
    for entity in entities:
        top_k_entities.append(Entity(**entity))
    
    return top_k_entities


def get_entity_by_name(client: pydgraph.DgraphClient, entity_name: str) -> List[Entity]:
    txn = client.txn()
    entities = []
    
    try:
        query = f"""
        {{
            getEntity(func: type(Entity)) @filter(eq(title, {entity_name})) {{
                id
                title
                short_id
                type
                description
                rank
                text_unit_ids
                community_ids
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        entities = ppl.get("getEntity")
    
    finally:
        txn.discard()
        
    return [Entity(**entity) for entity in entities]
        
        

def get_entity_by_attribute(
    entities: Iterable[Entity], attribute_name: str, attribute_value: Any
) -> List[Entity]:
    """Get entities by attribute."""
    return [
        entity
        for entity in entities
        if entity.attributes
        and entity.attributes.get(attribute_name) == attribute_value
    ]



def to_entity_dataframe(
    entities: List[Entity],
    include_entity_rank: bool = True,
    rank_description: str = "number of relationships"
) -> pd.DataFrame:
    """Convert a list of entities to a pandas dataframe."""
    if len(entities) == 0:
        return pd.DataFrame()
    
    header = ["id", "entity", "description"]
    if include_entity_rank:
        header.append(rank_description)
        
    attribute_cols = (
        list(entities[0].attributes.keys()) if entities[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)
    
    records = []
    for entity in entities:
        new_record = [
            entity.short_id if entity.short_id else "",
            entity.title,
            entity.description if entity.description else ""
        ]
        
        if include_entity_rank:
            new_record.append(str(entity.rank))
            
        for field in attribute_cols:
            field_value = (
                str(entity.attributes.get(field))
                if entity.attributes and entity.attributes.get(field)
                else ""
            )
            new_record.append(field_value)
        records.append(new_record)
    return pd.DataFrame(records, columns=cast(Any, header))



def is_valid_uuid(value: str) -> bool:
    """Determine if a string is a valid UUID"""
    
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    else:
        return True