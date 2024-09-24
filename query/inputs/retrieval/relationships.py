# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json
from typing import Any, List, cast

from models.entity import Entity
from models.relationship import Relationship

import pandas as pd
import pydgraph


def get_in_network_relationships(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
    ranking_attribute: str = "rank"
) -> List[Relationship]:
    """Get all directed relationships between selected entities, sorted by ranking_attribute."""
    selected_entity_names = [entity.title for entity in selected_entities]
    
    txn = client.txn()
    selected_relationships = []
    
    
    source_text = ' OR '.join([f'eq(source, "{name}")' for name in selected_entity_names])
    target_text = ' OR '.join([f'eq(target, "{name}")' for name in selected_entity_names])
    
    try:
        query = f"""
        {{
            getRelationships(func: type(Entity)) @filter(has(connect)) {{
                connect @facets(({source_text}) AND ({target_text})) @facets{{
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
                selected_relationships.append(Relationship(
                    id=con["connect|id"],
                    short_id=con["connect|short_id"],
                    source=con["connect|source"],
                    target=con["connect|target"],
                    weight=con["connect|weight"],
                    description=con["connect|description"]
                ))
        
    finally:
        txn.discard()
    
    
    if len(selected_relationships) <= 1:
        return selected_relationships
    
    # Sort by ranking attribute
    return sort_relationships_by_ranking_attribute(
        selected_relationships, selected_entities, ranking_attribute
    )
    

def get_out_network_relationships(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
    ranking_attribute: str = "rank"
) -> List[Relationship]:
    """Get relationships from selected entities to other entities that are not within the selected entities, sorted by ranking_attribute."""
    selected_entity_names = [entity.title for entity in selected_entities]
    
    txn = client.txn()
    
    source_relationships = []
    target_relationships = []
    
    source_text = ' OR '.join([f'eq(source, "{name}")' for name in selected_entity_names])
    target_text = ' OR '.join([f'eq(target, "{name}")' for name in selected_entity_names])
    
    not_source_text = ' AND '.join([f'NOT eq(source, "{name}")' for name in selected_entity_names])
    not_target_text = ' AND '.join([f'NOT eq(target, "{name}")' for name in selected_entity_names])
    
    try:
        query = f"""
        {{
            getRelationships(func: type(Entity)) @filter(has(connect)) {{
                connect @facets(({source_text}) AND ({not_target_text})) @facets{{
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
                source_relationships.append(Relationship(
                    id=con["connect|id"],
                    short_id=con["connect|short_id"],
                    source=con["connect|source"],
                    target=con["connect|target"],
                    weight=con["connect|weight"],
                    description=con["connect|description"]
                ))
        
        query = f"""
        {{
            getRelationships(func: type(Entity)) @filter(has(connect)) {{
                connect @facets(({not_source_text}) AND ({target_text})) @facets{{
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
                target_relationships.append(Relationship(
                    id=con["connect|id"],
                    short_id=con["connect|short_id"],
                    source=con["connect|source"],
                    target=con["connect|target"],
                    weight=con["connect|weight"],
                    description=con["connect|description"]
                ))
        
    finally:
        txn.discard()
    
    selected_relationships = source_relationships + target_relationships
    
    
    return sort_relationships_by_ranking_attribute(
        selected_relationships, selected_entities, ranking_attribute
    )
    
    
def get_candidate_relationships(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
) -> List[Relationship]:
    """Get all relationships that are associated with the selected entities."""
    selected_entity_names = [entity.title for entity in selected_entities]
    
    txn = client.txn()
    
    source_text = ' OR '.join([f'eq(source, "{name}")' for name in selected_entity_names])
    target_text = ' OR '.join([f'eq(target, "{name}")' for name in selected_entity_names])
    
    selected_relationships = []
    
    try:
        query = f"""
        {{
            getRelationships(func: type(Entity)) @filter(has(connect)) {{
                connect @facets(({source_text}) OR ({target_text})) @facets{{
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
                selected_relationships.append(Relationship(
                    id=con["connect|id"],
                    short_id=con["connect|short_id"],
                    source=con["connect|source"],
                    target=con["connect|target"],
                    weight=con["connect|weight"],
                    description=con["connect|description"]
                ))
    
    finally:
        txn.discard()
    
    
    return selected_relationships


def get_entities_from_relationships(
    client: pydgraph.DgraphClient,
    relationships: List[Relationship],
) -> List[Entity]:
    """Get all entities that are associated with the selected relationships."""
    selected_entity_names = [relationship.source for relationship in relationships] + [
        relationship.target for relationship in relationships
    ]
    
    title_text = ' OR '.join([f'eq(title, "{name}")' for name in selected_entity_names])
    
    txn = client.txn()
    
    resolve_entity = []
    try:
        query = f"""
        {{
            getEntity(func: type(Entity)) @filter({title_text}) {{
                id
                title
                short_id
                type
                description
                rank
                text_unit_ids
                community_ids
                attributes
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        entities = ppl.get("getEntity", [])
        
        
        for entity in entities:
            if entity.get("attributes", None):
                entity["attributes"] = json.loads(entity["attributes"]) if entity["attributes"] else None
            resolve_entity.append(Entity(**entity))
    finally:
        txn.discard()


    return resolve_entity
    
      

def calculate_relationship_combined_rank(
    relationships: List[Relationship],
    entities: List[Entity],
    ranking_attribute: str = "rank"
) -> List[Relationship]:
    """Calculate default rank for a relationship based on the combined rank of source and target entities."""
    entity_mappings = {entity.title: entity for entity in entities}

    for relationship in relationships:
        if relationship.attributes is None:
            relationship.attributes = {}
        source = entity_mappings.get(relationship.source)
        target = entity_mappings.get(relationship.target)
        source_rank = source.rank if source and source.rank else 0
        target_rank = target.rank if target and target.rank else 0
        relationship.attributes[ranking_attribute] = source_rank + target_rank  # type: ignore
    return relationships

    
def sort_relationships_by_ranking_attribute(
    relationships: List[Relationship],
    entities: List[Entity],
    ranking_attribute: str = "rank"
) -> List[Relationship]:
    """
    Sort relationships by a ranking_attribute.

    If no ranking attribute exists, sort by combined rank of source and target entities.
    """
    if len(relationships) == 0:
        return relationships
    
    # Sorted by ranking attribute
    attribute_names = (
        list(relationships[0].attributes.keys()) if relationships[0].attributes else []
    )
    if ranking_attribute in attribute_names:
        relationships.sort(
            key=lambda x: int(x.attributes[ranking_attribute]) if x.attributes else 0,
            reverse=True
        )
    elif ranking_attribute == "weight":
        relationships.sort(
            key=lambda x: x.weight if x.weight else 0.0,
            reverse=True
        )
    else:
        # ranking attribute do not exist, calculate rank = combined ranks of source and target
        relationships = calculate_relationship_combined_rank(
            relationships, entities, ranking_attribute
        )
        relationships.sort(
            key=lambda x: int(x.attributes[ranking_attribute]) if x.attributes else 0,
            reverse=True,
        )
    return relationships


def to_relationship_dataframe(
    relationships: List[Relationship], include_relationship_weight: bool = True
) -> pd.DataFrame:
    """Convert a list of relationship to a DataFrame"""
    if len(relationships) == 0:
        return pd.DataFrame()
    
    header = ["id", "source", "target", "description"]
    
    if include_relationship_weight:
        header.append("weight")
    
    attribute_cols = (
        list(relationships[0].attributes.keys()) if relationships[0].attributes else []
    )
    
    attribute_cols = [col for col in attribute_cols if col not in header]
    
    header.extend(attribute_cols)
    
    records = []
    for rel in relationships:
        new_record = [
            rel.short_id if rel.short_id else "",
            rel.source,
            rel.target,
            rel.description if rel.description else ""
        ]
        if include_relationship_weight:
            new_record.append(str(rel.weight if rel.weight else ""))
        for field in attribute_cols:
            field_value = (
                str(rel.attributes.get(field))
                if rel.attributes and rel.attributes.get(field)
                else ""
            )
            new_record.append(field_value)
        records.append(new_record)
    return pd.DataFrame(records, columns=cast(Any, header))
            