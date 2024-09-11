from collections import defaultdict
from typing import Any, List, Optional, Tuple, cast

import pandas as pd

from models.entity import Entity
from models.relationship import Relationship
from query.inputs.retrieval.relationships import get_in_network_relationships, get_out_network_relationships
from utils import list_of_token


def build_relationship_context(
    selected_entities: List[Entity],
    relationships: List[Relationship],
    token_encoder: Optional[str] = None,
    include_relationship_weight: bool = False,
    max_tokens: int = 8000,
    top_k_relationships: int = 10,
    relationship_ranking_attribute: str = "rank",
    column_delimiter: str = "|",
    context_name: str = "Relationships"
) -> Tuple[str, pd.DataFrame]:
    """Prepare relationships data tables as context data for system prompt"""
    
    # Get the most relevent relationships
    selected_relationships = filter_relationship(
        selected_entities=selected_entities,
        relationships=relationships,
        top_k_relationships=top_k_relationships,
        relationship_ranking_attribute=relationship_ranking_attribute
    )
    
    if len(selected_entities) == 0 or len(selected_relationships) == 0:
        return "", pd.DataFrame()
    
    current_context_text = f"-----{context_name}-----" + "\n"
    header = ["id", "source", "target", "description"]
    if include_relationship_weight:
        header.append("weight")
    attribute_cols = (
        list(selected_relationships[0].attributes.keys())
        if selected_relationships[0].attributes
        else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)
    
    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = len(list_of_token(current_context_text, token_encoder))

    all_context_records = [header]
    for rel in selected_relationships:
        new_context = [
            rel.short_id if rel.short_id else "",
            rel.source,
            rel.target,
            rel.description if rel.description else "",
        ]
        if include_relationship_weight:
            new_context.append(str(rel.weight if rel.weight else ""))
        for field in attribute_cols:
            field_value = (
                str(rel.attributes.get(field))
                if rel.attributes and rel.attributes.get(field)
                else ""
            )
            new_context.append(field_value)
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

    return current_context_text, record_df



# Get the most relevant relationships to entities
def filter_relationship(
    selected_entities: List[Entity],
    relationships: List[Relationship],
    top_k_relationships: int = 10,
    relationship_ranking_attribute: str = "rank",
) -> List[Relationship]:
    """Filter and sort relationships based on a set of entities and a ranking attribute."""
    # First priority: in-network relationships (i.e relationships between selected entities)
    in_network_relationships = get_in_network_relationships(
        selected_entities=selected_entities,
        relationships=relationships,
        ranking_attribute=relationship_ranking_attribute
    )
    
    # Second priority: out-network relationships
    # (i.e. relationships between selected entities and other entities that are not within the selected entities)
    out_network_relationships = get_out_network_relationships(
        selected_entities=selected_entities,
        relationships=relationships,
        ranking_attribute=relationship_ranking_attribute
    )
    
    if len(out_network_relationships) <= 1:
        return in_network_relationships + out_network_relationships
    
    # within out-network relationships, prioritize mutual relationships
    # (i.e. relationships with out-network entities that are shared with multiple selected entities)
    selected_entity_names = [entity.title for entity in selected_entities]
    out_network_source_names = [
        relationship.source
        for relationship in out_network_relationships
        if relationship.source not in selected_entity_names
    ]
    
    out_network_target_names = [
        relationship.target
        for relationship in out_network_relationships
        if relationship.target not in selected_entity_names
    ]
    
    out_network_entity_names = list(
        set(out_network_source_names + out_network_target_names)
    )
    
    out_network_entity_links = defaultdict(int)
    for entity_name in out_network_entity_names:
        targets = [
            relationship.target
            for relationship in out_network_relationships
            if relationship.source == entity_name
        ]
        
        sources = [
            relationship.source
            for relationship in out_network_relationships
            if relationship.target == entity_name
        ]
        
        out_network_entity_links[entity_name] = len(set(targets + sources))
        
    # sort out-network relationships by number of links and rank_attributes
    for rel in out_network_relationships:
        if rel.attributes is None:
            rel.attributes = {}
        rel.attributes["links"] = (
            out_network_entity_links[rel.source]
            if rel.source in out_network_entity_links
            else out_network_entity_links[rel.target]
        )
        
    # sort by attributes[links] first, then by ranking_attribute
    if relationship_ranking_attribute == "weight":
        out_network_relationships.sort(
            key=lambda x: (x.attributes["links"], x.weight),
            reverse=True
        )
    else:
        out_network_relationships.sort(
            key=lambda x: (
                x.attributes["links"],
                x.attributes[relationship_ranking_attribute]
            ),
            reverse=True
        )
        
    relationship_budget = top_k_relationships * len(selected_entities)
        
    return in_network_relationships + out_network_relationships[:relationship_budget]