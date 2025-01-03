from typing import Dict, List

import pandas as pd

from models.covariate import Covariate
from models.entity import Entity
from models.relationship import Relationship
from query.inputs.retrieval.covariates import get_candidate_covariates, to_covariate_dataframe
from query.inputs.retrieval.entities import to_entity_dataframe
from query.inputs.retrieval.relationships import get_candidate_relationships, get_entities_from_relationships, to_relationship_dataframe


def get_candidate_context(
    selected_entities: List[Entity],
    entities: List[Entity],
    relationships: List[Relationship],
    covariates: Dict[str, List[Covariate]],
    include_entity_rank: bool = True,
    entity_rank_description: str = "number of relationships",
    include_relationship_weight: bool = False
) -> Dict[str, pd.DataFrame]:
    """Prepare entity, relationships, and covariate data table for system prompt."""
    candidate_context = {}
    
    candidate_relationships = get_candidate_relationships(
        selected_entities=selected_entities,
        relationships=relationships
    )
    
    candidate_context["relationships"] = to_relationship_dataframe(
        relationships=candidate_relationships,
        include_relationship_weight=include_relationship_weight
    )
    
    candidate_entities = get_entities_from_relationships(
        relationships=candidate_relationships, entities=entities
    )
    
    candidate_context["entities"] = to_entity_dataframe(
        entities=candidate_entities,
        include_entity_rank=include_entity_rank,
        rank_description=entity_rank_description
    )
    
    for covariate in covariates:
        candidate_covariates = get_candidate_covariates(
            selected_entities=selected_entities,
            covariates=covariates[covariate]
        )
        
        candidate_context[covariate.lower()] = to_covariate_dataframe(
            covariates=candidate_covariates
        )
        
    return candidate_context