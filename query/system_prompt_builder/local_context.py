from typing import Dict, List

import pandas as pd
import pydgraph

from models.covariate import Covariate
from models.entity import Entity
from models.relationship import Relationship
from query.inputs.retrieval.covariates import get_candidate_covariates, to_covariate_dataframe
from query.inputs.retrieval.entities import to_entity_dataframe
from query.inputs.retrieval.relationships import get_candidate_relationships, get_entities_from_relationships, to_relationship_dataframe


def get_candidate_context(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
    include_entity_rank: bool = True,
    entity_rank_description: str = "number of relationships",
    include_relationship_weight: bool = False
) -> Dict[str, pd.DataFrame]:
    """Prepare entity, relationships, and covariate data table for system prompt."""
    candidate_context = {}
    
    candidate_relationships = get_candidate_relationships(
        client=client,
        selected_entities=selected_entities,
    )
    
    candidate_context["relationships"] = to_relationship_dataframe(
        relationships=candidate_relationships,
        include_relationship_weight=include_relationship_weight
    )
    
    candidate_entities = get_entities_from_relationships(
        client=client,
        relationships=candidate_relationships
    )
    
    
    candidate_context["entities"] = to_entity_dataframe(
        entities=candidate_entities,
        include_entity_rank=include_entity_rank,
        rank_description=entity_rank_description
    )
    

    candidate_covariates = get_candidate_covariates(
        client=client,
        selected_entities=selected_entities,
    )
    
    candidate_context["claims"] = to_covariate_dataframe(
        covariates=candidate_covariates
    )
    
    return candidate_context