import json
from typing import Any, List, cast

import pandas as pd
import pydgraph

from models.covariate import Covariate
from models.entity import Entity


def get_candidate_covariates(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
) -> List[Covariate]:
    """Get all covariates that are related to selected entities."""
    selected_entity_names = [entity.title for entity in selected_entities]
    
    source_text = ' OR '.join([f'eq(subject_id, "{name}")' for name in selected_entity_names])
    txn = client.txn()
    
    selected_covariates = []
    
    try:
        query = f"""{{
                queryEntity(func: type(Covariate)) @filter({source_text}){{
                    uid
                    object_id
                    object_type
                    id
                    type
                    status
                    start_date
                    end_date
                    short_id
                    subject_id
                    subject_type
                    covariate_type
                    description
                    source_text
                    text_unit_ids
                    document_ids
                    human_readable_id
                    claim_details
                    attributes
            }}
        }}
        """
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        covs = ppl["queryEntity"]
        for cov in covs:
            if cov.get("attributes", None):
                cov["attributes"] = json.loads(cov["attributes"]) if cov["attributes"] else None
            selected_covariates.append(Covariate(**cov))
            
    finally:
        txn.discard()
    
    
    return selected_covariates


def to_covariate_dataframe(covariates: List[Covariate]) -> pd.DataFrame:
    """Convert a list of covariates to a pandas dataframe."""
    if len(covariates) == 0:
        return pd.DataFrame()

    # add header
    header = ["id", "entity"]
    attributes = covariates[0].attributes or {} if len(covariates) > 0 else {}
    attribute_cols = list(attributes.keys()) if len(covariates) > 0 else []
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    records = []
    for covariate in covariates:
        new_record = [
            covariate.short_id if covariate.short_id else "",
            covariate.subject_id,
        ]
        for field in attribute_cols:
            field_value = (
                str(covariate.attributes.get(field))
                if covariate.attributes and covariate.attributes.get(field)
                else ""
            )
            new_record.append(field_value)
        records.append(new_record)
    return pd.DataFrame(records, columns=cast(Any, header))
