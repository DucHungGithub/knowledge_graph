import json
from typing import Any, List, Optional, Tuple, cast

import pandas as pd
import pydgraph

from models.covariate import Covariate
from models.entity import Entity
from utils import list_of_token

# Build the covariate table equivalent to the List of entities
def build_covariates_context(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
    token_encoder: Optional[str] = None,
    max_tokens: int = 8000,
    column_delimiter: str = "|",
    context_name: str = "Covariates",
) -> Tuple[str, pd.DataFrame]:
    """Prepare covariate data table as context data for system prompt"""
    
    if len(selected_entities) == 0:
        return "", pd.DataFrame()
    
    selected_covariates = []
    record_df = pd.DataFrame()
    
    current_context_text = f"-----{context_name}-----" + "\n"
    
    txn = client.txn()
    
    try:
        for entity in selected_entities:
            if entity.title:
                query = f"""{{
                        queryEntity(func: type(Covariate)) @filter(eq(subject_id,"{entity.title}")){{
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
    
    header = ["id", "entity"]
    attributes = selected_covariates[0].attributes or {} if len(selected_covariates) > 0 else {}
    attribute_cols = list(attributes.keys()) if len(selected_covariates) > 0 else []
    header.extend(attribute_cols)
    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = len(list_of_token(current_context_text, token_encoder))
    
    all_context_records = [header]
    
        
    for covariate in selected_covariates:
        new_context = [
            covariate.short_id if covariate.short_id else "",
            covariate.subject_id
        ]
        
        for field in attribute_cols:
            field_value = (
                str(covariate.attributes.get(field))
                if covariate.attributes and covariate.attributes.get(field)
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
            record_df - pd.DataFrame()
            
    return current_context_text, record_df