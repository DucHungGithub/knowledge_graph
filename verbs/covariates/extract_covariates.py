import json
import logging
from enum import Enum
import os
from typing import Any, Dict, List

import pandas as pd


from verbs.covariates.graph_intelligence.run import run_gi
from verbs.covariates.typing import Covariate, CovariateExtractStrategy, CovariateExtractionResult
import config as defs


logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

class ExtractClaimsStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"
    
    def __repr__(self) -> str:
        return f"{self.value}"



def save_covariates_to_json(covariates: CovariateExtractionResult, check_dir: str = defs.CHECKPOINT_DIR, old_covariates_file: str = "claims.json") -> None:
    # Convert covariates to a list of dictionaries
    cov_data = [cov.dict() for cov in covariates.covariate_data]
    

    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    # Save to JSON file
    with open(f"{check_dir}/{old_covariates_file}", 'w') as json_file:
        json.dump(cov_data, json_file, indent=4)


       
def load_covariates_from_json(file_path: str) -> List[Covariate] | None:
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    
    return [Covariate(**cov) for cov in data]


def combine_and_remove_duplicates(
    new_covariates: CovariateExtractionResult, 
    old_covariates_file: str = "claims.json"
) -> CovariateExtractionResult:
    # Load old covariates
    old_covariates = load_covariates_from_json(f"{defs.CHECKPOINT_DIR}/{old_covariates_file}")
    if old_covariates is None:
        save_covariates_to_json(new_covariates)
        return new_covariates
    
    # Convert old and new covariates to a DataFrame
    new_df = new_covariates.to_dataframe()
    old_df = pd.DataFrame([cov.dict() for cov in old_covariates])

    # Combine the dataframes
    combined_df = pd.concat([new_df, old_df]).reset_index(drop=True)

    # Combine descriptions, source_text, and doc_id for the same subject_id
    combined_df = combined_df.groupby('subject_id', as_index=False).agg({
        'description': lambda x: ' '.join([str(desc) for desc in x if desc is not None]),  # Handle None by filtering it out
        'source_text': lambda x: list(set([item for sublist in x if sublist is not None for item in sublist])),  # Flatten and filter None
        'doc_id': lambda x: ', '.join([doc for doc in x if doc is not None]),  # Combine doc_id, ignoring None
        'type': lambda x: next((t for t in x if t is not None), None),  # Return the first non-None 'type'
        'status': lambda x: next((s for s in x if s is not None), None),  # Return the first non-None 'status'
        'start_date': lambda x: next((sd for sd in x if sd is not None), None),  # Return the first non-None 'start_date'
        'end_date': lambda x: next((ed for ed in x if ed is not None), None),  # Return the first non-None 'end_date'
        'covariate_type': lambda x: next((ct for ct in x if ct is not None), None),  # Return the first non-None 'covariate_type'
        'subject_type': lambda x: next((st for st in x if st is not None), None),  # Return the first non-None 'subject_type'
        'object_id': lambda x: next((oid for oid in x if oid is not None), None),  # Return the first non-None 'object_id'
        'object_type': lambda x: next((ot for ot in x if ot is not None), None),  # Return the first non-None 'object_type'
        'record_id': lambda x: next((rid for rid in x if rid is not None), None),  # Return the first non-None 'record_id'
        'id': lambda x: next((id_ for id_ in x if id_ is not None), None)  # Return the first non-None 'id'
    })



    # Fill missing values
    combined_df = combined_df.fillna({
        'id': "",
        'short_id': None,
        'text_unit_ids': None,
        'document_ids': None,
        'attributes': None
    })
    
    # Convert back to Covariate objects
    combined_covariates = CovariateExtractionResult(
        covariate_data=[Covariate(**row) for _, row in combined_df.iterrows()]
    )
    
    # Save the combined covariates to the file
    save_covariates_to_json(combined_covariates)

    return combined_covariates



async def extract_covariates(
    texts: List[str],
    entity_types: List[str] | None = None,
    resolved_entities_map: Dict[str, Any] = None,
    strategy: Dict[str, Any] | None = None,
    **kwargs
) -> CovariateExtractionResult:
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    
    resolved_entities_map = {}
    
    strategy = strategy or {}
    
    strategy_exec = load_strategy(
        strategy.get("type", ExtractClaimsStrategyType.graph_intelligence)
    )
    
    config = {**strategy}
    
    if resolved_entities_map is None:
        resolved_entities_map = {}
    
    results = await strategy_exec(texts, entity_types, resolved_entities_map, config)
    
    results = combine_and_remove_duplicates(new_covariates=results)
    
    return results



def load_strategy(strategy_type: ExtractClaimsStrategyType) -> CovariateExtractStrategy:
    match strategy_type:
        case ExtractClaimsStrategyType.graph_intelligence:
            from .graph_intelligence.run import run_gi
            return run_gi
        case _:
            raise ValueError(f"Unknow claims type: {strategy_type}")