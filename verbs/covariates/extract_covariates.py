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
        'description': lambda x: ' '.join(x.unique()),  # Combine unique descriptions
        'source_text': lambda x: list(set([item for sublist in x for item in sublist])),  # Combine source_text into a unique list
        'doc_id': lambda x: ', '.join(x.unique()),  # Combine doc_id as a comma-separated string
        'type': 'first',  # Keep the first occurrence of 'type' (or choose other logic)
        'status': 'first',  # Keep the first occurrence of 'status'
        'start_date': 'first',  # Keep the first occurrence of 'start_date'
        'end_date': 'first',  # Keep the first occurrence of 'end_date'
        'covariate_type': 'first',  # Keep the first occurrence of 'covariate_type'
        'subject_type': 'first',  # Keep the first occurrence of 'subject_type'
        'object_id': 'first',  # Keep the first occurrence of 'object_id'
        'object_type': 'first',  # Keep the first occurrence of 'object_type'
        'record_id': 'first',  # Keep the first occurrence of 'record_id'
        'id': 'first'  # Keep the first occurrence of 'id' (or modify based on logic)
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
    save_covariates_to_json(combined_covariates, f"{defs.CHECKPOINT_DIR}/{old_covariates_file}")

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