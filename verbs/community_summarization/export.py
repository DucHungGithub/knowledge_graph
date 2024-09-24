from pathlib import Path
from typing import List, Optional
import os

import pandas as pd

from langchain_core.documents import Document

from models.entity import Entity
from models.relationship import Relationship
from utils import documentsToDataframe
from verbs.community_summarization.prepare_report.community_claims import prepare_community_reports_claims
from verbs.community_summarization.prepare_report.community_edges import prepare_community_reports_edges
from verbs.community_summarization.prepare_report.community_entities import prepare_community_reports_entities
from verbs.community_summarization.prepare_report.community_hierachy import restore_community_hierarchy
from verbs.community_summarization.prepare_report.community_nodes import prepare_community_reports_nodes
from verbs.community_summarization.prepare_report.community_reports import prepare_community_reports
from verbs.covariates.typing import CovariateExtractionResult

import config as defs


async def export_final_files(
    path: str | Path = defs.OUTPUT_FOLDER_PATH,
    text_units: Optional[List[Document]] = None,
    entities: Optional[List[Entity]] = None,
    rels: Optional[List[Relationship]] = None,
    claims: Optional[List[CovariateExtractionResult]] = None,
    community_report: Optional[pd.DataFrame] = None
) -> None:
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")
        
    if text_units is not None:
        text_unit_df = documentsToDataframe(text_units)
        text_unit_df.to_csv(f"{path}/text_unit.csv", index=False)
    
    if entities is not None:
        node_df = prepare_community_reports_nodes(entities=entities)
        
        entity_df = prepare_community_reports_entities(node_df=node_df)
        entity_df.to_csv(f"{path}/entity.csv", index=False)
        
        node_df.to_csv(f"{path}/node.csv", index=False)
    
    if rels is not None:
        rel_df = prepare_community_reports_edges(rels=rels, node_df=node_df)
        rel_df.to_csv(f"{path}/relationship.csv", index=False)
    
    if claims is not None:
        claim_df = prepare_community_reports_claims(covariate_data=claims)
        claim_df.to_csv(f"{path}/claims.csv", index=False)
    
    
    if community_report is not None:
        community_report.to_csv(f"{path}/community_report.csv", index=False)
        
        




