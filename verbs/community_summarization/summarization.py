import asyncio
import logging 
from enum import Enum
from typing import Any, Dict

import pandas as pd

import config as defs

from graph.community_reports.pre_community_report_context import prep_community_report_context
from graph.community_reports.utils import get_levels
from utils.uuid import gen_uuid
from verbs.community_summarization.prepare_report.community_hierachy import restore_community_hierarchy
from verbs.community_summarization.prepare_report.community_nodes import prepare_community_reports_nodes
from verbs.community_summarization.prepare_report.community_reports import prepare_community_reports
from verbs.community_summarization.typing import CommunityReport, CommunityReportsStrategy
from verbs.covariates.typing import CovariateExtractionResult

import config as defs
import graph.community_reports.schemas as schemas

logger = logging.getLogger(__name__)

class CreateCommunityReportsStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"
    
    def __repr__(self) -> str:
        return f"{self.value}"



async def create_community_reports(
    entities: Any,
    rels: Any,
    args: Dict[str, Any],
    claims: CovariateExtractionResult = CovariateExtractionResult(covariate_data=[]),
    num_threads: int = 4,
    **kwargs
) -> pd.DataFrame:
    logger.debug(f"Create entities for each row, and optionally a graph of those entities", exc_info=True)
    local_context = prepare_community_reports(
        entities=entities,
        rels=rels,
        claims=claims,
        use_claim=True
    )
    node_df = prepare_community_reports_nodes(
        entities=entities
    )
    community_hierarchy = restore_community_hierarchy(
        entities=entities
    )
    
    levels = get_levels(node_df)
    
    reports: list[CommunityReport | None] = []
    strategy_exec = load_strategy(args.get("COMMUNITY_REPORT_TYPE", CreateCommunityReportsStrategyType.graph_intelligence))
    
    semaphore = asyncio.Semaphore(num_threads)
    
    async def run_generate(record):
        async with semaphore:
            return await generate_report(
                runner=strategy_exec,
                community_id=record[schemas.NODE_COMMUNITY],
                level=record[schemas.COMMUNITY_LEVEL],
                context=record[schemas.CONTEXT_STRING],
                args=args,
            )
    
    for level in levels:
        level_contexts = prep_community_report_context(
            pd.DataFrame(reports),
            local_context_df=local_context,
            community_hierarchy_df=community_hierarchy,
            level=level,
            max_tokens=args.get(
                "max_input_tokens", defs.COMMUNITY_REPORT_MAX_INPUT_LENGTH
            ),
        )
        
        tasks = [run_generate(record) for record in level_contexts.to_dict('records')]
        local_reports = await asyncio.gather(*tasks)
        
        reports.extend([lr for lr in local_reports if lr is not None])
    
    
    data = []
    for _, rep in enumerate(reports):
        rep_dict = rep.dict()
        data.append(rep_dict)    

    return pd.DataFrame(data)
    


async def generate_report(
    runner: CommunityReportsStrategy,
    community_id: int | str,
    level: int,
    context: str,
    args: Dict[str, Any],
) -> CommunityReport | None:
       
    return await runner(community_id, context, level, args)




def load_strategy(
    strategy: CreateCommunityReportsStrategyType
) -> CommunityReportsStrategy:
    match strategy:
        case CreateCommunityReportsStrategyType.graph_intelligence:
            from .graph_intelligence import run_gi

            return run_gi
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")