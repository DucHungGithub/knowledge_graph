# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from typing import Any, cast
import pandas as pd
import logging

import graph.community_reports.schemas as schemas
from graph.community_reports.sort_context import sort_context
from graph.community_reports.utils import filter_claims_to_nodes, filter_edges_to_nodes, filter_nodes_to_level, get_levels, set_context_exceeds_flag, set_context_size

from verbs.community_summarization.prepare_report.community_claims import prepare_community_reports_claims
from verbs.community_summarization.prepare_report.community_edges import prepare_community_reports_edges
from verbs.community_summarization.prepare_report.community_nodes import prepare_community_reports_nodes
from verbs.covariates.typing import CovariateExtractionResult


import colorlog

# Set up basic logging configuration with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# Get the logger and add the handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.addHandler(handler)


def prepare_community_reports(
    entities: Any,
    rels: Any,
    claims: CovariateExtractionResult,
    max_tokens: int = 16000,
    use_claim: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Generate entities for each row, and optionally a graph of those entities."""
    # Prepare Community Reports
    node_df = prepare_community_reports_nodes(entities=entities)
    
    
    rel_df = prepare_community_reports_edges(rels=rels, node_df=node_df)
    
    
    claim_df = prepare_community_reports_claims(covariate_data=claims) if use_claim else None
    
    
    
    levels = get_levels(node_df, schemas.NODE_LEVEL)

    dfs = []
    
    for level in levels:
        communities_at_level_df = _prepare_reports_at_level(
            node_df=node_df,
            edge_df=rel_df,
            claim_df=claim_df,
            level=level,
            max_tokens=max_tokens,
            use_claim=use_claim
        )
        dfs.append(communities_at_level_df)

    return pd.concat(dfs)

def _prepare_reports_at_level(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    claim_df: pd.DataFrame | None,
    level: int,
    use_claim: bool,
    max_tokens: int = 16_000,
    community_id_column: str = schemas.COMMUNITY_ID,
    node_id_column: str = schemas.NODE_ID,
    node_name_column: str = schemas.NODE_NAME,
    node_details_column: str = schemas.NODE_DETAILS,
    node_level_column: str = schemas.NODE_LEVEL,
    node_degree_column: str = schemas.NODE_DEGREE,
    node_community_column: str = schemas.NODE_COMMUNITY,
    edge_id_column: str = schemas.EDGE_ID,
    edge_source_column: str = schemas.EDGE_SOURCE,
    edge_target_column: str = schemas.EDGE_TARGET,
    edge_degree_column: str = schemas.EDGE_DEGREE,
    edge_details_column: str = schemas.EDGE_DETAILS,
    claim_id_column: str = schemas.CLAIM_ID,
    claim_subject_column: str = schemas.CLAIM_SUBJECT,
    claim_details_column: str = schemas.CLAIM_DETAILS,
):
    def get_edge_details(node_df: pd.DataFrame, edge_df: pd.DataFrame, name_col: str):
        return node_df.merge(
            cast(
                pd.DataFrame,
                edge_df[[name_col, schemas.EDGE_DETAILS]],
            ).rename(columns={name_col: schemas.NODE_NAME}),
            on=schemas.NODE_NAME,
            how="left",
        )

    level_node_df = filter_nodes_to_level(node_df, level)
    logger.info("Number of nodes at level=%s => %s", level, len(level_node_df), exc_info=True)
    nodes = level_node_df[node_name_column].tolist()

    # Filter edges & claims to those containing the target nodes
    level_edge_df = filter_edges_to_nodes(edge_df, nodes)
    level_claim_df = None
    if use_claim and claim_df is not None:
        level_claim_df = filter_claims_to_nodes(claim_df, nodes)

    # concat all edge details per node
    merged_node_df = pd.concat(
        [
            get_edge_details(level_node_df, level_edge_df, edge_source_column),
            get_edge_details(level_node_df, level_edge_df, edge_target_column),
        ],
        axis=0,
    )
    merged_node_df = (
        merged_node_df.groupby([
            node_name_column,
            node_community_column,
            node_degree_column,
            node_level_column,
        ])
        .agg({node_details_column: "first", edge_details_column: list})
        .reset_index()
    )

    # concat claim details per node if use_claim is True
    if use_claim and level_claim_df is not None:
        merged_node_df = merged_node_df.merge(
            cast(
                pd.DataFrame,
                level_claim_df[[claim_subject_column, claim_details_column]],
            ).rename(columns={claim_subject_column: node_name_column}),
            on=node_name_column,
            how="left",
        )
        claim_agg = {claim_details_column: list}
    else:
        claim_agg = {}

    merged_node_df = (
        merged_node_df.groupby([
            node_name_column,
            node_community_column,
            node_level_column,
            node_degree_column,
        ])
        .agg({
            node_details_column: "first",
            edge_details_column: "first",
            **claim_agg,
        })
        .reset_index()
    )

    # concat all node details, including name, degree, node_details, edge_details, and claim_details if use_claim is True
    merged_node_df[schemas.ALL_CONTEXT] = merged_node_df.apply(
        lambda x: {
            node_name_column: x[node_name_column],
            node_degree_column: x[node_degree_column],
            node_details_column: x[node_details_column],
            edge_details_column: x[edge_details_column],
            **({claim_details_column: x[claim_details_column]} if use_claim and level_claim_df is not None else {}),
        },
        axis=1,
    )

    # group all node details by community
    community_df = (
        merged_node_df.groupby(node_community_column)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    community_df[schemas.CONTEXT_STRING] = community_df[schemas.ALL_CONTEXT].apply(
        lambda x: sort_context(
            x,
            node_id_column=node_id_column,
            node_name_column=node_name_column,
            node_details_column=node_details_column,
            edge_id_column=edge_id_column,
            edge_details_column=edge_details_column,
            edge_degree_column=edge_degree_column,
            edge_source_column=edge_source_column,
            edge_target_column=edge_target_column,
            claim_id_column=claim_id_column,
            claim_details_column=claim_details_column,
            community_id_column=community_id_column,
        )
    )
    set_context_size(community_df)
    set_context_exceeds_flag(community_df, max_tokens)

    community_df[schemas.COMMUNITY_LEVEL] = level
    return community_df
