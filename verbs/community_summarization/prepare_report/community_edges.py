# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from typing import Any

import pandas as pd

from graph.community_reports.schemas import (
    EDGE_DEGREE,
    EDGE_DESCRIPTION,
    EDGE_DETAILS,
    EDGE_ID,
    EDGE_SOURCE,
    EDGE_TARGET
)
from verbs.community_summarization.prepare_report.compute_edge_degree import compute_edge_combined_degree

_MISSING_DESCRIPTION = "No Description"


def prepare_community_reports_edges(
    rels: Any,
    node_df: pd.DataFrame,
    to: str = EDGE_DETAILS,
    id_column: str = EDGE_ID,
    source_column: str = EDGE_SOURCE,
    target_column: str = EDGE_TARGET,
    description_column: str = EDGE_DESCRIPTION,
    degree_column: str = EDGE_DEGREE,
    **kwargs
) -> pd.DataFrame:
    edge_df = relationships_to_dataframe(rels)
    edge_df = edge_df.fillna(value={description_column: _MISSING_DESCRIPTION})
    
    edge_df = compute_edge_combined_degree(edge_df=edge_df, node_df=node_df)
    
    edge_df[to] = edge_df.apply(
        lambda x: {
            id_column: x[id_column],
            source_column: x[source_column],
            target_column: x[target_column],
            description_column: x[description_column],
            degree_column: x[degree_column]
        },
        axis=1
    )
    
    
    return edge_df
    
def relationships_to_dataframe(relationships: Any):
    data = []
    for source, target, attributes in relationships:
        row = {
            'source': source,
            'target': target,
            'id': attributes.get('id'),
            'weight': attributes.get('weight'),
            'description': attributes.get('description'),
            'text_unit_ids': attributes.get('source_id'),
            'human_readable_id': attributes.get('human_readable_id'),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df