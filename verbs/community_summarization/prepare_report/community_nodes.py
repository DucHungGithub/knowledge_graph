# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from typing import Any
import pandas as pd

from graph.community_reports.schemas import (
    NODE_DEGREE,
    NODE_DESCRIPTION,
    NODE_DETAILS,
    NODE_ID,
    NODE_NAME
)

_MISSING_DESCRIPTION = "No Description"

def prepare_community_reports_nodes(
    entities: Any,
    to: str = NODE_DETAILS,
    id_column: str = NODE_ID,
    name_column: str = NODE_NAME,
    description_column: str = NODE_DESCRIPTION,
    degree_column: str = NODE_DEGREE,
    **kwargs
) -> pd.DataFrame:
    node_df = entities_to_dataframe(entities=entities)
    node_df = node_df.fillna(value={description_column: _MISSING_DESCRIPTION})

    node_df[to] = node_df.apply(
        lambda x: {
            id_column: x[id_column],
            name_column: x[name_column],
            description_column: x[description_column],
            degree_column: x[degree_column]
        },
        axis=1
    )
    return node_df

def entities_to_dataframe(entities: Any) -> pd.DataFrame:
    data = []
    for name, attributes in entities:
        row = {
            'id': attributes.get('id'),
            'title': name,
            'type': attributes.get('type'),
            'description': attributes.get('description'),
            'source_id': attributes.get('source_id'),
            'degree': attributes.get('degree'),
            'human_readable_id': attributes.get('human_readable_id')
        }
        
        row['community'] = attributes.get('cluster', None)
        row['level'] = attributes.get('level', 0)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df