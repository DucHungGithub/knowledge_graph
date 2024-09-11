# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from typing import Any
import logging

import pandas as pd

import graph.community_reports.schemas as schema
from verbs.community_summarization.prepare_report.community_nodes import prepare_community_reports_nodes

logger = logging.getLogger(__name__)

def restore_community_hierarchy(
    entities: Any,
    name_column: str = schema.NODE_NAME,
    community_column: str = schema.NODE_COMMUNITY,
    level_column: str = schema.NODE_LEVEL,
    **kwargs
) -> pd.DataFrame:
    node_df = prepare_community_reports_nodes(entities=entities)
    
    community_df = (
        node_df.groupby([community_column, level_column])
        .agg({name_column: list})
        .reset_index()
    )
    
    community_levels = {}
    for _, row in community_df.iterrows():
        level = row[level_column]
        name = row[name_column]
        community = row[community_column]

        if community_levels.get(level) is None:
            community_levels[level] = {}
        community_levels[level][community] = name

    # get unique levels, sorted in ascending order
    levels = sorted(community_levels.keys())

    community_hierarchy = []
    
    for idx in range(len(levels) - 1):
        level = levels[idx]
        logger.debug("Level: %s", level, exc_info=True)
        next_level = levels[idx + 1]
        current_level_communities = community_levels[level]
        next_level_communities = community_levels[next_level]
        logger.debug(
            "Number of communities at level %s: %s",
            level,
            len(current_level_communities),
            exc_info=True
        )

        for current_community in current_level_communities:
            current_entities = current_level_communities[current_community]

            # loop through next level's communities to find all the subcommunities
            entities_found = 0
            for next_level_community in next_level_communities:
                next_entities = next_level_communities[next_level_community]
                if set(next_entities).issubset(set(current_entities)):
                    community_hierarchy.append({
                        community_column: current_community,
                        schema.COMMUNITY_LEVEL: level,
                        schema.SUB_COMMUNITY: next_level_community,
                        schema.SUB_COMMUNITY_SIZE: len(next_entities),
                    })

                    entities_found += len(next_entities)
                    if entities_found == len(current_entities):
                        break
    return community_hierarchy