# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph, _get_node_attributes, _get_edge_attributes and _get_attribute_column_mapping methods definition."""

from typing import Any, cast

import pandas as pd


def compute_edge_combined_degree(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    to: str = "rank",
    node_name_column: str = "title",
    node_degree_column: str = "degree",
    edge_source_column: str = "source",
    edge_target_column: str = "target",
    **_kwargs,
) -> pd.DataFrame:
    """
    Compute the combined degree for each edge in a graph.

    Inputs Tables:
    - input: The edge table
    - nodes: The nodes table.

    Args:
    - to: The name of the column to output the combined degree to. Default="rank"
    """
    if to in edge_df.columns:
        return edge_df
    node_degree_df = _get_node_degree_table(node_df, node_name_column, node_degree_column)

    def join_to_degree(df: pd.DataFrame, column: str) -> pd.DataFrame:
        degree_column = _degree_colname(column)
        result = df.merge(
            node_degree_df.rename(
                columns={node_name_column: column, node_degree_column: degree_column}
            ),
            on=column,
            how="left",
        )
        result[degree_column] = result[degree_column].fillna(0)
        return result

    edge_df = join_to_degree(edge_df, edge_source_column)
    edge_df = join_to_degree(edge_df, edge_target_column)
    edge_df[to] = (
        edge_df[_degree_colname(edge_source_column)]
        + edge_df[_degree_colname(edge_target_column)]
    )

    return edge_df


def _degree_colname(column: str) -> str:
    return f"{column}_degree"


def _get_node_degree_table(
    node_df: pd.DataFrame, node_name_column: str, node_degree_column: str
) -> pd.DataFrame:
    return cast(pd.DataFrame, node_df[[node_name_column, node_degree_column]])
