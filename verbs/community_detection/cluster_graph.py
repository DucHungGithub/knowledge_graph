from enum import Enum
from random import Random
from typing import Any, Dict, List
import logging

import networkx as nx

from utils.graph import load_graph
from utils.uuid import gen_uuid
from verbs.community_detection.typing import Communities

logger = logging.getLogger(__name__)

class GraphCommunityStrategyType(str, Enum):
    leiden = "leiden"
    
    def __repr__(self) -> str:
        return f"{self.value}"
    

async def clustering_graph(
    graphml: str | nx.Graph,
    args: Dict[str, Any],
    level: int = 0,
    seed: int = 6969,
    **kwargs
) -> nx.Graph:
    communities = await run_layout(graphml, args)
    return await apply_clustering(communites=communities, graphml=graphml, level=level, seed=seed)

async def apply_clustering(
    graphml: str,
    communites: Communities,
    level: int,
    seed: int
) -> nx.Graph:
    random = Random(seed)
    graph = nx.parse_graphml(graphml)
    for community_level, community_id, nodes in communites:
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    # add node degree
    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    # add node uuid and incremental record id (a human readable id used as reference in the final report)
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    # add ids to edges
    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level
    return graph
  
  
async def run_layout(
    graphml_or_graph: str | nx.Graph,
    args: Dict[str, Any],
) -> Communities:
    graph = load_graph(graphml_or_graph)
    if len(graph.nodes) == 0:
        logger.warning("Graph has no nodes", exc_info=True)
        return []
    
    clusters: Dict[int, Dict[str, List[str]]] = {}
    
    strategy_type = args.get("type", GraphCommunityStrategyType.leiden)
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            from .clustering import run_community_dectection
            clusters = run_community_dectection(graph=graph, args=args)
            
        case _:
            raise ValueError(f"Unknown clustering strategy {strategy_type}")
        
    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level,cluster_id,nodes))
    return results