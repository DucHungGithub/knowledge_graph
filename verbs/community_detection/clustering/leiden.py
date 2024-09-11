from typing import Any, Dict, List
import logging

import networkx as nx
from graspologic.partition import hierarchical_leiden

from utils.graph import stable_largest_connected_component

logger = logging.getLogger(__name__)


def run_community_dectection(graph: nx.Graph, args: Dict[str, Any]) -> Dict[int, Dict[str, List[str]]]:
    max_cluster_size = args.get("max_cluster_size", 10)
    use_lcc = args.get("use_lcc", True)
    
    if args.get("verbose", False):
        logger.info(
            f"Running leiden with max_cluster_size={max_cluster_size}, lcc={use_lcc}", exc_info=True
        )
    
    node_id_to_community_map = compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 6969)
    )
    
    levels = args.get("levels")
    
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())
    
    results_by_level: Dict[int, Dict[str, List[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


def compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int = 6969
) -> Dict[int, Dict[str, int]]:
    if use_lcc:
        graph = stable_largest_connected_component(graph)
        
    community_mapping = hierarchical_leiden(graph, max_cluster_size=max_cluster_size,random_seed=seed)

    results: Dict[int, Dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster
        
    return results
