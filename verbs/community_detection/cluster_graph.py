from enum import Enum
from random import Random
from typing import Any, Dict, List
import logging

import networkx as nx

from utils.graph import combine_graphs_for_clustering, combine_graphs_for_extract, load_graph, load_graph_from_file, save_files_graphml
from utils.uuid import gen_uuid
from verbs.community_detection.typing import Communities
import config as defs

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
    
    graph = load_graph(graphml)
    
    old_graph = load_graph_from_file(checkpoint_dir=defs.CHECKPOINT_DIR, file_name="summarize_graph.graphml")
    if old_graph:
        graph = combine_graphs_for_extract(graph1=old_graph, graph2=graph, checkpoint_dir=defs.CHECKPOINT_DIR)
    else:
        save_files_graphml(checkpoint_dir=defs.CHECKPOINT_DIR, file_name="summarize_graph.graphml", graph_ml=graph)
    
    communities = await run_layout(graph, args)
    return await apply_clustering(communites=communities, graphml=graph, level=level, seed=seed)

async def apply_clustering(
    graphml: str | nx.Graph,
    communites: Communities,
    level: int,
    seed: int
) -> nx.Graph:
    random = Random(seed)
    graph = load_graph(graphml)
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
        
    old_graph = load_graph_from_file(checkpoint_dir=defs.CHECKPOINT_DIR, file_name="cluster_graph.graphml")
    if old_graph:
        graph = combine_graphs_for_clustering(graph1=old_graph, graph2=graph, checkpoint_dir=defs.CHECKPOINT_DIR)
    else:
        save_files_graphml(checkpoint_dir=defs.CHECKPOINT_DIR, file_name="cluster_graph.graphml", graph_ml=graph)
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