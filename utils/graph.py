from typing import cast

import html
import networkx as nx
from graspologic.utils import largest_connected_component

def load_graph(graphml: str | nx.Graph) -> nx.Graph:
    """Load a graph from a graphml file or a networkx graph."""
    return nx.parse_graphml(graphml) if isinstance(graphml, str) else graphml

 
def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    graph = graph.copy()
    graph = cast(nx.Graph, largest_connected_component(graph))
    graph = normalize_node_names(graph)
    return  stabilize_graph(graph)


# If the graph is undirected, we create the edges in a stable way, so we get the same results
# for example:
# A -> B
# in graph theory is the same as
# B -> A
# in an undirected graph
# however, this can lead to downstream issues because sometimes
# consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
# but they base some of their logic on the order of the nodes, so the order ends up being important
# so we sort the nodes in the edge in a stable way, so that we always get the same order
def stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    
    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
    
    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))
    
    if not graph.is_directed():
        
        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                source, target = target, source
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]
        
    edges = sorted(edges, key=lambda x: f"{x[0]} -> {x[1]}")
    
    fixed_graph.add_edges_from(edges)
    return fixed_graph



def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    return nx.relabel_nodes(graph, node_mapping)