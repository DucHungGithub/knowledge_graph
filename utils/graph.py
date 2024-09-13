import json
import os
from pathlib import Path
from typing import Any, Dict, cast

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




def combine_graphs_for_extract(graph1: nx.Graph, graph2: nx.Graph, checkpoint_dir: str) -> nx.Graph:
    """
    Combine two NetworkX graphs based on node IDs and edge source/target.
    
    Args:
    graph1 (nx.Graph): The first graph (old graph)
    graph2 (nx.Graph): The second graph (new graph)
    
    Returns:
    nx.Graph: The combined graph
    """
    combined_graph = nx.Graph()
    
    # Combine nodes
    for node, data in graph1.nodes(data=True):
        combined_graph.add_node(node, **data)
    
    for node, data in graph2.nodes(data=True):
        if node in combined_graph:
            # Node already exists, update attributes
            print("CHECK UPDATE NODE")
            combined_data = combined_graph.nodes[node]
            combined_data['source_id'] = f"{combined_data.get('source_id', '')},{data.get('source_id', '')}".strip(',')
            combined_data['description'] = f"Old Information---\n{combined_data.get('description', '')}\nNew Information---\n{data.get('description', '')}"
            print(combined_data)
        else:
            # New node, add it to the combined graph
            combined_graph.add_node(node, **data)
    
    # Combine edges
    for source, target, data in graph1.edges(data=True):
        combined_graph.add_edge(source, target, **data)
    
    for source, target, data in graph2.edges(data=True):
        if combined_graph.has_edge(source, target):
            # Edge already exists, update attributes
            print("CHECK UPDATE EDGE")
            combined_data = combined_graph.edges[source, target]
            combined_data['source_id'] = f"{combined_data.get('source_id', '')},{data.get('source_id', '')}".strip(',')
            combined_data['description'] = f"Old---\n{combined_data.get('description', '')}\nNew---\n{data.get('description', '')}"
            print(combined_data)
        else:
            # New edge, add it to the combined graph
            combined_graph.add_edge(source, target, **data)
    
    save_files_graphml(checkpoint_dir=checkpoint_dir, file_name="summarize_graph.graphml",graph_ml=combined_graph)
    
    return combined_graph



def combine_graphs_for_clustering(graph1: nx.Graph, graph2: nx.Graph, checkpoint_dir: str | Path) -> nx.Graph:
    """
    Combine two NetworkX graphs based on node IDs and edge source/target.
    
    Args:
    graph1 (nx.Graph): The first graph (old graph)
    graph2 (nx.Graph): The second graph (new graph)
    
    Returns:
    nx.Graph: The combined graph
    """
    combined_graph = nx.Graph()
    new_nodes = {}
    new_edges = {}
    update_nodes = {}
    update_edges = {}
    
    # Combine nodes
    for node, data in graph1.nodes(data=True):
        combined_graph.add_node(node, **data)
    
    for node, data in graph2.nodes(data=True):
        if node in combined_graph:
            # Node already exists, update attributes
            
            print("CHECK UPDATE 2:")
            combined_data = combined_graph.nodes[node]
            print("Old_Content: ----")
            print(combined_data)
            combined_data.update(data)
            combined_data["id"] = combined_data.get('id', data.get('id'))
            update_nodes[node] = combined_data
            print("New_Content: ------")
            print(combined_data)
        else:
            # New node, add it to the combined graph
            combined_graph.add_node(node, **data)
            new_nodes[node] = data
    
    # Combine edges
    for source, target, data in graph1.edges(data=True):
        combined_graph.add_edge(source, target, **data)
    
    for source, target, data in graph2.edges(data=True):
        if combined_graph.has_edge(source, target):
            # Edge already exists, update attributes
            combined_data = combined_graph.edges[source, target]
            combined_data.update(data)
            combined_data['id'] = combined_data.get('id', data.get('id'))
            update_edges[f"{source}-{target}"] = combined_data
        else:
            # New edge, add it to the combined graph
            combined_graph.add_edge(source, target, **data)
            new_edges[f"{source}-{target}"] = data
            
    
    save_checkpoints_and_graph(checkpoint_dir=checkpoint_dir, new_nodes=new_nodes, new_edges=new_edges, update_nodes=update_nodes, update_edges=update_edges, graph_ml=combined_graph)
    
    return combined_graph
    
    
def save_checkpoints_and_graph(
    checkpoint_dir: str,
    new_nodes: Dict[str, Any],
    new_edges: Dict[str, Any],
    update_nodes: Dict[str, Any],
    update_edges: Dict[str, Any],
    graph_ml: nx.graph
) ->  None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(os.path.join(checkpoint_dir, "new_nodes.json"), "w") as f:
        json.dump(new_nodes, f, indent=2)
    
    with open(os.path.join(checkpoint_dir, "new_edges.json"), "w") as f:
        json.dump(new_edges, f, indent=2)
    
    with open(os.path.join(checkpoint_dir, "update_nodes.json"), "w") as f:
        json.dump(update_nodes, f, indent=2)
    
    with open(os.path.join(checkpoint_dir, "update_edges.json"), "w") as f:
        json.dump(update_edges, f, indent=2)
    
    # Save combined graph
    save_files_graphml(checkpoint_dir=checkpoint_dir, file_name="cluster_graph.graphml",graph_ml=graph_ml)
    


def load_graph_from_file(checkpoint_dir: str, file_name: str) -> nx.Graph | None:
    file_path = os.path.join(checkpoint_dir, file_name)
    if not os.path.isfile(file_path):
        return None
    return nx.read_graphml(file_path)


def save_files_graphml(checkpoint_dir: str | Path, file_name: str, graph_ml: nx.Graph) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = checkpoint_dir / file_name
    
    nx.write_graphml(graph_ml, file_path)



def load_checkpoint(checkpoint_dir: str) -> tuple:
    """
    Load checkpoint information and combined graph.
    
    Args:
    checkpoint_dir (str): Directory containing checkpoint files
    
    Returns:
    tuple: (new_nodes, new_edges, update_nodes, update_edges, combined_graph)
    """
    with open(os.path.join(checkpoint_dir, "new_nodes.json"), "r") as f:
        new_nodes = json.load(f)
    
    with open(os.path.join(checkpoint_dir, "new_edges.json"), "r") as f:
        new_edges = json.load(f)
    
    with open(os.path.join(checkpoint_dir, "update_nodes.json"), "r") as f:
        update_nodes = json.load(f)
    
    with open(os.path.join(checkpoint_dir, "update_edges.json"), "r") as f:
        update_edges = json.load(f)
    
    
    return new_nodes, new_edges, update_nodes, update_edges


