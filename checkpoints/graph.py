import json
import os
import networkx as nx


class GraphCheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.new_nodes_file = os.path.join(checkpoint_dir, "new_nodes.json")
        self.new_edges_file = os.path.join(checkpoint_dir, "new_edges.json")

    def save_checkpoint(self, combined_graph: nx.Graph, old_graph: nx.Graph):
        new_nodes = {}
        new_edges = {}

        # Check for new nodes
        for node, data in combined_graph.nodes(data=True):
            if node not in old_graph.nodes:
                new_nodes[node] = data

        # Check for new edges
        for source, target, data in combined_graph.edges(data=True):
            if not old_graph.has_edge(source, target):
                new_edges[f"{source}-{target}"] = data

        # Save new nodes
        with open(self.new_nodes_file, 'w') as f:
            json.dump(new_nodes, f, indent=2)

        # Save new edges
        with open(self.new_edges_file, 'w') as f:
            json.dump(new_edges, f, indent=2)

    def load_checkpoint(self) -> tuple:
        new_nodes = {}
        new_edges = {}

        if os.path.exists(self.new_nodes_file):
            with open(self.new_nodes_file, 'r') as f:
                new_nodes = json.load(f)

        if os.path.exists(self.new_edges_file):
            with open(self.new_edges_file, 'r') as f:
                new_edges = json.load(f)

        return new_nodes, new_edges