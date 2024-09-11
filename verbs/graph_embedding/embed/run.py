from typing import Any, Dict, List

import networkx as nx

from graph.embedding.embedding import embed_nod2vec
from utils.graph import stable_largest_connected_component


def run_embedding(
    graph: nx.Graph,
    args: Dict[str, Any]
) -> Dict[str, List[float]]:
    if args.get("use_lcc", True):
        graph = stable_largest_connected_component(graph)
        
    embeddings = embed_nod2vec(
        graph=graph,
        dimensions=args.get("dimensions", 1536),
        num_walks=args.get("num_walks", 10),
        walk_length=args.get("walk_length", 40),
        window_size=args.get("window_size", 2),
        iterations= args.get("iterations", 3),
        random_seed=args.get("random_seed", 6969)
    )
    
    pairs = zip(embeddings.nodes, embeddings.embeddings,strict=True)
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    return dict(sorted_pairs)