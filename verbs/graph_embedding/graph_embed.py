from enum import Enum
from typing import Any, Dict, List
import networkx as nx

from utils.graph import load_graph

from .typing import NodeEmbedding

class EmbedGraphStrategyType(str, Enum):
    node2vec = "node2vec"
    
    def __repr__(self) -> str:
        return f"{self.value}"
    


async def embed_graph(
    raphml_or_graph: str | nx.Graph,
    strategy: Dict[str, Any],
    **kwargs
) -> NodeEmbedding:
    strategy_type = strategy.get("type", EmbedGraphStrategyType.node2vec)
    strategy_args = {**strategy}
    
    result = await run_embeddings(strategy_type, raphml_or_graph, strategy_args)
    return result

async def run_embeddings(
    strategy: EmbedGraphStrategyType,
    graphml_or_graph: str | nx.Graph,
    args: Dict[str, Any],
) -> NodeEmbedding:
    """Run embeddings method definition."""
    graph = load_graph(graphml_or_graph)
    match strategy:
        case EmbedGraphStrategyType.node2vec:
            from verbs.graph_embedding.embed.run import run_embedding
            return run_embedding(graph, args)
        case _:
            msg = f"Unknown strategy {strategy}"
            raise ValueError(msg)