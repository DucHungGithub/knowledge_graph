import asyncio
from enum import Enum
from typing import Any, Dict, List, Tuple

from langchain.pydantic_v1 import BaseModel
import networkx as nx

from verbs.entities.extraction.typing import EntityExtractionResult
from verbs.entities.summarization.typing import SummarizationStrategy
from utils.graph import load_graph

class DescriptionSummarizeRow(BaseModel):
    graph: Any

class SummarizeStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"
    
    def __repr__(self) -> str:
        return f"{self.value}"
    

async def summarize_description(
    entity_result: EntityExtractionResult,
    strategy: Dict[str, Any] | None = None,
    **kwargs
) -> str | None:
    strategy = strategy or {}
    strategy_exec = load_strategy(strategy.get("SUMMARIZE_ENTITY_RELATIONSHIP", SummarizeStrategyType.graph_intelligence))
    config = {**strategy}
    
    async def summarize_graph_items(graph: nx.Graph, semaphore: asyncio.Semaphore):
        async def process_item(item, desc):
            async with semaphore:
                return await strategy_exec(item, sorted(set(desc.split("\n"))), config)

        futures = []
        for node in graph.nodes():
            futures.append(process_item(node, graph.nodes[node].get("description", "")))
        for edge in graph.edges():
            futures.append(process_item(edge, graph.edges[edge].get("description", "")))
        
        results = await asyncio.gather(*futures)
        
        for result in results:
            item = result.items
            if isinstance(item, str) and item in graph.nodes():
                graph.nodes[item]["description"] = result.description
            elif isinstance(item, tuple) and item in graph.edges():
                graph.edges[item]["description"] = result.description
        
        return graph

    graph = load_graph(entity_result.graphml_graph)
    semaphore = asyncio.Semaphore(kwargs.get("num_threads", 4))
    updated_graph = await summarize_graph_items(graph, semaphore)
    
    graphml_generator = nx.generate_graphml(updated_graph)
    graphml_str = ''.join(graphml_generator)
    
    return graphml_str



def load_strategy(strategy_type: SummarizeStrategyType) -> SummarizationStrategy:
    """Load strategy method definition"""
    match strategy_type:
        case SummarizeStrategyType.graph_intelligence:
            from verbs.entities.summarization.graph_intelligence.run import run_gi
            return run_gi
        
        case _:
            raise ValueError(f"Unknown Strategy: {strategy_type}")
        
        
        
        
        

        
        
        