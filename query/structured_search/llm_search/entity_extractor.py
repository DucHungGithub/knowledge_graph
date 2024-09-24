

from langgraph.graph import StateGraph

from query.inputs.retrieval.entities import get_entity_by_key
from query.system_prompt_builder.entity_extraction import map_query_to_entities

def get_relavent_entity(graph: StateGraph):
    message = graph["messages"][-1]
    k = graph["k"]
    
    results = graph["entity_vectorstore"].similarity_search_with_relevance_scores(
        query=message,
        k=k
    )
    
    entities_matched = []
    
    for result in results:
        