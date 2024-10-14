
import json
from typing import List, Tuple
import pydgraph

from langchain_core.vectorstores import VectorStore

from models.community_report import CommunityReport
from models.entity import Entity

def reduce_fanouts_wrapper(a, b):
    def _combine(left, middle, right):
        left = left if isinstance(left, list) else [left] if left else []
        middle = middle if isinstance(middle, list) else [middle] if middle else []
        right = right if isinstance(right, list) else [right] if right else []
        return [item for item in left + middle + right]

    if isinstance(a, tuple) and len(a) == 2:
        return _combine(a[0], a[1], b)
    else:
        return _combine(a, None, b)
    
    
def retrieve_relevant_community_reports(db: VectorStore, client: pydgraph.DgraphClient, query: str) -> Tuple[List[CommunityReport], List[Entity]]:
    selected_communities = []
    selected_entities = []
    
    docs = db.similarity_search(query=query, k=10)
    
    
    
    txn = client.txn()
    
    try:
        for doc in docs:
            query = f"""
            {{
                getCommunity(func: type(CommunityReport)) @filter(eq(id, "{doc.metadata["id"]}"))  @recurse(depth: 1000) {{
                        id
                        title
                        short_id
                        community_id
                        type
                        description
                        summary
                        rank
                        text_unit_ids
                        community_ids
                        summary_embedding
                        full_content
                        full_content_embedding
                        attributes
                        ~belong @facets
                }}
            }}
            """
            
            res = txn.query(query=query)
            ppl = json.loads(res.json)
            
            coms = ppl.get("getCommunity", [])
            
            for com in coms:
                if com.get("attributes", None):
                    com["attributes"] = json.loads(com["attributes"]) if com["attributes"] else None
                selected_communities.append(CommunityReport(**com))
                entities = []
                if com.get("~belong", None):
                    entities = com["~belong"]
                for entity in entities:
                    selected_entities.append(Entity(**entity))
        
    finally:
        txn.discard()
        
    return selected_communities, selected_entities



