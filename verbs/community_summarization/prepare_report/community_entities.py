import pandas as pd

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

def prepare_community_reports_entities(
    node_df: pd.DataFrame
) -> pd.DataFrame:
    entities_data = []
    
    for _, node in node_df.iterrows():
        entity = {
            "id": node.get("id"),
            "name": node.get("title"),
            "type": node.get("type"),
            "description": node.get("description"),
            "human_readable_id": node.get("human_readable_id"),
            "graph_embedding": None,
            "text_unit_ids": node.get("source_id"),
            # "description_embedding": embeddings.embed_query(node.get("description")) if node.get("description") else None
        }
        entities_data.append(entity)
        
        

    entities_df = pd.DataFrame(entities_data)
    
    return entities_df
