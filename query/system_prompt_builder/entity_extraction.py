from enum import Enum
from typing import List, Optional

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


from models.entity import Entity
from models.relationship import Relationship
from query.inputs.retrieval.entities import get_entity_by_key, get_entity_by_name


class EntityVectorStoreKey(str, Enum):
    """Keys used as ids in the entity embedding vectorstores."""
    
    ID = "id"
    TITLE = "title"
    
    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        """Convert string to EntityVectorStoreKey."""
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE
        
        raise ValueError(f"Invalid EntityVectorStoreKey: {value}")
    
def map_query_to_entities(
    query: str,
    text_embedding_vectorstore: VectorStore,
    all_entities: List[Entity],
    text_embedder: Optional[Embeddings] = None,
    embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    include_entity_names: Optional[List[str]] = None,
    exclude_entity_names: Optional[List[str]] = None,
    k: int = 10,
    oversample_scaler: int = 2
) -> List[Entity]:
    """Extract entities that match a given query using semantic similarity of text embeddings of query and entity descriptions."""
    
    if include_entity_names is None:
        include_entity_names = []
    if exclude_entity_names is None:
        exclude_entity_names = []
        
    matched_entities = []
    if query != "":
        search_results = text_embedding_vectorstore.similarity_search_with_relevance_scores(
            query = query,
            k= k * oversample_scaler
        )
        
        for result in search_results:
            matched = get_entity_by_key(
                entities=all_entities,
                key=embedding_vectorstore_key,
                value=str(result[0].metadata["id"])
            )
            
            if matched:
                matched_entities.append(matched)
    else:
        all_entities.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
        matched_entities = all_entities[:k]    
        
    # filter out excluded entities
    if exclude_entity_names:
        matched_entities = [
            entity
            for entity in matched_entities
            if entity.title not in exclude_entity_names
        ]
        
    # add entities in the include_entity list
    included_entities = []
    for entity_name in include_entity_names:
        included_entities.extend(get_entity_by_name(all_entities, entity_name))
        
    return included_entities + matched_entities


def find_nearest_neighbors_by_graph_embeddings(
    entity_id: str,
    graph_embedding_vectorstore: VectorStore,
    all_entities: List[Entity],
    exclude_entity_names: Optional[List[str]] = None,
    embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    k: int = 10,
    oversample_scaler: int = 2
) -> List[Entity]:
    """Retrieve related entities by graph embeddings."""
    if exclude_entity_names is None:
        exclude_entity_names = []
        
    # Find nearest neighbors of this entity using graph embedding
    query_entity = get_entity_by_key(
        entities=all_entities, key=embedding_vectorstore_key, value=entity_id
    )
    query_embedding = query_entity.graph_embedding if query_entity else None
    
    # Oversample to account for excluded entities
    if query_embedding:
        matched_entities = []
        search_results = graph_embedding_vectorstore.similarity_search_by_vector(
            embedding=query_embedding, k = k*oversample_scaler
        )
        
        for result in search_results:
            matched = get_entity_by_key(
                entities=all_entities,
                key=embedding_vectorstore_key,
                value=result.id
            )
            if matched:
                matched_entities.append(matched)
                
        # Filter out excluded entities
        if exclude_entity_names:
            matched_entities = [
                entity 
                for entity in matched_entities
                if entity.title not in exclude_entity_names
            ]
        matched_entities.sort(key=lambda x: x.rank, reverse=True)
        return matched_entities[:k]
    
    return []


def find_nearest_neighbors_by_entity_rank(
    entity_name: str,
    all_entities: List[Entity],
    all_relationships: List[Relationship],
    exclude_entity_names: Optional[List[str]] = None,
    k: Optional[int] = 10
) -> List[Entity]:
    """Retrieve entities that have direct connections with the target entity, sorted by entity rank."""
    if exclude_entity_names is None:
        exclude_entity_names = []
        
    entity_relationships = [
        rel 
        for rel in all_relationships
        if rel.source == entity_name or rel.target == entity_name
    ]
    
    source_entity_names = {rel.source for rel in entity_relationships}
    target_entity_names = {rel.target for rel in entity_relationships}
    
    related_entity_names = (source_entity_names.union(target_entity_names)).difference(
        set(exclude_entity_names)
    )
    
    top_relations = [
        entity for entity in all_entities if entity.title in related_entity_names
    ]
    top_relations.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
    
    if k:
        return top_relations[:k]
    return top_relations