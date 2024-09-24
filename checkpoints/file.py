import pandas as pd
import networkx as nx
import os
import json
from typing import List, Dict, Any

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, entities: pd.DataFrame, relationships: pd.DataFrame, run_id: str):
        entities_file = os.path.join(self.checkpoint_dir, f"entities_{run_id}.csv")
        relationships_file = os.path.join(self.checkpoint_dir, f"relationships_{run_id}.csv")
        
        entities.to_csv(entities_file, index=False)
        relationships.to_csv(relationships_file, index=False)
        
    def load_checkpoint(self, run_id: str) -> tuple:
        entities_file = os.path.join(self.checkpoint_dir, f"entities_{run_id}.csv")
        relationships_file = os.path.join(self.checkpoint_dir, f"relationships_{run_id}.csv")
        
        if os.path.exists(entities_file) and os.path.exists(relationships_file):
            entities = pd.read_csv(entities_file)
            relationships = pd.read_csv(relationships_file)
            return entities, relationships
        else:
            return None, None
        
    def update_graph(self, current_entities: pd.DataFrame, current_relationships: pd.DataFrame, 
                     new_entities: List[Dict[str, Any]], new_relationships: List[Dict[str, Any]]) -> tuple:
        # Update entities
        new_entities_df = pd.DataFrame(new_entities)
        updated_entities = pd.concat([current_entities, new_entities_df]).drop_duplicates(subset=['id'])
        
        # Update relationships
        new_relationships_df = pd.DataFrame(new_relationships)
        updated_relationships = pd.concat([current_relationships, new_relationships_df]).drop_duplicates(subset=['id'])
        
        return updated_entities, updated_relationships
    
    def create_networkx_graph(self, entities: pd.DataFrame, relationships: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        
        for _, entity in entities.iterrows():
            G.add_node(entity['id'], **entity.to_dict())
        
        for _, rel in relationships.iterrows():
            G.add_edge(rel['source'], rel['target'], **rel.to_dict())
        
        return G







# def run_extraction_with_checkpoints(checkpoint_manager: CheckpointManager, run_id: str, 
#                                     entity_types: List[str], args: Dict[str, Any]):
#     # Load previous checkpoint if exists
#     prev_entities, prev_relationships = checkpoint_manager.load_checkpoint(run_id)
    
#     # Run new extraction
#     extraction_result = run_extract_entities(llm, docs, entity_types, args)
    
#     if prev_entities is not None and prev_relationships is not None:
#         # Update with new results
#         updated_entities, updated_relationships = checkpoint_manager.update_graph(
#             prev_entities, prev_relationships, extraction_result.entities, extraction_result.rels
#         )
#     else:
#         # First run, use new results directly
#         updated_entities = pd.DataFrame(extraction_result.entities)
#         updated_relationships = pd.DataFrame(extraction_result.rels)
    
#     # Save updated checkpoint
#     checkpoint_manager.save_checkpoint(updated_entities, updated_relationships, run_id)
    
#     # Create and return updated graph
#     return checkpoint_manager.create_networkx_graph(updated_entities, updated_relationships)
