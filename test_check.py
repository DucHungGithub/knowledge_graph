import json
from typing import List
import pydgraph

from models.entity import Entity

# Create a client stub
def create_client_stub() -> pydgraph.DgraphClientStub:
    return pydgraph.DgraphClientStub('localhost:9080')

# Create a client
def create_client(client_stub: pydgraph.DgraphClientStub) -> pydgraph.DgraphClient:
    return pydgraph.DgraphClient(client_stub)

def get_all_entities(client: pydgraph.DgraphClient) -> List[Entity]:
    txn = client.txn()
    
    entities = []
    
    try:
        query = f"""
        {{
            getEntities(func: type(Entity)){{
                id
                title
                short_id
                type
                description
                rank
                community_ids
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        get_entities = ppl["getEntities"]
        
        for entity in get_entities:
            entities.append(Entity(**entity))
    finally:
        txn.discard()
        
    return entities
        
def get_all_community_report(client: pydgraph.DgraphClient):
    txn = client.txn()
    
    try:
        query = f"""
        {{
            getCommunityReports(func: type(CommunityReport)){{
                expand(_all_)
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        # print(ppl["getCommunityReports"])
    finally:
        txn.discard()
    

def get_all_covariates(client: pydgraph.DgraphClient):
    txn = client.txn()
    
    try:
        query = f"""
        {{
            getCovariates(func: type(Covariate)){{
                expand(_all_)
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        # print(ppl["getCovariates"])
    finally:
        txn.discard()

def get_relationships(client: pydgraph.DgraphClient):
    txn = client.txn()
    
    try:
        query = f"""
        {{
            getEntitiesRelationships(func: type(Entity)){{
                connect @facets {{
                    expand(_all_)
                }}
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        # print(ppl["getEntitiesRelationships"])
    finally:
        txn.discard()

 
if __name__=="__main__":
    stub = create_client_stub()
    client = create_client(stub)
    
    # print(get_all_entities(client))
    # get_all_community_report(client)
    # get_all_covariates(client)
    # get_relationships(client)