import json
import pydgraph

# Create a client stub
def create_client_stub() -> pydgraph.DgraphClientStub:
    return pydgraph.DgraphClientStub('localhost:9080')

# Create a client
def create_client(client_stub: pydgraph.DgraphClientStub) -> pydgraph.DgraphClient:
    return pydgraph.DgraphClient(client_stub)

def get_all_entities(client: pydgraph.DgraphClient):
    txn = client.txn()
    try:
        query = f"""
        {{
            getEntities(func: type(Entity)){{
                expand(_all_)
            }}
        }}
        """
        
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        print(ppl["getEntities"])
    finally:
        txn.discard()
        
        
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
        
        print(ppl["getCommunityReports"])
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
        
        print(ppl["getCovariates"])
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
        
        print(ppl["getEntitiesRelationships"])
    finally:
        txn.discard()

 
if __name__=="__main__":
    stub = create_client_stub()
    client = create_client(stub)
    
    # get_all_entities(client)
    # get_all_community_report(client)
    get_all_covariates(client=client)