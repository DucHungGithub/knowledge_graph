import json
import logging
import os
from typing import List
import pydgraph

import pandas as pd
from models.community_report import CommunityReport
from models.entity import Entity
from models.relationship import Relationship
from models.text_unit import TextUnit
from models.covariate import Covariate
from query.inputs.loader.dfs import read_entities
from query.inputs.loader.indexer_adapters import read_indexer_covariates, read_indexer_entities, read_indexer_relationships, read_indexer_reports, read_indexer_text_units


import colorlog

# Set up basic logging configuration with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# Get the logger and add the handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.addHandler(handler)

INPUT_DIR = "outputs"

COMMUNITY_REPORT_TABLE = "community_report.csv"
ENTITY_TABLE = "node.csv"
ENTITY_EMBEDDING_TABLE = "entity.csv"
RELATIONSHIP_TABLE = "relationship.csv"
COVARIATE_TABLE = "claims.csv"
TEXT_UNIT_TABLE = "text_unit.csv"
TABLE_PATH = "/home/hungquan/build_kg/lancedb_store"
TABLE_NAME = "multimodal_test"
COMMUNITY_LEVEL = 2


# Set schema.
def set_schema(client: pydgraph.DgraphClient):
    schema = """
    id: string @index(exact) .
    title: string @index(exact) .
    name: string @index(exact) .
    type: string .
    description: string .
    human_readable_id: string .
    text_unit_ids: [string] .
    chunk: string .
    n_tokens: int .
    source: string .
    degree: int .
    community: int .
    level: int .
    connect: [uid] @reverse .
    summary: string .
    full_content: string .
    full_content_json: string .
    rank: float .
    rank_explanation: string .
    explanation: string .
    subject_id: string .
    object_id: string .
    source_text: [string] .
    from: [uid] @reverse .
    claim_details: string .
    claim: [uid] @reverse .
    attributes: string .
    relationship_ids: [string] .
    covariate_ids: [string] .
    
    friend: [uid] @reverse .
    age: int .
    married: bool .
    loc: geo .
    dob: datetime .
    
    
    
    type TextUnit {
        id
        chunk
        n_tokens
        source
    }
    
    type Entity {
        id
        title
        type
        description
        text_unit_ids
        degree
        human_readable_id
        community
        level
        connect
        from
        claim
    }
    
    
    type CommunityReport {
        id
        community
        title
        summary
        full_content
        full_content_json
        rank
        level
        rank_explanation
        
    }
    
    type Covariate {
        subject_id
        object_id
        type
        description
        source_text
        id
        human_readable_id
        claim_details
        from
    }
    
    """
    return client.alter(pydgraph.Operation(schema=schema))


# Create a client stub.
def create_client_stub() -> pydgraph.DgraphClientStub:
    return pydgraph.DgraphClientStub('localhost:9080')


# Create a client.
def create_client(client_stub) -> pydgraph.DgraphClient:
    return pydgraph.DgraphClient(client_stub)


def injest_text_units(client: pydgraph.DgraphClient, text_units: List[TextUnit]):
    txn = client.txn()
    try:
        mutations = []
        
        for tu in text_units:
            p = {**tu.dict(), "dgraph.type": "TextUnit"}
            p["attributes"] = json.dumps(p["attributes"]) if p["attributes"] else None
            mutations.append(txn.create_mutation(set_obj={**p, "dgraph.type": "TextUnit"}))
        
        request = txn.create_request(mutations=mutations, commit_now=True)
        response = txn.do_request(request)
        
        logger.info(f"Mutation response: {response}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        txn.discard()
    

def query_and_ingest_entity(client: pydgraph.DgraphClient, entities: List[Entity]):
    txn = client.txn()

    try:
        mutations = []
        for entity in entities:
            source_ids = []
            community_ids = []
            covariate_ids = []
            p = entity.dict()
            # p.pop("text_unit_ids")
            if entity.text_unit_ids:
                for tu in entity.text_unit_ids:
                    query = f"""
                    {{
                        getTextUnit(func: eq(id, "{tu}")) @filter(has(text)) {{
                            uid
                        }}
                    }}
                    """
                    res = txn.query(query=query)
                    ppl = json.loads(res.json)
                    
                    
                    rel = ppl['getTextUnit']
                    
                    source_ids.extend(rel)
                    
            if entity.community_ids:
                for cm in entity.community_ids:
                    if cm == "-1":
                        continue
                    query = f"""
                    {{
                        getCommunity(func: type(CommunityReport)) @filter(eq(community_id, {cm})) {{
                            uid
                        }}
                    }}
                    """
                    res = txn.query(query=query)
                    ppl = json.loads(res.json)
                    
                    
                    rel = ppl['getCommunity']
                    
                    community_ids.extend(rel)
                    
            
            if entity.title:
                query = f"""{{
                    getCovariates(func: type(Covariate)) @filter(eq(subject_id, "{entity.title}")){{
                        uid
                    }}
                }}
                """
                res = txn.query(query=query)
                ppl = json.loads(res.json)
                
                rel = ppl['getCovariates']
                
                covariate_ids.extend(rel)
                
            p["attributes"] = json.dumps(p["attributes"]) if p["attributes"] else None
            p["from"] = source_ids
            p["belong"] = community_ids
            p["claim"] = covariate_ids
            p["dgraph.type"] = "Entity"
            
            mutations.append(txn.create_mutation(set_obj={**p, "attributes": json.dumps(p["attributes"]) if p["attributes"] else None}))
            
        request = txn.create_request(mutations=mutations, commit_now=True)
        response = txn.do_request(request)
        
        logger.info(f"Mutation response: {response}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Clean up the transaction
        txn.discard()
            
    

def query_and_ingest_relationship(client: pydgraph.DgraphClient, relationships: List[Relationship]):
    txn = client.txn()
    
    try:
        for rel in relationships:
            query1 = f"""
            {{
                getEntity(func: eq(title, "{rel.source}")) {{
                    uid
                    connect @facets{{
                        uid
                    }}
                }}
            }}
            """
            res1 = txn.query(query=query1)
            ppl1 = json.loads(res1.json)
            
            
            entity1 = ppl1['getEntity'][0]
            uid1 = entity1['uid']
            existing_rel = entity1.get('connect', [])            
            
            query2 = f"""
            {{
                getEntity(func: eq(title, "{rel.target}")) {{
                    uid
                }}
            }}
            """
            res2 = txn.query(query=query2)
            ppl2 = json.loads(res2.json)
            entity2 = ppl2['getEntity'][0]
            uid2 = entity2['uid']
            
            p = {
                "uid": uid2
            }
            
            key_value = rel.dict()
            key_value["text_unit_ids"] = json.dumps(key_value["text_unit_ids"]) if json.dumps(key_value["text_unit_ids"]) else None
            for key, val in key_value.items():
                if isinstance(val,str) or isinstance(val, int) or isinstance(val, float) or isinstance(val,bool):
                    p[f"connect|{key}"] = val
        
            
            if p not in existing_rel:
                existing_rel.append(p)
            
            
            mutation = {
                'set': [
                    {
                        'uid': uid1,
                        'connect': existing_rel
                    }
                ]
            }
            
            txn.mutate(set_obj=mutation)
        
        txn.commit()
    finally:
        txn.discard()
    


def query_and_ingest_covariates(client: pydgraph.DgraphClient, covariates: List[Covariate]):
    txn = client.txn()
    
    try:
        mutations = []
        for cov in covariates:
            source_ids = []
            p = cov.dict()
            # print("CHECK---------COV-----------")
            # print(p)
            # p.pop("text_unit_ids")
            if cov.text_unit_ids:
                text_unit_ids_unique = list(set(cov.text_unit_ids))
                for tu in text_unit_ids_unique:
                    query = f"""
                    {{
                        getTextUnit(func: eq(id, "{tu}")) @filter(has(text)) {{
                            uid
                        }}
                    }}
                    """
                    res = txn.query(query=query)
                    ppl = json.loads(res.json)
                    
                    if not ppl['getTextUnit']:
                        logger.warning(f"No text unit found for id: {tu}")
                        continue
                    
                    
                    rel = ppl['getTextUnit'][0]
                    text_unit_uid = rel['uid']
                    logger.info(f"Text Unit UID found: {text_unit_uid}")
                    source_ids.append({"uid": text_unit_uid})
            
            p["attributes"] = json.dumps(p["attributes"]) if p["attributes"] else None
            p["from"] = source_ids
            p["dgraph.type"] = "Covariate"
            
            mutations.append(txn.create_mutation(set_obj=p))
                
        request = txn.create_request(mutations=mutations, commit_now=True)
        response = txn.do_request(request)
        
        logger.info(f"Mutation response: {response}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Clean up the transaction
        txn.discard()


def ingest_community_report(client: pydgraph.DgraphClient, community_reports: List[CommunityReport]):
    txn = client.txn()
    try:
        mutations = []
        for report in community_reports:
            p = {**report.dict(), "dgraph.type": "CommunityReport"}
            p["attributes"] = json.dumps(p["attributes"]) if p["attributes"] else None
            mutations.append(txn.create_mutation(set_obj={**p, "dgraph.type": "CommunityReport"}))
        
        request = txn.create_request(mutations=mutations, commit_now=True)
        response = txn.do_request(request)
        
        logger.info(f"Mutation response: {response}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        txn.discard()
    
    
if __name__ == "__main__":
    client_stub = create_client_stub()
    client = create_client(client_stub)
    
    set_schema(client)
    
    # TextUnit ----:
    text_unit_df = pd.read_csv(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}")
    text_units = read_indexer_text_units(text_unit_df)

    injest_text_units(client, text_units)
    
    
    # Community Report
    entity_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_TABLE}")
    report_df = None
    file_path = f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}"
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        report_df = pd.DataFrame()
    else:
        report_df = pd.read_csv(file_path)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    ingest_community_report(client, reports)
    
    
    # Covariate ----:
    covariate_df = pd.read_csv(f"{INPUT_DIR}/{COVARIATE_TABLE}")
    covariates = read_indexer_covariates(covariate_df)
    
    
    query_and_ingest_covariates(client, covariates)
    
    
    # Entity ----:
    entity_embedding_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}")

    entity_embedding_df["description"] = entity_embedding_df["description"].fillna("")
    entity_embedding_df["text_unit_ids"] = entity_embedding_df["text_unit_ids"].apply(lambda x: x.split(','))
    # entity_embedding_df["description_embedding"] = entity_embedding_df["description"].apply(lambda desc: embeddings.embed_query(desc))

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    query_and_ingest_entity(client, entities)
    
    
    # Relationship ----:
    relationship_df = pd.read_csv(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}")
    relationship_df["text_unit_ids"] = relationship_df["text_unit_ids"].apply(lambda x: x.split(','))
    relationships = read_indexer_relationships(relationship_df)
    
    query_and_ingest_relationship(client, relationships)
    
    client_stub.close()
    
        