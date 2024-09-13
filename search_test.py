import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_openai import ChatOpenAI
import lancedb

from query.structured_search.local_search.combine_context import LocalSearchMixedContext
from query.structured_search.local_search.search import LocalSearch
from query.system_prompt_builder.entity_extraction import EntityVectorStoreKey
from vectorstore import store_entity_semantic_embeddings
from query.inputs.loader.indexer_adapters import read_indexer_covariates, read_indexer_entities, read_indexer_relationships, read_indexer_reports, read_indexer_text_units


# Embedding ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

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




# Entity ----:
entity_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_TABLE}")
entity_embedding_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}")

entity_embedding_df["description"] = entity_embedding_df["description"].fillna("")
entity_embedding_df["text_unit_ids"] = entity_embedding_df["text_unit_ids"].apply(lambda x: x.split(','))
# entity_embedding_df["description_embedding"] = entity_embedding_df["description"].apply(lambda desc: embeddings.embed_query(desc))

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

print("check----------")
print(entities)

# VectorStore -----:
connection = lancedb.connect("/home/hungquan/build_kg/lancedb_store")
db = None
if TABLE_NAME not in connection.table_names():
    db = LanceDB(table_name=TABLE_NAME,embedding=embeddings, uri=TABLE_PATH)
    db = store_entity_semantic_embeddings(entities=entities, vectorstore=db)
else:
    db = LanceDB(connection=connection, embedding=embeddings, table_name=TABLE_NAME)



# Relationship ----:
relationship_df = pd.read_csv(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}")
relationship_df["text_unit_ids"] = relationship_df["text_unit_ids"].apply(lambda x: x.split(','))
relationships = read_indexer_relationships(relationship_df)

print(f"Relationship count: {len(relationship_df)}")



# Covariate ----:
covariate_df = pd.read_csv(f"{INPUT_DIR}/{COVARIATE_TABLE}")

claims = read_indexer_covariates(covariate_df)

covariates = {"claims": claims}


# Community Report ----:
file_path = f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}"
if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
    report_df = pd.DataFrame()
else:
    report_df = pd.read_csv(file_path)

reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)




# Text Unit ----:
text_unit_df = pd.read_csv(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}")
text_units = read_indexer_text_units(text_unit_df)



# Create local search context builder
context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=covariates, # If not use, set this to None
    entity_text_embeddings=db,
    text_embedder=embeddings,
)



# Local context params:
local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
    "max_tokens": 5000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
}


# LLM config params
llm_params = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
    "temperature": 0.0,
}

llm = ChatOpenAI(model="gpt-4o-mini", **llm_params)

token_encoder = None


# search engine:-----
search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
)


result = search_engine.search("facility: www.gutenberg.org.")
print(result)