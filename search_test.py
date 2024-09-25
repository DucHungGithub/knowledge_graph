import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_openai import ChatOpenAI
import lancedb
import pydgraph

from query_context.structured_search.local_search.combine_context import LocalSearchMixedContext
from query_context.structured_search.local_search.search import LocalSearch
from query_context.system_prompt_builder.entity_extraction import EntityVectorStoreKey
from vectorstore import store_entity_semantic_embeddings
from query_context.inputs.loader.indexer_adapters import read_indexer_covariates, read_indexer_entities, read_indexer_relationships, read_indexer_reports, read_indexer_text_units

# Create a client
def create_client_stub() -> pydgraph.DgraphClientStub:
    return pydgraph.DgraphClientStub('localhost:9080')


def create_client(client_stub: pydgraph.DgraphClientStub) -> pydgraph.DgraphClient:
    return pydgraph.DgraphClient(client_stub)

stub = create_client_stub()
client = create_client(stub)

# Embedding ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


TABLE_PATH = "/home/hungquan/build_kg/lancedb_store"
TABLE_NAME = "multimodal_test"

# VectorStore -----:
connection = lancedb.connect("/home/hungquan/build_kg/lancedb_store")
db = None
if TABLE_NAME not in connection.table_names():
    db = LanceDB(table_name=TABLE_NAME,embedding=embeddings, uri=TABLE_PATH)
    # db = store_entity_semantic_embeddings(entities=entities, vectorstore=db)
else:
    db = LanceDB(connection=connection, embedding=embeddings, table_name=TABLE_NAME)


# Create local search context builder
context_builder = LocalSearchMixedContext(
    entity_text_embeddings=db,
)



# Local context params:
local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.25,
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


result = search_engine.search(client=client, query="facility: www.gutenberg.org.")
print(result)