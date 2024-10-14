import os
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
import pandas as pd
import weaviate


from query_context.inputs.loader.indexer_adapters import read_indexer_reports
from vectorstore import store_community_semantic_embeddings

# Initialize embeddings and input directories
embedding = OpenAIEmbeddings()

INPUT_DIR = "outputs1"
COMMUNITY_REPORT_TABLE = "community_report.csv"
ENTITY_TABLE = "node.csv"
ENTITY_EMBEDDING_TABLE = "entity.csv"
RELATIONSHIP_TABLE = "relationship.csv"
COVARIATE_TABLE = "claims.csv"
TEXT_UNIT_TABLE = "text_unit.csv"
TABLE_PATH = "/home/hungquan/build_kg/lancedb_store_weaviate"
TABLE_NAME = "multimodal_test"
COMMUNITY_LEVEL = 2

# Load CSV files
entity_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_TABLE}")
report_df = None
file_path = f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}"
if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
    report_df = pd.DataFrame()
else:
    report_df = pd.read_csv(file_path)

# Read the reports
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

client = weaviate.connect_to_local()


# Index name and text key
index_name = "WeaviateText"
text_key = "contents"

client.collections.delete_all()

# Initialize Weaviate VectorStore using gRPC client
db = WeaviateVectorStore(client=client, index_name=index_name, text_key=text_key, embedding=embedding)

# Add new data
db = store_community_semantic_embeddings(reports, db)

# Test similarity search
print(db.similarity_search("Coca-Cola Beverages Vietnam and Service Outlets"))

# Close the client connection after usage
client.close()
