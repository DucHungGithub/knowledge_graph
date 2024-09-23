# Folder path configs
INPUT_FOLDER_PATH = "inputs"

OUTPUT_FOLDER_PATH = "outputs"

CHECKPOINT_DIR = "graph_save"


# encoding model config
ENCODING_MODEL = "cl100k_base"

MAX_MESSAGES = 5


# Text Unit config

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 100

ENTITY_EXTRACTION_MAX_GLEANINGS = 1


# Models configs
MODEL_CONFIG = "gpt-4o-mini"

TEMPERATURE = 0.3

CLAIM_MAX_GLEANINGS = 1

MAX_TOKENS = 1500



# Type strategy config
ENTITY_EXTRACT_TYPE = "graph_intelligence"

SUMMARIZE_ENTITY_RELATIONSHIP = "graph_intelligence"

GENERATE_REPORT_TYPE = "graph_intelligence"

COMMUNITY_REPORT_TYPE = "graph_intelligence"



# Community config
LEVEL = 0
SEED = 6969
COMMUNITY_REPORT_MAX_INPUT_LENGTH = 8000


# Optional
EXTRACT_COVARIATES = False

CLAIM_DESCRIPTION = "Any claims or facts that could be relevant to information discovery."


# Weaviate config
HTTP_HOST = ""
HTTP_PORT = 80
HTTP_SECURE = False
GRPC_HOST = ""
GRPC_PORT = 80
GRPC_SECURE = False
AUTH_CREDENTIALS = ""
INDEX_NAME = ""
TEXT_KEY = ""
EMBEDDING_MODEL = ""
