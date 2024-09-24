from langchain_openai import OpenAIEmbeddings
import weaviate
from weaviate.auth import AuthApiKey

from langchain_weaviate import WeaviateVectorStore


import config as defs

async def ingest_vectorstore() -> WeaviateVectorStore:    
    try:
        weaviate_client = weaviate.connect_to_custom(
            http_host=defs.HTTP_HOST,
            http_port=defs.HTTP_PORT,
            http_secure=defs.HTTP_SECURE,
            grpc_host=defs.GRPC_HOST,
            grpc_port=defs.GRPC_PORT,
            grpc_secure=defs.GRPC_SECURE,
            auth_credentials=AuthApiKey(defs.AUTH_CREDENTIALS)
        )
        
        vector_store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=defs.INDEX_NAME,
            text_key=defs.TEXT_KEY,
            embedding=OpenAIEmbeddings(model=defs.EMBEDDING_MODEL)
        )
        return vector_store
    except Exception as e:
        raise ConnectionError("Can not connect to weaviate: ", e)