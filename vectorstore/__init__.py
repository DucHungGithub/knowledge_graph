from typing import List
from langchain.vectorstores import VectorStore
from langchain_core.documents import Document

from models.community_report import CommunityReport
from models.entity import Entity

def store_entity_semantic_embeddings(
    entities: list[Entity],
    vectorstore: VectorStore,
) -> VectorStore:
    """Store entity semantic embeddings in a vectorstore."""

    documents = []
    
    for entity in entities:
        doc = Document(
            id=entity.id,
            page_content=entity.description if entity.description else "No Description",
            metadata={
                "id": entity.id,
                "title": entity.title,
                **(entity.attributes if entity.attributes else {}),
            }
        )
        
        documents.append(doc)
    
    print(f"Documents to be added: {documents}")
    vectorstore.add_documents(documents=documents)
    print(f"Documents to be added finished")

    return vectorstore



def store_community_semantic_embeddings(
    coms: List[CommunityReport],
    vectorstore: VectorStore,
) -> VectorStore:
    """Store entity semantic embeddings in a vectorstore."""

    documents = []
    
    for com in coms:
        doc = Document(
            id=com.id,
            page_content=com.summary if com.summary else "No Description",
            metadata={
                "ids": com.id,
                "title": com.title,
                **(com.attributes if com.attributes else {}),
            }
        )
        
        documents.append(doc)
    
    print(f"Documents to be added: {documents}")
    vectorstore.add_documents(documents=documents)
    print(f"Documents to be added finished")

    return vectorstore