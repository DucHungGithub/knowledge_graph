from langchain.vectorstores import VectorStore
from langchain_core.documents import Document

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