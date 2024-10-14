# Knowledge Graph Project

This project is inspired by the GraphRAG model from the paper ["Retrieval-Augmented Graph Neural Networks"](https://arxiv.org/pdf/2404.16130). It integrates **LangChain** with **Dgraph** to create an advanced knowledge graph system that supports efficient entity extraction, knowledge representation, and query processing.

## Features

1. **LangChain Integration**:
   - LangChain's language model framework is used to perform entity extraction, relation discovery, and question answering over the knowledge graph.
   - The framework supports various models and workflows for handling text data and natural language queries.

2. **Dgraph Backend**:
   - Dgraph is utilized as the backend graph database to store and manage entities, relationships, and semantic embeddings.
   - Enables graph querying, relationship extraction, and structured data retrieval.

3. **Entity and Relation Extraction**:
   - Text data is processed to extract entities and their relationships, which are dynamically stored in the knowledge graph.
   - Supports various types of documents, leveraging the capabilities of LangChain for document processing and splitting.

4. **Graph-Based Retrieval**:
   - Retrieval is augmented through both knowledge graph queries and text embeddings, enabling complex multi-hop reasoning.
   - Combines retrieval-augmented generation (RAG) with graph-based relationships to enhance knowledge discovery.

5. **Custom Querying System**:
   - The knowledge graph allows for structured querying over entities, relations, and their properties, providing powerful search and analytics capabilities.
   - Integrates with the LangChain prompt templates for easy natural language querying.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd knowledge_graph
   ```

2. **Install Dependencies**:
   Use the provided requirements file to install Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Dgraph Setup**:
   Follow the instructions from the official Dgraph documentation to set up and run your Dgraph instance. Ensure it is properly configured to interface with the LangChain setup.


## Usage

1. **Entity Extraction**:
   - Use the LangChain text processing pipeline to extract entities from a document.


2. **Storing Entities in Dgraph**:
   - Add extracted entities and relationships to the Dgraph backend.


3. **Querying the Knowledge Graph**:
   - Perform structured queries to retrieve entities and their relationships.
   

4. **Natural Language Queries**:
   - Leverage LangChain's question-answering module to query the graph using natural language.


## Contributions

Feel free to fork the repository, make pull requests, or submit issues for any feature requests or bugs you encounter.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

### References

- [Retrieval-Augmented Graph Neural Networks](https://arxiv.org/pdf/2404.16130)
- [LangChain Documentation](https://langchain.com/docs)
- [Dgraph Documentation](https://dgraph.io/docs)

