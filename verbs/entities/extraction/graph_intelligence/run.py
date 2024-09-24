from typing import Any, Dict, List

import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import TokenTextSplitter

from graph.extractors.graph_extractor import GraphExtractor
from llm import load_openai_llm
import config as defs
from verbs.entities.extraction.typing import EntityExtractionResult

    
    
async def run_gi(
    docs: List[Document],
    entity_types: List[str],
    args: Dict[str, Any]
) -> EntityExtractionResult:
    """Run the graph entity extraction"""
    model_config = args.get("llm", defs.MODEL_CONFIG)
    temperature = args.get("temperature", defs.TEMPERATURE)
    llm = load_openai_llm(model=model_config, temperature=temperature)
    return await run_extract_entities(llm=llm, docs=docs, entity_types=entity_types, args=args)


async def run_extract_entities(
    llm: BaseChatModel,
    docs: List[Document],
    entity_types: List[str],
    args: Dict[str, Any]
) -> EntityExtractionResult:
    import config as defs
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    

    # Chunking Arguments
    prechunked = args.get("prechunked", False)
    chunk_size = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)
    
    # Extraction Arguments
    tuple_delimiter = args.get("tuple_delimiter", None)
    record_delimiter = args.get("record_delimiter", None)
    completion_delimiter = args.get("completion_delimiter", None)
    extraction_prompt = args.get("extraction_prompt", None)
    encoding_model = args.get("encoding_name", None)
    max_gleanings = args.get("max_gleanings", defs.ENTITY_EXTRACTION_MAX_GLEANINGS)
    
    entity_extractor = GraphExtractor(
        llm=llm,
        extraction_prompt=extraction_prompt,
        encoding_model=encoding_model,
        max_gleanings=max_gleanings,
    )
    
    text_list = [doc.page_content.strip() for doc in docs]
    
    
    # splitter = TokenTextSplitter(
    #     encoding_name=encoding_name,
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    # )
    
    # if not prechunked:
    #     text_list =  splitter.split_text("\n".join(text_list))
        
    results = await entity_extractor.invoke(
        text_list,
        {
            "entity_types": entity_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter
        }
    )
    
    graph = results.output
    # Map the "source_id" back to the "id" field
    
    print("----------GRAPH NODEs--------")
    print(graph.nodes(data=True))
    
    
    for _, node in graph.nodes(data=True):
        if node is not None:
            node["source_id"] = ",".join(
                docs[int(id)].id for id in str(node["source_id"]).split(",")
            )
            
    for _, _, edge in graph.edges(data=True):
        if edge is not None:
            edge["source_id"] = ",".join(
                docs[int(id)].id for id in str(edge["source_id"]).split(",")
            )
    
    entities = [
        ({"name": item[0], **(item[1] or {})}) for item in graph.nodes(data=True)
        if item is not None
    ]
    
    rels = [
        ({"source": item[0], "target": item[1], **(item[2] or {})}) for item in graph.edges(data=True)
        if item is not None
    ]
    
    graph_data = "".join(nx.generate_graphml(graph))
    return EntityExtractionResult(entities=entities, rels=rels, graphml_graph=graph_data)
        
        
        
    
    