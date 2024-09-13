from typing import Dict, Any

from utils.graph import load_graph
from verbs.community_summarization.export import export_final_files
from verbs.documents import run_load_documents
from verbs.text_units import run_split_documents_text_units
from verbs.graph_embedding import graph_embedding
from verbs.entities import entity_extract, summarize_description
from verbs.covariates import extract_covariates
from verbs.community_detection import clustering_graph
from verbs.community_summarization.summarization import create_community_reports


import asyncio
import networkx as nx
import matplotlib.pyplot as plt



async def init_workflow(args: Dict[str, Any]):
    # Load documents from folder
    documents_list = await run_load_documents(args=args)
    print("####documents_list#####")
    print(documents_list)
    
    
    # Split documents to text units
    text_units = await run_split_documents_text_units(docs=documents_list, args=args)
    print("####text_units#####")
    print(text_units)
    
    
    # Extract entities and relationships
    ens_rels = await entity_extract(docs=text_units, args=args)
    print("####ENS_RELS#####")
    print(ens_rels)
    
    # Extract claims (covariates)
    text_list = [tu.page_content for tu in text_units]
    claims = await extract_covariates(texts=text_list)
    print("####covariates#####")
    print(claims)
    
    # Summarize description for entities and relationships
    ens_rels_summarize_graph = await summarize_description(entity_result=ens_rels, strategy=args)
    print("####ens_rels_summarize_graph#####")
    print(ens_rels_summarize_graph)
    graph_check = load_graph(ens_rels_summarize_graph)
    print("Community Nodes After ens_rels_summarize_graph: --------")
    print(graph_check.nodes(data=True))
    print("Community Edges After ens_rels_summarize_graph: --------")
    print(graph_check.edges(data=True))
    
    
    # Community detection (Clustering graph)
    communities_graph = await clustering_graph(graphml=ens_rels_summarize_graph, args=args)
    print("####communities_graph#####")
    print(communities_graph)
    nx.draw(communities_graph, with_labels = True)
    plt.savefig("communities_graph.png")
    
    print("Community Nodes After Summarize: --------")
    print(communities_graph.nodes(data=True))
    print("Community Edges After Summarize: --------")
    print(communities_graph.edges(data=True))
    
    # # Generate report for communities
    print("####community_reports#####")
    community_reports = await create_community_reports(
        entities=communities_graph.nodes(data=True),
        rels=communities_graph.edges(data=True),
        claims=claims,
        args=args
    )
    
    
    await export_final_files(
        text_units=text_units,
        entities=communities_graph.nodes(data=True),
        rels=communities_graph.edges(data=True),
        claims=claims,
        community_report=community_reports
    )
    