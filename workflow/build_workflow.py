import logging
from typing import Dict, Any

from utils.graph import load_graph
from verbs.community_summarization.export import export_final_files
from verbs.documents import run_load_documents
from verbs.text_units import run_split_documents_text_units
from verbs.entities import entity_extract, summarize_description
from verbs.covariates import extract_covariates
from verbs.community_detection import clustering_graph
from verbs.community_summarization.summarization import create_community_reports

import networkx as nx
import matplotlib.pyplot as plt

import colorlog

# Set up basic logging configuration with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

async def init_workflow(args: Dict[str, Any]):
    
    # Load documents from folder
    documents_list = await run_load_documents(args=args)
    logger.info("####documents_list#####")
    logger.info(documents_list)
    
    
    # Split documents to text units
    text_units = await run_split_documents_text_units(docs=documents_list, args=args)
    logger.info("####text_units#####")
    logger.info(text_units)
    
    # Extract claims (covariates)
    claims = await extract_covariates(texts=text_units)
    logger.info("####covariates#####")
    logger.info(claims)
    
    
    # Extract entities and relationships
    ens_rels = await entity_extract(docs=text_units, args=args)
    logger.info("####ENS_RELS#####")
    logger.info(ens_rels)
    
    
    # Summarize description for entities and relationships
    ens_rels_summarize_graph = await summarize_description(entity_result=ens_rels, strategy=args)
    logger.info("####ens_rels_summarize_graph#####")
    logger.info(ens_rels_summarize_graph)
    graph_check = load_graph(ens_rels_summarize_graph)
    logger.info("Community Nodes After ens_rels_summarize_graph: --------")
    logger.info(graph_check.nodes(data=True))
    logger.info("Community Edges After ens_rels_summarize_graph: --------")
    logger.info(graph_check.edges(data=True))
    
    
    # Community detection (Clustering graph)
    communities_graph = await clustering_graph(graphml=ens_rels_summarize_graph, args=args)
    logger.info("####communities_graph#####")
    logger.info(communities_graph)
    nx.draw(communities_graph, with_labels = True)
    plt.savefig("communities_graph.png")
    
    logger.info("Community Nodes After Summarize: --------")
    logger.info(communities_graph.nodes(data=True))
    logger.info("Community Edges After Summarize: --------")
    logger.info(communities_graph.edges(data=True))
    
    # # Generate report for communities
    logger.info("####community_reports#####")
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
    