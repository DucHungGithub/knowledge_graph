import asyncio
from typing import Annotated, List, Literal, TypedDict
import pandas as pd
import logging
import os

import colorlog
import lancedb
import pydgraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import pydgraph.client_stub

from langgraph_group.prompts import QUESTION_SYSTEM_PROMPT, REWRITE_PROMPT
from langgraph_group.output_type import OutputResponse, Queries, RewriteResponse
from langchain_community.vectorstores import LanceDB
from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage



from langgraph_group.utils import reduce_fanouts_wrapper
from models.community_report import CommunityReport
from models.covariate import Covariate
from models.entity import Entity

from models.relationship import Relationship
from models.text_unit import TextUnit
from query_context.inputs.loader.indexer_adapters import read_indexer_entities
from query_context.inputs.retrieval.community_reports import get_candidate_community_reports
from query_context.inputs.retrieval.covariates import get_candidate_covariates
from query_context.inputs.retrieval.entities import get_entity_by_key
from query_context.inputs.retrieval.relationships import get_candidate_relationships
from query_context.inputs.retrieval.text_units import get_candidate_textUnit
from query_context.system_prompt_builder.history.conversation_history import ConversationHistory
from query_context.system_prompt_builder.history.typing import ConversationRole
from query_context.system_prompt_builder.process_context.community_context import build_community_context
from query_context.system_prompt_builder.process_context.covariate_context import build_claims_context
from query_context.system_prompt_builder.process_context.entity_context import build_entity_context
from query_context.system_prompt_builder.process_context.relationship_context import build_relationships_context
from query_context.system_prompt_builder.process_context.textunit_context import build_text_unit_context
from vectorstore import store_entity_semantic_embeddings




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



client_stub = pydgraph.DgraphClientStub()
client = pydgraph.DgraphClient(client_stub)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


# TABLE_PATH = "/home/hungquan/build_kg/lancedb_store"
# TABLE_NAME = "multimodal_test"
INPUT_DIR = "outputs1"
COMMUNITY_REPORT_TABLE = "community_report.csv"
ENTITY_TABLE = "node.csv"
ENTITY_EMBEDDING_TABLE = "entity.csv"
RELATIONSHIP_TABLE = "relationship.csv"
COVARIATE_TABLE = "claims.csv"
TEXT_UNIT_TABLE = "text_unit.csv"
TABLE_PATH = "/home/hungquan/build_kg/lancedb_store_entity"
TABLE_NAME = "multimodal_test"
COMMUNITY_LEVEL = 2

# VectorStore -----:
entity_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_TABLE}")
entity_embedding_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}")

entity_embedding_df["description"] = entity_embedding_df["description"].fillna("")
entity_embedding_df["text_unit_ids"] = entity_embedding_df["text_unit_ids"].apply(lambda x: x.split(','))
# entity_embedding_df["description_embedding"] = entity_embedding_df["description"].apply(lambda desc: embeddings.embed_query(desc))

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
connection = lancedb.connect(TABLE_PATH)
db = None
if TABLE_NAME not in connection.table_names():
    db = LanceDB(table_name=TABLE_NAME,embedding=embeddings, uri=TABLE_PATH)
    db = store_entity_semantic_embeddings(entities=entities, vectorstore=db)
else:
    db = LanceDB(connection=connection, embedding=embeddings, table_name=TABLE_NAME)



async def retrieve_related_entities(query: str) -> List[Entity]:
    search_results = db.similarity_search(query=query, k=10)
    matched_entities = []
    
    for result in search_results:
        matched = get_entity_by_key(
            client=client,
            key="id",
            value=str(result.metadata["id"])
        )
        if matched and matched not in matched_entities:
            matched_entities.append(matched)
    
    return matched_entities


async def search_queries(queries: List[str]) -> List[List[Entity]]:
    tasks = [retrieve_related_entities(w) for w in queries]
    results = await asyncio.gather(*tasks)
    return results
            
    
            
class LocalSearchWorkFlowState(TypedDict):
    max_retry: int
    conversation_history: ConversationHistory
    check_add_user_query: bool
    messages: Annotated[List[BaseMessage], reduce_fanouts_wrapper]
    search_queries: Annotated[List[str], reduce_fanouts_wrapper] = []
    search_entities: Annotated[List[Entity], reduce_fanouts_wrapper] = []
    search_rels: Annotated[List[Relationship], reduce_fanouts_wrapper] = []
    search_covariates: Annotated[List[Covariate], reduce_fanouts_wrapper] = []
    search_textUnit: Annotated[List[TextUnit], reduce_fanouts_wrapper] = []
    search_communityReport: Annotated[List[CommunityReport], reduce_fanouts_wrapper] = []
    total_context: Annotated[List[str], reduce_fanouts_wrapper] = []
    check_extract_query: Annotated[List[str], reduce_fanouts_wrapper] = []
    
    
    
    @classmethod
    def generate_history_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------generate_history_agent")
        messages = state['messages'][-1]
        entities = state["search_entities"]
        check_add = state["check_add_user_query"]
        
        conversation_history = ConversationHistory.load_histories(client=client, entities=entities)
        
        if check_add:
            conversation_history.add_turn(role=ConversationRole.from_string("user"), content=messages.content)
        
        return {"conversation_history": conversation_history, "check_add_user_query": False} 
    
    
    @classmethod
    def generate_question_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------generate_question_agent")
        messages = state['messages']
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    QUESTION_SYSTEM_PROMPT,
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        
        chain = prompt | model.with_structured_output(Queries)
        results = chain.invoke(messages)
        
        if "search_queries" not in state:
            state["search_queries"] = []
            
        return {"search_queries": results.queries}
    
    
    @classmethod
    def search_entities_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------search_entities_agent")
        queries = state["search_queries"]
        check_extract_query = state["check_extract_query"]
        
        queries = [query for query in queries if query not in check_extract_query]
        
        results = asyncio.run(search_queries(queries))
        entities_list = []
        for entities in results:
            entities_list.extend(entities)
        
        return {"search_entities": entities_list, "check_extract_query": queries}
    
    
    @classmethod
    def search_relationships_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------search_relationships_agent")
        entities = state["search_entities"]
        rels = get_candidate_relationships(client, entities)
        
        if "search_rels" not in state:
            state["search_rels"] = []
        
        return {"search_rels": rels}
    
    @classmethod
    def search_covariates_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------search_covariates_agent")
        entities = state["search_entities"]
        covariates = get_candidate_covariates(client, entities)
        
        if "search_covariates" not in state:
            state["search_covariates"] = []
        
        return {"search_covariates": covariates}
    
    @classmethod
    def search_communities_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------search_communities_agent")
        entities = state["search_entities"]
        community_reports = get_candidate_community_reports(client, entities)
        if "search_communityReport" not in state:
            state["search_communityReport"] = []
        return {"search_communityReport": community_reports}
    
    @classmethod
    def search_text_unit_agent(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------search_text_unit_agent")
        entities = state["search_entities"]
        text_units = get_candidate_textUnit(client, entities)
        if "search_textUnit" not in state:
            state["search_textUnit"] = []
        
        return {"search_textUnit": text_units}
    
    @classmethod
    def check_answers(cls, state: "LocalSearchWorkFlowState") -> Literal[END, "REWRITE_QUERY"]: # type: ignore
        context = state["total_context"]
        question = state["messages"][-1]
        max_retry = state.get("max_retry", 3) 
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                REWRITE_PROMPT
            ),
        ])
        
        input_prompt = prompt.format(context=context, question=question)
        new_model = model.with_structured_output(OutputResponse)
        response = new_model.invoke(input_prompt)
        
        logger.critical(response.is_enough)
        logger.warning(response.reason_why)
        
        
        if str(response.is_enough).lower() == "yes" or max_retry <= 1:
            return END
        
        return "REWRITE_QUERY"
    
    @classmethod
    def provide_query(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        context = state["total_context"]
        question = state["messages"][-1]
        max_retry = state["max_retry"]
        search_queries = state["search_queries"]
        
        if search_queries is None:
            search_queries = []
            
        if question.content not in search_queries:
            search_queries.append(question.content)
        questions = "\n".join(search_queries)
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                REWRITE_PROMPT
            ),
        ])
        
        input_prompt = prompt.format(context=context, question=questions)
        new_model = model.with_structured_output(RewriteResponse)
        response = new_model.invoke(input_prompt)
        
        
        logger.info(f"response.new_query: {response.new_query}")
        logger.info(f"response.reason_why: {response.reason_why}")
        
        return {"search_queries": response.new_query, "max_retry": max_retry - 1}
        
    @classmethod
    def context_combine(cls, state: "LocalSearchWorkFlowState") -> "LocalSearchWorkFlowState":
        print("-------------context_combine")
        entities = state["search_entities"]
        relationships = state["search_rels"]
        covariates = state["search_covariates"]
        text_units = state["search_textUnit"]
        community_reports = state["search_communityReport"]
        histories = state["conversation_history"]
        
        if entities is None:
            entities = []
        if relationships is None:
            relationships = []
        if covariates is None: 
            covariates = []
        if text_units is None:
            text_units = []
        if community_reports is None:
            community_reports = []
            
            
        history_context, _ = histories.build_context(include_user_turns_only=False, max_tokens=1500)
        text_unit_context, _ = build_text_unit_context(text_units, max_tokens=3500)
        community_context, _ = build_community_context(community_reports, entities, max_tokens=1000)
        claim_context, _ = build_claims_context(covariates, max_tokens=1000)
        entity_context, _ = build_entity_context(entities, max_tokens=1500)
        relationship_context, _ = build_relationships_context(relationships, max_tokens=1500)
        
        
        context_community = ""
        if isinstance(community_context, list):
            context_community = "\n\n".join(community_context)
        else:
            context_community = str(community_context)
        
        
        total_context = history_context + "\n\n" + text_unit_context + "\n\n" + context_community + "\n\n" + claim_context + "\n\n" + entity_context + "\n\n" + relationship_context
        
        return {"total_context": total_context}
    
        
local_context_workflow = StateGraph(LocalSearchWorkFlowState)

local_context_workflow.add_node("QUESTION_GENERATION", LocalSearchWorkFlowState.generate_question_agent)
local_context_workflow.add_node("ENTITY_EXTRACT", LocalSearchWorkFlowState.search_entities_agent)
local_context_workflow.add_node("HISTORY_EXTRACT", LocalSearchWorkFlowState.generate_history_agent)
local_context_workflow.add_node("RELATIONSHIP_EXTRACT", LocalSearchWorkFlowState.search_relationships_agent)
local_context_workflow.add_node("COVARIATE_EXTRACT", LocalSearchWorkFlowState.search_covariates_agent)
local_context_workflow.add_node("COMMUNITY_REPORT_EXTRACT", LocalSearchWorkFlowState.search_communities_agent)
local_context_workflow.add_node("TEXTUNIT_EXTRACT", LocalSearchWorkFlowState.search_text_unit_agent)
local_context_workflow.add_node("CONTEXT_COMBINE", LocalSearchWorkFlowState.context_combine)
local_context_workflow.add_node("REWRITE_QUERY", LocalSearchWorkFlowState.provide_query)

local_context_workflow.add_conditional_edges("CONTEXT_COMBINE", LocalSearchWorkFlowState.check_answers)


local_context_workflow.set_entry_point("QUESTION_GENERATION")

local_context_workflow.add_edge("QUESTION_GENERATION", "ENTITY_EXTRACT")
local_context_workflow.add_edge("ENTITY_EXTRACT", "RELATIONSHIP_EXTRACT")
local_context_workflow.add_edge("ENTITY_EXTRACT", "HISTORY_EXTRACT")
local_context_workflow.add_edge("ENTITY_EXTRACT", "COVARIATE_EXTRACT")
local_context_workflow.add_edge("ENTITY_EXTRACT", "COMMUNITY_REPORT_EXTRACT")
local_context_workflow.add_edge("ENTITY_EXTRACT", "TEXTUNIT_EXTRACT")
local_context_workflow.add_edge("REWRITE_QUERY", "ENTITY_EXTRACT")


local_context_workflow.add_edge(["HISTORY_EXTRACT", "RELATIONSHIP_EXTRACT", "COVARIATE_EXTRACT", "COMMUNITY_REPORT_EXTRACT", "TEXTUNIT_EXTRACT"], "CONTEXT_COMBINE")

graph_context = local_context_workflow.compile()


png_graph = graph_context.get_graph().draw_mermaid_png()
with open("my_graph_new_local.png", "wb") as f:
    f.write(png_graph)

print(f"Graph saved as 'my_graph_new.png' in {os.getcwd()}")


text = graph_context.invoke({"messages": [HumanMessage(content="CORE BRONZE?")], "max_retry": 3, "check_add_user_query": True})

print(text["total_context"][0])
print(text["search_queries"])


