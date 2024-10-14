import asyncio
from typing import Annotated, List, Literal, Tuple, TypedDict
import pandas as pd 
import logging
import os


import colorlog
import lancedb
import pydgraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import LanceDB


from langgraph_group.output_type import OutputResponse, Queries, RewriteResponse
from langgraph_group.prompts import QUESTION_SYSTEM_PROMPT, REWRITE_PROMPT
from langgraph_group.utils import reduce_fanouts_wrapper, retrieve_relevant_community_reports
from langchain_openai import OpenAIEmbeddings

from models.community_report import CommunityReport
from models.covariate import Covariate
from models.entity import Entity
from models.relationship import Relationship
from models.text_unit import TextUnit
from query_context.inputs.loader.indexer_adapters import read_indexer_entities, read_indexer_reports
from query_context.inputs.retrieval.covariates import get_candidate_covariates
from query_context.inputs.retrieval.relationships import get_candidate_relationships
from query_context.inputs.retrieval.text_units import get_candidate_textUnit
from query_context.system_prompt_builder.history.conversation_history import ConversationHistory
from query_context.system_prompt_builder.history.typing import ConversationRole
from query_context.system_prompt_builder.process_context.community_context import build_community_context
from query_context.system_prompt_builder.process_context.covariate_context import build_claims_context
from query_context.system_prompt_builder.process_context.entity_context import build_entity_context
from query_context.system_prompt_builder.process_context.relationship_context import build_relationships_context
from query_context.system_prompt_builder.process_context.textunit_context import build_text_unit_context
from vectorstore import store_community_semantic_embeddings


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
TABLE_PATH = "/home/hungquan/build_kg/lancedb_store_community"
TABLE_NAME = "multimodal_test"
COMMUNITY_LEVEL = 2

entity_df = pd.read_csv(f"{INPUT_DIR}/{ENTITY_TABLE}")
report_df = None
file_path = f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}"

if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
    report_df = pd.DataFrame()
else:
    report_df = pd.read_csv(file_path)
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

connection = lancedb.connect(TABLE_PATH)
db = None
if TABLE_NAME not in connection.table_names():
    db = LanceDB(table_name=TABLE_NAME,embedding=embeddings, uri=TABLE_PATH)
    db = store_community_semantic_embeddings(coms=reports, vectorstore=db)
else:
    db = LanceDB(connection=connection, embedding=embeddings, table_name=TABLE_NAME)


async def retrieve_related_community_report(query: str) -> Tuple[List[CommunityReport], List[Entity]]:
    communities, entities = retrieve_relevant_community_reports(db=db, client=client, query=query)
    return communities, entities
    
    

async def search_queries(queries: List[str]) -> List[Tuple[List[CommunityReport], List[Entity]]]:
    tasks = [retrieve_related_community_report(w) for w in queries]
    results = await asyncio.gather(*tasks)
    return results
    

class GlobalSearchWorkFlowState(TypedDict):
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
    def generate_history_agent(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
        print("-------------generate_history_agent")
        messages = state['messages'][-1]
        entities = state["search_entities"]
        check_add = state["check_add_user_query"]
        
        conversation_history = ConversationHistory.load_histories(client=client, entities=entities)
        
        if check_add:
            conversation_history.add_turn(role=ConversationRole.from_string("user"), content=messages.content)
        
        return {"conversation_history": conversation_history, "check_add_user_query": False} 
    
        
        
    @classmethod
    def generate_question_agent(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
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
    def search_community_agent(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
        print("-------------search_community_agent")
        queries = state['search_queries']
        check_extract_query = state["check_extract_query"]
        
        queries = [query for query in queries if query not in check_extract_query]

        results = asyncio.run(search_queries(queries))
        
        community_list = []
        entity_list = []
        for com in results:
            new_communities = [c for c in com[0] if c.id not in {existing.id for existing in community_list}]
            community_list.extend(new_communities)
            
            new_entities = [e for e in com[1] if e.id not in {existing.id for existing in entity_list}]
            entity_list.extend(new_entities)
        return {"search_communityReport": community_list, "search_entities": entity_list, "check_extract_query": queries}
    
    @classmethod
    def search_relationships_agent(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
        print("-------------search_relationships_agent")
        entities = state["search_entities"]
        rels = get_candidate_relationships(client, entities)
        
        if "search_rels" not in state:
            state["search_rels"] = []
        
        return {"search_rels": rels} 
    
    @classmethod
    def search_covariates_agent(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
        print("-------------search_covariates_agent")
        entities = state["search_entities"]
        covariates = get_candidate_covariates(client, entities)
        
        if "search_covariates" not in state:
            state["search_covariates"] = []
        
        return {"search_covariates": covariates}
    
    @classmethod
    def search_text_unit_agent(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
        print("-------------search_text_unit_agent")
        entities = state["search_entities"]
        text_units = get_candidate_textUnit(client, entities)
        if "search_textUnit" not in state:
            state["search_textUnit"] = []
        
        return {"search_textUnit": text_units}
    
    
    @classmethod
    def check_answers(cls, state: "GlobalSearchWorkFlowState") -> Literal[END, "REWRITE_QUERY"]: # type: ignore
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
        
        
        if str(response.is_enough).lower() == "yes" or max_retry <= 1:
            return END
        
        return "REWRITE_QUERY"
    
    @classmethod
    def provide_query(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
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
        
        
        return {"search_queries": response.new_query, "max_retry": max_retry - 1}
    
    @classmethod
    def context_combine(cls, state: "GlobalSearchWorkFlowState") -> "GlobalSearchWorkFlowState":
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
        text_unit_context, _ = build_text_unit_context(text_units, max_tokens=3000)
        community_context, _ = build_community_context(community_reports, entities, max_tokens=1000)
        claim_context, _ = build_claims_context(covariates, max_tokens=1500)
        entity_context, _ = build_entity_context(entities, max_tokens=1500)
        relationship_context, _ = build_relationships_context(relationships, max_tokens=1500)
        
        
        context_community = ""
        if isinstance(community_context, list):
            context_community = "\n\n".join(community_context)
        else:
            context_community = str(community_context)
        
        
        total_context = history_context + "\n\n" + text_unit_context + "\n\n" + context_community + "\n\n" + claim_context + "\n\n" + entity_context + "\n\n" + relationship_context
        
        
        return {"total_context": total_context}
    

global_context_workflow = StateGraph(GlobalSearchWorkFlowState)

global_context_workflow.add_node("QUESTION_GENERATION", GlobalSearchWorkFlowState.generate_question_agent)
global_context_workflow.add_node("HISTORY_EXTRACT", GlobalSearchWorkFlowState.generate_history_agent)
global_context_workflow.add_node("COMMUNITY_ENTITY_EXTRACT", GlobalSearchWorkFlowState.search_community_agent)
global_context_workflow.add_node("RELATIONSHIP_EXTRACT", GlobalSearchWorkFlowState.search_relationships_agent)
global_context_workflow.add_node("COVARIATE_EXTRACT", GlobalSearchWorkFlowState.search_covariates_agent)
global_context_workflow.add_node("TEXTUNIT_EXTRACT", GlobalSearchWorkFlowState.search_text_unit_agent)
global_context_workflow.add_node("CONTEXT_COMBINE", GlobalSearchWorkFlowState.context_combine)
global_context_workflow.add_node("REWRITE_QUERY", GlobalSearchWorkFlowState.provide_query)

global_context_workflow.add_conditional_edges("CONTEXT_COMBINE", GlobalSearchWorkFlowState.check_answers)

global_context_workflow.set_entry_point("QUESTION_GENERATION")

global_context_workflow.add_edge("QUESTION_GENERATION", "COMMUNITY_ENTITY_EXTRACT")
global_context_workflow.add_edge("COMMUNITY_ENTITY_EXTRACT", "HISTORY_EXTRACT")
global_context_workflow.add_edge("COMMUNITY_ENTITY_EXTRACT", "RELATIONSHIP_EXTRACT")
global_context_workflow.add_edge("COMMUNITY_ENTITY_EXTRACT", "COVARIATE_EXTRACT")
global_context_workflow.add_edge("COMMUNITY_ENTITY_EXTRACT", "TEXTUNIT_EXTRACT")


global_context_workflow.add_edge("REWRITE_QUERY", "COMMUNITY_ENTITY_EXTRACT")

global_context_workflow.add_edge(["HISTORY_EXTRACT", "RELATIONSHIP_EXTRACT", "COVARIATE_EXTRACT", "TEXTUNIT_EXTRACT"], "CONTEXT_COMBINE")


graph_context = global_context_workflow.compile()


png_graph = graph_context.get_graph().draw_mermaid_png()
with open("my_graph_new_global.png", "wb") as f:
    f.write(png_graph)

print(f"Graph saved as 'my_graph_new.png' in {os.getcwd()}")


text = graph_context.invoke({"messages": [HumanMessage(content="Consumer SKU from May 2023?")], "max_retry": 3, "check_add_user_query": True})

print(text["total_context"][0])
print(text["search_queries"])

