from langgraph.graph import StateGraph, END, START


from query_context.structured_search.llm_search.type import AgentState
from query_context.structured_search.llm_search.supervisor import supervisor_agent
from query_context.structured_search.llm_search.type import members


workflow = StateGraph(AgentState)


workflow.add_node("Supervisor", supervisor_agent)
workflow.add_node("EntityExtract",)
workflow.add_node("RelationshipExtract",)

for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in member}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)

workflow.add_edge(START, "Supervisor")

graph = workflow.compile()
    