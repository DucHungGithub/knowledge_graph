from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.graph import StateGraph

from generate_answers.prompts import supervisor_prompt, routing_prompt
from generate_answers.type import options, members, RouteResponse


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", routing_prompt)
    ]
).partial(options=str(options), members=", ".join(members))


llm = ChatOpenAI(model="gpt-4o-mini")


def supervisor_agent(state: StateGraph):
    chain = (
        prompt
        | llm.with_structured_output(RouteResponse)
    )
    
    return chain.invoke(state)

