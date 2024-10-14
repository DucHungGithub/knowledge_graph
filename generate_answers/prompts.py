

supervisor_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

routing_prompt = (
    "Given the conversation above, who should act next?"
    " Or should we FINISH? Select one of: {options}"
)

rewrite_prompt = """
You are an expert in analyzing context and questions. Your task is to determine whether the current context has enough information, keywords, and data necessary to answer the question. If the context is sufficient to provide an answer, respond with "yes." If the context is insufficient, respond with "no" and rewrite the query to specify what additional information is needed to accurately answer the question.

**Context:** {context}

**User Question:** {question}


1. Evaluate the current context:
   - Are the relevant keywords present?
   - Is there enough data to answer the question?
   - Have all critical information points been covered?

2. If sufficient, respond with "yes"
3. If not sufficient, respond with "no" and suggest a specific query to gather the additional data required.

**Output:**
- Your evaluation: (yes/no)
- If "no," rewrite the query to specify the additional information needed to accurately answer the question.

"""