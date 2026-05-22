from langchain_core.messages import SystemMessage


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a planning agent. Decompose the user's goal into a numbered list of "
    "discrete, executable tasks. Assign each task to one of: researcher, writer, coder. "
    "Output only the task list — no preamble."
))


def planner_node(llm):
    def node(state):
        messages = [SYSTEM_PROMPT] + state["messages"]
        return {"messages": [llm.invoke(messages)]}
    return node
