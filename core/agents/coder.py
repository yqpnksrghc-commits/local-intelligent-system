from langchain_core.messages import SystemMessage


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a coding agent. Write clean, correct, minimal code. "
    "Always explain what the code does in one sentence before the block."
))


def coder_node(llm):
    def node(state):
        messages = [SYSTEM_PROMPT] + state["messages"]
        return {"messages": [llm.invoke(messages)]}
    return node
