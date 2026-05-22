from langchain_core.messages import SystemMessage


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a writing agent. Take research and context provided and produce "
    "clear, well-structured prose. Match the requested tone and format exactly."
))


def writer_node(llm):
    def node(state):
        messages = [SYSTEM_PROMPT] + state["messages"]
        return {"messages": [llm.invoke(messages)]}
    return node
