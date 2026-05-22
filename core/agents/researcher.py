from langchain_core.messages import SystemMessage


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research agent. Your job is to gather accurate, current information "
    "using available tools. Cite sources where possible. Be thorough and concise."
))


def researcher_node(llm, tools):
    bound = llm.bind_tools(tools)
    def node(state):
        messages = [SYSTEM_PROMPT] + state["messages"]
        return {"messages": [bound.invoke(messages)]}
    return node
