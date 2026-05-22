from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return DuckDuckGoSearchRun().run(query)


@tool
def search_wiki(query: str) -> str:
    """Search Wikipedia for factual background."""
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(query)


tools = [search_web, search_wiki]
tool_node = ToolNode(tools)


def _agent_node(llm):
    bound = llm.bind_tools(tools)
    def node(state: MessagesState):
        return {"messages": [bound.invoke(state["messages"])]}
    return node


def _route(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def create_orchestrator(llm, vectorstore=None):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node(llm))
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route)
    graph.add_edge("tools", "agent")
    return graph.compile()
