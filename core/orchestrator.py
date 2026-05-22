from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from core.memory import store


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


def _build_context(vectorstore, query: str) -> str:
    parts = []

    memories = store.recent(10)
    if memories:
        mem_text = "\n".join(f"{m['role']}: {m['content']}" for m in memories)
        parts.append(f"[Long-term memory]\n{mem_text}")

    if vectorstore:
        try:
            docs = vectorstore.similarity_search(query, k=3)
            if docs:
                rag_text = "\n---\n".join(d.page_content for d in docs)
                parts.append(f"[Knowledge base]\n{rag_text}")
        except Exception:
            pass

    return "\n\n".join(parts)


def _agent_node(llm, vectorstore):
    bound = llm.bind_tools(tools)

    def node(state: MessagesState):
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        context = _build_context(vectorstore, last_human)
        messages = list(state["messages"])
        if context:
            messages = [SystemMessage(content=context)] + messages
        response = bound.invoke(messages)

        # persist the exchange
        if last_human:
            store.save("user", last_human)
        store.save("assistant", response.content or "")

        return {"messages": [response]}

    return node


def _route(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def create_orchestrator(llm, vectorstore=None):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node(llm, vectorstore))
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route)
    graph.add_edge("tools", "agent")
    return graph.compile()
