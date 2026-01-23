from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("--WEB SEARCH--")
    question = state["question"]
    if "documents" in state:  # if the route to web search gives error in first run
        documents = state["documents"]
    else:
        documents = None

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    # after getting the list of docs, take content from all elements and combine into one LangChain document
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    # if we have document in our state, append relevent documents to documents list
    if documents is not None:
        documents.append(web_results)
    # if no relevant docs were found, append the web_results
    else:
        documents = [web_results]

    return {
        "documents": documents,
        "question": question,
    }  # return updated state of graph execution with documents and the original question


if __name__ == "__main__":
    # in case no relevant docs were found
    web_search(state={"question": "online violence against women", "documents": None})
