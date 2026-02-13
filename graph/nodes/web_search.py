from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("--WEB SEARCH--")
    question = state["question"]
    documents = state.get("documents", [])
    attempts = state.get("web_search_attempts", 0) + 1

    tavily_results = web_search_tool.invoke({"query": question})["results"]

    # create a list to store documents resulting from web search
    web_documents = []
    for result in tavily_results:
        # make a LangChain document for each web result and add metadata for source extraction in UI
        doc = Document(
            page_content=result["content"],
            metadata={
                "source": result.get("url"),
                "title": result.get("title"),
            },
        )
        web_documents.append(doc)

    documents.extend(web_documents)

    return {
        "documents": documents,
        "question": question,
        "web_search_attempts": attempts,
    }  # return updated state of graph execution with documents and the original question


if __name__ == "__main__":
    # in case no relevant docs were found
    web_search(state={"question": "online violence against women", "documents": None})
