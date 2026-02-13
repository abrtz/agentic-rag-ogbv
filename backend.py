from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


from graph.graph import app

def get_sources(state)-> List[str]:
    """
    Format sources to be displayed in UI.
    :param state: the state graph
    :return list containing url of documents that answer is grounded on
    "rtype: List[str]
    """
    sources = []
    for doc in state.get("documents", []):
        if "source" in doc.metadata:
            url = doc.metadata["source"]
            if url not in sources:
                sources.append(url)
    sources = list(set(sources)) #to make sure sources are not duplicated
    return sources

def run_llm(question: str) -> Dict[str, Any]:
    """
    Run the graph pipeline to answer a query using the retrieved documentation.

    :param query: the user's question.
    :type query: str
    :return: dictionary containing:
        - question: the user's question.
        - generation: the generated answer.
        - source: list of documents used to generate answer to user.
    :rtype: Dict[str, Any]
    """

    state = app.invoke(input={"question": question})

    # get document sources
    sources = get_sources(state)

    return {
        "question": state["question"],
        "generation": state["generation"],
        "source": sources,
    }
