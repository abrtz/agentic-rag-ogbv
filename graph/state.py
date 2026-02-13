from typing import List, TypedDict


class GraphState(TypedDict):
    """ "
    Represent the state of graph.

    Params:
    - question: User query.
    - generation: LLM-generated answer.
    - web_search: Whether to add search.
    - documents: List of documents.
    - web_search_attempts: Number of times web search was called.
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    web_search_attempts: 0
