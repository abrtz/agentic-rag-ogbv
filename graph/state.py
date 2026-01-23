from typing import List, TypedDict


class GraphState(TypedDict):
    """ "
    Represent the state of graph.

    Params:
    - question: User query.
    - generation: LLM-generated answer.
    - web_search: Whether to add search.
    - documents: List of documents.
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
