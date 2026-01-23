from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether the retrieved documents are relevant to the question.
    If any document is not relevant, set a flag to run web search.

    Params:
    - state (dict): The current graph state.

    Return:
    - state (dict): Filtered out irrelevant documents and updated web_search state.
    """

    print("--CHECK DOCUMENT RELEVANCE TO QUESTION--")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []  # list to append relevant docs
    web_search = (
        False  # every time we find doc not relevant to the question, change to True
    )

    # iterate through all documents and call retrieval_grader chain
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )  # get score whether it's relevant or not
        grade = score.binary_score
        if grade.lower() == "yes":
            print("--GRADE: DOCUMENT RELEVANT--")
            filtered_docs.append(d)
        else:
            print("--GRADE: DOCUMENT NOT RELEVANT--")
            web_search = True  # if not relevant, change web_search to True
            continue

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }  # update graph state with filtered_docs, original question and updated web_search flag
