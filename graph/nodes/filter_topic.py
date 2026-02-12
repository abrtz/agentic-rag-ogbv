from typing import Any, Dict

from graph.chains.topic_grader import topic_grader
from graph.state import GraphState


def filter_topic(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether the user query is relevant to allowed topics.
    If it is not, return answer to user.

    Params:
    - state (dict): The current graph state.

    Return:
    - state (dict): Answer to user.
    """

    print("--ASSESS QUESTION RELEVANCE--")
    question = state["question"]

    # get score whether it's relevant or not
    score = topic_grader.invoke({"question": question})
    grade = score.binary_score
    if grade.lower() == "yes":
        print("--GRADE: TOPIC RELEVANT--")
        relevance_status = True
    else:
        print("--GRADE: TOPIC NOT RELEVANT--")
        relevance_status = False

    return {
        "question": question,
        "relevance": relevance_status,
    }  # update graph state with relevance Boolean


if __name__ == "__main__":
    filter_topic(state={"question": "crochet"})
