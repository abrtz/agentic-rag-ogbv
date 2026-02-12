from typing import Any, Dict

from graph.state import GraphState


def reject(state: GraphState) -> Dict[str, Any]:
    """
    Handle irrelevant user queries.
    Return an updated state with a denial response.

    Params:
    - state (dict): The current graph state.

    Return:
    - state (dict): Answer to user.
    """
    question = state["question"]
    response = """I can't help with that request.

This assistant only provides information about:
- online violence against women and girls,
- the manosphere,
- deepfakes,
- laws and regulations related to gender-based violence,
- measures to mitigate gender-based violence.
"""

    return {
        "question": question,
        "generation": response,
    }  # update graph state with hard response to user
