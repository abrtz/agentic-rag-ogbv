from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


# take docs from state and run the chain
def generate(state: GraphState) -> Dict[str, Any]:
    print("--GENERATE--")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }  # update generation key in graph state to be the answer from LLM
