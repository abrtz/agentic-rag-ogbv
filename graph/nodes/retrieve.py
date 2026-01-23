from typing import Any, Dict

from graph.state import GraphState  # input for the node, what it updates
from ingestion import retriever  # references vector store


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--RETRIEVE--")
    question = state["question"]  # extract question from current state

    documents = retriever.invoke(question)  # do semantic search and get relevant docs
    return {
        "documents": documents,
        "question": question,
    }  # update the field of documents in current state with retrieve docs and original question to be cautious
