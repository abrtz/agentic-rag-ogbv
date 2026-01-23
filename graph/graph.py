from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router

# import all consts (nodes names) and the nodes created
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState


def decide_to_generate(state):
    print("--ASSESS GRADED DOCUMENTS--")

    # if web_search True (at least one doc is not relevant), run web search
    if state["web_search"]:
        print("--DECISION: NOT ALL DOCUMENTS ARE RELEVANT, RUNNING WEB SEARCH")
        return WEBSEARCH
    # if not, all retrieved docs are relevant to query
    else:
        print("--DECISION: GENERATE--")
        return GENERATE


# conditional edge function
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Receive state.
    Return string of which node to go next.
    """
    print("--CHECK HALLUCINATIONS--")
    # get question, documents and LLM generation
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # retreived docs with or without search and LLM generation. response has binary score attribute
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # value is true -> no hallucination
    if hallucination_grade := score.binary_score:
        print("--DECISION: GENERATION IS GROUNDED IN DOCUMENTS--")
        # grade generation to whether it answers question or not
        print("--GRADE GENERATION vs. QUESTION--")
        score = answer_grader.invoke({"question": question, "generation": generation})
        # if answer_grade score is true, generation does address the question
        if answer_grade := score.binary_score:
            print("--DECISION: GENERATION ADDRESSES THE QUESTION--")
            return "useful"  # for LangGraph mapping
        # if answer does not address question, but it is grounded in docs
        else:
            print("--DECISION: GENERATION DOES NOT ADDRESS THE QUESTION--")
            return "not useful"  # info from vector store not sufficent to answer question so we use external search

    # if answer is not even grounded in docs, we regenerate it again from docs
    else:
        print("--DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY--")
        return "not supported"


def route_question(state: GraphState) -> str:
    """
    Receive state.
    Return a string with next node to execute.
    """
    print("--ROUTE QUESTION--")
    question = state["question"]  # get question from graph state
    # run question router chain with user question and save result in source variable of RouteQuery type
    source: RouteQuery = question_router.invoke({"question": question})
    # if data source is websearch, route it to the websearch node
    if source.datasource == WEBSEARCH:
        print("--ROUTE QUESTION TO WEB SEARCH--")
        return WEBSEARCH
    # if the data source is vector store, route it to retrieve node for retrieval augmentation
    elif source.datasource == "vectorstore":
        print("--ROUTE QUESTION TO RAG--")
        return RETRIEVE


# connect everything together
workflow = StateGraph(GraphState)

# add nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# # add retrieve node as entry point
# workflow.set_entry_point(RETRIEVE)

# add conditional entry point
workflow.set_conditional_entry_point(
    route_question, path_map={WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE}
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
# add conditional edge
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,  # source node
    decide_to_generate,  # function to decide which node to execute
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },  # it is also possible to go back to each node
)

workflow.add_conditional_edges(
    GENERATE,  # source node
    grade_generation_grounded_in_documents_and_question,  # function to decide which node to execute
    path_map={
        "not supported": GENERATE,  # regenerate after answer was not grounded in docs
        "useful": END,  # go to end node and return answer to user
        "not useful": WEBSEARCH,  # the vector store did not contain information enough to answer question
    },  # map strings returning from previous function to real node names since they do not represent a real node, strings are displayed in the edges.
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)  # last edge

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print(f"Graph: \n {app.get_graph().draw_mermaid()}\n")  # to get mermaid
