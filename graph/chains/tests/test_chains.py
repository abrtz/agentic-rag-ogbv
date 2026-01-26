from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from graph.chains.topic_grader import TopicGate, topic_grader
from ingestion import retriever


# check whether we get relevant document
def test_retrieval_grader_answer_yes() -> None:
    # take example question and use retriever to get relevant docs
    question = "online violence against women"
    docs = retriever.invoke(question)
    # get the first doc that we find and take the content
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )  # question: original question - document: the retrieved doc

    # assert that the result with binary score is yes
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    # question not related to retrieved docs
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


# check that everything is working as expected
def test_generation_chain() -> None:
    question = "online violence against women"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "online violence against women"
    docs = retriever.invoke(question)  # retrieve docs

    # provide retrieve docs as context and question so answer should be grounded in docs
    generation = generation_chain.invoke({"context": docs, "question": question})
    # run hallucination grader chain with retrieved docs and generation - result should be 'yes'
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


# when we do have hallucinated answer
def test_hallucination_grader_answer_no() -> None:
    question = "online violence against women"
    docs = retriever.invoke(question)  # retrieve docs

    # run hallucination grader chain with retrieved docs and rubbish generation - result should be 'no'
    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "Crochet can reduce the risk of Alzeheimer's disease",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "online violence against women"

    # invoke router question chain to get RouteQuery obj
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "crochet"  # sth unrelated should lead to websearch

    # invoke router question chain to get RouteQuery obj
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"


def test_topic_grader_answer_yes() -> None:
    question = "online violence against women"

    # run topic grader chain with user query - result should be 'yes'
    res: TopicGate = topic_grader.invoke({"question": question})
    assert res.binary_score == "yes"


def test_topic_grader_answer_no() -> None:
    question = "crochet"

    # run topic grader chain with out of topic user query - result should be 'no'
    res: TopicGate = topic_grader.invoke({"question": question})
    assert res.binary_score == "no"
