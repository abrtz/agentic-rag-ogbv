from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# llm that supports function calling
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )  # LLM will leverage description to decide whether doc is relevant or not


# LangChain will use function calling and for every LLM call, return a pydantic object and LLM will return in the schema we want
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# create chain
retrieval_grader = grade_prompt | structured_llm_grader
