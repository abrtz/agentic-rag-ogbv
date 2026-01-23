from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)


class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer.
    """

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )  # set binary score type hint as Boolean, the LangChain output parsers will pass the LLM answer into Boolean


# answer from LLM will be formatted as pydantic class of GradeHallucination with one attribute of binary score (Boolean)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# system prompt for chain
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # tuple: first elemnt is role, second element is content
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# chain: take hallucionation prompt and pipe it to structed_llm_grader to get answer yes or no if the answer is grounded in docs
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
