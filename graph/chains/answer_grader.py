from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
# same LLM with the grade answer structured output
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# create system prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
# create prompt template
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: \n\n {question} \n\n LLM generation: {generation}",
        ),  # answer LLM generated
    ]
)

# answer grader chain which takes the answer prompt and pipe it into structued_llm_grader
# to get GradeAnswer class obj with True or False whether it answer question or not
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
