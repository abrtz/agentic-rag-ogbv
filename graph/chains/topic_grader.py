from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# llm that supports function calling
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)


class TopicGate(BaseModel):
    """
    Binary score for relevance check on user query.
    """

    binary_score: str = Field(
        description="User query is relevant to the allowed topics, 'yes' or 'no'"
    )  # LLM will leverage description to decide whether query is relevant or not


# LangChain will use function calling and for every LLM call, return a pydantic object and LLM will return in the schema we want
structured_llm_grader = llm.with_structured_output(TopicGate)

allowed_topics = [
    "online violence against women and girls",
    "manosphere",
    "deepfakes",
    "laws and regulations related to gender-based violence",
    "measures to mitigate gender-based violence",
]

system = f"""You are a strict topic classifier assessing relevance of a user question to a topic. \n 
    Allowed topics: {', '.join(allowed_topics)}. \n
    If the user question is about these topics, allow it. \n
    Give a binary score 'yes' or 'no' to indicate whether the user question is related to the allowed topics."""

topic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

# create chain
topic_grader = topic_prompt | structured_llm_grader
