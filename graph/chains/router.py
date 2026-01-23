from typing import (
    Literal,
)  # provide way to specify a variable can take one of predefined set of values

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant data source.
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore."
    )


llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
structure_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to online violence against women and girls, manosphere, deepfakes, existing laws and regulations, and measure to mitigate gender-based violence.
Use the vectorstore for questions on these topics. For all else, use web search."""
route_propmt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

# create chain to take route prompt and pipe it into structured llm router
question_router = route_propmt | structure_llm_router
