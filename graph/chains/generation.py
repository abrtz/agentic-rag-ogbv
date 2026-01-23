from langchain_classic import hub
from langchain_core.output_parsers import (
    StrOutputParser,
)  # take message, get the content and turn it into a string
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
