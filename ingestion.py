from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

urls = [
    "https://www.unwomen.org/en/articles/faqs/ai-powered-online-abuse-how-ai-is-amplifying-violence-against-women-and-what-can-stop-it",
    "https://www.unwomen.org/en/articles/explainer/online-safety-101-what-every-woman-and-girl-should-know",
    "https://www.unwomen.org/en/news-stories/feature-story/2025/12/from-survivors-to-changemakers-how-women-are-fighting-digital-violence-in-mexico-and-bolivia",
    "https://www.unwomen.org/en/articles/facts-and-figures/facts-and-figures-ending-violence-against-women",
    "https://www.unwomen.org/en/articles/glossary/glossary-the-manosphere",
]

pdfs = [
    "https://www.unwomen.org/sites/default/files/2025-06/normative-advances-on-technology-facilitated-violence-against-women-and-girls-en.pdf",
    "https://www.isdglobal.org/wp-content/uploads/2023/09/Misogynistic-Pathways-to-Radicalisation-Recommended-Measures-for-Platforms-to-Assess-and-Mitigate-Online-Gender-Based-Violence.pdf",
]
# load urls and pdfs
docs = [WebBaseLoader(url).load() for url in urls] + [
    PyPDFLoader(pdf).load() for pdf in pdfs
]

# create a list of unified docs
doc_list = [item for sublist in docs for item in sublist]

# load langchain text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0, separators=["\n\n", "\n", ".", " ", ""]
)
# split docs into chunks
doc_splits = text_splitter.split_documents(doc_list)

# initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # chunk_size=50,
    retry_min_seconds=10,
)

# comminting to avoid that it is loaded every time the code runs
# index doc chunks into vector store
# vectorstore = PineconeVectorStore.from_documents(
#     doc_splits, embeddings, index_name="ogbv-rag",
# )

retriever = PineconeVectorStore(
    index_name="ogbv-rag",
    embedding=embeddings,
).as_retriever()
