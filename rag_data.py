import os
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

URL = "https://python.langchain.com/docs/how_to/recursive_text_splitter/"

loader = WebBaseLoader(web_paths=(URL,), requests_kwargs={"timeout": 15})
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name="promt_engineering",      
    persist_directory="./chroma_db",
)
