from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load text
loader = PyPDFLoader("/Users/vikas.kumar/SQL Server intro.pdf")
docs = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings
"""
Open AI Embedding
embeddings = OpenAIEmbeddings(openai_api_key="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")
"""

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)

# Save
vector_db.save_local("vector_index")