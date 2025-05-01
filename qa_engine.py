from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient

# Load vector store
vector_db = FAISS.load_local("vector_index", HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

# Search
query = "what is staging push?"
docs = vector_db.similarity_search(query, k=2)

print("\n Top 2 closest prompts are as follows")
print("\n 1. " + docs[0].page_content)
print("\n 2. " + docs[1].page_content)

context = "\n\n".join([doc.page_content for doc in docs])
# prompt = f"""Use the context below to answer the question.
prompt = f"""You are a helpful assistant. Based on the context below, return the complete paragraphs and not just one statement that most directly answers the question.

Context:
{context}

Question: {query}

Answer:"""

client = InferenceClient(model="google/flan-t5-large", token="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")

response = client.text_generation(
    prompt=prompt,
    max_new_tokens=245,
    temperature=0.3,
    do_sample=False,
)

# Answer
# llm = OpenAI(openai_api_key="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")
# chain = load_qa_chain(llm, chain_type="stuff")
# response = chain.run(input_documents=docs, question=query)

# llm = HuggingFaceEndpoint(repo_id="google/flan-t5-large", task="text2text-generation", huggingfacehub_api_token="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", temperature=0.5)
# retriever = vector_db.as_retriever()
# response = RetrievalQA.from_chain_type(llm=llm, retriever=retriever).invoke({"query": "Staging push mechanism?"})

# retriever = FAISS.load_local("faiss_index", HuggingFaceEmbeddings()).as_retriever()
#
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
#
# response = qa_chain.invoke({"query": query})

print("\n Response:" + response)