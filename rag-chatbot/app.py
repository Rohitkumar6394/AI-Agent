from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

print("Step 1: Loading PDF")

loader = PyPDFLoader("data/data.pdf")
documents = loader.load()

print("Step 2: Splitting")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = splitter.split_documents(documents)

print("Step 3: Embeddings")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Step 4: Creating Vector Store")

vectorstore = FAISS.from_documents(
    texts,
    embeddings
)

print("Step 5: Creating Retriever")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Step 6: Loading Local LLM")

pipe = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150
)

llm = HuggingFacePipeline(pipeline=pipe)

print("\n✅ RAG Chatbot Ready")
print("Type 'exit' to quit")

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = llm.invoke(prompt)

        print("\nAnswer:")
        print(response)

    except Exception as e:
        print("Error:", e)