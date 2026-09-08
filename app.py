import os
import streamlit as st
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

@st.cache_resource
def load_vectordb():
    documents = []
    data_folder = "data"

    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            path = os.path.join(data_folder, file)
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file
                doc.metadata["page"] = doc.metadata.get("page", "Unknown")

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    print("Number of documents:", len(documents))
    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(
            f"Chunk {i}: "
            f"{len(chunk.page_content)} characters, "
            f"page={chunk.metadata.get('page')}"
        )
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if os.path.exists("./chroma_db"):
        vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding
        )
    else:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory="./chroma_db"
        )
    return vectordb

st.title("Enterprise Knowledge Assistant")
st.write("Ask questions about uploaded documents.")
if st.button("Rebuild Database"):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    st.cache_resource.clear()
    st.rerun()
vectordb = load_vectordb()
query = st.text_input("Ask a question")
retriever = vectordb.as_retriever(
    search_kwargs={
        "k": 3}
)
if st.button("Submit") and query.strip():
    docs = retriever.invoke(query)
    if not docs:
        st.warning("No relevant documents found.")
        st.stop()
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are a document assistant.

    Rules:
    1. Answer only from provided context
    2. Do not invent facts
    3. If insufficient context, say:
    "I could not find relevant information."

    Context:
    {context}

    Question:
    {query}
    """

    try:
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

    except Exception as e:
        st.error(f"Groq Error: {e}")
        st.stop()

    answer = response.choices[0].message.content
    st.write("### Answer")
    st.write(answer)
    if "could not find relevant information." not in answer:
      st.write("### Sources")
      seen = set()
      for doc in docs:
          source = doc.metadata.get("source_file", "Unknown")
          page = doc.metadata.get("page", "Unknown")
          identifier = (source, page)
          if identifier not in seen:
              st.write(f"{source} (page {page})")
              seen.add(identifier)


