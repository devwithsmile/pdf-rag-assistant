# app.py
import streamlit as st
import os
import logging
import uuid
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import asyncio
import nest_asyncio

#configure asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "multi-doc-rag"
PERSIST_DIRECTORY = "./chroma_db"

@st.cache_resource
def initialize_components():
    """Cache heavy resources"""
    ollama.pull(EMBEDDING_MODEL)
    ollama.pull(MODEL_NAME)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOllama(model=MODEL_NAME)
    return embedding, llm

def process_pdf(file):
    """Process PDF from bytes without saving"""
    with NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp.seek(0)
        loader = PyPDFLoader(tmp.name)
        pages = loader.load_and_split()
    return pages

@st.cache_data(show_spinner=False)
def split_documents(_documents):
    """Split documents into chunks with caching"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=300
    )
    return text_splitter.split_documents(_documents)

def update_vector_db(_docs, _embedding):
    """Update existing vector store with new documents"""
    vector_db = Chroma(
        collection_name=VECTOR_STORE_NAME,
        embedding_function=_embedding,
        persist_directory=PERSIST_DIRECTORY,
    )
    
    existing_ids = set(vector_db.get()['ids'])
    new_docs = [doc for doc in _docs if doc.metadata['doc_id'] not in existing_ids]
    
    if new_docs:
        vector_db.add_documents(new_docs)
        vector_db.persist()
        logging.info(f"Added {len(new_docs)} new chunks to vector DB")

def handle_file_uploads():
    """Process uploaded PDFs and update vector DB"""
    pdf_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if not pdf_files:
        return None

    embedding, _ = initialize_components()
    all_chunks = []

    for pdf_file in pdf_files:
        # Process PDF in memory
        pages = process_pdf(pdf_file)
        
        # Generate unique document ID
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pdf_file.name))
        
        # Add metadata to chunks
        chunks = split_documents(pages)
        for chunk in chunks:
            chunk.metadata.update({
                "doc_id": doc_id,
                "source": pdf_file.name
            })
        all_chunks.extend(chunks)

    # Update vector DB with new chunks
    update_vector_db(all_chunks, embedding)
    return Chroma(
        collection_name=VECTOR_STORE_NAME,
        embedding_function=embedding,
        persist_directory=PERSIST_DIRECTORY,
    )

def create_retriever(_llm):
    """Create multi-query retriever with session caching"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        [Your existing prompt template here]
        """
    )
    
    vector_db = Chroma(
        collection_name=VECTOR_STORE_NAME,
        embedding_function=initialize_components()[0],
        persist_directory=PERSIST_DIRECTORY,
    )
    
    return MultiQueryRetriever.from_llm(
        vector_db.as_retriever(search_kwargs={'k': 8}),
        _llm,
        prompt=QUERY_PROMPT
    )

def main():
    st.title("Multi-Document Assistant")
    
    # Initialize cached components
    embedding, llm = initialize_components()
    
    # Handle PDF uploads and update vector DB
    vector_db = handle_file_uploads()
    
    # User input
    query = st.text_input("Ask about your documents:")
    
    if query and vector_db:
        with st.spinner("Analyzing documents..."):
            try:
                # Create processing chain
                retriever = create_retriever(llm)
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template(
                        "Answer using ONLY these documents:\n\n{context}\n\nQuestion: {question}"
                    )
                    | llm
                    | StrOutputParser()
                )
                
                # Stream response
                st.write("**Answer:**")
                response = chain.invoke(query)
                st.write(response)
                
                # Show sources
                docs = vector_db.similarity_search(query, k=3)
                st.write("**Relevant Sources:**")
                for doc in docs:
                    st.write(f"- {doc.metadata['source']} (page {doc.metadata.get('page', '?')})")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
    main()
