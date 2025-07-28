# Practicals Part 1:
#  - Use FAISS for all vectors created previously using two chunking methods
#  - Generate responses for the queries identified at the beginning of this exercise
#  - Note any differences in responses when tried out with two chunking methods


# app.py
# Import necessary libraries
import streamlit as st
import os

# Import utility functions for chunking, embeddings, FAISS indexing, and response generation
from chunk_utils import character_split, recursive_split
from embedding_utils import get_embeddings
from faiss_utils import build_faiss_index, retrieve_similar_chunks
from response_utils import generate_answer

# Function to load the dataset file
def load_file():
    if os.path.exists("dataset.txt"):
        with open("dataset.txt", "r", encoding="utf-8") as file:
            return file.read()              # Return the entire text content
    st.error("dataset.txt not found.")      # Show error if file is missing
    st.stop()                               # Stop execution

# Streamlit page configuration
st.set_page_config(page_title="Practical 1: FAISS RAG Chunking Comparison", layout="wide")
st.title("🧠 Practical 1: FAISS-Based RAG: Chunking Comparison")

# Load the dataset from file
text = load_file()

# Let the user choose the chunking method
chunk_method = st.radio("Choose chunking method:", ["Character Split", "Recursive Split"], horizontal=True)

# Let the user adjust chunk size and overlap using sliders
chunk_size = st.slider("Chunk size", 50, 300, 100, step=10)
chunk_overlap = st.slider("Chunk overlap", 0, 100, 20, step=10)

# Perform text chunking based on user selection
if chunk_method == "Character Split":
    chunks = character_split(text, chunk_size, chunk_overlap)
else:
    chunks = recursive_split(text, chunk_size, chunk_overlap)

# Display the chunked text for verification
st.markdown("### 📚 Chunked Text")
for i, chunk in enumerate(chunks):
    st.markdown(f"**Chunk {i+1}:** {chunk}")

# Generate embeddings and build FAISS index when button is clicked
# Embeddings + FAISS
if st.button("🔄 Generate Embeddings and Build FAISS Index"):
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)
    
    # Save in session state
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.index = index

    st.success("✅ Embeddings and FAISS index ready!")

    # 🔍 Display embedding vectors (first 5 dimensions only)
    st.markdown("### 🧬 Embedding Vectors (Dimensions)")
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        st.markdown(f"**Chunk {i+1}:** {chunk[:100]}...")  # First 100 chars for readability
        st.code(f"{vector[:5].tolist()}")  # Show first 5 dimensions only


# Query
query = st.text_input("🔎 Enter your query:")
if query and "index" in st.session_state:
    query_embedding = get_embeddings([query])
    indices = retrieve_similar_chunks(st.session_state.index, st.session_state.embeddings, query_embedding)
    best_chunk = st.session_state.chunks[indices[0]]
    st.markdown("### 📌 Matched Chunk")
    st.info(best_chunk)

    if st.button("🧠 Generate Answer"):
        answer = generate_answer(best_chunk, query)
        st.success("Generated Response:")
        st.write(answer)
