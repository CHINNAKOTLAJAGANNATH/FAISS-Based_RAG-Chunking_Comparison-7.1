import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.cpu().detach().numpy())
    return index

def retrieve_similar_chunks(index, embeddings, query_embedding, top_k=1):
    query_vector = query_embedding.cpu().detach().numpy()
    D, I = index.search(query_vector, top_k)
    return I[0]
