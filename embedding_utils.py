# embedding_utils.py
from sentence_transformers import SentenceTransformer
import torch

def get_embeddings(chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model = model.to(device)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings
