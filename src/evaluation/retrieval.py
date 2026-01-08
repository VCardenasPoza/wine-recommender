from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2] 

EMBED_PATH = ROOT / "models/embeddings" 


emb = np.load(EMBED_PATH / "wine_embeddings.npy")

nn = NearestNeighbors(metric="cosine").fit(emb)

def top_k(wine_id:int, k:int =10):
    return nn.kneighbors(emb[wine_id][None], k+1)[1][0][1:]