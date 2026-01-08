from umap import UMAP
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] 

EMBED_PATH = ROOT / "models/embeddings"

def umap_2d_embedding():

    Z = np.load(EMBED_PATH / "wine_embeddings.npy")

    umap = UMAP(n_components=2, random_state=0)
    Z_2 = umap.fit_transform(Z)

    np.save(EMBED_PATH / "2d_embedding",Z_2)
    
    return Z_2

if __name__ == "__main__":
    umap_2d_embedding()


