import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.visualization.latent_space import umap_2d_embedding


def main():
    umap_2d_embedding()
    
if __name__ == "__main__":
    main()