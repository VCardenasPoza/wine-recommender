import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.make_dataset import make_dataset
from src.models.train_autoencoder import train_autoencoder
from src.models.embed_wines import embed_wines


n_cepas = 10 

def main():
    make_dataset(n_cepas)
    
    history = train_autoencoder(n_cepas=n_cepas)
    embed_wines()
    
if __name__ == "__main__":
    main()