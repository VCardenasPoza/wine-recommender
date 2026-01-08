import tensorflow as tf
import numpy as np
from pathlib import Path

from src.utils.io import load_csv
from src.features.build_features import build_features

ROOT = Path(__file__).resolve().parents[2] 

DATA_PATH = ROOT / "data/processed/wines_clean.csv"
MODEL_PATH = ROOT / "models/autoencoder/encoder.keras"
OUT_PATH = ROOT / "models/embeddings"

def embed_wines():
    # ---------- data ----------
    df = load_csv(DATA_PATH)
    X_num, X_cepa, X_mar, X_txt, artifacts = build_features(df)

    # ---------- model ----------
    encoder = tf.keras.models.load_model(MODEL_PATH)

    # ---------- embeddings ----------
    Z = encoder.predict(
        [X_num, X_cepa, X_mar, X_txt],
        batch_size=256,
        verbose=1
    )  # shape (N, latent_dim)
    Z = tf.math.l2_normalize(Z, axis=1)
    # ---------- save ----------
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH / "wine_embeddings.npy", Z)

    return Z, artifacts


if __name__ == "__main__":
    embed_wines()
