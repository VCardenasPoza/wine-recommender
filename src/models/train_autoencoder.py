import tensorflow as tf
import numpy as np
from pathlib import Path

from src.models.autoencoder import build_autoencoder
from src.features.build_features import build_features
from src.utils.io import load_csv


DATA_PATH = Path("data/processed/wines_clean.csv")
MODEL_PATH = Path("models/autoencoder")


def train_autoencoder(
    n_cepas: int,
    latent_dim: int = 32,
    batch_size: int = 128,
    epochs: int = 50,
):
    # ---------- data ----------
    df = load_csv(DATA_PATH)

    X_num, X_cepa, X_mar, X_txt, artifacts = build_features(df)

    # targets
    y_num = X_num
    y_mar = X_mar
    y_txt = tf.math.l2_normalize(X_txt, axis=1)
    y_cepa = X_cepa

    # ---------- model ----------
    autoencoder, encoder = build_autoencoder(
        n_cepas=n_cepas,
        latent_dim=latent_dim
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "num": "mse",
            "cepa": "sparse_categorical_crossentropy",
            "mar": "binary_crossentropy",
            "txt": "mse",
        },
        loss_weights={
            "num": 1.0,
            "cepa": 1.0,
            "mar": 1.0,
            "txt": 2.0,
        },
    )

    autoencoder.summary()

    # ---------- training ----------
    history = autoencoder.fit(
        x=[X_num, X_cepa, X_mar, X_txt],
        y={
            "num": y_num,
            "cepa": y_cepa,
            "mar": y_mar,
            "txt": y_txt,
        },
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        shuffle=True,
    )

    # ---------- save ----------
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    autoencoder.save(MODEL_PATH / "autoencoder.keras")
    encoder.save(MODEL_PATH / "encoder.keras")

    return history


if __name__ == "__main__":
    raise RuntimeError(
        "Execute train_autoencoder in a notebook or script."
    )
