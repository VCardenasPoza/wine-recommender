# src/features/build_features.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer


NUM_COLS = ["precio", "puntaje", "grados", "frescor", "cuerpo", "dulzor","astringencia"]
CEPA_COL = "cepa"
MARIDAJE_COLS = ["m_carnesrojas",
        "m_aves",
        "m_quesos",
        "m_carnescaza",
        "m_pescados",
        "m_comidachilena",
        "m_cerdo",
        "m_cordero",
        "m_pasta",
        "m_crustaceos",
        "m_mariscos",
        "m_comidaoriental",
        "m_postres"]
TEXT_COL = "nota_cata"


def build_features(df: pd.DataFrame):
    """
    Build stacks of features for the autoencoder.
    """

    # ---------- numerics ----------
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[NUM_COLS])
    # shape (N, 7)

    # ---------- cepa (int) ----------
    le = LabelEncoder()
    X_cepa = le.fit_transform(df[CEPA_COL])
    # shape (N,)

    # ---------- maridaje (multi-hot) ----------
    X_maridaje = df[MARIDAJE_COLS].values.astype("float32")
    # shape (N,13)

    # ---------- texto (embeddings) ----------
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_text = model.encode(
        df[TEXT_COL].tolist(),
        show_progress_bar=True
    )
    # shape (N, 384)

    artifacts = {
        "scaler": scaler,
        "label_encoder_cepa": le,
        "text_model": model
    }

    return X_num, X_cepa, X_maridaje, X_text, artifacts
