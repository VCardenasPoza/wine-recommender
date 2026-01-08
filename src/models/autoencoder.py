import tensorflow as tf
from tensorflow.keras import layers, Model

def build_autoencoder(n_cepas: int, latent_dim: int = 32):
    # ---------- encoder ----------
    # --- inputs ---
    in_num = layers.Input(shape=(7,))
    in_cepa = layers.Input(shape=(1,), dtype="int32")
    in_mar = layers.Input(shape=(13,))
    in_txt = layers.Input(shape=(384,))

    # --- num ---
    x_num = layers.Dense(32, activation="relu")(in_num)
    x_num = layers.Dense(16, activation="relu")(x_num)

    # --- cepa ---
    x_cepa = layers.Embedding(input_dim=n_cepas, output_dim=16)(in_cepa)
    x_cepa = layers.Flatten()(x_cepa)

    # --- maridaje ---
    x_mar = layers.Dense(32, activation="relu")(in_mar)

    x_cat = layers.Concatenate()([x_cepa, x_mar])
    x_cat = layers.Dense(32, activation="relu")(x_cat)

    # --- texto ---
    x_txt = layers.Dense(256, activation="relu")(in_txt)
    x_txt = layers.Dense(128, activation="relu")(x_txt)

    # --- fusión ---
    x = layers.Concatenate()([x_num, x_cat, x_txt])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    z = layers.Dense(latent_dim, name="z")(x)   # latent space
    #z = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(z)

    # ---------- decoder ----------
    x = layers.Dense(64, activation="relu")(z)
    x = layers.Dense(128, activation="relu")(x)

    out_num = layers.Dense(7, name="num")(x)
    out_cepa = layers.Dense(n_cepas, activation="softmax", name="cepa")(x)
    out_mar = layers.Dense(13, activation="sigmoid", name="mar")(x)
    out_txt = layers.Dense(384, name="txt")(x)


    autoencoder = Model(
        inputs=[in_num, in_cepa, in_mar, in_txt],
        outputs=[out_num, out_cepa, out_mar, out_txt],
        name="autoencoder"
    )

    encoder = Model(
        inputs=[in_num, in_cepa, in_mar, in_txt],
        outputs=z,
        name="encoder"
    )

    return autoencoder, encoder
