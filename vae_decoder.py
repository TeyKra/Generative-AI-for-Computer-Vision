import tensorflow as tf
from tensorflow.keras import layers, Model

def create_decoder(latent_dim):
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
