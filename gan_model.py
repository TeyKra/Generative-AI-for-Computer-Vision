import tensorflow as tf
from tensorflow.keras import layers, Model
from vae_decoder import create_decoder

def create_gan(latent_dim):
    # Generator (uses the VAE decoder as the generator)
    generator = create_decoder(latent_dim)

    # Discriminator
    discriminator_inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(discriminator_inputs)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    discriminator_outputs = layers.Dense(1, activation="sigmoid")(x)
    discriminator = Model(discriminator_inputs, discriminator_outputs, name="discriminator")

    # GAN Model (Generator + Discriminator)
    latent_inputs = layers.Input(shape=(latent_dim,))
    generated_image = generator(latent_inputs)
    discriminator.trainable = False  # Do not train the discriminator in the GAN model as stated in the instructions
    validity = discriminator(generated_image)

    gan = Model(latent_inputs, validity, name="gan")
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")

    return generator, discriminator, gan
