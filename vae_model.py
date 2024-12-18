import tensorflow as tf
from tensorflow.keras import layers, Model
from vae_encoder import create_encoder
from vae_decoder import create_decoder

def create_vae(latent_dim, optimizer, loss_function="binary_crossentropy"):
    # Load Encoder and Decoder
    encoder = create_encoder(latent_dim)
    decoder = create_decoder(latent_dim)

    # VAE Model
    encoder_inputs = encoder.input
    z_mean, z_log_var, z = encoder(encoder_inputs)
    vae_outputs = decoder(z)

    # KL Divergence Loss as a Layer
    class VAELoss(layers.Layer):
        def call(self, inputs, outputs):
            z_mean, z_log_var, reconstruction = inputs
            reconstruction_loss = tf.keras.losses.get(loss_function)(outputs, reconstruction)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss) * 28 * 28
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
            self.add_loss(total_loss)
            return reconstruction

    vae_loss = VAELoss()
    vae_outputs = vae_loss([z_mean, z_log_var, vae_outputs], encoder_inputs)

    vae = Model(encoder_inputs, vae_outputs, name="vae")
    vae.compile(optimizer=optimizer, loss=None)  # Uses the optimizer passed as an argument
    return encoder, decoder, vae
