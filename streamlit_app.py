import streamlit as st
from train import train_vae, run_gan

# Main title
st.title("Variational Autoencoder (VAE) and Generative Adversarial Network (GAN) on MNIST")
st.sidebar.header("Configuration Options")

# Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Select a model to use:", ["VAE", "GAN"])

# Common parameters
st.sidebar.subheader("Common Parameters")
latent_dim = st.sidebar.slider("Latent Dimension", min_value=2, max_value=128, value=16, step=2)

# If the chosen model is VAE
if model_choice == "VAE":
    # VAE-specific parameters
    st.sidebar.subheader("VAE-Specific Parameters")
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adagrad"])
    learning_rate = st.sidebar.number_input(
        "Learning Rate", 
        min_value=1e-5, 
        max_value=1e-2, 
        value=1e-3, 
        step=1e-5, 
        format="%.5f"
    )
    loss_function = st.sidebar.selectbox("Loss Function", ["binary_crossentropy", "mse", "mae", "huber"])
    callbacks_options = st.sidebar.multiselect(
        "Select callbacks to use:",
        options=["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TensorBoard"],
    )

    # Button to train the VAE
    if st.button("Start VAE Training"):
        st.write("Training with the following parameters:")
        st.json({
            "Latent Dimension": latent_dim, 
            "Epochs": epochs, 
            "Batch Size": batch_size, 
            "Optimizer": optimizer, 
            "Learning Rate": learning_rate,
            "Loss Function": loss_function,
            "Callbacks": callbacks_options
        })
        train_vae(latent_dim, epochs, batch_size, callbacks_options, loss_function, optimizer, learning_rate)
        st.success("Training completed!")

# If the chosen model is GAN
elif model_choice == "GAN":
    # Instructions for running the GAN
    st.sidebar.info("GAN does not require training.")
    num_images = st.sidebar.slider("Number of images to generate", min_value=1, max_value=20, value=10)

    # Button to run the GAN
    if st.button("Run GAN"):
        st.write(f"Running GAN with a latent dimension of {latent_dim} and generating {num_images} images.")
        run_gan(latent_dim, num_images=num_images)
        st.success("Execution completed!")
