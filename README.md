# Lab 1: Variational Autoencoders and Generative Adversarial Networks

## Objective
In this lab, you will:
1. Implement a Variational Autoencoder (VAE) to learn how to reconstruct images by sampling from a latent space.
2. Understand how the components of a VAE can be adapted to conceptualize and implement a Generative Adversarial Network (GAN).
3. Compare the roles of VAEs and GANs in generative modeling.
4. Reflect on the strengths and weaknesses of each approach.

## How to use
- Download the present repository
- build the venv by double-clicking the built.bat file
- launch the streamlit UI by double-clicking the launch.bat file
- remember to train before generating

## Background
### Variational Autoencoders (VAEs)
A VAE is a generative model that reconstructs input data by encoding it into a latent space and
then decoding it back. The latent space is a compressed representation that captures the most important features of the input.
**The key elements of a VAE:**:
- Encoder: Compresses input data into a latent distribution (mean and variance).
- Latent Space: The encoded representation space.
- Reparameterization Trick: Allows gradients to flow through the stochastic latent space.
- Decoder: Reconstructs the input from the latent space.
**Generative Adversarial Networks (GANs)**
A GAN is a generative model consisting of two components:
- Generator: Produces fake data from random noise (latent space).
- Discriminator: Distinguishes between real and fake data. GANs are trained using an adversarial process where the generator tries to fool the discriminator, and the discriminator tries to detect fake data.
**Reversing VAEs into GANs**
In a conceptual sense:
- A VAE decoder can be seen as a GAN generator.
- The GAN discriminator replaces the VAE encoder by determining the quality of generated samples instead of encoding input data.

## Part 1: Implementing a Variational Autoencoder (VAE)

## Questions 1:
1. Why do we use the reparameterization trick in VAEs?
## Answer:
We use the **reparameterization trick** in Variational Autoencoders (VAEs) to allow backpropagation through the stochastic sampling process. 

VAEs involve sampling a latent variable $z$ from a distribution (e.g., Gaussian) defined by parameters $\mu$ (mean) and $\sigma$ (variance), which are outputs of the encoder network. Without the trick, the sampling step is non-differentiable, and gradients cannot propagate back to the encoder.

## Reparameterization Trick:
Instead of sampling $z$ directly as $z \sim \mathcal{N}(\mu, \sigma^2)$, we rewrite $z$ as:

\[
    z = \mu + \sigma \cdot \epsilon
\]

where $\epsilon \sim \mathcal{N}(0, 1)$ is a noise term sampled from a standard normal distribution. This reformulation separates the stochastic part ($\epsilon$) from the deterministic parameters ($\mu$, $\sigma$), making the process differentiable.

### Benefits:
- Enables gradient-based optimization (e.g., using backpropagation).
- Makes training VAEs practical using standard deep learning frameworks.

In short, the reparameterization trick bridges the gap between stochastic sampling and deterministic optimization, ensuring the VAE is trainable.

## Questions 2:
### 2. How does the KL divergence loss affect the latent space?

### Answer:
The **KL divergence loss** in Variational Autoencoders (VAEs) encourages the latent space to closely match a predefined prior distribution, typically a standard Gaussian distribution ($\mathcal{N}(0, I)$). Here's how it affects the latent space:

### 1. **Regularizes the Latent Space:**
- The KL divergence term measures the difference between the encoder's learned latent distribution $q(z|x)$ and the prior $p(z)$.
- Minimizing this term pushes $q(z|x)$ to be as close as possible to $p(z)$, ensuring a structured and smooth latent space.

### 2. **Promotes Generalization:**
- By regularizing the latent space, the KL divergence prevents overfitting. This allows the VAE to generate meaningful samples even for points not explicitly seen during training.

### 3. **Controls Latent Space Overlap:**
- Ensures that the latent space for different inputs overlaps sufficiently. This overlap helps the decoder generalize and produce coherent reconstructions and generations.

### Trade-off:
- Too much weight on the KL divergence loss can lead to underfitting, where the encoder ignores the data ($q(z|x) \approx p(z)$).
- A proper balance (via the ELBO loss) maintains reconstruction quality while keeping a well-regularized latent space.

In summary, the KL divergence loss shapes the latent space into a smooth, organized structure that aligns with the prior, promoting meaningful interpolation and generalization.

## Questions 3:
### 3. How does changing the latent space dimension (latent_dim) impact the reconstruction quality?

### Answer:
Changing the **latent space dimension** ($\text{latent\_dim}$) in a Variational Autoencoder (VAE) directly impacts the **reconstruction quality** as follows:

### 1. **Low Latent Dimension:**
- **Impact:** If $\text{latent\_dim}$ is too small, the model may not have enough capacity to encode all the important features of the input data.
- **Result:** Poor reconstruction quality, as the latent space is too constrained to capture the full variability of the data.
- **Use Case:** Useful for discovering low-dimensional representations when the data is inherently simple or highly structured.

### 2. **High Latent Dimension:**
- **Impact:** If $\text{latent\_dim}$ is too large, the model has excessive capacity, leading to less regularized latent representations.
- **Result:** The VAE might overfit to the training data, potentially leading to good reconstruction but poorer generative performance, as the latent space may deviate from the prior distribution.
- **Use Case:** Suitable for complex datasets with high variability but risks losing the meaningful structure of the latent space.

### 3. **Optimal Latent Dimension:**
- Balancing $\text{latent\_dim}$ is critical. An appropriately sized latent space allows the model to:
  - Capture essential data features for accurate reconstruction.
  - Maintain a structured and regularized latent space for generalization.

### Summary:
- **Small $\text{latent\_dim}$:** Poor reconstruction, good generalization (if over-restricted).
- **Large $\text{latent\_dim}$:** Good reconstruction, risk of poor generalization.
- **Choosing $\text{latent\_dim}$:** Depends on the complexity of the dataset and the task (e.g., reconstruction vs. generation).

## Part 2: From VAE to GAN

### Question 1:
1. Conceptual Discussion:
    - Explain how the VAE decoder can be used as a GAN generator.
    - Discuss the differences between the VAE encoder and the GAN discriminator.

### Answer:

#### **1. How the VAE Decoder Can Be Used as a GAN Generator**

The **VAE decoder** can function as a **GAN generator** because both share the same goal: generating samples from a learned data distribution. Here's how this works conceptually:

- **VAE Decoder:**
  - The decoder takes a latent vector $z$ sampled from a prior distribution (e.g., $\mathcal{N}(0, 1)$) and maps it to the data space $x$.
  - Its training is focused on reconstructing data while regularizing the latent space to match the prior distribution.

- **GAN Generator:**
  - The generator in a GAN also takes a random latent vector $z$ (sampled from a prior distribution, typically $\mathcal{N}(0, 1)$) and maps it to the data space $x$.
  - Its training focuses on "fooling" the discriminator into classifying the generated samples as real.

**Why the VAE Decoder Fits:**
- After training a VAE, the decoder is effectively a learned mapping from $z$ to realistic data $x$, much like a GAN generator. It can be directly used to generate new samples by sampling $z$ from the prior distribution.

**Key Difference:**
- The VAE decoder's outputs are typically optimized for reconstruction quality and latent space organization, while a GAN generator prioritizes producing samples indistinguishable from real data according to the discriminator.

---

#### **2. Differences Between the VAE Encoder and the GAN Discriminator**

| **Aspect**             | **VAE Encoder**                          | **GAN Discriminator**                  |
|-------------------------|------------------------------------------|----------------------------------------|
| **Role in Model**       | Maps input \( x \) to a latent space \( z \). | Distinguishes between real and generated samples. |
| **Output**              | Parameters of a latent distribution (e.g., \( \mu \), \( \sigma \)). | A binary probability (real vs. fake). |
| **Training Objective**  | Minimize the ELBO loss (reconstruction + KL divergence). | Minimize the adversarial loss, learning to classify correctly. |
| **Focus**               | Capturing essential features of the data for reconstruction. | Detecting realistic patterns in data for adversarial training. |
| **Type of Learning**    | Probabilistic (latent distribution modeling). | Discriminative (binary classification). |
| **Interaction with \( z \)** | Learns to model the relationship between the input data and a compact, meaningful representation in a latent space. | Does not involve or manipulate any latent representation; its focus is on distinguishing real from fake samples. |


---

### Summary:
- The **VAE decoder** can serve as a GAN generator because it learns to map latent vectors \( z \) to data \( x \), much like a generator. However, it prioritizes reconstruction and regularization over adversarial objectives.
- The **VAE encoder** differs from the GAN discriminator in purpose and output: the encoder maps data to latent space distributions, while the discriminator classifies real vs. fake samples without directly interacting with a latent space.


