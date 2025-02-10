# Lab 2: GAN & Attention Mechanism

This lab is divided into two main sections:

1. [GAN](#gan)
2. [Attention Mechanism](#attention-mechanism)

---

## GAN

_All implementations and code are available in the `GAN.ipynb` notebook._

### Objectives

In this lab, you will:
- **Implement a CNN-based GAN** to generate realistic images.
- **Explore the use of Transformer-based architectures** for generative modeling.
- **Compare the performance and visual results** of different GAN architectures.
- **Reflect on the strengths and challenges** of using CNNs vs. Transformers in GANs.

---

### Part 1: CNN-based GAN

#### Instructions

1. **Implement a CNN Generator:**
   - Use **transpose convolution layers** to upsample noise into an image.
   - Use **ReLU activation** for hidden layers and **Tanh** for the output layer.

2. **Implement a CNN Discriminator:**
   - Use **convolution layers** to downsample the input image.
   - Use **LeakyReLU activation** for hidden layers and **Sigmoid** for the output layer.

3. **Train the GAN:**
   - Train the GAN on the MNIST dataset.
   - Generate images from the latent noise vector.

#### Result on each epochs
![CNN-based GAN](gif/CNN-based_GAN.gif)

---

### GAN Questions (CNN-based)

1. **What is Transpose Convolution and why do we use it in the Generator?**

   > *Answer:*  
   Transpose convolution is an operation used to increase the resolution of an image (upsampling). In a GAN generator, it transforms a low-dimensional representation (e.g., a random latent vector) into a higher-dimensional image. Unlike simple resizing, this operation learns to reconstruct detailed structures while preserving spatial information, which is crucial for generating realistic images.

2. **What are LeakyReLU and Sigmoid, and why do we use them?**

   > *Answer:*  
        - **LeakyReLU:** This is a variant of ReLU that allows a small portion of negative values to pass through (typically with a small coefficient, e.g., 0.01). It helps prevent the "dying neurons" problem by ensuring that neurons continue to update even for negative inputs. It is used in hidden layers to maintain information flow.
        - **Sigmoid:** This function compresses input values into the range \([0,1]\). It is used in the output layer (typically in the discriminator) to produce a probability estimate indicating whether an image is real or generated, making the output easily interpretable in a probabilistic framework.

3. **Read and Comment the Code:**
   - Refer to the `GAN.ipynb` notebook and comment on every line to fully understand the implementation.

---

### Part 2: Transformer-based GAN

#### Instructions

1. **Implement a Transformer Generator:**
   - Use **Multi-Head Self-Attention (MHSA)** and positional encodings.
   - Upsample the latent space using feedforward layers.

2. **Implement a Transformer Discriminator:**
   - Use **MHSA** to analyze global relationships in the input.
   - Classify images as "real" or "fake."

3. **Train the Transformer-based GAN:**
   - Train the Transformer-based GAN on the MNIST dataset.
   - Compare its performance with the CNN-based GAN.

#### Result on each epochs
![Transformer-based GAN](gif\Transformer-based GAN.gif)

---

### Part 3: GAN Model and Training

- The GAN model combines the Transformer Generator and Discriminator.
- **Objective:** Implement a complete GAN model using both components with adversarial training to improve both the generator and the discriminator.
- **Hint:** Take inspiration from the solution provided for the CNN-based GAN.

#### Result on each epochs
![GAN Model and Training](gif\GAN Model and Training.gif)

---

## Attention Mechanism

_All implementations and code are available in the `Self_Attention_mechanism.ipynb` notebook._

### Objective

Implement a basic self-attention mechanism in TensorFlow and apply it to a short sequence of words.

#### Input Details

- **Vocabulary:** `['le', 'chat', 'est', 'sur', 'le', 'tapis']`
- **Input sequence:** `['le', 'chat', 'est', 'sur', 'le', 'tapis']`
- **Input shape:** `(2, 10, 64)`
- **Output shape:** `(2, 10, 64)`

---

### Visualization with Heatmap

The generated heatmap represents the attention scores for the first sample and the first attention head in the self-attention mechanism. Each cell in the matrix indicates the degree of attention a word (query position) pays to another word (key position) in the input sequence.

- **Color Interpretation:**
  - **Yellow:** Indicates higher attention (the word receives or gives more importance).
  - **Violet:** Indicates lower attention.

In this heatmap, the values typically range between approximately 0.07 and 0.12, suggesting a relatively balanced distribution of attention.

---

### Experimenting with Different Hyperparameters

You can test various configurations to observe their impact on the attention weight distribution:

1. **Heatmap 1 (embed_dim=32, num_heads=2):**
   - The distribution is relatively homogeneous with a few peaks reaching around 0.14.
   - Certain key positions (e.g., column 8) capture more attention.

2. **Heatmap 2 (embed_dim=32, num_heads=4):**
   - The distribution remains balanced with maximum values around 0.13, indicating better modulation of attention weights.

3. **Heatmap 3 (embed_dim=32, num_heads=8):**
   - The attention distribution becomes even more uniform, with slightly reduced maximum values (approximately 0.13).

4. **Heatmap 4 (embed_dim=64, num_heads=2):**
   - There is a more pronounced focus on specific positions, with scores exceeding 0.14, accentuating certain word relationships.

5. **Heatmap 5 (embed_dim=64, num_heads=4):**
   - The distribution is more refined with a notable peak at about 0.15, improving the precision of contextual relationships.

6. **Heatmap 6 (embed_dim=64, num_heads=8):**
   - The values are more balanced with maximum scores around 0.13, suggesting a good distribution of attention.

7. **Heatmap 7 (embed_dim=128, num_heads=2):**
   - Increasing the embedding dimension to 128 with only 2 heads results in a more marked focus on specific positions (around 0.15), better capturing global dependencies.

8. **Heatmap 8 (embed_dim=128, num_heads=4):**
   - A good balance is achieved between focus and dispersion, with high values reaching around 0.17 at certain positions.

9. **Heatmap 9 (embed_dim=128, num_heads=8):**
   - The attention is uniformly distributed with a few focal points (around 0.13), demonstrating the model's flexibility in capturing complex relationships.

#### General Interpretation

- **Number of Heads:**  
  Increasing the number of heads allows the model to capture different relationships, but may dilute individual attention weights.

- **Embedding Dimension:**  
  A higher embedding dimension allows for richer contextual representations, but might result in over-concentration on specific positions if the number of heads is too low.

**Suggested Optimal Configuration:**  
Using an embedding dimension of 64 or 128 with 4 to 8 heads offers a good compromise between precision and generalization.

---

### Conclusion and Reflection

Implementing the self-attention mechanism in TensorFlow has been a highly educational experience. It involved the challenge of correctly managing tensor dimensions and splitting embeddings into multiple heads to capture contextual relationships. This process is reminiscent of the GAN lab, where using multiple heads allowed for exploring different perspectives in image generation. By experimenting with various hyperparameters, the impact on the distribution of attention weights becomes evident, enhancing the model's ability to capture significant dependencies in the data. This experience deepened the understanding of attention mechanisms and underscored the importance of fine-tuning and careful tensor management in complex architectures.

---

*This code is not perfect and has room for improvement and adaptation. Feel free to adapt it to your needs and explore the fascinating world of GEN AI.*