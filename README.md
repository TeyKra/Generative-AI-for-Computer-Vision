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

#### Generator and Discriminator Loss During Training
![CNN-Based GAN](images/CNN-Based%20GAN%20Loss.png)

We first observe significant fluctuations in the Generator’s loss (blue curve) as well as the Discriminator’s loss (orange curve). Over the course of the iterations, the Discriminator’s loss tends to decrease and then generally stabilizes at lower values, while the Generator’s loss remains higher and shows occasional spikes. These oscillations are normal in GAN training, because the Generator is constantly trying to fool the Discriminator, which adapts in response. The graph indicates that neither model completely dominates the other, suggesting a kind of equilibrium in their confrontation and a gradual learning process.

#### GIF showing the Generator's progress over epochs
![CNN-based GAN](gif/CNN-based%20GAN.gif)


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

#### Generator and Discriminator Loss During Training
![Transformer-based GAN](images/Transformer-based%20GAN.png)

Initially, the Generator’s loss increases sharply, indicating that it struggles to produce realistic images, while the Discriminator’s loss decreases rapidly, showing that it effectively distinguishes real images from fake ones. After this unstable phase, a relative equilibrium sets in, with typical GAN oscillations. The Generator’s loss remains generally higher, suggesting that it still has difficulty deceiving the Discriminator, but the trend indicates a gradual stabilization of the learning process.

At the beginning, both loss curves (G and D) fluctuate significantly, signaling that the Discriminator and Generator are adjusting to each other. Gradually, the Discriminator’s loss (D) decreases and stabilizes further, indicating that it is becoming more efficient at detecting fake samples. 
Meanwhile, the Generator’s loss (G) oscillates at relatively higher values, showing that it must continuously improve its samples to deceive an increasingly effective Discriminator. These oscillations demonstrate that both models push each other to improve continuously.

#### GIF showing the Generator's progress over epochs
![Transformer-based GAN](gif/Transformer-based%20GAN.gif)

---

### Part 3: GAN Model and Training

- The GAN model combines the Transformer Generator and Discriminator.
- **Objective:** Implement a complete GAN model using both components with adversarial training to improve both the generator and the discriminator.
- **Hint:** Take inspiration from the solution provided for the CNN-based GAN.

#### Generator and Discriminator Loss During Training
![GAN Model and Training](images/GAN%20Model%20and%20Training.png)

At the beginning, the Generator’s loss increases sharply, indicating that it struggles to produce convincing images. Meanwhile, the Discriminator’s loss decreases, showing that it easily detects generated images as fake.

After this initial phase, the Generator’s loss stabilizes and oscillates, while the Discriminator’s loss remains relatively low but fluctuates. This suggests that the GAN is reaching an equilibrium where the Generator gradually improves to deceive the Discriminator, while the latter adjusts its ability to distinguish real from fake images.
Towards the end of training, the losses remain within relatively stable ranges, indicating that both networks continue to refine themselves, though the oscillations reflect an ongoing competitive dynamic.

#### GIF showing the Generator's progress over epochs
![GAN Model and Training](gif/GAN%20Model%20and%20Training.gif)

---
### Conclusion Part 1,2,3
Overall, the three approaches successfully generate realistic handwritten digits on MNIST, but each has its own characteristics. The first method, based on CNNs (with transposed convolutions for the Generator and standard convolutions for the Discriminator), offers a relatively simple architecture to train and quickly produces high-quality images. The second method, which replaces CNNs with Transformers in both the Generator and the Discriminator, focuses more on global relationships (via multi-head attention); however, it can be more unstable during training, even though it ultimately manages to generate credible samples. The third method, which also employs a Transformer-based Generator–Discriminator pair, closely resembles the second in its implementation and yields visually comparable results while reaffirming the potential of Transformers for image generation. On this relatively simple dataset, the differences in quality between the generated images remain modest, but the Transformer approach demonstrates its ability to capture a global view of the image and offers promising prospects for extending to more complex tasks.

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
![Heatmap of Attention Scores](images\First%20Head.png)

The generated heatmap represents the attention scores for the first sample and the first attention head in the self-attention mechanism. Each cell in the matrix indicates the degree of attention a word (query position) pays to another word (key position) in the input sequence.

- **Color Interpretation:**
  - **Yellow:** Indicates higher attention (the word receives or gives more importance).
  - **Violet:** Indicates lower attention.

In this heatmap, the values typically range between approximately 0.07 and 0.12, suggesting a relatively balanced distribution of attention.

---

### Experimenting with Different Hyperparameters

You can test various configurations to observe their impact on the attention weight distribution:

![Heatmap of Attention Scores](images\embed32-heads2.png)
1. **Heatmap 1 (embed_dim=32, num_heads=2):**
   - The distribution is relatively homogeneous with a few peaks reaching around 0.14.
   - Certain key positions (e.g., column 8) capture more attention.

![Heatmap of Attention Scores](images\embed32-heads4.png)
2. **Heatmap 2 (embed_dim=32, num_heads=4):**
   - The distribution remains balanced with maximum values around 0.13, indicating better modulation of attention weights.

![Heatmap of Attention Scores](images\embed32-heads8.png)
3. **Heatmap 3 (embed_dim=32, num_heads=8):**
   - The attention distribution becomes even more uniform, with slightly reduced maximum values (approximately 0.13).

![Heatmap of Attention Scores](images\embed64-heads2.png)
4. **Heatmap 4 (embed_dim=64, num_heads=2):**
   - There is a more pronounced focus on specific positions, with scores exceeding 0.14, accentuating certain word relationships.

![Heatmap of Attention Scores](images\embed64-heads4.png)
5. **Heatmap 5 (embed_dim=64, num_heads=4):**
   - The distribution is more refined with a notable peak at about 0.15, improving the precision of contextual relationships.

![Heatmap of Attention Scores](images\embed64-heads8.png)
6. **Heatmap 6 (embed_dim=64, num_heads=8):**
   - The values are more balanced with maximum scores around 0.13, suggesting a good distribution of attention.

![Heatmap of Attention Scores](images\embed128-heads2.png)
7. **Heatmap 7 (embed_dim=128, num_heads=2):**
   - Increasing the embedding dimension to 128 with only 2 heads results in a more marked focus on specific positions (around 0.15), better capturing global dependencies.

![Heatmap of Attention Scores](images\embed128-heads4.png)
8. **Heatmap 8 (embed_dim=128, num_heads=4):**
   - A good balance is achieved between focus and dispersion, with high values reaching around 0.17 at certain positions.

![Heatmap of Attention Scores](images\embed128-heads8.png)
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