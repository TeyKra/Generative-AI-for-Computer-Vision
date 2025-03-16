# Generative AI for Computer Vision

This repository provides resources and labs to explore the foundations of Generative AI techniques applied to computer vision. It includes practical implementations and comparisons of state-of-the-art generative models.

## Branches

### `lab1`: VAE and GAN
In this lab, you will implement a Variational Autoencoder (VAE) to reconstruct images and explore its latent space. You will also conceptualize and implement a Generative Adversarial Network (GAN) to understand their differences and reflect on their respective strengths and weaknesses in generative modeling.

### `lab2`: GAN and Attention Mechanism
In this lab, you will explore advanced generative modeling by integrating attention mechanisms into GAN architectures. This lab is divided into two sub-labs:

- ***1. Advanced Generative Adversarial Networks (GANs):***
  1. **Implement a CNN-based GAN** to generate realistic images.
  2. **Integrate Transformer-based architectures** into the GAN framework to leverage the power of attention for generative modeling.
  3. **Compare the performance and visual outputs** of both CNN-based and Transformer-based GANs.
  4. **Reflect on the strengths and challenges** of using CNNs versus Transformers in the context of adversarial training and image generation.

- ***2. Implementing Self-Attention in TensorFlow:***
  1. **Implement a basic self-attention mechanism** using TensorFlow.
  2. **Apply the self-attention module** to a small sequence of words to observe how attention enhances feature representation.
  3. **Analyze the impact** of self-attention on model performance and interpretability in sequence modeling tasks.

### `lab3`: Transformers NLP VisionTasks
In this lab, you will dive deeper into the core architecture of the Transformer model and then explore its applications in both natural language processing and computer vision tasks.

1. **Understand the core architecture** of the Transformer model, including the attention mechanism, positional encoding, and encoder-decoder structure.
2. **Implement a Transformer for an NLP task** such as text classification or machine translation.
3. **Apply a Vision Transformer (ViT) for image classification**, understanding how Transformer concepts transfer to visual data.
4. **Reflect on the differences** between Transformer architectures in text and vision applications, discussing potential benefits and challenges in each domain.

### `lab4`: Stable Diffusion Models
In this lab, you will explore how neural networks can be used to approximate the inverse of a function, delve into the fundamental principles of diffusion models, and implement a basic diffusion model for image generation.

1. **Understand how neural networks** can approximate the inverse of a function and analyze the implications of this capability.
2. **Explore diffusion models** and investigate their applications in image generation.
3. **Implement a basic diffusion model** to generate images and compare its performance to previously explored generative models.
