# Lab 4 : Stable Diffusion Models

---

## Objective
In this lab, you will:
1. Understand how neural networks can approximate the inverse of a function.
2. Explore diffusion models and their applications in image generation.
3. Implement a basic diffusion model to generate images.

---

## Part 1: Preliminary Activity - Neural Network for Function Inversion Concept
Neural networks can approximate complex functions, but can they learn to estimate the inverse
of a function? We'll test this by training a model to approximate the inverse of y=sin(x), i.e.,
x=arcsin(y).

>Check the Stable Diffusion Models.ipynb file

---

## Part 2: Diffusion Models on Images
Diffusion models learn to generate images by gradually denoising a noisy input. We'll implement
a simple diffusion model to understand the process.

>Check the Stable Diffusion Models.ipynb file

---

## Final Reflection

### How does function inversion relate to diffusion models?

**Answer:**  
Function inversion in the context of diffusion models involves reversing the noise-adding process. During training, the model learns how data is corrupted step by step (the forward process). During sampling, the model “inverts” this function by removing noise iteratively, effectively reconstructing the original image from a noisy input. This reversal is key to generating new samples that match the distribution of the training data.

### How does iterative noise removal help generate realistic images?

**Answer:**  
By gradually removing noise at each step, the model refines the image progressively. Early steps remove large-scale distortions, while later steps focus on subtle details. This staged denoising process allows the model to carefully reconstruct features, leading to crisp and coherent results. Each iteration uses the learned knowledge of the data distribution to guide the noise removal, producing more realistic images over time.

### Potential applications of diffusion models (e.g., text-to-image generation like Stable Diffusion)

**Answer:**  
Diffusion models are versatile and can be applied to a wide range of tasks. They excel in text-to-image generation, where they transform text prompts into high-quality images, as demonstrated by systems like Stable Diffusion. Beyond that, they can be used for image inpainting (filling in missing parts), super-resolution (enhancing image quality), and other creative applications such as style transfer or artistic image generation.

*This code is not perfect and has room for improvement and adaptation. Feel free to modify it to suit your needs and explore the fascinating world of GEN AI.*