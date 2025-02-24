# Lab: Exploring Transformers for Natural Language Processing and Vision Tasks

---

## Objective

In this lab, you will:

1. Understand the basic architecture of the Transformer model (attention mechanism, positional encoding, encoder-decoder structure).
2. Implement a Transformer for an NLP task, such as text classification or machine translation.
3. Apply a Vision Transformer (ViT) for image classification.
4. Reflect on the differences between Transformer architectures for text processing and vision.

---

## Part 1: Theoretical Overview

### 1. Fundamental Components of Transformers

#### a. Background and Reference

Summary of the main concepts introduced in the research paper [“Attention Is All You Need”](https://arxiv.org/pdf/1706.03762).

---

### 2. The Self-Attention Mechanism

#### 2.1 Principle

Self-attention (or “intra-attention”) allows each position in a sequence (word, sub-word, token) to “look at” all other positions in the sequence to extract the most relevant information. Unlike recurrent neural networks (RNNs) that process sequences iteratively, self-attention operates in one matrix step by matching **all** positions with **all** others.

Each token is represented by a vector (embedding) and, to compute attention, three matrices are created: **Q** (Query), **K** (Key), and **V** (Value), each obtained by a linear transformation of the initial embedding.

#### 2.2 Calculation

The basis for calculating the self-attention matrix is the “Scaled Dot-Product Attention”:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr) \, V
$$

- **Q** has dimensions $(n \times d_k)$ for $n$ tokens with a key dimension $d_k$.
- **K** has dimensions $(n \times d_k)$.
- **V** has dimensions $(n \times d_v)$.

Dividing by $\sqrt{d_k}$ stabilizes the gradients by preventing overly large dot products.

#### 2.3 Simplified Example

Assume a sequence of 3 tokens $\{t_1, t_2, t_3\}$. After converting into embeddings, you obtain vectors of dimension $d_{model}$. Each embedding is then projected to obtain $Q$, $K$, and $V$ (for example with $d_k = 2$):

$$
Q = \begin{bmatrix}
q_{11} & q_{12} \\
q_{21} & q_{22} \\
q_{31} & q_{32}
\end{bmatrix}
$$

$$
K = \begin{bmatrix}
k_{11} & k_{12} \\
k_{21} & k_{22} \\
k_{31} & k_{32}
\end{bmatrix}
$$

$$
V = \begin{bmatrix}
v_{11} & v_{12} \\
v_{21} & v_{22} \\
v_{31} & v_{32}
\end{bmatrix}
$$

The product $QK^\top$ results in a $3 \times 3$ matrix. After normalization by $\sqrt{d_k}= \sqrt{2}$ and applying a row-wise softmax, you obtain the attention weights used to combine $V$ linearly.

---

### 3. Multi-Head Attention

#### 3.1 Principle

Instead of using a single attention matrix, the Transformer employs multiple attention heads in parallel (multi-head attention). Each head performs its own self-attention calculation with distinct linear projections, enabling the extraction of various correlations (e.g., syntax, anaphora, long-range dependencies).

#### 3.2 Calculation

For $h$ heads, $h$ self-attention calculations are performed, with matrices $Q$, $K$, and $V$ projected into smaller subspaces. For example, with $d_{model}=512$ and $h=8$, each head works with vectors of size $d_k = \frac{512}{8} = 64$. The outputs of the heads are then concatenated, and a final projection is applied:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) \; W^O,
$$

where 

$$
\text{head}_i = \text{Attention}(Q\,W_i^Q,\;K\,W_i^K,\;V\,W_i^V).
$$

The matrices $W_i^Q, W_i^K, W_i^V$ are specific to each head, and $W^O$ performs the final projection.

#### 3.3 Example

With 8 heads, each head operates in a space of dimension 64. 
For instance, one head may learn to capture subject-verb dependencies 
while another focuses on punctuation or rare words. 
Each head outputs a matrix of size `n × d_v`. 
After concatenation (resulting in a size `n × (h × d_v)`), 
a matrix `W^O` (of size `(h × d_v) × d_model`) 
is used to map back to the `d_model` dimension.


---

### 4. Positional Encoding

#### 4.1 The Order Problem

Transformers, lacking convolutions or recurrence, do not have an inherent notion of sequence order. To allow the model to differentiate, for example, a token at the beginning of a sentence from one at the end, positional encoding is added to the embeddings.

#### 4.2 Sine/Cosine Form

The method proposed in the paper uses sinusoids of increasing frequencies:

$$
\text{PE}(pos,\,2i) = \sin\Bigl(\frac{pos}{10000^{\,\frac{2i}{d_{model}}}}\Bigr)
\quad;\quad
\text{PE}(pos,\,2i+1) = \cos\Bigl(\frac{pos}{10000^{\,\frac{2i}{d_{model}}}}\Bigr),
$$

where $pos$ represents the position (0, 1, 2, …) and $i$ is the dimension index. Each dimension of the position vector follows a different sinusoid. These values are then added to the token embedding to create a unique positional signature.

The sinusoids are constructed so that their wavelengths follow a geometric progression, ranging from $2\pi$ to $10000 \cdot 2\pi$. This progression allows each dimension to capture positional information at different scales.

#### 4.3 Role and Benefits of Sinusoidal Encoding

Using sinusoids enables the model to capture relative relationships between tokens. For an offset $k$, the encoding of position $pos + k$ can be expressed as a linear combination of $PE(pos)$. This facilitates relative attention and makes learning dependency relationships in the sequence more effective.

For example, in a 10-token sentence, position 0 receives a specific vector, position 1 a shifted vector, and so on. This mechanism allows the model to differentiate tokens by their position, even if they are identical words.

---

### 5. Feedforward Layers (FFN)

#### 5.1 Principle

After each attention layer, the Transformer applies a fully connected feedforward network independently to each position in the sequence. For each token (with dimension $d_{model}$), two linear transformations are applied, separated by a non-linear activation (often ReLU).

#### 5.2 General Form

The classic formula is:

$$
\text{FFN}(x) = \max(0,\,xW_1 + b_1)\,W_2 + b_2.
$$

- $W_1$ has dimensions $(d_{model} \times d_{ff})$ and $W_2$ has dimensions $(d_{ff} \times d_{model})$.
- Typically, $d_{ff}$ is larger than $d_{model}$ (for example, 2048 vs 512) to increase the network's capacity.

#### 5.3 Example

If $d_{model}=512$ and $d_{ff}=2048$:
1. Multiply by $W_1$ (512 × 2048), add $b_1$, and apply the ReLU activation.
2. Multiply by $W_2$ (2048 × 512) and add $b_2$.

Each token is transformed identically using the same parameters.

---

## Key Points

The Transformer relies on these mechanisms:

- **Self-attention** to capture global dependencies between tokens.
- **Multi-head attention** to extract different types of relationships simultaneously.
- **Positional encoding** to incorporate the notion of order.
- **Feedforward layers** to locally enrich each token’s representation.

---

## Connection Between Convolution and Attention Mechanisms

### 1. Projection into Non-Orthonormal Spaces

In attention mechanisms, three linear projections are defined:

- **$W^k$** to project the embedding into the “key space”,
- **$W^v$** to project into the “value space”,
- **$W^q$** to project into the “query space”.

These projections transform the initial embedding (token or image patch) into a subspace relevant for computing similarity via dot product. Notably, **$W^k$** shapes the space for better discrimination in the dot product.

### 2. The Link with Convolution

#### 2.1 Classical Convolution

In a 2D convolution, for a patch $\mathcal{P}_{i,j}$ around position $(i, j)$, the weighted sum is:

```math
\mathrm{Convolution}(\mathcal{P}_{i,j})
= \sum_{(k,l)\in \mathcal{P}_{i,j}}
a_{i,j,(k,l)} \, V(k,l).
```

- $\mathcal{P}_{i,j}$ represents the pixels in the neighborhood around $(i, j)$.
- The coefficients $a_{i,j,(k,l)}$ come from the convolution kernel.
- $V(k,l)$ represents the value (or embedding) of the pixel at $(k, l)$.

Each position $(i,j)$ obtains a linear combination of the local neighborhood values.

#### 2.2 Transition to Attention

With self-attention applied to an image, the image is first divided into patches (as in a Vision Transformer). The attention mechanism computes, for each patch, a linear combination of all other patches. Thus, if patch $i$ is very similar to patch $j$ (based on the dot product between keys and queries), the weight will be high and $j$’s contribution will be significant.

#### 2.3 Similarity and Weighting

- In **convolution**, the weights $a_{i,j,(k,l)}$ are fixed by the kernel and applied uniformly across the image (weight sharing).
- In **attention**, the weights are dynamically computed via $\langle Q_i, K_j \rangle$ and softmax, allowing extraction of non-local relationships.

### Conceptual Summary

- **Convolution**: Computes a local neighborhood with a weighted sum.
- **Self-attention**: Computes a global neighborhood where weights are derived from the dot product, capturing long-range relationships.

The matrices **$W^k$**, **$W^v$**, and **$W^q$** guide the attention so that each patch can interact meaningfully with all others.

> **Note:**  
> In non-orthonormal spaces, **$W^k$** is used. Conversely, in the orthonormal case, there is no obligation to use **$W^k$** since **$W^v$** can serve as both **v** and **k**.  
> One can determine whether the space is orthonormal by exploring the relationships between variables. **$W^k$** thus promotes the use of the dot product.

---

## The [CLS] Token for Classification

### 1. What is [CLS]?

1. **Special Token:** The `[CLS]` token (for classification) is inserted at the beginning of the sequence.
2. **Aggregated Representation:** Through the Transformer layers, this token captures the relevant information from all tokens.
3. **Classification Head:** At the end, the representation associated with `[CLS]` is used by one or more MLP layers to generate a score or probability.

### 2. How to Classify Using `[CLS]`?

- **Formation of a Global Embedding:** During training, the model learns to condense essential information into the `[CLS]` token.
- **Attention Mechanism:** The `[CLS]` token “attends” to all other tokens and integrates their information via self-attention.
- **Supervision:** In fine-tuning for a classification task (e.g., sentiment analysis), the output of `[CLS]` is connected to a classification layer (often with a softmax function) and the model is optimized using a loss function (cross-entropy).

### 3. Why Does It Work?

1. **Global Position:** The constant presence of the `[CLS]` token helps the model understand that it must extract a condensed representation.
2. **Freedom of Attention:** The self-attention mechanism allows the model to capture information from the entire sequence.
3. **Multi-task Efficiency:** This standard approach (used in BERT, RoBERTa, etc.) efficiently handles classification or regression tasks.

**Summary:** The `[CLS]` token provides a unique, aggregated representation of the sequence for classification or regression through attention dynamics and supervised learning.

---

## Variants of Transformers

- **BERT (Bidirectional Encoder Representations from Transformers)**  
  A language model based on the Transformer architecture, pre-trained on large volumes of text. BERT captures bidirectional context and, through fine-tuning, adapts to various NLP tasks (sentence classification, question answering, sentiment analysis, etc.).

- **Vision Transformers (ViT)**  
  The application of Transformers to the vision domain. ViT divides the image into patches, each treated as a token with positional encoding to preserve spatial information. Through self-attention, the model learns to extract and combine relevant visual features for tasks such as image classification.

---

## Part 2: Implementing a Transformer for NLP

>Check the Exploring Transformers for Natural Language Processing and Vision Tasks.ipynb file

## Part 3: Applying Vision Transformers (ViT)

>Check the Exploring Transformers for Natural Language Processing and Vision Tasks.ipynb file

## Part 4: Reflection and Discussion

Discussion Points:
1. What are the differences in how Transformers process text versus images?  
   Transformers process text by converting tokens into embeddings enriched with positional information to maintain sequential order, whereas for images, the approach involves dividing the image into linearized patches, each assigned a suitable positional embedding to preserve spatial structure.
2. How does the self-attention mechanism adapt to different data modalities?  
   The self-attention mechanism evaluates the similarities between all elements of a sequence whether words or image patches and adapts by incorporating context specific features of each modality, enabling the extraction of complex and relevant dependencies.
3. What are the limitations of Transformers, and how can they be mitigated?  
   Main limitations include quadratic complexity with respect to sequence length, heavy reliance on large volumes of data for optimal training, and sensitivity to initial biases; these challenges can be mitigated by using approximate attention techniques, pre-training on diverse corpora, and integrating hybrid mechanisms such as convolutional layers to better capture local structures.

*This code is not perfect and has room for improvement and adaptation. Feel free to modify it to suit your needs and explore the fascinating world of GEN AI.*