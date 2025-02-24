# Lab: Exploring Transformers for Natural Language Processing and Vision Tasks

---

## Objective

In this lab, you will:

1. Understand the basic architecture of the Transformer model (attention mechanism, positional encoding, encoder-decoder structure).
2. Implement a Transformer for an NLP task, such as text classification or machine translation.
3. Apply a Vision Transformer (ViT) for image classification.
4. Reflect on the differences between Transformer architectures for text and for vision.

---

## Part 1: Theoretical Overview

### 1. The Fundamental Components of Transformers

#### a. Context and Reference

A summary of the main concepts introduced in the research paper [“Attention Is All You Need”](https://arxiv.org/pdf/1706.03762).

---

### 2. The Self-Attention Mechanism

#### 2.1 Principle

Self-attention (or "intra-attention") allows each position in a sequence (word, subword, token) to "look" at every other position in the same sequence to extract the most relevant information. Unlike recurrent networks (RNNs) that process the sequence iteratively, self-attention operates in a single matrix step by matching **all** positions with **all** others.

Each token is represented by a vector (embedding) and, to compute the attention, three matrices are created: **Q** (Query), **K** (Key), and **V** (Value), each obtained by a linear multiplication of the initial embedding.

#### 2.2 Calculation

The basis for computing a self-attention matrix is the "Scaled Dot-Product Attention":

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr) \, V
\]

- **Q** has dimensions \((n \times d_k)\) for \(n\) tokens and a key dimension \(d_k\).
- **K** has dimensions \((n \times d_k)\).
- **V** has dimensions \((n \times d_v)\).

Dividing by \(\sqrt{d_k}\) stabilizes the gradients by preventing overly large dot products.

#### 2.3 Simplified Example

Suppose a sequence of 3 tokens \(\{t_1, t_2, t_3\}\). After transformation into embeddings, we obtain vectors of dimension \(d_{model}\). Then, each embedding is projected to obtain \(Q\), \(K\), and \(V\) (for example with \(d_k = 2\)):

- \(Q = \begin{bmatrix}q_{11} & q_{12}\\[5pt] q_{21} & q_{22}\\[5pt] q_{31} & q_{32}\end{bmatrix}\)
- \(K = \begin{bmatrix}k_{11} & k_{12}\\[5pt] k_{21} & k_{22}\\[5pt] k_{31} & k_{32}\end{bmatrix}\)
- \(V = \begin{bmatrix}v_{11} & v_{12}\\[5pt] v_{21} & v_{22}\\[5pt] v_{31} & v_{32}\end{bmatrix}\)

The product \(QK^\top\) will yield a \(3 \times 3\) matrix. After normalization by \(\sqrt{d_k}= \sqrt{2}\) and applying a row-wise softmax, we obtain the attention weights that allow the linear combination with \(V\).

---

### 3. Multi-Head Attention

#### 3.1 Principle

Instead of a single attention matrix, the Transformer uses several attention heads in parallel (multi-head attention). Each head performs its own version of the self-attention calculation with distinct linear projections, enabling it to extract varied correlations (syntax, anaphora, long-range dependencies, etc.).

#### 3.2 Calculation

For \(h\) heads, \(h\) attention computations are performed, with the matrices \(Q\), \(K\), and \(V\) projected into smaller subspaces. For example, with \(d_{model}=512\) and \(h=8\), each head works on vectors of size \(d_k = \frac{512}{8} = 64\). The outputs of the heads are then concatenated and a final projection is applied:

\[
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) \; W^O,
\]
where 
\[
\text{head}_i = \text{Attention}(Q\,W_i^Q,\;K\,W_i^K,\;V\,W_i^V).
\]

The matrices \(W_i^Q, W_i^K, W_i^V\) are specific to each head, and \(W^O\) performs the final projection.

#### 3.3 Example

With 8 heads, each operates in a space of dimension 64. For example, one head may learn to capture subject-verb dependencies, while another focuses on punctuation or rare words. Each head returns an output matrix of size \(n \times d_v\). After concatenation (resulting in a size \(n \times (h \cdot d_v)\)), a matrix \(W^O\) (of size \((h \cdot d_v) \times d_{model}\)) is used to return to the \(d_{model}\) dimension.

---

### 4. Positional Encoding

#### 4.1 The Order Problem

Transformers, lacking convolutions or recurrence, do not have an inherent notion of sequential order. To allow the model to differentiate, for instance, a token at the beginning of a sentence from a token at the end, positional encoding is added to the embeddings.

#### 4.2 Sine/Cosine Form

The method proposed in the paper involves using sinusoids of increasing frequencies:

\[
\text{PE}(pos,\,2i) = \sin\Bigl(\frac{pos}{10000^{\,\frac{2i}{d_{model}}}}\Bigr)
\quad;\quad
\text{PE}(pos,\,2i+1) = \cos\Bigl(\frac{pos}{10000^{\,\frac{2i}{d_{model}}}}\Bigr),
\]

where \(pos\) represents the position (0, 1, 2, …) and \(i\) is the dimension index. Each dimension of the position vector thus follows a different sinusoid. These values are then added to the token embedding to create a unique positional signature.

These sinusoids are constructed so that their wavelengths follow a geometric progression, ranging from \(2\pi\) to \(10000 \cdot 2\pi\). This progression allows each dimension to capture different scales of positional information.

#### 4.3 Role and Advantages of Sinusoidal Encoding

Using sinusoids allows the model to capture relative relationships between tokens. Indeed, for an offset \( k \), the encoding of the position \( pos + k \) can be expressed as a linear combination of \( PE(pos) \). This facilitates relative attention, making it more effective to learn dependency relationships in the sequence.

For example, in a sentence of 10 tokens, position 0 receives a specific vector, position 1 a shifted vector, and so on. This mechanism enables the model to distinguish tokens by their position, even if they are the same words.

---

### 5. Feedforward Layers (FFN)

#### 5.1 Principle

After each attention layer, the Transformer applies a fully connected neural network (feedforward) independently to each position of the sequence. For each token (with dimension \(d_{model}\)), two linear transformations are applied, separated by a non-linear activation (often ReLU).

#### 5.2 General Form

The classic formula is:

\[
\text{FFN}(x) = \max(0,\,xW_1 + b_1)\,W_2 + b_2.
\]

- \(W_1\) is of dimension \((d_{model} \times d_{ff})\) and \(W_2\) is of dimension \((d_{ff} \times d_{model})\).
- Typically, \(d_{ff}\) is larger than \(d_{model}\) (for example, 2048 vs 512) to increase the transformation capacity.

#### 5.3 Example

If \(d_{model}=512\) and \(d_{ff}=2048\):

1. A multiplication by \(W_1\) (512 × 2048) is performed, followed by the addition of \(b_1\) and a ReLU activation.
2. A multiplication by \(W_2\) (2048 × 512) is performed, followed by the addition of \(b_2\).

Each token is transformed identically with the same parameters.

---

## Key Points

The Transformer relies on these mechanisms:

- **Self-attention** to capture global dependencies between tokens.
- **Multi-head attention** to simultaneously extract different types of relationships.
- **Positional encoding** to integrate the notion of order.
- **Feedforward layers** to locally enrich the representation of each token.

---

## The Link Between Convolution and Attention Mechanisms

### 1. Projection into Non-Orthonormal Spaces

In attention mechanisms, three linear projections are defined:

- **Wᵏ** to project the embedding into the "key space",
- **Wᵛ** to project into the "value space",
- **Wᑫ** to project into the "query space".

These projections transform the initial embedding (token or image patch) into a subspace relevant for calculating similarity via the dot product. In particular, **Wᵏ** shapes the space for better discrimination during the dot product.

### 2. The Link with Convolution

#### 2.1 Classic Convolution

In a 2D convolution, for a patch \(\mathcal{P}_{i,j}\) around the position \((i, j)\), the weighted sum is:

\[
(\text{Convolution})(\mathcal{P}_{i,j}) \;=\; \sum_{(k,l)\in \mathcal{P}_{i,j}} a_{i,j,(k,l)} \cdot V(k,l).
\]

- \(\mathcal{P}_{i,j}\) represents the pixels in the neighborhood around \((i, j)\).
- The coefficients \(a_{i,j,(k,l)}\) come from the convolution kernel.
- \(V(k,l)\) represents the value (or embedding) of the pixel \((k, l)\).

Each position \((i,j)\) thus obtains a linear combination of the values in its local neighborhood.

#### 2.2 Transition to Attention

With self-attention applied to the image, it is first divided into patches (as in a Vision Transformer). The attention mechanism computes, for each patch, a linear combination of all the other patches. Thus, if patch \(i\) is very similar to patch \(j\) (according to the dot product between keys and queries), the weighting will be high and the contribution of \(j\) will be significant in the final sum.

#### 2.3 Similarity and Weights

- In **convolution**, the weights \(a_{i,j,(k,l)}\) are fixed by the kernel and applied uniformly across the image (weight sharing).
- In **attention**, the weightings are dynamically calculated via \(\langle Q_i, K_j \rangle\) and softmax, allowing the extraction of non-local relationships.

### Conceptual Summary

- **Convolution**: Calculation of a local neighborhood with a weighted sum.
- **Self-attention**: Calculation of a global neighborhood where the weights come from the dot product, enabling the capture of long-range relationships.

The matrices **\(W^k\)**, **\(W^v\)**, and **\(W^q\)** direct the attention so that each patch can interact meaningfully with all the others.

> **Note:**  
> In cases where we are in a non-orthonormal space, **Wᵏ** is used. Conversely, in the opposite case, we are not obliged to use **Wᵏ** because **Wᵛ** serves as both **v** and **k**. One can determine whether the space is orthonormal by exploring the relationships between the variables. **Wᵏ** thus promotes the use of the dot product.

---

## The [CLS] Token for Classification

### 1. What is the [CLS] Token?

1. **Special Token**: The `[CLS]` token (for *classification*) is inserted at the beginning of the sequence.
2. **Aggregated Representation**: Through the Transformer's layers, this token captures the relevant information from all tokens.
3. **Classification Head**: At the end, the representation associated with `[CLS]` is used by one or more MLP layers to generate a score or probability.

### 2. How to Classify from `[CLS]`?

- **Formation of a Global Embedding**: During training, the model learns to condense essential information into the `[CLS]` token.
- **Attention Mechanism**: The `[CLS]` token "looks" at all the other tokens and integrates their information through self-attention.
- **Supervision**: During the fine-tuning phase for a classification task (for example, sentiment analysis), the output of `[CLS]` is connected to a classification layer (often with a softmax function) and the model is optimized using a loss function (cross-entropy).

### 3. Why Does This Work?

1. **Global Position**: The continuous presence of the `[CLS]` token allows the model to understand that it must extract a condensed representation.
2. **Attention Freedom**: The self-attention mechanism enables capturing information from the entire sequence.
3. **Multi-task Efficiency**: This standard approach (used in BERT, RoBERTa, etc.) allows for efficient handling of classification or regression tasks.

**Summary**: The `[CLS]` token provides a unique and aggregated representation of the sequence for classification or regression, thanks to the attention dynamics and supervised learning.

---

## Variants of Transformers

- **BERT (Bidirectional Encoder Representations from Transformers)**  
  A language model based on the Transformer architecture, pre-trained on large volumes of text. BERT captures bidirectional context and, via a fine-tuning phase, adapts to various NLP tasks (sentence classification, question-answering, sentiment analysis, etc.).

- **Vision Transformers (ViT)**  
  Application of the Transformer in the field of vision. ViT divides the image into patches, with each patch treated as a token with positional encoding to preserve spatial information. Through self-attention, the model learns to extract and combine relevant visual information for tasks such as image classification.

---

## Part 2: Implementing a Transformer for NLP 

>Check Exploring Transformers for Natural Language Processing and Vision Tasks.ipynb

## Part 3: Applying Vision Transformers (ViT)

>Check Exploring Transformers for Natural Language Processing and Vision Tasks.ipynb

## Part 4: Reflection and Discussion

Discussion Points:
1. What are the differences in how Transformers process text versus images?  
   Transformers process text by converting tokens into embeddings enriched with positional information to maintain sequential order, whereas for images, the approach involves dividing the image into linearized patches which are then converted into vectors, with each patch being assigned an appropriate positional embedding to preserve the spatial structure.
2. How does the self-attention mechanism adapt to different data modalities?  
   The self-attention mechanism evaluates similarities between all elements of a sequence, whether they are words or image patches, and adapts by integrating contextual characteristics specific to each modality, allowing it to extract complex and relevant dependencies in various types of data.
3. What are the limitations of Transformers, and how can they be mitigated?  
   The main limitations include quadratic complexity with respect to sequence length, heavy dependence on large volumes of data for optimal training, and sensitivity to initial biases; these challenges can be mitigated through the use of approximate attention techniques, pre-training on diverse corpora, and integrating hybrid mechanisms such as convolutional layers to better capture local structures.

---

*This code and its interpretation are not perfect; there is room for improvement and adaptation. Feel free to modify it to suit your needs and explore the fascinating world of Gen AI.*
