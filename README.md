<div align="center">

# 🖼️ Vision Transformer (ViT) From Scratch in PyTorch

### Re-implementing the Vision Transformer Architecture from the Groundbreaking Google Research Paper

![ViT Architecture](assets/vit_architecture.png)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)]()
[![Transformer](https://img.shields.io/badge/Architecture-Vision_Transformer-orange)]()
[![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-green)]()
[![Research](https://img.shields.io/badge/Type-Paper_Implementation-purple)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

</div>

---

# 📌 Overview

This project presents a complete implementation of the Vision Transformer (ViT) architecture from scratch using PyTorch.

Unlike conventional Convolutional Neural Networks (CNNs), Vision Transformers process images as sequences of visual tokens, enabling the Transformer architecture originally designed for Natural Language Processing to be applied directly to image recognition tasks.

The implementation closely follows the seminal research paper:

**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**

and recreates all major components, including:

* Patch Embeddings
* Positional Encoding
* CLS Token
* Multi-Head Self Attention
* Transformer Encoder Blocks
* Classification Head

The model is trained and evaluated on the MNIST handwritten digit dataset.

---


# 📖 Research Paper

Paper:

https://arxiv.org/abs/2010.11929

Authors:

* Alexey Dosovitskiy
* Lucas Beyer
* Alexander Kolesnikov
* Dirk Weissenborn
* Xiaohua Zhai
* Thomas Unterthiner
* Mostafa Dehghani
* Matthias Minderer
* Georg Heigold
* Sylvain Gelly
* Jakob Uszkoreit
* Neil Houlsby

Google Research
# 🖼️ Vision Transformer (ViT) From Scratch in PyTorch

<div align="center">

<img src="assets/vit_architecture.png" width="900"/>

**Figure 1:** Original Vision Transformer (ViT) architecture proposed by Dosovitskiy et al. (2020)

</div>
---

# 🎯 Project Objective

The objective of this project is to understand and implement the internal mechanics of Vision Transformers rather than relying on pretrained libraries.

The implementation recreates the architecture manually to demonstrate:

* Transformer-based image processing
* Self-attention mechanisms
* Patch tokenization
* Positional embeddings
* Multi-head attention
* Deep learning model construction from first principles

---

# 🏗️ Model Architecture

## Vision Transformer Pipeline

```text
Input Image
      │
      ▼
Patch Generation
      │
      ▼
Patch Embeddings
      │
      ▼
Add CLS Token
      │
      ▼
Add Positional Encoding
      │
      ▼
Transformer Encoder Stack
      │
      ▼
CLS Token Representation
      │
      ▼
Classification Head
      │
      ▼
Predicted Class
```
# 🧠 Understanding the Vision Transformer Architecture

Unlike Convolutional Neural Networks (CNNs), which process local image regions using convolutional kernels, Vision Transformers treat an image as a sequence of visual tokens and learn global relationships using self-attention.

The architecture can be divided into five major components:

---

## 1️⃣ Patch Extraction

The first challenge is converting an image into a format suitable for Transformers.

Given an image of size:

```text
32 × 32 × 1
```

and a patch size of:

```text
16 × 16
```

the image is divided into non-overlapping patches.

```text
┌─────┬─────┐
│ P1  │ P2  │
├─────┼─────┤
│ P3  │ P4  │
└─────┴─────┘
```

Each patch becomes an independent visual token.

### Why is this necessary?

Transformers are designed to process sequences.

By splitting an image into patches, the image becomes analogous to a sentence:

```text
Image  →  Sequence of Patches

Sentence → Sequence of Words
```

This allows the Transformer architecture to operate on images.

---

## 2️⃣ Patch Embedding Layer

Each image patch is flattened into a vector.

Example:

```text
16 × 16 Patch
      ↓
256 Values
```

These values are then projected into a higher-dimensional feature space using a learnable linear layer.

```text
Flattened Patch
      ↓
Linear Projection
      ↓
Embedding Vector
```

### Purpose

The embedding layer converts raw pixel information into a dense representation that the Transformer can learn from.

Without embeddings, the model would only see isolated pixel values rather than meaningful visual features.

---

## 3️⃣ CLS Token

A learnable classification token is added to the beginning of the patch sequence.

```text
[CLS] P1 P2 P3 P4
```

The CLS token acts as a global information collector.

During self-attention:

* It interacts with all patches
* Receives information from all image regions
* Learns a compact representation of the entire image

After the final Transformer layer:

```text
CLS Token
     ↓
MLP Head
     ↓
Prediction
```

### Why not use all patch embeddings?

Using a single CLS token creates a fixed-size representation regardless of image size.

This simplifies classification.

---

## 4️⃣ Positional Embeddings

Transformers do not inherently understand spatial structure.

For example:

```text
Patch A
Patch B
```

and

```text
Patch B
Patch A
```

appear identical to the model.

To solve this problem, learnable positional embeddings are added.

```text
Patch Embedding
+
Position Embedding
=
Input Token
```

### Why is this important?

Position embeddings allow the model to understand:

* Left vs Right
* Top vs Bottom
* Spatial arrangements
* Object structure

Without positional information, the image would effectively become a shuffled collection of patches.

---

## 5️⃣ Multi-Head Self Attention

This is the core innovation behind Vision Transformers.

Instead of using convolutional kernels, the model learns which patches should influence one another.

For every token:

```text
Input
 ├── Query
 ├── Key
 └── Value
```

Attention scores are computed using:

```text
Attention(Q,K,V)
=
Softmax(QKᵀ / √d)
V
```

### Intuition

Suppose the image contains the digit:

```text
8
```

A patch containing the top loop may need information from a patch containing the bottom loop.

Self-attention allows these distant patches to communicate directly.

CNNs require many convolution layers to achieve the same global receptive field.

---

## 6️⃣ Multi-Head Attention

Instead of learning one attention pattern, multiple attention heads operate simultaneously.

```text
Head 1 → Shape Features

Head 2 → Edges

Head 3 → Texture

Head 4 → Structural Patterns
```

Outputs from all heads are concatenated and projected.

### Benefit

Different heads learn different visual relationships.

This creates richer image representations.

---

## 7️⃣ Transformer Encoder Block

The encoder block is the fundamental building block of the Vision Transformer.

Each block contains:

```text
LayerNorm
      ↓
Multi-Head Attention
      ↓
Residual Connection
      ↓
LayerNorm
      ↓
Feed Forward Network
      ↓
Residual Connection
```

### Residual Connections

Residual connections help:

* Prevent vanishing gradients
* Improve optimization
* Enable deeper networks

### Layer Normalization

Normalization stabilizes training and improves convergence.

---

## 8️⃣ Feed Forward Network (MLP)

After attention, every token is independently processed through a Multi-Layer Perceptron.

```text
Linear
   ↓
GELU
   ↓
Linear
```

The MLP allows the network to learn nonlinear transformations beyond attention.

---

## 9️⃣ Classification Head

After all Transformer layers:

```text
CLS Token
     ↓
Linear Layer
     ↓
Softmax
     ↓
Digit Prediction
```

The output probabilities correspond to the ten MNIST classes:

```text
0,1,2,3,4,5,6,7,8,9
```

The class with the highest probability becomes the final prediction.

---

# 📊 Why Vision Transformers Matter

Vision Transformers introduced a paradigm shift in computer vision by demonstrating that:

* Convolutions are not strictly necessary for image recognition.
* Self-attention can model long-range visual dependencies.
* Transformers can outperform CNNs when trained on sufficient data.

This architecture inspired numerous follow-up models including:

* DeiT
* Swin Transformer
* BEiT
* ViT-H
* DINOv2
* Segment Anything Model (SAM)

making Vision Transformers one of the most influential developments in modern computer vision.

---



# 📂 Dataset

## MNIST

The model is trained on:

```text
70,000 handwritten digit images
```

Classes:

```text
0 - 9
```

Image Size:

```text
28 × 28
```

Resized To:

```text
32 × 32
```

---

# 🚀 Training Pipeline

```text
MNIST Dataset
      │
      ▼
Image Resize
      │
      ▼
Patch Embedding
      │
      ▼
Vision Transformer
      │
      ▼
Cross Entropy Loss
      │
      ▼
Adam Optimizer
      │
      ▼
Parameter Updates
```

---

# 📊 Training Components

## Loss Function

```python
CrossEntropyLoss()
```

Used for multi-class classification.

---

## Optimizer

```python
Adam
```

Learning Rate:

```python
1e-4
```

---

# 📈 Evaluation

Model performance is evaluated on the MNIST test set.

Metrics:

* Accuracy
* Classification performance
* Generalization capability

---

# 🔬 Key Learnings

This project demonstrates practical understanding of:

### Computer Vision

* Image processing
* Patch tokenization
* Visual representations

### Deep Learning

* Neural network construction
* PyTorch implementation
* Training pipelines

### Transformer Architecture

* Self Attention
* Multi Head Attention
* Positional Encoding
* Encoder Blocks

### Research Reproduction

* Paper implementation
* Architecture recreation
* Experimental validation

---

# 📦 Installation

```bash
git clone https://github.com/yourusername/vision-transformer-from-scratch.git

cd vision-transformer-from-scratch
```

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

---

# 📁 Repository Structure

```text
vision-transformer-from-scratch/

│
├── assets/
│   └── vit_architecture.png
│
├── notebooks/
│   └── Building_a_Vision_transformer_from_scratch.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 🔮 Future Improvements

* CIFAR-10 Training
* CIFAR-100 Training
* Tiny ImageNet Experiments
* DeiT Implementation
* Swin Transformer Implementation
* Attention Visualization Maps
* Transfer Learning Support
* Hybrid CNN-ViT Architectures

---

# 👨‍💻 Author

## Soham Dutta

Electronics & Communication Engineering

Machine Learning • Deep Learning • Computer Vision • Transformers • AI Research

---

# ⭐ Acknowledgements

This implementation is inspired by Google's original Vision Transformer paper:

"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

and serves as an educational reproduction to understand transformer-based computer vision architectures from first principles.

If you found this project useful, consider giving it a ⭐ on GitHub.
