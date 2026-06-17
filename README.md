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

and recreates all major components including:

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

---

# 🔍 Understanding Vision Transformers

Traditional CNNs process images using:

```text
Convolution Layers
Pooling Layers
Feature Maps
```

Vision Transformers instead treat an image as a sequence of patches.

For example:

```text
32 × 32 Image
```

can be divided into:

```text
16 × 16 patches
```

Result:

```text
4 image patches
```

Each patch becomes a visual token analogous to a word token in NLP.

---

# 🧩 Patch Embedding Layer

The first stage converts image patches into learnable vector representations.

## Patch Extraction

Given:

```text
Image Size = 32 × 32
Patch Size = 16 × 16
```

Number of patches:

```text
(32 × 32)/(16 × 16)
=
4 patches
```

Each patch is flattened and projected into:

```text
d_model dimensional space
```

using a learnable linear projection.

---

# 🎟️ CLS Token

A learnable classification token is prepended to the patch sequence.

```text
[CLS] Patch1 Patch2 Patch3 Patch4
```

The CLS token acts as a global representation of the image.

After passing through all transformer layers, the final CLS embedding is used for classification.

---

# 📍 Positional Encoding

Transformers have no inherent understanding of spatial position.

Positional embeddings are therefore added to each token.

```text
Token Embedding
+
Position Embedding
=
Final Input Representation
```

This allows the model to preserve spatial information.

---

# 🧠 Multi-Head Self Attention

The core innovation of the Transformer architecture is self-attention.

Each patch learns:

* What other patches matter
* Where important information exists
* Which regions influence classification

---

## Query-Key-Value Mechanism

For every patch:

```text
Input
 ├── Query
 ├── Key
 └── Value
```

Attention score:

```text
Softmax(QKᵀ / √dk)
```

This determines how much focus one patch should place on another.

---

# 🎯 Attention Head Implementation

The project implements custom attention heads from scratch.

Each head learns different visual relationships such as:

* Shape
* Texture
* Edges
* Digit structure

Multiple attention heads allow the model to learn diverse representations simultaneously.

---

# 🔄 Multi-Head Attention

Multiple attention heads operate in parallel.

```text
Head 1
Head 2
Head 3
Head 4
      │
      ▼
Concatenate
      │
      ▼
Linear Projection
```

This enables richer feature extraction.

---

# ⚙️ Transformer Encoder Block

Each encoder block consists of:

```text
LayerNorm
   ↓
Multi Head Attention
   ↓
Residual Connection
   ↓
LayerNorm
   ↓
Feed Forward Network
   ↓
Residual Connection
```

---

## Feed Forward Network

Architecture:

```text
Linear
  ↓
GELU
  ↓
Linear
```

The implementation uses:

```text
r_mlp = 4
```

which expands the hidden representation before projecting it back.

---

# 🧪 Model Configuration

```python
d_model = 32
n_heads = 4
n_layers = 3
patch_size = (16,16)
img_size = (32,32)
batch_size = 128
epochs = 20
learning_rate = 0.0001
```

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
