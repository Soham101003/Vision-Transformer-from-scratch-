This repository contains a from-scratch implementation of a Vision Transformer (ViT) in PyTorch, trained and evaluated on the MNIST handwritten digit dataset. The project focuses on understanding ViT internals, training stability, and proper evaluation metrics rather than maximizing benchmark scores.

📊 Final Results (MNIST)

Accuracy: 96.80%

Macro Precision: 0.9678

Macro Recall: 0.9678

Macro F1-score: 0.9677

Macro-averaged metrics are reported to provide a balanced evaluation across all digit classes.

**🧠 Architecture**

The full ViT pipeline is implemented from the ground up:

**Patch Embedding** — splits input images into fixed-size patches and linearly projects them into an embedding space

**Positional Encoding** — learnable 1D positional embeddings added to patch tokens

**[CLS] Token** — prepended classification token whose final state is used for prediction

**Transformer Encoder Blocks** — multi-head self-attention + MLP with LayerNorm and residual connections

**Classification Head** — MLP head on top of the [CLS] token output

🧠 Key Highlights

Custom patch embedding, positional encoding, and transformer encoder blocks

Stable training with AdamW optimizer and tuned learning rate

Analysis using training vs validation loss curves

Evaluation beyond accuracy using precision, recall, and F1-score

📁 Contents

End-to-end Vision Transformer implementation

Training and evaluation pipeline

Metric computation and loss visualization

🔗 More Details

For full implementation details, experiments, and explanations, please refer to the accompanying Colab notebook : https://colab.research.google.com/drive/1e62aw1I72fIIWo2agy_ewP5zEwYUP7HG?usp=sharing
