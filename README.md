# Dessert Classification

## 1. Introduction

The **Dessert Classification** project aims to classify images of popular desserts into five categories: **cannoli**, **donuts**, **pancakes**, **tiramisu**, and **waffles**. This project is an extension of the [Replicating-ViT project](https://github.com/abhayrokkam/replicating-vit) and uses two different approaches for classification:

1. **Training a custom Vision Transformer (ViT) model from scratch.**
2. **Fine-tuning a pre-trained Vision Transformer (ViT) model from `torchvision`.**

### 1.1. Highlights

- Classifying five different types of desserts.
- Exploring two training techniques: from-scratch and fine-tuning.
- Using a custom built vision transformer model for training from scratch.
- Explore the training results for both the techniques.

---

- [Dessert Classification](#dessert-classification)
  - [1. Introduction](#1-introduction)
    - [1.1. Highlights](#11-highlights)
  - [2. Training and Fine-tuning](#2-training-and-fine-tuning)
    - [2.1. Hyperparameters](#21-hyperparameters)
    - [2.2. Difference In Model Architecture](#22-difference-in-model-architecture)
      - [2.2.1. Training from Scratch](#221-training-from-scratch)
      - [2.2.2. Fine-tuning](#222-fine-tuning)

---

## 2. Training and Fine-tuning

### 2.1. Hyperparameters

In this project, we leverage the Vision Transformer (ViT) architecture, specifically the ViT-B/16 model. This variant uses a patch size of 16, which is the most basic configuration described in the original [Dosovitskiy et. al.](https://arxiv.org/abs/2010.11929) paper on Vision Transformers.

Below are the key hyperparameters for the ViT-B/16 model:

```python
# Hyperparameters for the ViT-B/16 model
PATCH_SIZE = (16, 16)  # Size of each image patch
NUM_PATCHES = 196      # Total number of patches (height / patch_size) * (width / patch_size)

EMBED_DIMS = 768       # Dimensionality of the embeddings
NUM_ATTN_HEADS = 12    # Number of attention heads in the multi-head attention mechanism
RATIO_HIDDEN_MLP = 4   # Hidden layer expansion ratio for MLP in the MLP block of encoder and classifier head
NUM_ENC_BLOCKS = 12    # Number of encoder blocks in the transformer

```

These hyperparameters define the core structure and functionality of the Vision Transformer used in this project. `EMBED_DIMS` corresponds to the embedding dimension, while `NUM_ATTN_HEADS` defines the number of attention heads used in each transformer block. The `RATIO_HIDDEN_MLP` controls the size of the hidden layer in the classifier, and `NUM_ENC_BLOCKS` specifies how many transformer blocks are used in the encoder part of the model.

### 2.2. Difference In Model Architecture

Both models in this project share the same Vision Transformer architecture. However, the key distinction lies in how the classification head is implemented, which is determined by the training approach used. Look at the text given below from the ViT paper.

> The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

#### 2.2.1. Training from Scratch

When training the model from scratch, the classification head is implemented with a multi-layer perceptron (MLP) consisting of two linear layers. This allows the model to learn richer feature representations during training, providing more flexibility and capacity to learn from the data.

```python
# Trianing from scratch: classifier head
classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=embed_dims,
                    out_features=ratio_hidden_mlp*embed_dims),
    torch.nn.Linear(in_features=ratio_hidden_mlp*embed_dims,
                    out_features=out_dims)
)
```

Here, the first layer expands the `EMBED_DIMS` by a factor of `RATIO_HIDDEN_MLP`, and the second layer maps the result to the desired number of `out_dims`.

#### 2.2.2. Fine-tuning

When fine-tuning a pre-trained Vision Transformer, the classification head is simplified to a single linear layer. This is because the model has already learned useful representations from its pre-training, so only a minimal adjustment is needed during fine-tuning.

```python
# Fine-tuning: classifier head
classifier = torch.nn.Linear(in_features=embed_dims,
                             out_features=out_dims)
```

This simplified classifier head significantly reduces the complexity of the model during fine-tuning, making it faster and more efficient while still leveraging the powerful feature representations learned from pre-training.

---