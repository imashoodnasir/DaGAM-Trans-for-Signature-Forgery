
# DaGAM-Trans: Dual Graph Attention Module-based Transformer for Offline Signature Forgery Detection

This repository contains a PyTorch implementation of **DaGAM-Trans**, a novel deep learning model that combines Vision Transformers (ViT) with dual Graph Attention Modules for robust offline signature verification and forgery detection.

## 🔍 Overview

DaGAM-Trans is designed to:
- Convert signature images into patch embeddings for ViT processing.
- Apply dual graph attention modules:
  - **Node-level GAT** on patch tokens.
  - **Channel-level GAT** for channel-wise semantic reasoning.
- Use **Global Multi-head Self-Attention Pooling (GMSAPool)** for graph-level aggregation.
- Enhance classification accuracy with a final prediction layer.

---

## 📁 Project Structure

```bash
├── 1_dataset_preparation.py       # Load and patchify grayscale signature images
├── 2_patch_embedding.py           # Patch embedding and positional encoding
├── 3_multihead_self_attention.py  # ViT-style multi-head attention block
├── 4_graph_attention_node.py      # Node-level graph attention network
├── 5_graph_attention_channel.py   # Channel-level attention via GAT
├── 6_gmsa_pooling.py              # Global pooling using multi-head attention
├── 7_dagam_trans_model.py         # Integrated DaGAM-Trans model
├── 8_train_eval.py                # Training and evaluation scripts
```

---

## 🛠 Requirements

- Python 3.8+
- PyTorch 1.11+
- torchvision
- einops
- numpy
- PIL

Install dependencies:

```bash
pip install torch torchvision einops numpy pillow
```

---

## 🚀 Training

Prepare your dataset and update `1_dataset_preparation.py` with your image directory.

Run the training script:

```bash
python 8_train_eval.py
```

---

## 📊 Evaluation Metrics

The model supports the following metrics for signature verification:

- FAR: False Acceptance Rate
- FRR: False Rejection Rate
- EER: Equal Error Rate
- Accuracy
