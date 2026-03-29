# GATv2-GRN: Link Prediction for Gene Regulatory Networks

This repository contains a specialized pipeline for predicting gene regulatory interactions using **GATv2 (Graph Attention Networks)** and **ESM-2 (Evolutionary Scale Modeling)** protein embeddings. 

The project integrates high-dimensional expression data with protein language model features to infer complex biological relationships.

---

## 🚀 Features
* **Protein Embeddings:** Leverages the `ESM-2` transformer model to generate functional representations of protein sequences.
* **Graph Neural Networks:** Implements `GATv2Conv` for link prediction on Gene Regulatory Networks (GRN).
* **Multi-Method Integration:** A robust `DataLoader` to aggregate predictions from various GRN inference methods (GENIE3, SWING, PEAK, etc.).
* **PCA-Enhanced Features:** Combines scaled expression time-series data with PLM embeddings for rich node characterization.

---

## 📦 Installation

### Prerequisites
* Python 3.8+
* CUDA-capable GPU (Recommended for ESM-2 embedding generation)

### Dependencies
Install the required packages via pip:
```bash
pip install torch torch-geometric pandas numpy scikit-learn biopython transformers
