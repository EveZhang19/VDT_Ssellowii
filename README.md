# _Selaginella sellowii_ Gene Regulatory Networks Reveal Distinct Transcriptional Strategies for Dehydration Stress and Recovery

This repository provides the computational framework used to decode the regulatory blueprint of vegetative desiccation tolerance (VDT) in the resurrection plant *Selaginella sellowii*. 

## 📝 Abstract

Vegetative desiccation tolerance (VDT) enables resurrection plants to withstand extreme water loss. By integrating **Time-series RNA-seq**, **Ensemble Network Inference**, **Graph Attention Networks (GATs)**, and **Protein Language Models (PLMs)**, we reveal two distinct transcriptional strategies:
* **Intermediate Dehydration**: Active damage control through complex transcriptional responses.
* **Extreme Dehydration**: Transition to a minimalist, stable transcriptome.
* **Recovery**: Identification of 12 persistent regulators driving rapid cellular reconstruction through context-specific functional reprogramming.
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

```

---
## 📂 Project Structure

```text
├── data/                   # Place your input files here (FASTA, CSV, NPY)
├── output/                 # Model checkpoints and final embeddings
├── scripts/
│   ├── extract_plm.py      # ESM-2 embedding generation script
│   ├── train_gatv2_core.py        # GATv2 training and evaluation script
│   └── data_loader.py      # Data processing and method integration
└── requirements.txt        # Environment dependencies
```

