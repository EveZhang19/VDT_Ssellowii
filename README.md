# _Selaginella sellowii_ Gene Regulatory Networks Reveal Distinct Transcriptional Strategies for Dehydration Stress and Recovery

This repository provides the computational framework used to decode the regulatory blueprint of vegetative desiccation tolerance (VDT) in the resurrection plant *Selaginella sellowii*. 

## 📝 Abstract

Vegetative desiccation tolerance enables resurrection plants to withstand extreme water loss. By employing time-series RNA-seq and a GNN-based computational pipeline, we identified two distinct strategies for dehydration tolerance:
1. **Intermediate Dehydration**: Active damage control via complex transcriptional responses.
2. **Extreme Dehydration**: A transition to a minimalist, stable transcriptome.

Our network analysis identified **12 persistent regulators** that bridge quiescence and recovery, undergoing context-specific functional reprogramming to drive cellular reconstruction.

---

## 🚀 Features

- **Multi-modal Integration**: Combines time-series transcriptomics with sequence-based protein embeddings (ESM-2).
- **GATv2 Architecture**: Employs Graph Attention Networks for high-accuracy regulatory link prediction.
- **Workflow Automation**: Scripts for embedding extraction, data preprocessing, and model training/evaluation.
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

