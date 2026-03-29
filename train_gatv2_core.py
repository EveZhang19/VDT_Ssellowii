#!/usr/bin/env python
import os
import torch
import numpy as np
import pandas as pd
from torch.nn import Linear, Sequential, ReLU

from torch_geometric.nn import GATv2Conv 
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURATION
# ==========================================
MODE = "DEHYDRATION" # Options: "DEHYDRATION" or "REHYDRATION"

# Use relative paths for portability and privacy
DATA_DIR = "./data"
if MODE == "DEHYDRATION":
    print(f"--- CONFIGURING FOR DEHYDRATION GATv2 MODEL ---")
    EXPRESSION_PATH = os.path.join(DATA_DIR, "expression_dehy.csv")
    EDGE_PATH = os.path.join(DATA_DIR, "edges_dehy.csv")
    PLM_PATH = os.path.join(DATA_DIR, "embeddings_dehy.npy")
    OUTPUT_DIR = "output_dehy"
    
elif MODE == "REHYDRATION":
    print(f"--- CONFIGURING FOR REHYDRATION GATv2 MODEL ---")
    EXPRESSION_PATH = os.path.join(DATA_DIR, "expression_rehy.csv")
    EDGE_PATH = os.path.join(DATA_DIR, "edges_rehy.csv")
    PLM_PATH = os.path.join(DATA_DIR, "embeddings_rehy.npy")
    OUTPUT_DIR = "output_rehy"

# Hyperparameters
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
ATTENTION_HEADS = 8
DROPOUT_RATE = 0.25
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
EPOCHS = 300
PATIENCE = 30
# ==========================================

class PCAEmbedder:
    def __init__(self, n_components=4, use_standardized=True):
        self.n_components = n_components
        self.use_standardized = use_standardized
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def fit_transform(self, expression_df):
        exp_data = expression_df.iloc[:, 1:].values
        n_features = exp_data.shape[1]
        if self.n_components > n_features:
            self.n_components = n_features
            self.pca = PCA(n_components=n_features)

        if self.use_standardized:
            exp_data = self.scaler.fit_transform(exp_data)
        
        return self.pca.fit_transform(exp_data) 


class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.25):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1)

        self.mlp_decoder = Sequential(
            Linear(out_channels * 2, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1)
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_feat = torch.cat([src, dst], dim=-1)
        return self.mlp_decoder(edge_feat).view(-1)

def prepare_data():
    print("Loading datasets...")
    try:
        plm_features = torch.tensor(np.load(PLM_PATH), dtype=torch.float)
    except FileNotFoundError:
        print(f"File not found: {PLM_PATH}")
        return None, None, None, None

    if not os.path.exists(EXPRESSION_PATH) or not os.path.exists(EDGE_PATH):
        return None, None, None, None
        
    expr_df = pd.read_csv(EXPRESSION_PATH)
    gene_list = expr_df.iloc[:, 0].tolist()
    gene_to_idx = {name: i for i, name in enumerate(gene_list)}
    
    n_time_points = expr_df.shape[1] - 1
    pca_embedder = PCAEmbedder(n_components=n_time_points) 
    pca_features = torch.tensor(pca_embedder.fit_transform(expr_df), dtype=torch.float)

    node_features = torch.cat([pca_features, plm_features], dim=1)
    
    edge_df = pd.read_csv(EDGE_PATH)
    src = edge_df.iloc[:, 0].map(gene_to_idx).values
    dst = edge_df.iloc[:, 1].map(gene_to_idx).values
    
    valid_mask = ~np.isnan(src) & ~np.isnan(dst)
    edge_index = torch.tensor([src[valid_mask].astype(int), dst[valid_mask].astype(int)], dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)
    transform = RandomLinkSplit(
        is_undirected=False, 
        num_val=0.1, num_test=0.1, 
        add_negative_train_samples=True
    )
    return transform(data) + (node_features.shape[1],)

def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index)
    loss = torch.nn.BCEWithLogitsLoss()(out, data.edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index)
    return roc_auc_score(data.edge_label.cpu(), out.cpu())

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data, val_data, test_data, in_channels = prepare_data()
    
    if train_data is None: return

    model = GATLinkPredictor(in_channels, HIDDEN_CHANNELS, OUT_CHANNELS, ATTENTION_HEADS, DROPOUT_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, train_data)
        val_auc = test(model, val_data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss {loss:.4f}, Val AUC {val_auc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val AUC: {best_val_auc:.4f}")
            break

    model.load_state_dict(torch.load(best_model_path))
    test_auc = test(model, test_data)
    print(f"\nFinal Test AUC: {test_auc:.4f}")

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(train_data.x, train_data.edge_index).cpu().numpy()
    
    emb_path = os.path.join(OUTPUT_DIR, "final_embeddings.npy")
    np.save(emb_path, final_embeddings)
    print(f"Embeddings saved to {emb_path}")

if __name__ == "__main__":
    main()
