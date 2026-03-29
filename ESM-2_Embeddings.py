#!/usr/bin/env python
import os
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# ==========================================
# CONFIGURATION
# ==========================================
# Use relative paths for portability and privacy
INPUT_FASTA = "./data/protein_sequences.fasta"
OUTPUT_EMBEDDINGS = "./data/embeddings.npy"
MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
BATCH_SIZE = 8  
# ==========================================

def generate_embeddings():
    """
    Generates per-protein embeddings using ESM-2 model with Mean Pooling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    model.to(device)
    model.eval() 
    print("Model loaded successfully.")

    try:
        sequences = [str(record.seq) for record in SeqIO.parse(INPUT_FASTA, "fasta")]
    except FileNotFoundError:
        print(f"ERROR: FASTA file not found at: {INPUT_FASTA}")
        return
        
    print(f"Loaded {len(sequences)} sequences.")
    
    all_embeddings = []
    start_time = time.time()

    print("Processing batches...")
    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_seqs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean Pooling logic
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        # Result: per-protein embedding
        mean_embeddings = sum_embeddings / sum_mask
        all_embeddings.append(mean_embeddings.cpu())

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Batch {i // BATCH_SIZE} / {len(sequences) // BATCH_SIZE} completed.")

    # Concatenate and save results
    final_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    final_embeddings_np = final_embeddings_tensor.numpy()

    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS), exist_ok=True)
    np.save(OUTPUT_EMBEDDINGS, final_embeddings_np)
    
    total_time = time.time() - start_time
    print(f"\nSuccess! 🚀")
    print(f"Final shape: {final_embeddings_np.shape}")
    print(f"Saved to: {OUTPUT_EMBEDDINGS}")
    print(f"Total time: {total_time:.2f}s")

if __name__ == "__main__":
    generate_embeddings()
