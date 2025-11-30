import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vae_octuples import OctupleVAE
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
import numpy as np
import os
from vae_training import OctupleDataset, load_octuples_from_folder, collate_fn

# Run VAE to get latents for Jazz and Classical pieces
train_jazz_octuples = load_octuples_from_folder("train_octuples")
train_classical_octuples = load_octuples_from_folder("train_octuples/classical_octuples")
val_jazz_octuples = load_octuples_from_folder("val_octuples")
val_classical_octuples = load_octuples_from_folder("val_octuples/classical_octuples")
test_jazz_octuples = load_octuples_from_folder("test_octuples")
test_classical_octuples = load_octuples_from_folder("test_octuples/classical_octuples")

print(f"Loaded {len(train_jazz_octuples)} Jazz training pieces")

# Hyperparameters
batch_size = 64
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]

# Create datasets and dataloaders
train_jazz_dataset = OctupleDataset(train_jazz_octuples)
train_classical_dataset = OctupleDataset(train_classical_octuples)
val_jazz_dataset = OctupleDataset(val_jazz_octuples)
val_classical_dataset = OctupleDataset(val_classical_octuples)
test_jazz_dataset = OctupleDataset(test_jazz_octuples)
test_classical_dataset = OctupleDataset(test_classical_octuples)

train_jazz_loader = DataLoader(train_jazz_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
train_classical_loader = DataLoader(train_classical_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_jazz_loader = DataLoader(val_jazz_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_classical_loader = DataLoader(val_classical_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_jazz_loader = DataLoader(test_jazz_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_classical_loader = DataLoader(test_classical_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=device)
checkpoint = torch.load("vae_hierarchical_finetune_best.pt", map_location=device)
vae.load_state_dict(checkpoint["model"])
vae.to(device)
vae.eval()

with torch.no_grad():
    # Obtain latents for Jazz pieces
    jazz_latents = []
    for batch, seq_lens in train_jazz_loader:
        batch = batch.to(device)
        mu, logvar = vae.encode(batch)
        jazz_latents.append(mu.cpu())
    jazz_latents = torch.cat(jazz_latents, dim=0)

    # Obtain latents for Classical pieces
    classical_latents = []
    for batch, seq_lens in train_classical_loader:
        batch = batch.to(device)
        mu, logvar = vae.encode(batch)
        classical_latents.append(mu.cpu())
    classical_latents = torch.cat(classical_latents, dim=0)

    # Obtain latents for Jazz validation pieces
    val_jazz_latents = []
    for batch, seq_lens in val_jazz_loader:
        batch = batch.to(device)
        mu, logvar = vae.encode(batch)
        val_jazz_latents.append(mu.cpu())
    val_jazz_latents = torch.cat(val_jazz_latents, dim=0)

    # Obtain latents for Classical validation pieces
    val_classical_latents = []
    for batch, seq_lens in val_classical_loader:
        batch = batch.to(device)
        mu, logvar = vae.encode(batch)
        val_classical_latents.append(mu.cpu())
    val_classical_latents = torch.cat(val_classical_latents, dim=0)

    # Obtain latents for Jazz test pieces
    test_jazz_latents = []
    for batch, seq_lens in test_jazz_loader:
        batch = batch.to(device)
        mu, logvar = vae.encode(batch)
        test_jazz_latents.append(mu.cpu())
    test_jazz_latents = torch.cat(test_jazz_latents, dim=0)

    # Obtain latents for Classical test pieces
    test_classical_latents = []
    for batch, seq_lens in test_classical_loader:
        batch = batch.to(device)
        mu, logvar = vae.encode(batch)
        test_classical_latents.append(mu.cpu())
    test_classical_latents = torch.cat(test_classical_latents, dim=0)

# Create latents directory and save latents to disk
os.makedirs("latents", exist_ok=True)
torch.save(jazz_latents, "latents/train_jazz_latents.pt")
torch.save(classical_latents, "latents/train_classical_latents.pt")
torch.save(val_jazz_latents, "latents/val_jazz_latents.pt")
torch.save(val_classical_latents, "latents/val_classical_latents.pt")
torch.save(test_jazz_latents, "latents/test_jazz_latents.pt")
torch.save(test_classical_latents, "latents/test_classical_latents.pt")