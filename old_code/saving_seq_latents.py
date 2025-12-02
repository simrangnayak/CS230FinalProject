import os
import torch
from torch.utils.data import DataLoader
from vae_octuples import OctupleVAE
from vae_training import OctupleDataset, load_octuples_from_folder, collate_fn

# Run VAE to get latents for Jazz and Classical pieces
train_jazz_octuples = load_octuples_from_folder("train_octuples")
train_bach_octuples = 

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

vae = OctupleVAE(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, device=device)
vae.load_state_dict(torch.load("vae_nonhierarchical.pt", map_location=device))
vae.eval()

os.makedirs("latents", exist_ok=True)

def collect_seq_latents(data_loader):
    seq_latents = []  # list of tensors with variable lengths
    mu_latents = []
    with torch.no_grad():
        for batch, seq_lens in data_loader:
            batch = batch.to(device)
            h_enc, mu, logvar = vae.encode_with_sequence(batch)
            # h_enc: [batch, max_len, hidden*2]
            for i, L in enumerate(seq_lens):
                seq_latents.append(h_enc[i, :L, :].cpu())
            mu_latents.append(mu.cpu())
    mu_latents = torch.cat(mu_latents, dim=0)
    return seq_latents, mu_latents

print("Collecting train jazz latents (sequences + mu)...")
train_jazz_seq, train_jazz_mu = collect_seq_latents(train_jazz_loader)
print("Collecting train classical latents...")
train_classical_seq, train_classical_mu = collect_seq_latents(train_classical_loader)
print("Collecting val jazz latents...")
val_jazz_seq, val_jazz_mu = collect_seq_latents(val_jazz_loader)
print("Collecting val classical latents...")
val_classical_seq, val_classical_mu = collect_seq_latents(val_classical_loader)
print("Collecting test jazz latents...")
test_jazz_seq, test_jazz_mu = collect_seq_latents(test_jazz_loader)
print("Collecting test classical latents...")
test_classical_seq, test_classical_mu = collect_seq_latents(test_classical_loader)

# Save mu latents (unchanged behavior)
torch.save(train_jazz_mu, "latents/train_jazz_latents.pt")
torch.save(train_classical_mu, "latents/train_classical_latents.pt")
torch.save(val_jazz_mu, "latents/val_jazz_latents.pt")
torch.save(val_classical_mu, "latents/val_classical_latents.pt")
torch.save(test_jazz_mu, "latents/test_jazz_latents.pt")
torch.save(test_classical_mu, "latents/test_classical_latents.pt")

# Save full encoder sequences as lists
torch.save(train_jazz_seq, "latents/train_jazz_seq_latents.pt")
torch.save(train_classical_seq, "latents/train_classical_seq_latents.pt")
torch.save(val_jazz_seq, "latents/val_jazz_seq_latents.pt")
torch.save(val_classical_seq, "latents/val_classical_seq_latents.pt")
torch.save(test_jazz_seq, "latents/test_jazz_seq_latents.pt")
torch.save(test_classical_seq, "latents/test_classical_seq_latents.pt")

print("Saved sequence latents and mu latents to latents/ directory.")