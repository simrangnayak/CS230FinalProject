import torch
from torch.utils.data import DataLoader, Dataset
from vae_octuples import OctupleVAE, vae_loss
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
import os
from glob import glob

# --- Dataset ---
class OctupleDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

# --- Collate function for padding ---
def collate_fn(batch):
    seq_lens = [len(seq) for seq in batch]
    max_len = max(seq_lens)
    padded = []
    for seq in batch:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, seq.shape[1]), dtype=torch.long)
            padded.append(torch.cat([seq, pad_tensor], dim=0))
        else:
            padded.append(seq)
    return torch.stack(padded), torch.tensor(seq_lens)

# --- Load octuples from folder ---
def load_octuples_from_folder(folder_path, tokens_per_note=8):
    sequences = []
    txt_files = glob(os.path.join(folder_path, '*.txt'))
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = [t for t in tokens if '<s>' not in t and '</s>' not in t]
                if not tokens:
                    continue
                encoding = [int(t.split('-')[1].rstrip('>')) for t in tokens]
                octuples = [tuple(encoding[i:i+tokens_per_note])
                            for i in range(0, len(encoding), tokens_per_note)]
                sequences.append(octuples)
    return sequences

# --- Hyperparameters ---
batch_size = 64
learning_rate = 1e-3
epochs = 3
train_dir = 'train_octuples'  # Folder containing octuple text files
test_dir = 'test_octuples'
val_dir = 'val_octuples' 

# --- Load dataset ---
train_sequences = load_octuples_from_folder(train_dir)
test_sequences = load_octuples_from_folder(test_dir)
val_sequences = load_octuples_from_folder(val_dir)
print(f"Loaded {len(train_sequences)} sequences for training.")
print(f"Loaded {len(test_sequences)} sequences for testing.")
print(f"Loaded {len(val_sequences)} sequences for validation.")
train_dataset = OctupleDataset(train_sequences)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataset = OctupleDataset(val_sequences)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataset = OctupleDataset(test_sequences)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# --- Instantiate model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]
vae = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=4, device=device)
vae = vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# --- Training loop ---
for epoch in range(epochs):
    vae.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    print(f"--- Epoch {epoch+1}/{epochs} ---")
    for batch_idx, (batch, seq_lens) in enumerate(train_dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs, mu, logvar = vae(batch)
        loss, recon_loss, kl_loss = vae_loss(outputs, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(train_dataloader)} | "
                  f"Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")

    print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/len(train_dataloader):.4f} "
          f"| Recon Loss: {total_recon/len(train_dataloader):.4f} | KL Loss: {total_kl/len(train_dataloader):.4f}")
    
    # --- Validation ---
    vae.eval()
    val_loss = 0
    val_recon = 0
    val_kl = 0
    with torch.no_grad():
        for batch, seq_lens in val_dataloader:
            batch = batch.to(device)
            outputs, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss = vae_loss(outputs, batch, mu, logvar)
            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()
        
    print(f"Validation | Total Loss: {val_loss/len(val_dataloader):.4f} "
          f"| Recon Loss: {val_recon/len(val_dataloader):.4f} | KL Loss: {val_kl/len(val_dataloader):.4f}")
    

# --- Testing ---
vae.eval()
test_loss = 0
test_recon = 0
test_kl = 0
with torch.no_grad():
    for batch, seq_lens in test_dataloader:
        batch = batch.to(device)
        outputs, mu, logvar = vae(batch)
        loss, recon_loss, kl_loss = vae_loss(outputs, batch, mu, logvar)
        test_loss += loss.item()
        test_recon += recon_loss.item()
        test_kl += kl_loss.item()

print(f"Test | Total Loss: {test_loss/len(test_dataloader):.4f} "
      f"| Recon Loss: {test_recon/len(test_dataloader):.4f} | KL Loss: {test_kl/len(test_dataloader):.4f}")