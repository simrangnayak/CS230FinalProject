"""
Fine-tune VAE with aggressive scheduled sampling to force decoder to rely on z.
Start from existing checkpoint, ramp AR ratio quickly, and maintain high KL weight.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from old_code.vae_octuples import OctupleVAE, vae_loss
from vae_training import OctupleDataset, collate_fn, load_octuples_from_folder
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# vocab sizes for each of the 8 channels
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]

# Load datasets
print("Loading training data...")
train_jazz_seqs = load_octuples_from_folder("train_octuples/jazz_octuples/")
train_classical_seqs = load_octuples_from_folder("train_octuples/classical_octuples/")
train_jazz = OctupleDataset(train_jazz_seqs)
train_classical = OctupleDataset(train_classical_seqs)
train_dataset = ConcatDataset([train_jazz, train_classical])

print("Loading validation data...")
val_jazz_seqs = load_octuples_from_folder("val_octuples/jazz_octuples/")
val_classical_seqs = load_octuples_from_folder("val_octuples/classical_octuples/")
val_jazz = OctupleDataset(val_jazz_seqs)
val_classical = OctupleDataset(val_classical_seqs)
val_dataset = ConcatDataset([val_jazz, val_classical])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")

# Load existing VAE
vae = OctupleVAE(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, device=device).to(device)
checkpoint_path = "vae_finetuned_final.pt"  # Start from previous fine-tune
if os.path.exists(checkpoint_path):
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"WARNING: {checkpoint_path} not found, trying vae_nonhierarchical.pt")
    vae.load_state_dict(torch.load("vae_nonhierarchical.pt", map_location=device))

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)  # Lower LR

# Fine-tune hyperparams: pure AR with warm-up prefix
num_epochs = 3
final_kl_weight = 0.5
kl_weight = final_kl_weight
ar_schedule_start = 0.95  # mostly AR
ar_schedule_end = 1.0     # pure AR
prefix_len = 8  # warm-up prefix length

def get_ar_ratio(epoch, step, total_steps):
    """Linearly ramp AR ratio across all fine-tune epochs."""
    global_step = epoch * total_steps + step
    total_global_steps = num_epochs * total_steps
    progress = global_step / total_global_steps
    return ar_schedule_start + progress * (ar_schedule_end - ar_schedule_start)

# Training loop
history = {"train_loss": [], "train_recon": [], "train_kl": [], "train_ar": [],
           "val_loss": [], "val_recon": [], "val_kl": [], "val_ar": []}

for epoch in range(num_epochs):
    vae.train()
    train_loss_sum, train_recon_sum, train_kl_sum, train_ar_sum = 0, 0, 0, 0
    total_steps = len(train_loader)
    
    for step, batch in enumerate(train_loader):
        x, seq_lens = batch
        x = x.to(device)
        optimizer.zero_grad()
        
        # Get AR ratio
        ar_ratio = get_ar_ratio(epoch, step, total_steps)
        use_ar = (torch.rand(1).item() < ar_ratio)
        
        # Encode
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        
        # Decode with teacher forcing or autoregressive (with warm-up prefix)
        if use_ar:
            # Use a short teacher-forced prefix to warm up hidden state
            plen = min(prefix_len, x.size(1) // 4)
            x_prefix = x[:, :plen, :] if plen > 0 else None
            outputs = vae.decode(z, seq_len=x.size(1), autoregressive=True, x_prefix=x_prefix)
        else:
            outputs = vae.decode(z, seq_len=x.size(1), x=x)
        
        # Loss
        loss, recon_loss, kl_loss = vae_loss(outputs, x, mu, logvar, KL_weight=kl_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss_sum += loss.item()
        train_recon_sum += recon_loss.item()
        train_kl_sum += kl_loss.item()
        if use_ar:
            train_ar_sum += recon_loss.item()
        
        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{total_steps}, "
                  f"AR ratio: {ar_ratio:.2f}, Loss: {loss.item():.4f}, "
                  f"Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
    
    train_loss_avg = train_loss_sum / total_steps
    train_recon_avg = train_recon_sum / total_steps
    train_kl_avg = train_kl_sum / total_steps
    train_ar_avg = train_ar_sum / max(1, sum(1 for _ in train_loader if torch.rand(1).item() < get_ar_ratio(epoch, 0, total_steps)))
    
    # Validation with AR decoding
    vae.eval()
    val_loss_sum, val_recon_sum, val_kl_sum, val_ar_sum = 0, 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x, seq_lens = batch
            x = x.to(device)
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
            
            # Teacher-forced loss
            outputs_tf = vae.decode(z, seq_len=x.size(1), x=x)
            loss_tf, recon_tf, kl_tf = vae_loss(outputs_tf, x, mu, logvar, KL_weight=kl_weight)
            
            # AR loss
            outputs_ar = vae.decode(z, seq_len=x.size(1), autoregressive=True)
            loss_ar, recon_ar, _ = vae_loss(outputs_ar, x, mu, logvar, KL_weight=kl_weight)
            
            val_loss_sum += loss_tf.item()
            val_recon_sum += recon_tf.item()
            val_kl_sum += kl_tf.item()
            val_ar_sum += recon_ar.item()
    
    val_loss_avg = val_loss_sum / len(val_loader)
    val_recon_avg = val_recon_sum / len(val_loader)
    val_kl_avg = val_kl_sum / len(val_loader)
    val_ar_avg = val_ar_sum / len(val_loader)
    
    history["train_loss"].append(train_loss_avg)
    history["train_recon"].append(train_recon_avg)
    history["train_kl"].append(train_kl_avg)
    history["train_ar"].append(train_ar_avg)
    history["val_loss"].append(val_loss_avg)
    history["val_recon"].append(val_recon_avg)
    history["val_kl"].append(val_kl_avg)
    history["val_ar"].append(val_ar_avg)
    
    print(f"Epoch {epoch+1} complete: Train Loss={train_loss_avg:.4f}, Val Loss={val_loss_avg:.4f}, "
          f"Val AR Recon={val_ar_avg:.4f}, Val KL={val_kl_avg:.4f}")
    
    # Save checkpoint
    torch.save(vae.state_dict(), f"vae_finetuned_v2_epoch{epoch+1}.pt")

# Save final model and history
torch.save(vae.state_dict(), "vae_finetuned_v2_final.pt")
with open("finetune_v2_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("Fine-tuning v2 complete. Model saved to vae_finetuned_v2_final.pt")
