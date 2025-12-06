import torch
from torch.utils.data import DataLoader, Dataset
from vae_octuples import OctupleVAE, vae_loss
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
import os
from glob import glob
import matplotlib.pyplot as plt

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

__all__ = [
    'OctupleDataset',
    'collate_fn',
    'load_octuples_from_folder'
]


def main():
    # --- Hyperparameters ---
    batch_size = 64
    epochs = 3
    embed_dim = 64
    chunks = 4
    hidden_dim = 256
    latent_dims = [32, 64, 128]  # Different latent dimensions to experiment with
    learning_rates = [1e-3]  # Different learning rates to experiment with
    KL_weights = [0.5, 1.0]  # Different KL weights to experiment with
    train_dir = 'train_octuples'
    test_dir = 'test_octuples'
    val_dir = 'val_octuples'

    # --- Load dataset ---
    train_classical = load_octuples_from_folder('train_octuples/classical_octuples')
    train_jazz = load_octuples_from_folder('train_octuples')

    val_classical = load_octuples_from_folder('val_octuples/classical_octuples')
    val_jazz = load_octuples_from_folder('val_octuples')

    test_classical = load_octuples_from_folder('test_octuples/classical_octuples')
    test_jazz = load_octuples_from_folder('test_octuples')

    train_sequences = train_jazz + train_classical
    test_sequences = test_jazz + test_classical
    val_sequences = val_jazz + val_classical
    
    # train_sequences = load_octuples_from_folder(train_dir)
    # test_sequences = load_octuples_from_folder(test_dir)
    # val_sequences = load_octuples_from_folder(val_dir)
    print(f"Loaded {len(train_sequences)} sequences for training.")
    print(f"Loaded {len(test_sequences)} sequences for testing.")
    print(f"Loaded {len(val_sequences)} sequences for validation.")

    # test_dataset = OctupleDataset(test_sequences)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Instantiate model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vocab sizes for each of the 8 channels
    vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]

# Read in datasets
    train_dataset = OctupleDataset(train_sequences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = OctupleDataset(val_sequences)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---- Training Hierarchical Model ----

    # Instantiate hierarchical model
    hierarchical_epochs = 10  # Fewer epochs for fine-tuning
    hierarchical_train_losses, hierarchical_test_losses = [], []
    hierarchical_train_recon_losses, hierarchical_train_kl_losses = [], []
    hierarchical_test_recon_losses, hierarchical_test_kl_losses = [], []
    hierarchical_vae = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=device)
    hierarchical_vae = hierarchical_vae.to(device)
    hierarchical_optimizer = torch.optim.Adam(hierarchical_vae.parameters(), lr=1e-4)  # Lower LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hierarchical_optimizer, T_max=hierarchical_epochs, eta_min=1e-5)
    start_epoch = 0

    import glob as glob_module
    # Load final trained model for fine-tuning
    ckpt_path = "vae_hierarchical_params/vae_hierarchical_final_epoch5.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading trained model: {ckpt_path}")
        hierarchical_vae.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Model loaded for fine-tuning!")
    else:
        print("No trained model found. Looking for latest checkpoint...")
        checkpoints = sorted(glob_module.glob("vae_hierarchical_large_epoch*.pt"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            hierarchical_vae.load_state_dict(torch.load(latest_checkpoint, map_location=device))
            print(f"Loaded {latest_checkpoint} for fine-tuning")
        else:
            print("No checkpoints found! Starting from scratch.")
    
    best_test_loss = float('inf')
    
    for epoch in range(start_epoch, hierarchical_epochs):
        hierarchical_vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
    
        # Fine-tuning schedule: fixed KL, very low TF
        kl_weight = 0.15  # Keep KL fixed
    
        print(f"--- Fine-tune Epoch {epoch+1}/{hierarchical_epochs} (KL={kl_weight:.3f}) ---")
        for batch_idx, (batch, seq_lens) in enumerate(train_dataloader):
            batch = batch.to(device)

            # random crop to stabilize AR
            B, L, _ = batch.size()
            max_len = 256
            if L > max_len:
                start = torch.randint(0, L - max_len, (1,)).item()
                batch = batch[:, start:start+max_len, :]
                L = max_len

            hierarchical_optimizer.zero_grad()
        
            mu, logvar = hierarchical_vae.encode(batch)
            z = hierarchical_vae.reparameterize(mu, logvar)

            # 1. Teacher forcing forward pass for main loss
            outputs_tf = hierarchical_vae.decode(z, seq_len=L, x=batch)
            tf_loss, tf_recon, tf_kl = vae_loss(outputs_tf, batch, mu, logvar, KL_weight=kl_weight)
            
            # 2. Short AR burst for AR training (every few batches)
            ar_loss_val = 0.0
            if batch_idx % 4 == 0:  # Every 4th batch
                ar_len = min(64, L // 2)  # Short AR sequence
                # Randomly select start within batch
                ar_start = torch.randint(0, L - ar_len + 1, (1,)).item()
                x_prefix_len = min(16, ar_len // 4)  # small warm-up prefix
                x_prefix = batch[:, ar_start:ar_start + x_prefix_len, :]

                outputs_ar = hierarchical_vae.decode(z, seq_len=ar_len, autoregressive=True, x_prefix=x_prefix)
                # Compare AR output to ground truth subsequence
                target_subseq = batch[:, ar_start:ar_start + ar_len, :]
                ar_loss, _, _ = vae_loss(outputs_ar, target_subseq, mu, logvar, KL_weight=0.0)  # No KL penalty for AR
                ar_loss_val = ar_loss.item()
                
                # Combined loss: main TF loss + small AR loss
                batch_loss = tf_loss + 0.1 * ar_loss
            else:
                # Only TF loss
                batch_loss = tf_loss
                ar_loss_val = 0.0
                
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(hierarchical_vae.parameters(), max_norm=1.0)
            hierarchical_optimizer.step()

            total_loss += batch_loss.item()
            total_recon += tf_recon.item()
            total_kl += tf_kl.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(train_dataloader)} | "
                    f"TF Loss: {tf_loss.item():.4f} | AR Loss: {ar_loss_val:.4f} "
                    f"| Recon: {tf_recon.item():.4f} | KL: {tf_kl:.4f}")
        
        scheduler.step()

        avg_recon = total_recon / len(train_dataloader)
        avg_kl = total_kl / len(train_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)
        hierarchical_train_losses.append(avg_train_loss)
        hierarchical_train_recon_losses.append(avg_recon)
        hierarchical_train_kl_losses.append(avg_kl)
        print(f"Hierarchical Epoch {epoch+1}/{hierarchical_epochs} | Total Loss: {total_loss/len(train_dataloader):.4f} "
              f"| Recon Loss: {total_recon/len(train_dataloader):.4f} | KL Loss: {total_kl/len(train_dataloader):.4f}")
    
        # Test evaluation with AR
        hierarchical_vae.eval()
        test_loss = 0
        test_recon = 0
        test_kl = 0
        with torch.no_grad():
            for batch, seq_lens in val_dataloader:
                batch = batch.to(device)
                mu, logvar = hierarchical_vae.encode(batch)
                z = hierarchical_vae.reparameterize(mu, logvar)
                plen = min(16, batch.size(1))
                x_prefix = batch[:, :plen, :]
                outputs = hierarchical_vae.decode(z, seq_len=batch.size(1), autoregressive=True, x_prefix=x_prefix)
                loss, recon_loss, kl_loss = vae_loss(outputs, batch, mu, logvar, KL_weight=kl_weight)
                test_loss += loss.item()
                test_recon += recon_loss.item()
                test_kl += kl_loss.item()
        
        avg_test_recon = test_recon / len(val_dataloader)
        avg_test_kl = test_kl / len(val_dataloader)
        avg_test_loss = test_loss / len(val_dataloader)
        hierarchical_test_losses.append(avg_test_loss)
        hierarchical_test_recon_losses.append(avg_test_recon)
        hierarchical_test_kl_losses.append(avg_test_kl)


        print(f"Test after Epoch {epoch+1} [AR] | Total Loss: {avg_test_loss:.4f} "
              f"| Recon Loss: {avg_test_recon:.4f} | KL Loss: {avg_test_kl:.4f}")

        # --- Write losses to CSV ---
        csv_path = "hierarchical_finetune_losses_final.csv"
        header = "epoch,train_loss,train_recon,train_kl,test_loss,test_recon,test_kl\n"
        row = f"{epoch+1},{avg_train_loss:.6f},{avg_recon:.6f},{avg_kl:.6f},{avg_test_loss:.6f},{avg_test_recon:.6f},{avg_test_kl:.6f}\n"
        if epoch == 0 and not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write(header)
                f.write(row)
        else:
            with open(csv_path, "a") as f:
                f.write(row)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"vae_hierarchical_finetune_epoch{epoch+1}_final.pt"
            torch.save({'model': hierarchical_vae.state_dict()}, ckpt_path)
            print(f"  → Fine-tune checkpoint saved: {ckpt_path}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                {"model": hierarchical_vae.state_dict(), "optimizer": hierarchical_optimizer.state_dict(), "epoch": epoch},
                "vae_hierarchical_finetune_best_final.pt",
            )
            print("  ✓ Saved new BEST model.")
    

    # save training history
    import json
    history = {
        "train_losses": hierarchical_train_losses,
        "test_losses": hierarchical_test_losses,
        "train_recon_losses": hierarchical_train_recon_losses,
        "train_kl_losses": hierarchical_train_kl_losses,
        "test_recon_losses": hierarchical_test_recon_losses,
        "test_kl_losses": hierarchical_test_kl_losses,
    }
    with open("hierarchical_finetune_history_final.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(hierarchical_vae.state_dict(), "vae_hierarchical_finetuned_final.pt")
    print("Fine-tuning complete! Model saved as vae_hierarchical_finetuned_final.pt")

if __name__ == '__main__':
    main()
