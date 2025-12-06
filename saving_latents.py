import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vae_octuples import OctupleVAE
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
import numpy as np
import os
from vae_training import OctupleDataset, load_octuples_from_folder, collate_fn

def load_octuples_from_file(file_path, tokens_per_note=8):
    """Load octuples from a single txt file."""
    sequences = []
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


# Run VAE to get latents for Jazz and Classical pieces
train_jazz_octuples = load_octuples_from_folder("train_octuples")
train_bach_octuples = load_octuples_from_file("train_octuples/classical_octuples/bach_midi_train.txt")
train_beethoven_octuples = load_octuples_from_file("train_octuples/classical_octuples/beethoven_midi_train.txt")
train_brahms_octuples = load_octuples_from_file("train_octuples/classical_octuples/brahms_midi_train.txt")
train_cambini_octuples = load_octuples_from_file("train_octuples/classical_octuples/cambini_midi_train.txt")
train_dvorak_octuples = load_octuples_from_file("train_octuples/classical_octuples/dvorak_midi_train.txt")
train_faure_octuples = load_octuples_from_file("train_octuples/classical_octuples/faure_midi_train.txt")
train_haydn_octuples = load_octuples_from_file("train_octuples/classical_octuples/haydn_midi_train.txt")
train_mozart_octuples = load_octuples_from_file("train_octuples/classical_octuples/mozart_midi_train.txt")
train_ravel_octuples = load_octuples_from_file("train_octuples/classical_octuples/ravel_midi_train.txt")
train_schubert_octuples = load_octuples_from_file("train_octuples/classical_octuples/schubert_midi_train.txt")

val_jazz_octuples = load_octuples_from_folder("val_octuples")
val_bach_octuples = load_octuples_from_file("val_octuples/classical_octuples/bach_midi_valid.txt")
val_beethoven_octuples = load_octuples_from_file("val_octuples/classical_octuples/beethoven_midi_valid.txt")


test_jazz_octuples = load_octuples_from_folder("test_octuples")
test_bach_octuples = load_octuples_from_file("test_octuples/classical_octuples/bach_midi_test.txt")
test_brahms_octuples = load_octuples_from_file("test_octuples/classical_octuples/brahms_midi_test.txt")
test_cambini_octuples = load_octuples_from_file("test_octuples/classical_octuples/cambini_midi_test.txt")
test_faure_octuples = load_octuples_from_file("test_octuples/classical_octuples/faure_midi_test.txt")
test_haydn_octuples = load_octuples_from_file("test_octuples/classical_octuples/haydn_midi_test.txt")
test_mozart_octuples = load_octuples_from_file("test_octuples/classical_octuples/mozart_midi_test.txt")
test_ravel_octuples = load_octuples_from_file("test_octuples/classical_octuples/ravel_midi_test.txt")
test_schubert_octuples = load_octuples_from_file("test_octuples/classical_octuples/schubert_midi_test.txt")

print(f"Loaded {len(train_jazz_octuples)} Jazz training pieces")

# Hyperparameters
batch_size = 64
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]

# Create datasets and dataloaders for each composer
train_bach_dataset = OctupleDataset(train_bach_octuples)
train_beethoven_dataset = OctupleDataset(train_beethoven_octuples)
train_brahms_dataset = OctupleDataset(train_brahms_octuples)
train_cambini_dataset = OctupleDataset(train_cambini_octuples)
train_dvorak_dataset = OctupleDataset(train_dvorak_octuples)
train_faure_dataset = OctupleDataset(train_faure_octuples)
train_haydn_dataset = OctupleDataset(train_haydn_octuples)
train_mozart_dataset = OctupleDataset(train_mozart_octuples)
train_ravel_dataset = OctupleDataset(train_ravel_octuples)
train_schubert_dataset = OctupleDataset(train_schubert_octuples)

# Combine all classical training data
all_train_classical = (train_bach_octuples + train_beethoven_octuples + train_brahms_octuples + 
                      train_cambini_octuples + train_dvorak_octuples + train_faure_octuples + 
                      train_haydn_octuples + train_mozart_octuples + train_ravel_octuples + train_schubert_octuples)

# Create datasets and dataloaders
train_jazz_dataset = OctupleDataset(train_jazz_octuples)
train_classical_dataset = OctupleDataset(all_train_classical)
# Create validation datasets for each composer
val_bach_dataset = OctupleDataset(val_bach_octuples)
val_beethoven_dataset = OctupleDataset(val_beethoven_octuples)

# Combine validation classical data
all_val_classical = val_bach_octuples + val_beethoven_octuples
val_jazz_dataset = OctupleDataset(val_jazz_octuples)
val_classical_dataset = OctupleDataset(all_val_classical)

# Create test datasets for each composer
test_bach_dataset = OctupleDataset(test_bach_octuples)
test_brahms_dataset = OctupleDataset(test_brahms_octuples)
test_cambini_dataset = OctupleDataset(test_cambini_octuples)
test_faure_dataset = OctupleDataset(test_faure_octuples)
test_haydn_dataset = OctupleDataset(test_haydn_octuples)
test_mozart_dataset = OctupleDataset(test_mozart_octuples)
test_ravel_dataset = OctupleDataset(test_ravel_octuples)
test_schubert_dataset = OctupleDataset(test_schubert_octuples)

# Combine test classical data  
all_test_classical = (test_bach_octuples + test_brahms_octuples + test_cambini_octuples + 
                     test_faure_octuples + test_haydn_octuples + test_mozart_octuples + 
                     test_ravel_octuples + test_schubert_octuples)


test_jazz_dataset = OctupleDataset(test_jazz_octuples)
test_classical_dataset = OctupleDataset(all_test_classical)

# Create dataloaders for individual composers
train_bach_loader = DataLoader(train_bach_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_beethoven_loader = DataLoader(train_beethoven_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_brahms_loader = DataLoader(train_brahms_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_cambini_loader = DataLoader(train_cambini_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_dvorak_loader = DataLoader(train_dvorak_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_faure_loader = DataLoader(train_faure_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_haydn_loader = DataLoader(train_haydn_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_mozart_loader = DataLoader(train_mozart_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_ravel_loader = DataLoader(train_ravel_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_schubert_loader = DataLoader(train_schubert_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Create dataloaders for val/test composers
val_bach_loader = DataLoader(val_bach_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_beethoven_loader = DataLoader(val_beethoven_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

test_bach_loader = DataLoader(test_bach_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_brahms_loader = DataLoader(test_brahms_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_cambini_loader = DataLoader(test_cambini_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_faure_loader = DataLoader(test_faure_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_haydn_loader = DataLoader(test_haydn_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_mozart_loader = DataLoader(test_mozart_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_ravel_loader = DataLoader(test_ravel_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_schubert_loader = DataLoader(test_schubert_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

train_jazz_loader = DataLoader(train_jazz_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
train_classical_loader = DataLoader(train_classical_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_jazz_loader = DataLoader(val_jazz_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_classical_loader = DataLoader(val_classical_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_jazz_loader = DataLoader(test_jazz_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_classical_loader = DataLoader(test_classical_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=device)
checkpoint = torch.load("vae_hierarchical_params/vae_hierarchical_finetune_best_final.pt", map_location=device)
vae.load_state_dict(checkpoint["model"])
vae.to(device)
vae.eval()

with torch.no_grad():
    # Function to extract latents from a dataloader
    def extract_latents(dataloader, name):
        latents = []
        for batch, seq_lens in dataloader:
            batch = batch.to(device)
            mu, logvar = vae.encode(batch)
            latents.append(mu.cpu())
        return torch.cat(latents, dim=0) if latents else torch.empty(0, 128)
    
    # Extract latents for each composer
    print("Extracting latents train...")
    train_bach_latents = extract_latents(train_bach_loader, "Bach")
    train_beethoven_latents = extract_latents(train_beethoven_loader, "Beethoven")
    train_brahms_latents = extract_latents(train_brahms_loader, "Brahms")
    train_cambini_latents = extract_latents(train_cambini_loader, "Cambini")
    train_dvorak_latents = extract_latents(train_dvorak_loader, "Dvorak")
    train_faure_latents = extract_latents(train_faure_loader, "Faure")
    train_haydn_latents = extract_latents(train_haydn_loader, "Haydn")
    train_mozart_latents = extract_latents(train_mozart_loader, "Mozart")
    train_ravel_latents = extract_latents(train_ravel_loader, "Ravel")
    train_schubert_latents = extract_latents(train_schubert_loader, "Schubert")
    
    print(f"Bach: {train_bach_latents.shape[0]} pieces")
    print(f"Beethoven: {train_beethoven_latents.shape[0]} pieces")
    print(f"Brahms: {train_brahms_latents.shape[0]} pieces")
    print(f"Mozart: {train_mozart_latents.shape[0]} pieces")

    # Obtain latents for Jazz pieces
    train_jazz_latents = extract_latents(train_jazz_loader, "Jazz")

    # Obtain latents for Classical pieces (combined)
    train_classical_latents = extract_latents(train_classical_loader, "Classical")

    # Obtain latents for validation sets
    val_jazz_latents = extract_latents(val_jazz_loader, "Val Jazz")
    val_classical_latents = extract_latents(val_classical_loader, "Val Classical")

    # Extract validation latents for individual composers
    print("Extracting latents validation...")
    val_bach_latents = extract_latents(val_bach_loader, "Val Bach")
    val_beethoven_latents = extract_latents(val_beethoven_loader, "Val Beethoven")
    
    # Extract test latents for individual composers
    print("Extracting latents test...")
    test_bach_latents = extract_latents(test_bach_loader, "Test Bach")
    test_brahms_latents = extract_latents(test_brahms_loader, "Test Brahms")
    test_cambini_latents = extract_latents(test_cambini_loader, "Test Cambini")
    test_faure_latents = extract_latents(test_faure_loader, "Test Faure")
    test_haydn_latents = extract_latents(test_haydn_loader, "Test Haydn")
    test_mozart_latents = extract_latents(test_mozart_loader, "Test Mozart")
    test_ravel_latents = extract_latents(test_ravel_loader, "Test Ravel")
    test_schubert_latents = extract_latents(test_schubert_loader, "Test Schubert")

    # Obtain latents for test sets
    test_jazz_latents = extract_latents(test_jazz_loader, "Test Jazz")
    test_classical_latents = extract_latents(test_classical_loader, "Test Classical")

# Create latents directory and save latents to disk
os.makedirs("new_latents", exist_ok=True)

# Save individual composer latents - TRAINING
torch.save(train_bach_latents, "new_latents/train_bach_latents.pt")
torch.save(train_beethoven_latents, "new_latents/train_beethoven_latents.pt")
torch.save(train_brahms_latents, "new_latents/train_brahms_latents.pt")
torch.save(train_cambini_latents, "new_latents/train_cambini_latents.pt")
torch.save(train_dvorak_latents, "new_latents/train_dvorak_latents.pt")
torch.save(train_faure_latents, "new_latents/train_faure_latents.pt")
torch.save(train_haydn_latents, "new_latents/train_haydn_latents.pt")
torch.save(train_mozart_latents, "new_latents/train_mozart_latents.pt")
torch.save(train_ravel_latents, "new_latents/train_ravel_latents.pt")
torch.save(train_schubert_latents, "new_latents/train_schubert_latents.pt")

# Save individual composer latents - VALIDATION
torch.save(val_bach_latents, "new_latents/val_bach_latents.pt")
torch.save(val_beethoven_latents, "new_latents/val_beethoven_latents.pt")

# Save individual composer latents - TEST
torch.save(test_bach_latents, "new_latents/test_bach_latents.pt")
torch.save(test_brahms_latents, "new_latents/test_brahms_latents.pt")
torch.save(test_cambini_latents, "new_latents/test_cambini_latents.pt")
torch.save(test_faure_latents, "new_latents/test_faure_latents.pt")
torch.save(test_haydn_latents, "new_latents/test_haydn_latents.pt")
torch.save(test_mozart_latents, "new_latents/test_mozart_latents.pt")
torch.save(test_ravel_latents, "new_latents/test_ravel_latents.pt")
torch.save(test_schubert_latents, "new_latents/test_schubert_latents.pt")

# Save combined latents (for backward compatibility)
torch.save(train_jazz_latents, "new_latents/train_jazz_latents.pt")
torch.save(train_classical_latents, "new_latents/train_classical_latents.pt")
torch.save(val_jazz_latents, "new_latents/val_jazz_latents.pt")
torch.save(val_classical_latents, "new_latents/val_classical_latents.pt")
torch.save(test_jazz_latents, "new_latents/test_jazz_latents.pt")
torch.save(test_classical_latents, "new_latents/test_classical_latents.pt")

print("Saved individual composer latents:")
print(f"  Train Bach: {train_bach_latents.shape}")
print(f"  Train Beethoven: {train_beethoven_latents.shape}")
print(f"  Train Mozart: {train_mozart_latents.shape}")
print(f"  Val Bach: {val_bach_latents.shape}")
print(f"  Test Bach: {test_bach_latents.shape}")
print(f"  And others...")
print("Saved combined classical/jazz latents for training/val/test")