import os
import torch
import glob
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
from vae_training import OctupleDataset, load_octuples_from_folder, collate_fn


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZES = [256, 128, 129, 256, 128, 33, 128, 49]
MODEL_PATH = os.environ.get("VAE_MODEL_PATH", "vae_hierarchical_params/vae_hierarchical_finetune_best.pt")

LATENTS_DIR = "latents"

# Map filenames to split names
LATENT_FILES = {
    "train_jazz": os.path.join(LATENTS_DIR, "train_jazz_latents.pt"),
    "train_classical": os.path.join(LATENTS_DIR, "train_classical_latents.pt"),
    "val_jazz": os.path.join(LATENTS_DIR, "val_jazz_latents.pt"),
    "val_classical": os.path.join(LATENTS_DIR, "val_classical_latents.pt"),
    "test_jazz": os.path.join(LATENTS_DIR, "test_jazz_latents.pt"),
    "test_classical": os.path.join(LATENTS_DIR, "test_classical_latents.pt"),
}

def plot_pca(latents_dict, title="PCA of Latent Vectors", save_path="pca_latents.png"):
    """Plot PCA of latent vectors colored by genre (jazz vs classical)"""
    
    # Concatenate all latents and create color labels
    all_latents = []
    colors = []
    
    for key, latents in latents_dict.items():
        all_latents.append(latents)
        # Blue for jazz, red for classical
        color = 'blue' if 'jazz' in key else 'red'
        colors.extend([color] * latents.size(0))
    
    all_latents = torch.cat(all_latents, dim=0)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_latents.cpu().numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Separate jazz and classical for proper legend
    jazz_mask = np.array([c == 'blue' for c in colors])
    classical_mask = np.array([c == 'red' for c in colors])
    
    ax.scatter(reduced[jazz_mask, 0], reduced[jazz_mask, 1], c='blue', alpha=0.5, label='Jazz', s=30)
    ax.scatter(reduced[classical_mask, 0], reduced[classical_mask, 1], c='red', alpha=0.5, label='Classical', s=30)
    
    ax.set_title(title)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved PCA plot to {save_path}")


def gather_latents(vae, octuples_list, batch_size=64):
    dataset = OctupleDataset(octuples_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    mus = []
    with torch.no_grad():
        for batch, seq_lens in loader:
            batch = batch.to(DEVICE)
            mu, logvar = vae.encode(batch)
            mus.append(mu.cpu())
    return torch.cat(mus, dim=0)  # [N, latent_dim]

def stats(t: torch.Tensor):
    return {
        "count": t.size(0),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
    }

def pairwise_distance_summary(a: torch.Tensor, b: torch.Tensor, sample_cap=2000):
    # random subsample if huge
    if a.size(0) > sample_cap:
        a = a[torch.randperm(a.size(0))[:sample_cap]]
    if b.size(0) > sample_cap:
        b = b[torch.randperm(b.size(0))[:sample_cap]]
    # compute distances jazz->nearest classical
    dists = []
    for v in a:  # each jazz latent
        diff = b - v.unsqueeze(0)
        dist = torch.norm(diff, dim=1).min().item()
        dists.append(dist)
    d = torch.tensor(dists)
    return {
        "mean_nearest": d.mean().item(),
        "std_nearest": d.std().item(),
        "min_nearest": d.min().item(),
        "max_nearest": d.max().item(),
    }

def main():
    print(f"Loading latents from {LATENTS_DIR}/...")
    
    # Check if latents directory exists
    if not os.path.exists(LATENTS_DIR):
        print(f"Error: {LATENTS_DIR}/ directory not found")
        print("Please run saving_latents.py first to generate latent files")
        return
    
    # Load all latent files
    latents = {}
    for name, filepath in LATENT_FILES.items():
        if os.path.exists(filepath):
            latents[name] = torch.load(filepath, map_location=DEVICE)
            s = stats(latents[name])
            print(f"  {name}: count={s['count']} mean={s['mean']:.3f} std={s['std']:.3f} range=[{s['min']:.3f},{s['max']:.3f}]")
        else:
            print(f"  ⚠️  {name}: NOT FOUND ({filepath})")
    
    if not latents:
        print("Error: No latent files found. Please run saving_latents.py first")
        return
    
    print(f"\nLoaded {len(latents)} latent files\n")

    # Distance summaries jazz->classical per split
    for split in ["train", "val", "test"]:
        jazz_key = f"{split}_jazz"
        classical_key = f"{split}_classical"
        if jazz_key in latents and classical_key in latents:
            print(f"Computing nearest classical distance summary for {split} split...")
            dist_summary = pairwise_distance_summary(latents[jazz_key], latents[classical_key])
            print(f"  {split} nearest classical distances: mean={dist_summary['mean_nearest']:.3f} std={dist_summary['std_nearest']:.3f} min={dist_summary['min_nearest']:.3f} max={dist_summary['max_nearest']:.3f}")

    # Overlap check: cosine similarities sample
    
    for split in ["train", "val", "test"]:
        jazz_key = f"{split}_jazz"
        classical_key = f"{split}_classical"
        if jazz_key in latents and classical_key in latents:
            jazz_sample = latents[jazz_key][:256]
            classical_sample = latents[classical_key][:256]
            # Normalize
            jazz_norm = F.normalize(jazz_sample, dim=1)
            classical_norm = F.normalize(classical_sample, dim=1)
            sims = torch.mm(jazz_norm, classical_norm.T)  # [J,C]
            max_sims, _ = sims.max(dim=1)
            print(f"{split} cosine similarity to nearest classical (mean={max_sims.mean().item():.3f}, std={max_sims.std().item():.3f}, min={max_sims.min().item():.3f}, max={max_sims.max().item():.3f})")

    # PCA plot
    print("Generating PCA plot of latents...")
    plot_pca(latents, title="PCA of Latent Vectors (Jazz vs Classical)", save_path="pca_latents.png")

    print("\nDone. Use these stats to verify latent space is non-collapsed (std not ~0, ranges reasonable).")

if __name__ == "__main__":
    main()
