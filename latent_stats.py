import os
import torch
from torch.utils.data import DataLoader

from vae_octuples import OctupleVAE
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
from vae_training import OctupleDataset, load_octuples_from_folder, collate_fn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZES = [256, 128, 129, 256, 128, 33, 128, 49]
MODEL_PATH = os.environ.get("VAE_MODEL_PATH", "vae_hierarchical_large.pt")  # override with env var if needed

SPLITS = {
    "train_jazz": "train_octuples",
    "train_classical": "train_octuples/classical_octuples",
    "val_jazz": "val_octuples",
    "val_classical": "val_octuples/classical_octuples",
    "test_jazz": "test_octuples",
    "test_classical": "test_octuples/classical_octuples",
}

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
    print(f"Loading VAE model from {MODEL_PATH}")
    vae = OctupleVAE_HierarchicalDecoder(vocab_sizes=VOCAB_SIZES, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=DEVICE)
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    vae.to(DEVICE)
    vae.eval()

    # Load octuple lists per split
    loaded = {name: load_octuples_from_folder(path) for name, path in SPLITS.items()}
    print("Loaded splits:")
    for k, v in loaded.items():
        print(f"  {k}: {len(v)} pieces")

    latents = {}
    for name, octuples in loaded.items():
        print(f"Encoding {name}...")
        latents[name] = gather_latents(vae, octuples)
        s = stats(latents[name])
        print(f"  {name} stats -> count={s['count']} mean={s['mean']:.3f} std={s['std']:.3f} range=[{s['min']:.3f},{s['max']:.3f}]")

    # Distance summaries jazz->classical per split
    for split in ["train", "val", "test"]:
        jazz_key = f"{split}_jazz"
        classical_key = f"{split}_classical"
        if jazz_key in latents and classical_key in latents:
            print(f"Computing nearest classical distance summary for {split} split...")
            dist_summary = pairwise_distance_summary(latents[jazz_key], latents[classical_key])
            print(f"  {split} nearest classical distances: mean={dist_summary['mean_nearest']:.3f} std={dist_summary['std_nearest']:.3f} min={dist_summary['min_nearest']:.3f} max={dist_summary['max_nearest']:.3f}")

    # Overlap check: cosine similarities sample
    import torch.nn.functional as F
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

    print("\nDone. Use these stats to verify latent space is non-collapsed (std not ~0, ranges reasonable).")

if __name__ == "__main__":
    main()
