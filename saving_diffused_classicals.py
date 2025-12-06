import torch
import sys
import os
import random

from musicbert.preprocess import encoding_to_MIDI
from old_code.vae_octuples import OctupleVAE
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
from vae_training import load_octuples_from_folder, OctupleDataset
from musicbert.preprocess import encoding_to_MIDI, str_to_encoding

# this script runs an example classical to jazz style transfer using latent diffusion
# assumes we have diffuser models already trained and saved as .pt files

from MLPdiffuser import (
    MLPDiffuser, 
    get_index_from_list, 
    betas, 
    sqrt_one_minus_alphas_cumprod, 
    sqrt_recip_alphas, 
    posterior_variance,
    T
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vocab sizes for each of the 8 channels
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]

# diffuse all classical latents to jazz latents and save them in a folder called diffused_classicals


@torch.no_grad()
def sample_timestep(x, t, model, content_latent):
    """Calls model to predict noise in x at timestep t, and returns denoised x at t-1.
    x: [batch, latent_dim]
    t: [batch] long tensor
    content_latent: [batch, latent_dim]
    """
    betas_t = get_index_from_list(betas.to(device), t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod.to(device), t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas.to(device), t, x.shape)

    # Predict noise using content conditioning
    noise_pred = model(x, t, content_latent)
    
    # Compute mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance.to(device), t, x.shape)
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
@torch.no_grad()
def denoise_latent(style_latents, classical_latents, diffuser, num_steps=None):
    """Denoises the style_latents conditioned on classical_latents using DDPM sampling.
    style_latents: [batch, latent_dim] - pure noise initially
    classical_latents: [batch, latent_dim] - content conditioning
    """
    if num_steps is None:
        num_steps = T
    
    x = style_latents
    
    # print(f"Running {num_steps} denoising steps...")
    for i in reversed(range(0, num_steps)):
        # if i % 50 == 0:
        #     print(f"  Step {num_steps-i}/{num_steps}")
        t = torch.full((x.size(0),), i, device=device, dtype=torch.long)
        x = sample_timestep(x, t, diffuser, classical_latents)
    
    return x



def main():
    print("\nStarting diffusion-based style transfer...")
    torch.manual_seed(42)

    # Load all classical composer latents and combine them, then take 10% for diffusion
    latent_dim = 128
    
    # Load trained diffuser model first
    diffuser = MLPDiffuser(time_embed_dim=128, in_dim=latent_dim, out_dim=latent_dim)
    diffuser.load_state_dict(torch.load("diffuser_params/mlp_diffuser_best_small.pt", map_location=device))
    diffuser.to(device)
    diffuser.eval()

    # Create output directory
    os.makedirs("diffused_classicals", exist_ok=True)
    
    # Define all classical composer files (train, val, test)
    composer_files = {
        "bach": ["latents/train_bach_latents.pt", "latents/val_bach_latents.pt", "latents/test_bach_latents.pt"],
        "beethoven": ["latents/train_beethoven_latents.pt", "latents/val_beethoven_latents.pt", "latents/test_beethoven_latents.pt"],
        "brahms": ["latents/train_brahms_latents.pt", "latents/val_brahms_latents.pt", "latents/test_brahms_latents.pt"],
        "cambini": ["latents/train_cambini_latents.pt", "latents/val_cambini_latents.pt", "latents/test_cambini_latents.pt"],
        "dvorak": ["latents/train_dvorak_latents.pt", "latents/val_dvorak_latents.pt", "latents/test_dvorak_latents.pt"],
        "faure": ["latents/train_faure_latents.pt", "latents/val_faure_latents.pt", "latents/test_faure_latents.pt"],
        "haydn": ["latents/train_haydn_latents.pt", "latents/val_haydn_latents.pt", "latents/test_haydn_latents.pt"],
        "mozart": ["latents/train_mozart_latents.pt", "latents/val_mozart_latents.pt", "latents/test_mozart_latents.pt"],
        "ravel": ["latents/train_ravel_latents.pt", "latents/val_ravel_latents.pt", "latents/test_ravel_latents.pt"],
        "schubert": ["latents/train_schubert_latents.pt", "latents/val_schubert_latents.pt", "latents/test_schubert_latents.pt"]
    }
    
    # Process each composer
    for composer_name, file_paths in composer_files.items():
        print(f"\nProcessing composer: {composer_name}")
        
        # Load and combine all splits for this composer
        all_latents = []
        total_sequences = 0
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                latents = torch.load(file_path).to(device)
                all_latents.append(latents)
                total_sequences += latents.size(0)
                print(f"  Loaded {latents.size(0)} sequences from {file_path}")
        
        if not all_latents:
            print(f"  No files found for {composer_name}, skipping...")
            continue
            
        # Combine all latents
        combined_latents = torch.cat(all_latents, dim=0)

        # print total sequences
        print(f"  Total sequences for {composer_name}: {total_sequences}")
        
        # Take 10% for diffusion (minimum 5 sequences)
        num_to_diffuse = max(5, total_sequences // 10)
        
        # Randomly select sequences to diffuse
        indices = torch.randperm(total_sequences)[:num_to_diffuse]
        selected_latents = combined_latents[indices]
        
        print(f"  Diffusing {num_to_diffuse}/{total_sequences} sequences ({(num_to_diffuse/total_sequences)*100:.1f}%)")
        
        diffused_latents = []
        for i in range(selected_latents.size(0)):
            # if i % 10 == 0:
            #     print(f"    Processing sequence {i+1}/{num_to_diffuse}")
                
            z_classical = selected_latents[i:i+1, :]  # [1, latent_dim]

            # random gaussian noise vector as style latent - same shape as content latents
            z_jazz_start = torch.randn(1, latent_dim).to(device)

            # denoise to get jazz latent
            z_jazz = denoise_latent(z_jazz_start, z_classical, diffuser, num_steps=300)  # [1, latent_dim]

            diffused_latents.append(z_jazz)
        
        diffused_latents_tensor = torch.cat(diffused_latents, dim=0)  # [num_sequences, latent_dim]
        save_path = f"diffused_classicals/{composer_name}_diffused_latents.pt"
        torch.save(diffused_latents_tensor.cpu(), save_path)
        print(f"  Saved diffused latents for {composer_name} to {save_path}")

if __name__ == "__main__":
    main()

