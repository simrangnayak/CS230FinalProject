# this script is a latent diffuser model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import time

# The diffusion model will be trained on our Jazz latents ($z_{\text{style}}$). For each step, we will add Gaussian noise to create a noisy latent vector $z_t$. We will pair it with the closest content latent $z^*_\text{content} =  \arg \min_{z_\text{content}} || z_{\text{style}} - z_\text{content}||^2 $, and feed both into our diffusion model. The training objective will be to predict the noise added to $z_t$ using MSE loss

# using this colab notebook as reference: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=Rj17psVw7Shg

# noise scheduler

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals while considering the batch dimension
    """
    # vals: [T]; t: [batch]; direct advanced indexing is simpler than gather
    out = vals[t]  # [batch]
    # reshape to broadcast across remaining dimensions of x_shape
    return out.view(-1, *([1] * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x0, t, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Backward compatibility helper using global precomputed schedule."""
    noise = torch.randn_like(x0)
    sqrt_alpha_bar = get_index_from_list(sqrt_alphas_cumprod.to(device), t, x0.shape)
    sqrt_one_minus = get_index_from_list(sqrt_one_minus_alphas_cumprod.to(device), t, x0.shape)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
    return x_t, noise

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.fc(x))))

class SinusoidealPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # use float constant on the correct device to avoid dtype/device issues
        factor = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -factor)
        t = time.float()
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)
    
class MLPDiffuser(nn.Module):
    def __init__(self, time_embed_dim, in_dim=128, out_dim=128, hidden_dims=[512]):
        super().__init__()

        # previous: [512, 1024, 512]
        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidealPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        # Input: noisy style (in_dim) + content conditioning (in_dim) + time embedding (time_embed_dim)
        input_size = 2 * in_dim + time_embed_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_size
        for h_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, h_dim))
            prev_dim = h_dim
        
        # Output layer (no activation, predicts noise)
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, style_latent, t, content_latent):
        # style_latent, content_latent: [batch, latent_dim]
        t_emb = self.time_mlp(t)  # [batch, time_embed_dim]
        x = torch.cat([style_latent, content_latent, t_emb], dim=1)
        return self.mlp(x)
    

def get_loss(model, style_latent, content_latent, t, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Compute denoising loss.
    style_latent: [batch, latent_dim]
    content_latent: same shape (conditioning)
    t: [batch] int timesteps
    """
    style_latent = style_latent.to(device)
    content_latent = content_latent.to(device)
    t = t.to(device)
    
    x_noisy, noise = forward_diffusion_sample(style_latent, t, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    noise_pred = model(x_noisy, t, content_latent)
    return F.mse_loss(noise_pred, noise)


class LatentDiffusionDataset(Dataset):
    def __init__(self, style_latents, content_latents):
        """
        style_latents: list of [latent_dim] float tensors
        content_latents: list of [latent_dim] float tensors
        """
        self.style_latents = style_latents
        self.content_latents = content_latents
    
    def __len__(self):
        return len(self.style_latents)
    
    def __getitem__(self, idx):
        style_latent = self.style_latents[idx]

        # find closest content latent
        distances = [torch.norm(style_latent - content_latent) for content_latent in self.content_latents]

        min_idx = torch.argmin(torch.tensor(distances))
        closest_content_latent = self.content_latents[min_idx]

        return style_latent, closest_content_latent
    

# @torch.no_grad()
# def sample_timestep(x, t, model):
#     """Calls model to predict noise in x at timestep t, and returns denoised x at t-1."""
#     betas_t = get_index_from_list(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
#     sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

#     # Predict noise
#     model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
#     posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
#     if t[0] == 0:
#         return model_mean
#     else:
#         noise = torch.randn_like(x)
#         return model_mean + torch.sqrt(posterior_variance_t) * noise
    

# Define beta schedule and precompute constants at module level for import
T = 300  # Reduced from 1000 to prevent explosion
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def main():
    # Hyperparameters for training
    batch_size = 64
    latent_dim = 128
    num_epochs = 50  # Increased significantly for diffusion training
    learning_rate = 1e-4  # Reduced from 1e-3 for stability
    time_embed_dim = 128

    # save losses for plotting
    train_losses = []
    val_losses = []

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # Load latents saved from saving_latents.py
    train_jazz_latents = torch.load("new_latents/train_jazz_latents.pt")  # [num_pieces, latent_dim]
    train_classical_latents = torch.load("new_latents/train_classical_latents.pt")  # [num_pieces, latent_dim]
    val_jazz_latents = torch.load("new_latents/val_jazz_latents.pt")  # [num_pieces, latent_dim]
    val_classical_latents = torch.load("new_latents/val_classical_latents.pt")  # [num_pieces, latent_dim]
    test_jazz_latents = torch.load("new_latents/test_jazz_latents.pt")  # [num_pieces, latent_dim]
    test_classical_latents = torch.load("new_latents/test_classical_latents.pt")  # [num_pieces, latent_dim]
    print("loaded latents")
    print("Training jazz latents shape:", train_jazz_latents.shape)
    print("Training classical latents shape:", train_classical_latents.shape)
    print("Validation jazz latents shape:", val_jazz_latents.shape)
    print("Validation classical latents shape:", val_classical_latents.shape)
    print("Test jazz latents shape:", test_jazz_latents.shape)
    print("Test classical latents shape:", test_classical_latents.shape)

    # Create datasets and dataloaders
    train_dataset = LatentDiffusionDataset(train_jazz_latents, train_classical_latents)
    val_dataset = LatentDiffusionDataset(val_jazz_latents, val_classical_latents)
    test_dataset = LatentDiffusionDataset(test_jazz_latents, test_classical_latents)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Training dataset size: {len(train_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPDiffuser(time_embed_dim=time_embed_dim, in_dim=latent_dim, out_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Track best validation loss for early stopping
    best_val_loss = float('inf')
    patience = 10  # Increased patience for diffusion training
    patience_counter = 0


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        start_time = time.time()

        for batch_idx, (batch_style_latent, batch_content_latent) in enumerate(train_loader):
            batch_style_latent = batch_style_latent.to(device)
            batch_content_latent = batch_content_latent.to(device)

            # sample timesteps uniformly in [0, T-1]
            t = torch.randint(0, T, (batch_style_latent.size(0),), device=device).long()
            optimizer.zero_grad()
            loss = get_loss(model, batch_style_latent, batch_content_latent, t, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
        
        end = time.time()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Time: {end - start_time:.2f}s")
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_style_latent, batch_content_latent in val_loader:
                batch_style_latent = batch_style_latent.to(device)
                batch_content_latent = batch_content_latent.to(device)

                t = torch.randint(0, T, (batch_style_latent.size(0),), device=device).long()
                loss = get_loss(model, batch_style_latent, batch_content_latent, t, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "diffuser_params/mlp_diffuser_new_best.pt")
            print(f"  → New best model saved!")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break    # Save final model (and copy best as default)

    # Save losses for plotting as csv
    np.savetxt("diffuser_params/mlp_diffuser_train_losses_new.csv", np.array(train_losses), delimiter=",")
    np.savetxt("diffuser_params/mlp_diffuser_val_losses_new.csv", np.array(val_losses), delimiter=",")

if __name__ == "__main__":
    main()