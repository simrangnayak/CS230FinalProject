# this script is a latent diffuser model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
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


def forward_diffusion_sample(x0, t, device):
    """Backward compatibility helper using global precomputed schedule."""
    noise = torch.randn_like(x0)
    sqrt_alpha_bar = get_index_from_list(sqrt_alphas_cumprod.to(device), t, x0.shape)
    sqrt_one_minus = get_index_from_list(sqrt_one_minus_alphas_cumprod.to(device), t, x0.shape)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
    return x_t, noise

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_dim)
        self.conv1 = nn.Conv1d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_dim, out_dim, 3, 1, 1)
        self.down = nn.Conv1d(out_dim, out_dim, 4, 2, 1)  # stride 2 downsample
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.relu(self.bn1(self.conv1(x)))
        t_emb = self.relu(self.time_mlp(t)).unsqueeze(-1)
        h = h + t_emb
        h = self.relu(self.bn2(self.conv2(h)))
        skip = h  # store pre-downsample features
        out = self.down(h)
        return out, skip

class UpBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_dim)
        self.conv1 = nn.Conv1d(in_dim + skip_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_dim, out_dim, 3, 1, 1)
        self.up = nn.ConvTranspose1d(out_dim, out_dim, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, skip, t):
        x = torch.cat([x, skip], dim=1)
        h = self.relu(self.bn1(self.conv1(x)))
        t_emb = self.relu(self.time_mlp(t)).unsqueeze(-1)
        h = h + t_emb
        h = self.relu(self.bn2(self.conv2(h)))
        out = self.up(h)
        return out

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
    
class SimpleUnet(nn.Module):
    def __init__(self, time_embed_dim, in_dim=128, out_dim=128):
        super().__init__()
        # Channel plan
        self.down_channels = [256, 512]  # after initial concat+proj -> 256, then 512
        self.up_channels = [512, 256]

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidealPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        # initial concat projection (style + content)
        self.input_proj = nn.Conv1d(2 * in_dim, self.down_channels[0], 3, 1, 1)

        # down path
        self.down_blocks = nn.ModuleList([
            DownBlock(self.down_channels[0], self.down_channels[1], time_embed_dim)
        ])

        # up path (mirror) - current x channels start at 512, skip from previous down has 512
        self.up_blocks = nn.ModuleList([
            UpBlock(self.up_channels[0], self.up_channels[0], self.up_channels[1], time_embed_dim)
        ])

        self.out_conv = nn.Conv1d(self.up_channels[-1], out_dim, 1)

    def forward(self, style_latent, t, content_latent):
        # Expect style_latent/content_latent shape: [batch, seq_len, in_dim]
        # If provided as [batch, in_dim] (mu only), expand to seq_len=1
        if style_latent.dim() == 2:
            style_latent = style_latent.unsqueeze(1)
        if content_latent.dim() == 2:
            content_latent = content_latent.unsqueeze(1)
        # permute to channels-first for Conv1d
        style_latent = style_latent.permute(0, 2, 1)  # [B, C, L]
        content_latent = content_latent.permute(0, 2, 1)
        t_emb = self.time_mlp(t)  # [batch, time_embed_dim]
        x = torch.cat([style_latent, content_latent], dim=1)
        x = self.input_proj(x)

        skips = []
        for down in self.down_blocks:
            x, skip = down(x, t_emb)
            skips.append(skip)

        # Up path
        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        out = self.out_conv(x)  # [B, C, L]
        # return to original latent shape [batch, seq_len, latent_dim]
        return out.permute(0, 2, 1)
    

def get_loss(model, style_latent, content_latent, t, device):
    """Compute denoising loss.
    style_latent/content_latent: [batch, seq_len, latent_dim] OR [batch, latent_dim] (mu only)
    t: [batch]
    """
    style_latent = style_latent.to(device)
    content_latent = content_latent.to(device)
    t = t.to(device)

    # If mu-only vectors, treat seq_len=1 for diffusion
    if style_latent.dim() == 2:
        style_latent = style_latent.unsqueeze(1)
    if content_latent.dim() == 2:
        content_latent = content_latent.unsqueeze(1)

    # channels-first for noise addition
    x0 = style_latent.permute(0, 2, 1)  # [B, C, L]
    x_noisy, noise = forward_diffusion_sample(x0, t, device)
    # model expects original (possibly seq) layout; convert back
    x_noisy_seq = x_noisy.permute(0, 2, 1)
    noise_pred_seq = model(x_noisy_seq, t, content_latent)
    # Convert prediction to channels-first to compare with noise
    noise_pred = noise_pred_seq.permute(0, 2, 1)
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
    


# Define beta schedule
T = 1000
betas = linear_beta_schedule(timesteps=T)

# Hyperparameters for validation
batch_size = 64
latent_dim = 128
num_epochs = 5
learning_rate = 1e-3
time_embed_dim = 128
max_noise_level = T  # maximum timestep matches schedule length

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Load latents saved from saving_latents.py
train_jazz_latents = torch.load("latents/train_jazz_latents.pt")  # [num_pieces, latent_dim]
train_classical_latents = torch.load("latents/train_classical_latents.pt")  # [num_pieces, latent_dim]
val_jazz_latents = torch.load("latents/val_jazz_latents.pt")  # [num_pieces, latent_dim]
val_classical_latents = torch.load("latents/val_classical_latents.pt")  # [num_pieces, latent_dim]
test_jazz_latents = torch.load("latents/test_jazz_latents.pt")  # [num_pieces, latent_dim]
test_classical_latents = torch.load("latents/test_classical_latents.pt")  # [num_pieces, latent_dim]
print("loaded latents")

# Create datasets and dataloaders
train_dataset = LatentDiffusionDataset(train_jazz_latents, train_classical_latents)
val_dataset = LatentDiffusionDataset(val_jazz_latents, val_classical_latents)
test_dataset = LatentDiffusionDataset(test_jazz_latents, test_classical_latents)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Training dataset size: {len(train_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUnet(time_embed_dim=time_embed_dim, in_dim=latent_dim, out_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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
        loss = get_loss(model, batch_style_latent, batch_content_latent, t, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    end = time.time()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Time: {end - start_time:.2f}s")

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_style_latent, batch_content_latent in val_loader:
            batch_style_latent = batch_style_latent.to(device)
            batch_content_latent = batch_content_latent.to(device)

            t = torch.randint(0, T, (batch_style_latent.size(0),), device=device).long()
            loss = get_loss(model, batch_style_latent, batch_content_latent, t, device)
            val_loss += loss.item()
        
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")


# Save trained model
torch.save(model.state_dict(), "mlp_diffuser.pt")