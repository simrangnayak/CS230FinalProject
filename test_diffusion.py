import torch
import sys
import os
import random

from musicbert.preprocess import encoding_to_MIDI
from vae_octuples import OctupleVAE
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


# Load a real classical test piece and encode it

with open("test_octuples/classical_octuples/cambini_midi_test.txt") as f:
    lines = f.readlines()
    test_seq = random.choice(lines).strip()

octuples = str_to_encoding(test_seq)
seq_len = len(octuples)
print(f"Original sequence length: {seq_len}")
print(f"First few octuples: {octuples[:5]}")

# Normalize bar indices to start from 0
if octuples:
    min_bar = min(note[0] for note in octuples)
    octuples = [(note[0] - min_bar,) + note[1:] for note in octuples]

# save test seq as MIDI
midi_obj = encoding_to_MIDI([tuple(x) for x in octuples])
midi_obj.dump("original_diffusion.mid")

seq = torch.tensor(octuples).unsqueeze(0).to(device)  # [1, seq_len, 8]
print(f"Sequence shape: {seq.shape}")

# Ensure sequence length is divisible by chunks (8)
original_len = seq.size(1)
chunk_size = 8
padded_len = ((original_len + chunk_size - 1) // chunk_size) * chunk_size
if padded_len != original_len:
    padding = torch.zeros(1, padded_len - original_len, 8, dtype=torch.long, device=device)
    seq = torch.cat([seq, padding], dim=1)
    print(f"Padded sequence from {original_len} to {padded_len}")


vae = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=device)
vae.load_state_dict(torch.load("vae_hierarchical_finetune_best.pt", map_location=device)["model"])
vae.eval()


# Encode to get classical latent
with torch.no_grad():
    mu, logvar = vae.encode(seq)
    z_classical = mu  # use mean for deterministic output
    print(f"Classical latent stats: mean={z_classical.mean().item():.3f}, std={z_classical.std().item():.3f}")
    print(f"Classical latent range: [{z_classical.min().item():.3f}, {z_classical.max().item():.3f}]")

    # decode to save content classical reconstruction
    outputs = vae.decode(z_classical, seq_len=seq.size(1), x = seq, autoregressive=False)

latent_dim = z_classical.size(-1)

decoded = []
for out in outputs:
    # out: [1, seq_len, vocab_size]
    pred = torch.argmax(out, dim=-1)  # [1, seq_len]
    decoded.append(pred.squeeze(0).cpu().tolist())

# convert back to octuples
decoded = list(zip(*decoded))  # list of (seq_len, 8)
decoded_tuples = [tuple(x) for x in decoded]


# sanitize reconstruction tuples
def clamp_tuple_vals(t):
    return tuple(max(0, min(127, int(v))) for v in t)
decoded_tuples = [clamp_tuple_vals(t) for t in decoded_tuples]

# convert to MIDI and save
midi_obj = encoding_to_MIDI(decoded_tuples)
midi_obj.dump("reconstructed_diffusion.mid")


# --- STYLE TRANSFER VIA DIFFUSION ---
print("\nStarting diffusion-based style transfer...")
# random gaussian noise vector as style latent - same shape as content latents
style_latents = torch.randn_like(z_classical).to(device)  # [1, latent_dim]

# load trained diffuser model
diffuser = MLPDiffuser(time_embed_dim=128, in_dim=latent_dim, out_dim=latent_dim)
diffuser.load_state_dict(torch.load("mlp_diffuser_best_small.pt", map_location=device))
diffuser.to(device)
diffuser.eval()

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
    
    print(f"Running {num_steps} denoising steps...")
    for i in reversed(range(0, num_steps)):
        if i % 50 == 0:
            print(f"  Step {num_steps-i}/{num_steps}")
        t = torch.full((x.size(0),), i, device=device, dtype=torch.long)
        x = sample_timestep(x, t, diffuser, classical_latents)
    
    return x


# # For decoding, feed prefix from closest jazz latent
# # (to stabilize generation, as in training)

# train_jazz_latents = torch.load("latents/train_jazz_latents.pt")
# val_jazz_latents = torch.load("latents/val_jazz_latents.pt")
# test_jazz_latents = torch.load("latents/test_jazz_latents.pt")
# all_jazz_latents = torch.cat([train_jazz_latents, val_jazz_latents, test_jazz_latents], dim=0)

# # find closest jazz latent in dataset to z_classical
# distances = torch.norm(all_jazz_latents - z_classical, dim=1)  # [N]
# closest_idx = torch.argmin(distances)
# z_jazz_start = all_jazz_latents[closest_idx].unsqueeze(0)

# start with random noise
z_jazz_start = torch.randn(1, latent_dim).to(device)


# Denoise from z_jazz_start
jazz_style_latent = denoise_latent(z_jazz_start, z_classical, diffuser, num_steps=T)  # Use full T=300 steps

print(f"\nJazz latent stats: mean={jazz_style_latent.mean().item():.3f}, std={jazz_style_latent.std().item():.3f}")
print(f"Jazz latent range: [{jazz_style_latent.min().item():.3f}, {jazz_style_latent.max().item():.3f}]")
print(f"Latent distance from classical: {torch.norm(jazz_style_latent - z_classical).item():.3f}")

# Print element-wise comparison
print(f"\nFirst 10 dimensions comparison:")
print(f"Classical: {z_classical[0, :10].cpu().numpy()}")
print(f"Jazz:      {jazz_style_latent[0, :10].cpu().numpy()}")


# Decode jazz-style latent to MIDI using prefix from random jazz octuple
jazz_octuples = load_octuples_from_folder("test_octuples")
if not jazz_octuples:
    print("No jazz octuples found! Using classical sequence as prefix instead.")
    random_jazz_octuple = octuples  # Use the original classical sequence
else:
    random_jazz_octuple = random.choice(jazz_octuples)
jazz_len = len(random_jazz_octuple)
print(f"\nUsing jazz piece of length {jazz_len} for prefix.")
jazz_seq = torch.tensor(random_jazz_octuple).unsqueeze(0).to(device)  # [1, jazz_len, 8]

jazz_len = jazz_seq.size(1)
prefix_len = min(jazz_len // 2, jazz_len)
x_prefix = jazz_seq[:, :prefix_len, :]
with torch.no_grad():
    # Use autoregressive decoding (same as classical)
    outputs = vae.decode(jazz_style_latent, seq_len=jazz_len, autoregressive=True, x_prefix=x_prefix)
    
    print(f"Decoder output shapes: {[out.shape for out in outputs]}")
    
    # Check if decoder outputs are still collapsed
    for i, out in enumerate(outputs[:4]):  # Check first 4 channels
        logits = out[0, :10, :]  # First 10 timesteps, all vocab
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        print(f"Channel {i} max probs (first 10 steps): {max_probs.cpu().numpy()}")
        if (max_probs > 0.99).sum() > 8:  # If most are >99% confident
            print(f"  ⚠️  Channel {i} decoder still over-confident")

decoded = []
for out in outputs:
    pred = torch.argmax(out, dim=-1)  # [batch, seq_len]
    decoded.append(pred.cpu())

# Stack to [batch, seq_len, 8], then extract first batch item
decoded_stacked = torch.stack(decoded, dim=-1)  # [batch, seq_len, 8]
decoded_array = decoded_stacked[0].numpy()  # [seq_len, 8]

# Clamp values to valid vocab ranges for each channel
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]
for i in range(8):
    decoded_array[:, i] = decoded_array[:, i].clip(0, vocab_sizes[i] - 1)

# Additional MIDI value validation - ensure all values are valid MIDI data bytes
# For MIDI compatibility, most values should be 0-127
for i in range(decoded_array.shape[0]):
    for j in range(8):
        val = decoded_array[i, j]
        # Clamp to MIDI-safe range for most channels
        if j in [0, 3]:  # Bar and pitch - can be larger
            decoded_array[i, j] = max(0, min(127, val))
        else:  # Other channels should be 0-127
            decoded_array[i, j] = max(0, min(127, val))

print(f"Value ranges after clamping:")
for i in range(8):
    channel_vals = decoded_array[:, i]
    print(f"  Channel {i}: [{channel_vals.min():.0f}, {channel_vals.max():.0f}]")

# Check for silence (all zeros or constant values)
unique_pitches = len(set(decoded_array[:, 3]))  # pitch is channel 3
print(f"Unique pitch values in jazz output: {unique_pitches}")
if unique_pitches < 5:
    print("WARNING: Very few unique pitches detected - output may be silence/noise")

# Convert to list of tuples (each tuple is one octuple)
decoded_tuples = [tuple(int(x) for x in row) for row in decoded_array]

# Final validation - ensure all values are MIDI-compatible
for i, octuple in enumerate(decoded_tuples):
    fixed_octuple = []
    for ch, val in enumerate(octuple):
        # Ensure all values are within MIDI range
        clamped_val = max(0, min(127, int(val)))
        fixed_octuple.append(clamped_val)
    decoded_tuples[i] = tuple(fixed_octuple)

# Verify no invalid values remain
for i, octuple in enumerate(decoded_tuples[:5]):  # Check first 5
    for ch, val in enumerate(octuple):
        if val < 0 or val > 127:
            print(f"ERROR: Invalid MIDI value at position {i}, channel {ch}: {val}")
            
print(f"Sample output tuples (first 3): {decoded_tuples[:3]}")

midi_obj = encoding_to_MIDI(decoded_tuples)
midi_obj.dump("style_transferred_jazz.mid")
print(f"Generated style_transferred_jazz.mid with {len(decoded_tuples)} octuples")
print("\nDone! Compare reconstructed_diffusion.mid (original) vs style_transferred_jazz.mid (jazz style)")