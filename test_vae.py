from musicbert.preprocess import str_to_encoding, encoding_to_MIDI
import torch
from old_code.vae_octuples import OctupleVAE
from vae_octuples_hierarchical import OctupleVAE_HierarchicalDecoder
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vocab sizes for each of the 8 channels
vocab_sizes = [256, 128, 129, 256, 128, 33, 128, 49]

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
midi_obj.dump("original.mid")


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

vae_jazz_only = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=device)
vae_jazz_only.load_state_dict(torch.load("vae_hierarchical_params/vae_hierarchical_finetune_best.pt", map_location=device)["model"])
vae_jazz_only.eval()

vae_finetune = OctupleVAE_HierarchicalDecoder(vocab_sizes=vocab_sizes, embed_dim=64, hidden_dim=256, latent_dim=128, chunks=8, device=device)
vae_finetune.load_state_dict(torch.load("vae_hierarchical_finetune_best_final.pt", map_location=device)["model"])
vae_finetune.eval()

# --- Deterministic reconstruction ---
with torch.no_grad():
    mu, logvar = vae_jazz_only.encode(seq)
    z = mu  # use mean for deterministic output
    outputs = vae_jazz_only.decode(z, seq_len=seq.size(1), x=seq, autoregressive=False)

    mu_finetune, logvar_finetune = vae_finetune.encode(seq)
    z_finetune = mu_finetune  # use mean for deterministic output
    outputs_finetune = vae_finetune.decode(z_finetune, seq_len=seq.size(1), x=seq, autoregressive=False)

    print(f"Latent means difference (no finetune vs finetune): {(mu - mu_finetune).abs().mean().item():.6f}")

decoded = []
for out in outputs:
    # out: [1, seq_len, vocab_size]
    pred = torch.argmax(out, dim=-1)  # [1, seq_len]
    decoded.append(pred.squeeze(0).cpu().tolist())

decoded_finetune = []
for out in outputs_finetune:
    pred = torch.argmax(out, dim=-1)
    decoded_finetune.append(pred.squeeze(0).cpu().tolist())

# convert back to octuples
decoded = list(zip(*decoded))  # list of (seq_len, 8)
decoded_tuples = [tuple(x) for x in decoded]

decoded_finetune = list(zip(*decoded_finetune))
decoded_finetune_tuples = [tuple(x) for x in decoded_finetune]


# sanitize reconstruction tuples
def clamp_tuple_vals(t):
    return tuple(max(0, min(127, int(v))) for v in t)
decoded_tuples = [clamp_tuple_vals(t) for t in decoded_tuples]
decoded_finetune_tuples = [clamp_tuple_vals(t) for t in decoded_finetune_tuples]

# convert to MIDI and save
midi_obj = encoding_to_MIDI(decoded_tuples)
midi_obj.dump("reconstructed_finetune.mid")
midi_obj = encoding_to_MIDI(decoded_finetune_tuples)
midi_obj.dump("reconstructed_finetune_final.mid")


# --- Autoregressive generation from latent (longer prefix for stability) ---
generation_len = seq.size(1)  # Use actual padded length
# use smaller prefix for generation
prefix_len = min(generation_len // 2, seq_len)
x_prefix = seq[:, :prefix_len, :]
print(f"Using prefix length: {prefix_len}, generating length: {generation_len}")
print(f"PREFIX BOUNDARY: prefix_len = {prefix_len}, generation begins at token {prefix_len}")

with torch.no_grad():
    # Use the learned mean as latent for coherent generation
    z = mu  # [1, latent_dim]
    # # Warm-up with a longer prefix and use first frame as start token
    ar_outputs = vae_jazz_only.decode(
        z,
        seq_len=generation_len,
        autoregressive=True,
        x_prefix=x_prefix,
        temperature=0.7,
        top_k=8
    )

    z_finetune = mu_finetune
    ar_outputs_finetune = vae_finetune.decode(
        z_finetune,
        seq_len=generation_len,
        autoregressive=True,
        x_prefix=x_prefix,
        temperature=0.7,
        top_k=8
    )

ar_decoded = []
for out in ar_outputs:
    pred = torch.argmax(out, dim=-1)  # [1, seq_len]
    ar_decoded.append(pred.squeeze(0).cpu().tolist())

ar_decoded_finetune = []
for out in ar_outputs_finetune:
    pred = torch.argmax(out, dim=-1)
    ar_decoded_finetune.append(pred.squeeze(0).cpu().tolist())

ar_decoded = list(zip(*ar_decoded))
ar_decoded_tuples = [tuple(x) for x in ar_decoded]
ar_decoded_tuples = [clamp_tuple_vals(t) for t in ar_decoded_tuples]

ar_decoded_finetune = list(zip(*ar_decoded_finetune))
ar_decoded_finetune_tuples = [tuple(x) for x in ar_decoded_finetune]
ar_decoded_finetune_tuples = [clamp_tuple_vals(t) for t in ar_decoded_finetune_tuples]

# Print comparison between prefix and generated portions
print(f"\n=== PREFIX vs GENERATED ANALYSIS ===")
print(f"Prefix length: {prefix_len}, Total length: {len(ar_decoded_finetune_tuples)}")
print(f"Prefix (tokens 0-{prefix_len-1}):")
for i in range(min(5, prefix_len)):
    print(f"  Token {i}: Original={octuples[i]} | AR={ar_decoded_finetune_tuples[i]}")

print(f"Generated (tokens {prefix_len}-{prefix_len+4}):")
for i in range(prefix_len, min(prefix_len + 5, len(ar_decoded_finetune_tuples))):
    orig_val = octuples[i] if i < len(octuples) else "N/A (padding)"
    print(f"  Token {i}: Original={orig_val} | AR={ar_decoded_finetune_tuples[i]}")

encoding_to_MIDI(ar_decoded_tuples).dump("generated_from_mu_finetune_ar.mid")
encoding_to_MIDI(ar_decoded_finetune_tuples).dump("generated_from_mu_finetune_ar_final.mid")


# # --- Pure prior-sampled generation ---
# with torch.no_grad():
#     # Sample z ~ N(0, I) for unconditional generation
#     z_prior = torch.randn_like(mu_finetune)  # [1, latent_dim]
#     # Use same 16-token prefix for stability
#     # prior_outputs = vae_no_finetune.decode(
#     #     z_prior,
#     #     seq_len=generation_len,
#     #     autoregressive=True,
#     #     x_prefix=x_prefix,
#     #     temperature=1.2,
#     #     top_k=12
#     # )

#     prior_outputs_finetune = vae_finetune.decode(
#         z_prior,
#         seq_len=generation_len,
#         autoregressive=True,
#         x_prefix=x_prefix,
#         temperature=0.9,
#         top_k=16
#     )

# # prior_decoded = []
# # for out in prior_outputs:
# #     pred = torch.argmax(out, dim=-1)
# #     prior_decoded.append(pred.squeeze(0).cpu().tolist())

# prior_decoded_finetune = []
# for out in prior_outputs_finetune:
#     pred = torch.argmax(out, dim=-1)
#     prior_decoded_finetune.append(pred.squeeze(0).cpu().tolist())

# # prior_decoded = list(zip(*prior_decoded))
# # prior_decoded_tuples = [tuple(x) for x in prior_decoded]
# # prior_decoded_tuples = [clamp_tuple_vals(t) for t in prior_decoded_tuples]

# prior_decoded_finetune = list(zip(*prior_decoded_finetune))
# prior_decoded_finetune_tuples = [tuple(x) for x in prior_decoded_finetune]
# prior_decoded_finetune_tuples = [clamp_tuple_vals(t) for t in prior_decoded_finetune_tuples]

# # Print comparison between prefix and generated portions for prior sampling
# print(f"\n=== PRIOR SAMPLING PREFIX vs GENERATED ANALYSIS ===")
# print(f"Prefix length: {prefix_len}, Total length: {len(prior_decoded_finetune_tuples)}")
# print(f"Prefix (tokens 0-{prefix_len-1}):")
# for i in range(min(3, prefix_len)):
#     print(f"  Token {i}: Original={octuples[i]} | Prior={prior_decoded_finetune_tuples[i]}")

# print(f"Generated (tokens {prefix_len}-{prefix_len+4}):")
# for i in range(prefix_len, min(prefix_len + 5, len(prior_decoded_finetune_tuples))):
#     orig_val = octuples[i] if i < len(octuples) else "N/A (padding)"
#     print(f"  Token {i}: Original={orig_val} | Prior={prior_decoded_finetune_tuples[i]}")

# # encoding_to_MIDI(prior_decoded_tuples).dump("generated_from_prior_ar.mid")
# encoding_to_MIDI(prior_decoded_finetune_tuples).dump("generated_from_prior_finetune_ar_final.mid")

# # --- Sanitize helpers to avoid invalid MIDI data bytes ---
# def clamp_tuple_vals(t):
#     # Clamp all channels to 0..127 to satisfy MIDI data byte constraints
#     return tuple(max(0, min(127, int(v))) for v in t)

# # Apply clamping to AR and prior generations
# ar_decoded_tuples = [clamp_tuple_vals(t) for t in ar_decoded_tuples]
# prior_decoded_tuples = [clamp_tuple_vals(t) for t in prior_decoded_tuples]

# midi_obj = encoding_to_MIDI(ar_decoded_tuples)
# midi_obj.dump("generated_ar_from_mu.mid")

# midi_obj = encoding_to_MIDI(prior_decoded_tuples)
# midi_obj.dump("generated_ar_from_prior.mid")

