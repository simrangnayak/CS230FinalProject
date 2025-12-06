import torch
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import librosa
import librosa.display
from scipy.io.wavfile import write
import pretty_midi

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

def plot_musical_comparison(original_seq, diffused_seq, save_path="musical_comparison.png"):
    """Create visual comparison plots of octuple distributions between original and style-transferred sequences
    Note that this code was created with help from ChatGPT
    """
    orig = original_seq.cpu().numpy()
    diff = diffused_seq.cpu().numpy()
    
    # Flatten for plotting
    orig_flat = orig.reshape(-1, orig.shape[-1])
    diff_flat = diff.reshape(-1, diff.shape[-1])
    
    # Channel names for clarity
    channel_names = ['Measure', 'Position', 'Program', 'Pitch', 'Duration', 'Velocity', 'TimeSig', 'Tempo']
    
    # Create 2x4 subplot for all 8 octuple channels
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()  # Make it easier to iterate
    
    for i in range(8):
        # Plot histogram for each channel
        axes[i].hist(orig_flat[:, i], alpha=0.7, label='Original Classical', bins=min(30, len(np.unique(orig_flat[:, i]))), color='blue', density=True)
        axes[i].hist(diff_flat[:, i], alpha=0.7, label='Style Transferred', bins=min(30, len(np.unique(diff_flat[:, i]))), color='red', density=True)
        
        axes[i].set_title(f'{channel_names[i]} Distribution')
        axes[i].set_xlabel(f'{channel_names[i]} Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        orig_mean = orig_flat[:, i].mean()
        diff_mean = diff_flat[:, i].mean()
        orig_std = orig_flat[:, i].std()
        diff_std = diff_flat[:, i].std()
        
        stats_text = f'Orig: μ={orig_mean:.1f}, σ={orig_std:.1f}\nTransf: μ={diff_mean:.1f}, σ={diff_std:.1f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved octuple distribution comparison to {save_path}")
    
    print("\nDetailed Octuple Statistics:")
    print("-" * 50)
    for i, name in enumerate(channel_names):
        orig_vals = orig_flat[:, i]
        diff_vals = diff_flat[:, i]
        
        print(f"{name:10} | Orig: mean={orig_vals.mean():6.2f}, std={orig_vals.std():6.2f}, range=[{orig_vals.min():3.0f}, {orig_vals.max():3.0f}]")
        print(f"{'':10} | Diff: mean={diff_vals.mean():6.2f}, std={diff_vals.std():6.2f}, range=[{diff_vals.min():3.0f}, {diff_vals.max():3.0f}]")
        print(f"{'':<10} | Change: Δμ={diff_vals.mean()-orig_vals.mean():+6.2f}, Δσ={diff_vals.std()-orig_vals.std():+6.2f}")
        print("-" * 50)


def plot_spectrogram_comparison(original_midi_file, transferred_midi_file, save_path="spectrogram_comparison.png"):
    """Create spectrogram comparison between original and style-transferred MIDI
    
    Note that this code was created with help from ChatGPT
    """
    try:
        # Load MIDI files and convert to audio
        def midi_to_audio(midi_file, sr=22050, duration=30):
            """Convert MIDI file to audio array"""
            try:
                midi = pretty_midi.PrettyMIDI(midi_file)
                print(f"  MIDI file: {midi_file}")
                print(f"    Total time: {midi.get_end_time():.2f}s")
                print(f"    Number of instruments: {len(midi.instruments)}")
                
                total_notes = sum(len(inst.notes) for inst in midi.instruments)
                print(f"    Total notes: {total_notes}")
                
                if total_notes == 0:
                    print(f"    ⚠️  WARNING: No notes found in {midi_file}")
                    return None, sr
                
                # Show note ranges for debugging
                if total_notes > 0:
                    all_pitches = [note.pitch for inst in midi.instruments for note in inst.notes]
                    all_velocities = [note.velocity for inst in midi.instruments for note in inst.notes]
                    print(f"    Pitch range: {min(all_pitches)} - {max(all_pitches)}")
                    print(f"    Velocity range: {min(all_velocities)} - {max(all_velocities)}")
                
                # Synthesize audio (first 30 seconds to keep manageable)
                try:
                    # Try FluidSynth with basic parameters (pretty_midi doesn't support synth_kwargs)
                    audio = midi.fluidsynth(fs=sr)
                    print(f"    FluidSynth synthesis successful")
                except Exception as synth_error:
                    print(f"    ⚠️  FluidSynth failed: {synth_error}")
                    print(f"    Trying alternative synthesis method...")
                    
                    # Fallback: Create simple sine wave synthesis
                    audio_length = int(min(duration, midi.get_end_time()) * sr)
                    audio = np.zeros(audio_length)
                    
                    print(f"    Synthesizing {len(midi.instruments)} instruments...")
                    
                    for inst_idx, instrument in enumerate(midi.instruments):
                        print(f"      Instrument {inst_idx}: {len(instrument.notes)} notes")
                        
                        for note_idx, note in enumerate(instrument.notes):
                            if note.end > duration:
                                continue
                                
                            start_sample = int(note.start * sr)
                            end_sample = int(min(note.end * sr, audio_length))
                            
                            if start_sample < audio_length and end_sample > start_sample:
                                # Generate simple sine wave for this note
                                freq = 440 * (2 ** ((note.pitch - 69) / 12))  # A4 = 440Hz
                                note_duration = (end_sample - start_sample) / sr
                                t = np.linspace(0, note_duration, end_sample - start_sample)
                                
                                # Better envelope and amplitude
                                envelope = np.exp(-t * 1.5)  # Decay envelope
                                amplitude = 0.05 * (note.velocity / 127)  # Amplitude from velocity
                                sine_wave = amplitude * envelope * np.sin(2 * np.pi * freq * t)
                                
                                # Add to audio with bounds checking
                                end_idx = min(start_sample + len(sine_wave), len(audio))
                                sine_len = end_idx - start_sample
                                if sine_len > 0:
                                    audio[start_sample:end_idx] += sine_wave[:sine_len]
                    
                    print(f"    Fallback synthesis complete, checking audio...")
                    if len(audio) == 0:
                        print(f"    ❌ Generated empty audio array")
                        return None, sr
                    
                    # Check if we actually generated any sound
                    audio_max = np.max(np.abs(audio))
                    print(f"    Generated audio max amplitude: {audio_max:.6f}")
                    
                    if audio_max < 1e-8:
                        print(f"    ❌ Generated audio is too quiet")
                        # Try a simpler approach - just make some test tones
                        print(f"    Creating test tones based on note data...")
                        audio = np.zeros(audio_length)
                        
                        # Create simple test tones for first few notes
                        test_notes = []
                        for instrument in midi.instruments:
                            test_notes.extend(instrument.notes[:10])  # First 10 notes per instrument
                        
                        for i, note in enumerate(test_notes[:50]):  # Max 50 test notes
                            if note.start > duration:
                                continue
                            start_sample = int(note.start * sr)
                            # Make each note 0.5 seconds long
                            duration_samples = int(0.5 * sr)
                            end_sample = min(start_sample + duration_samples, audio_length)
                            
                            if start_sample < audio_length:
                                freq = 440 * (2 ** ((note.pitch - 69) / 12))
                                t = np.linspace(0, 0.5, duration_samples)
                                envelope = np.exp(-t * 2)
                                sine_wave = 0.1 * envelope * np.sin(2 * np.pi * freq * t)
                                
                                end_idx = min(start_sample + len(sine_wave), len(audio))
                                sine_len = end_idx - start_sample
                                if sine_len > 0:
                                    audio[start_sample:end_idx] += sine_wave[:sine_len]
                        
                        final_max = np.max(np.abs(audio))
                        print(f"    Test tone generation: max amplitude = {final_max:.6f}")
                    
                    # Normalize audio to prevent clipping
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio)) * 0.8
                
                # Check audio quality
                if len(audio) == 0:
                    print(f"    ⚠️  WARNING: Audio synthesis produced empty array")
                    return None, sr
                    
                audio_rms = np.sqrt(np.mean(audio**2))
                print(f"    Audio RMS: {audio_rms:.6f}")
                print(f"    Audio length: {len(audio)/sr:.2f}s, Max: {np.max(np.abs(audio)):.6f}")
                
                if audio_rms < 1e-6:
                    print(f"    ⚠️  WARNING: Audio is essentially silent (RMS < 1e-6)")
                
                # Trim to specified duration
                max_samples = sr * duration
                if len(audio) > max_samples:
                    print(f"    Trimming audio from {len(audio)/sr:.2f}s to {duration}s")
                    audio = audio[:max_samples]
                    
                    # Check audio after trimming
                    trimmed_rms = np.sqrt(np.mean(audio**2))
                    print(f"    After trimming - RMS: {trimmed_rms:.6f}, Max: {np.max(np.abs(audio)):.6f}")
                else:
                    print(f"    No trimming needed - audio is {len(audio)/sr:.2f}s")
                
                return audio, sr
                
            except Exception as e:
                print(f"    ❌ Error converting {midi_file} to audio: {e}")
                return None, sr
        
        print("Converting MIDI files to audio...")
        
        # Convert both MIDI files to audio
        orig_audio, sr = midi_to_audio(original_midi_file)
        trans_audio, _ = midi_to_audio(transferred_midi_file)
        
        if orig_audio is None or trans_audio is None:
            print("Could not generate spectrograms - MIDI to audio conversion failed")
            return
        
        print(f"\nAfter MIDI conversion:")
        print(f"  Original: {len(orig_audio)} samples, RMS: {np.sqrt(np.mean(orig_audio**2)):.6f}")
        print(f"  Transferred: {len(trans_audio)} samples, RMS: {np.sqrt(np.mean(trans_audio**2)):.6f}")
        
        # Ensure both audio arrays have the same length
        min_len = min(len(orig_audio), len(trans_audio))
        print(f"  Trimming both to {min_len} samples ({min_len/sr:.2f}s)")
        
        orig_audio = orig_audio[:min_len]
        trans_audio = trans_audio[:min_len]
        
        print(f"\nAfter length matching:")
        print(f"  Original: RMS: {np.sqrt(np.mean(orig_audio**2)):.6f}, Max: {np.max(np.abs(orig_audio)):.6f}")
        print(f"  Transferred: RMS: {np.sqrt(np.mean(trans_audio**2)):.6f}, Max: {np.max(np.abs(trans_audio)):.6f}")
        
        # Check if transferred audio is all zeros
        if np.all(trans_audio == 0):
            print(f"  ❌ Transferred audio is all zeros!")
        elif np.std(trans_audio) < 1e-10:
            print(f"  ❌ Transferred audio has near-zero variance!")
        else:
            print(f"  ✅ Transferred audio has valid content")
            print(f"    Non-zero samples: {np.count_nonzero(trans_audio)}/{len(trans_audio)}")
            print(f"    Sample values range: [{trans_audio.min():.6f}, {trans_audio.max():.6f}]")
        
        # Additional audio debugging
        print(f"\nAudio Quality Check:")
        print(f"  Original audio: max={np.max(np.abs(orig_audio)):.4f}, std={np.std(orig_audio):.4f}")
        print(f"  Transferred audio: max={np.max(np.abs(trans_audio)):.4f}, std={np.std(trans_audio):.4f}")
        
        # Compute spectrograms
        orig_stft = librosa.stft(orig_audio, n_fft=2048, hop_length=512)
        trans_stft = librosa.stft(trans_audio, n_fft=2048, hop_length=512)
        
        # Convert to magnitude (dB)
        orig_spec_db = librosa.amplitude_to_db(np.abs(orig_stft), ref=np.max)
        trans_spec_db = librosa.amplitude_to_db(np.abs(trans_stft), ref=np.max)
        
        print(f"  Spectrogram ranges: orig [{orig_spec_db.min():.1f}, {orig_spec_db.max():.1f}] dB")
        print(f"                      trans [{trans_spec_db.min():.1f}, {trans_spec_db.max():.1f}] dB")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Original spectrogram
        img1 = librosa.display.specshow(orig_spec_db, sr=sr, hop_length=512, 
                                       x_axis='time', y_axis='hz', ax=axes[0])
        axes[0].set_title('Original Classical - Spectrogram')
        axes[0].set_ylabel('Frequency (Hz)')
        plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        # Style transferred spectrogram  
        img2 = librosa.display.specshow(trans_spec_db, sr=sr, hop_length=512,
                                       x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('Style Transferred Jazz - Spectrogram')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved spectrogram comparison to {save_path}")
        
        # Print some audio statistics
        print(f"\nAudio Analysis:")
        print(f"  Audio length: {len(orig_audio)/sr:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Original RMS: {np.sqrt(np.mean(orig_audio**2)):.4f}")
        print(f"  Transferred RMS: {np.sqrt(np.mean(trans_audio**2)):.4f}")
        
    except ImportError as e:
        print(f"Missing required libraries for spectrogram: {e}")
        print("Please install: pip install librosa pretty_midi")
    except Exception as e:
        print(f"Error creating spectrogram: {e}")


# Load a real classical test piece and encode it
with open("test_octuples/classical_octuples/bach_midi_test.txt") as f:
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
vae.load_state_dict(torch.load("vae_hierarchical_params/vae_hierarchical_finetune_best.pt", map_location=device)["model"])
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
diffuser.load_state_dict(torch.load("diffuser_params/mlp_diffuser_best_small.pt", map_location=device))
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
# jazz_octuples = load_octuples_from_folder("test_octuples")
# if not jazz_octuples:
#     print("No jazz octuples found! Using classical sequence as prefix instead.")
#     random_jazz_octuple = octuples  # Use the original classical sequence
# else:
#     random_jazz_octuple = random.choice(jazz_octuples)
# jazz_len = len(random_jazz_octuple)
# print(f"\nUsing jazz piece of length {jazz_len} for prefix.")
# jazz_seq = torch.tensor(random_jazz_octuple).unsqueeze(0).to(device)  # [1, jazz_len, 8]

# jazz_len = jazz_seq.size(1)
# prefix_len = min(jazz_len // 2, jazz_len)
# x_prefix = jazz_seq[:, :prefix_len, :]

# decode jazz-style latent to MIDI using prefix from classical piece
prefix_len = min(seq.size(1) // 4, seq.size(1))
x_prefix = seq[:, :prefix_len, :]

use_seq_len = seq_len # change to jazz_len if using jazz prefix

with torch.no_grad():
    # Use autoregressive decoding (same as classical) using no prefix
    outputs = vae.decode(jazz_style_latent, seq_len=use_seq_len, autoregressive=True, x_prefix=None)
    
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

# ==================== QUALITATIVE ANALYSIS ====================
print("\n" + "="*50)
print("QUALITATIVE ANALYSIS: ORIGINAL vs STYLE TRANSFERRED")
print("="*50)

# Convert sequences to tensors for analysis
original_tensor = seq  # Original sequence [1, seq_len, 8]
transferred_tensor = torch.tensor(decoded_tuples).unsqueeze(0).to(device)  # [1, seq_len, 8]

# Ensure same length for comparison
min_len = min(original_tensor.size(1), transferred_tensor.size(1))
original_tensor = original_tensor[:, :min_len, :]
transferred_tensor = transferred_tensor[:, :min_len, :]

print(f"Comparing sequences of length: {min_len}")

plot_musical_comparison(original_tensor, transferred_tensor, "style_transfer_comparison.png")

# 5. Spectrogram Comparison
print("\n5. CREATING SPECTROGRAM COMPARISON:")
print("-" * 35)
plot_spectrogram_comparison("reconstructed_diffusion.mid", "style_transferred_jazz.mid", "spectrogram_comparison_noprefix.png")

print("\nDone! Compare reconstructed_diffusion.mid (original) vs style_transferred_jazz.mid (jazz style)")