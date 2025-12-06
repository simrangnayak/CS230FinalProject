# MelodyMorph: Translating Classical Artists to Jazz Tunes

A machine learning system for translating classical music compositions into jazz renditions while attempting to preserve the original composer's musical personality.

## 📋 Overview

This project implements a generative approach to music style transfer that combines:
- **Hierarchical VAE**: Compresses raw MIDI data into a learned latent space that separates content and style
- **MLP Diffusion Model**: Performs style transfer by denoising jazz-style latents conditioned on classical content
- **Multi-modal Evaluation**: Assesses both style transfer success and content preservation

### Key Results

- **Style Transfer Success**: 89.7% of outputs classified as jazz
- **Genre Classification on Latents**: 98.05% accuracy (VAE embeddings effectively capture genre)
- **Composer Identity Retention**: 29.35% (major challenge - indicates style transfer degrades composer-specific features)

## 🏗️ Architecture

### 1. OctupleMIDI Representation

Raw MIDI files are converted to an 8-dimensional discrete token sequence:
- **Dimension 0**: Measure (0-255)
- **Dimension 1**: Position in measure (0-127)
- **Dimension 2**: Program/Instrument (0-128)
- **Dimension 3**: Pitch (0-127)
- **Dimension 4**: Duration (0-255, log-scaled)
- **Dimension 5**: Velocity (0-31, quantized by 4)
- **Dimension 6**: Time Signature (0-128)
- **Dimension 7**: Tempo (0-49, log-scaled)

**Preprocessing**: Each discrete feature is embedded to 64D continuous vectors, then concatenated into 512D representation.

### 2. Hierarchical VAE (`vae_octuples_hierarchical.py`)

**Architecture**:
- **Encoder**: 
  - 8 independent embedding layers (one per feature) → 512D concatenated input
  - Bidirectional GRU (256D hidden) → 128D latent space (via reparameterization)
  
- **Decoder** (Hierarchical):
  - High-level RNN processes latent into 4 chunks
  - Low-level RNN generates each chunk autoregressively
  - 8 output heads for each feature dimension

**Latent Space**: 128-dimensional continuous vectors where:
- Classical and Jazz music occupy different regions
- Individual composers create composer-specific subregions

### 3. MLP Diffusion Model (`MLPdiffuser.py`)

**Architecture**:
- Time embedding: Sinusoidal positional encoding (128D)
- Input: Concatenate noisy style latent + content latent + time embedding (384D)
- Hidden layers: [512, 128] with batch normalization and dropout
- Output: Noise prediction (128D, same as latent dimension)

**Diffusion Schedule**: 300 timesteps with linear beta schedule (0.0001 to 0.02)

## 📊 Datasets

### Classical Music
- **Source**: 331 MIDI files from composers (Bach, Beethoven, Chopin, Mozart, Schubert, etc.)
- **Preprocessing**: Split into overlapping windows → 4,794 training sequences

### Jazz Music  
- **Source**: 935 MIDI files from various jazz compositions
- **Preprocessing**: Split into overlapping windows → 18,244 training sequences

**Total**: 22,038 tokenized music sequences for training

## 🔧 Training

### VAE Training
```bash
python vae_training.py
```
- Scheduled sampling: Decay TF ratio from 1.0 to 0.1 over 25 epochs
- KL annealing: Gradually increase weight from 0 to 0.15
- Batch size: 64, learning rate: 0.001, epochs: 30
- Best checkpoint: `vae_hierarchical_params/vae_hierarchical_large.pt`


```bash
python vae_training_finetune.py
```
- Learning rate: 1e-4 with cosine scheduling to 1e-5
- KL fixed at 0.15
- AR bursts every 4 batches
- Batch size: 64, epochs: 10
- Best checkpoint: `vae_hierarchical_params/vae_hierarchical_finetune_best.pt`

### Diffusion Model Training
```bash
python MLPdiffuser.py
```
- Learning rate: 1e-4 with cosine scheduling to 1e-6
- T = 300, linear schedule for beta from 0.0001 to 0.02
- Batch size: 64, epochs: 50, patience = 10
- Loss function: MSE (noise prediction)
- Best checkpoint: `diffuser_params/mlp_diffuser_best_small.pt`


## 🎵 Usage

### Single Style Transfer Example
```bash
python test_vae.py
```
This will:
1. Load a random classical MIDI from test set
2. Encode it to a 128D latent via VAE
4. Decode the transferred latent back to MIDI (either using a prefix or pure AR)

**Output files**:
- `original.mid` - Original classical MIDI
- `reconstructed.mid` - VAE reconstruction using TF
- `generated_from_mu.mid` - VAE reconstruction using prefix for AR generation


```bash
python test_diffusion.py
```

This will:
1. Load a random classical MIDI from test set
2. Encode it to a 128D latent via VAE
3. Diffuse a random noise vector conditioned on the classical latent
4. Decode the transferred latent back to MIDI
5. Generate qualitative analysis and spectrograms

**Output files**:
- `original_diffusion.mid` - Original classical MIDI
- `reconstructed_diffusion.mid` - VAE reconstruction
- `style_transferred_jazz.mid` - Diffusion-based style transfer
- `style_transfer_comparison.png` - 8-channel octuple distributions
- `spectrogram_comparison.png` - Audio spectrograms


### Batch Style Transfer
```bash
python saving_latents.py
```
Processes all octuples and saves into latent space for diffusion training.

```bash
python latent_stats.py
```
Checks if latent space is non-collapsed and genres are well defined. Also plots PCA of latent vectors.

```bash
python saving_diffused_classicals.py
```
Processes all classical latents and saves diffused latents for evaluation.

### Evaluation

```bash
python test_diffused_classicals.py
```

Evaluates style transfer success using trained genre and composer classifiers.

## 📈 Evaluation Metrics

### 1. Genre Classification
- **Metric**: Percentage classified as jazz by logistic classifier
- **Success**: Higher percentage = better style transfer
- **Result**: 89.7% ✅

### 2. Composer Identity Retention
- **Metric**: Accuracy of composer prediction on diffused latents
- **Success**: Higher accuracy = better content preservation
- **Result**: 29.35% ❌ (baseline: ~1/10 composers ≈ 10%)
- **Interpretation**: Style transfer significantly alters composer-specific features


## 📁 Project Structure

```
CS230FinalProject/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── vae_octuples_hierarchical.py          # Hierarchical VAE model
├── vae_training_finetune.py              # VAE training script
├── vae_hierarchical_params/               # VAE checkpoints
│   └── vae_hierarchical_finetune_best.pt
│
├── MLPdiffuser.py                         # Diffusion model & training script
├── diffuser_params/                       # Diffusion checkpoints & plots
│   ├── mlp_diffuser_best_small.pt
│   ├── mlp_diffuser_train_losses_small.csv
│   └── mlp_diffuser_val_losses_small.csv
│
├── test_vae.py                      # Single example VAE encode/decode 
├── saving_latents.py                      # Batch saving latents
├── latent_stats.py                      # Check latent space
│
├── test_diffusion.py                      # Single example style transfer + analysis
├── saving_diffused_classicals.py          # Batch processing script
├── test_diffused_classicals.py            # Evaluation on batch of results
│
├── logistic_models/                       # Classifier models
│   ├── binary_classifier.py               # Jazz vs Classical
│   └── Composer_Prediction/
│       └── multi_composer.py              # Composer identification
│
├── musicbert/                             # MIDI preprocessing (from MusicBERT)
│   ├── preprocess.py                      # OctupleMIDI conversion
│   └── README.md
│
├── midi_raw/                              # Raw MIDI files
│   ├── Classical_Midi/                    # 331 classical MIDI files
│   └── Jazz_Midi/                         # 935 jazz MIDI files
│
└── test_octuples/                         # Preprocessed test sequences
    ├── classical_octuples/
    └── jazz_octuples/
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
pip install librosa pretty_midi  # For spectrogram visualization
```

### 2. Preprocess MIDI Data
```bash
cd musicbert
python preprocess.py
# Input: MIDI directory path
# Output: OctupleMIDI text files
cd ..
```

### 3. Train Models (Optional - use pretrained)
```bash
# Train VAE
python vae_training_finetune.py

# Train Diffusion Model
python MLPdiffuser.py

# Train classifiers
python logistic_models/binary_classifier.py
python logistic_models/Composer_Prediction/multi_composer.py
```

## 📚 References

This project builds on:
- **MusicBERT** ([Wang et al., 2020](https://arxiv.org/abs/2106.05630)): OctupleMIDI representation and preprocessing
- **VAE** ([Kingma & Walterman, 2014](https://arxiv.org/abs/1312.6114)): Latent space learning
- **Diffusion Models** ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)): Conditional generation in latent space

## 📝 License

This project is part of a Stanford CS 230 final project.

## 👤 Authors

Created by Simran Nayak and Anya Hansen for CS 230: Deep Learning

---

**For questions or issues, please refer to the project documentation or contact the author.**
