import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("vae_hierarchical_params/hierarchical_epoch_losses.csv")

# Handle duplicated epoch row if present
df = df.groupby("epoch", as_index=False).mean()

# Figure style
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# -------------------------
# Panel 1: Total Loss
# -------------------------
axes[0].plot(df["epoch"], df["train_loss"], label="Train Total Loss", linewidth=2, color='blue')
axes[0].plot(df["epoch"], df["test_loss"], label="Val Total Loss", linewidth=2, color='orange')

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Total Loss")
#axes[0].set_title("Training vs Val Loss Across Epochs")
axes[0].legend()

# -------------------------
# Panel 2: KL Divergence
# -------------------------
axes[1].plot(df["epoch"], df["train_kl"], label="Train KL", linewidth=2, color='blue')
axes[1].plot(df["epoch"], df["test_kl"], label="Val KL", linewidth=2, color='orange')

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("KL Divergence")
axes[1].legend()

# -------------------------
# Panel 3: Reconstruction Loss
# -------------------------
axes[2].plot(df["epoch"], df["train_recon"], label="Train Reconstruction Loss", linewidth=2, color='blue')
axes[2].plot(df["epoch"], df["test_recon"], label="Val Reconstruction Loss", linewidth=2, color='orange')

axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Reconstruction Loss")
axes[2].legend()

plt.tight_layout()
plt.show()

# save figure
fig.savefig("hierarchical_vae_loss_plots.png")
