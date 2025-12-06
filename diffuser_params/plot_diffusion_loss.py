import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the loss files
train_losses = pd.read_csv("mlp_diffuser_train_losses_small.csv", header=None, names=['train_loss'])
val_losses = pd.read_csv("mlp_diffuser_val_losses_small.csv", header=None, names=['val_loss'])

# Create epoch numbers
epochs = np.arange(1, len(train_losses) + 1)

# Create DataFrame for easier handling
df = pd.DataFrame({
    'epoch': epochs,
    'train_loss': train_losses['train_loss'],
    'val_loss': val_losses['val_loss']
})

# Figure style
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot training and validation losses
ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2, color='blue')
ax.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2, color='orange')

ax.set_xlabel("Epoch")
ax.set_ylabel("Diffusion Loss")
ax.set_title("MLP Diffusion Model: Training vs Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Add some statistics
min_val_loss = df["val_loss"].min()
min_val_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
ax.axvline(x=min_val_epoch, color='red', linestyle=':', alpha=0.7, label=f'Best Val Epoch: {min_val_epoch}')
ax.legend()

plt.tight_layout()
plt.show()

# Save figure
fig.savefig("diffusion_loss_plots.png", dpi=300, bbox_inches='tight')

# Print some statistics
print(f"\n=== Diffusion Training Statistics ===")
print(f"Total epochs: {len(df)}")
print(f"Final train loss: {df['train_loss'].iloc[-1]:.6f}")
print(f"Final val loss: {df['val_loss'].iloc[-1]:.6f}")
print(f"Best val loss: {min_val_loss:.6f} (epoch {min_val_epoch})")
print(f"Initial train loss: {df['train_loss'].iloc[0]:.6f}")
print(f"Loss reduction: {(df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]) / df['train_loss'].iloc[0] * 100:.1f}%")

# Check for overfitting
final_gap = df['val_loss'].iloc[-1] - df['train_loss'].iloc[-1]
print(f"Final train-val gap: {final_gap:.6f}")
if final_gap > 0.05:
    print("⚠️  Potential overfitting detected (val > train by >0.05)")
else:
    print("✅ No significant overfitting detected")