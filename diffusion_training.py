import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusermodel import MLPLatentDiffuser, GRULatentDiffuser, TransformerLatentDiffuser, add_noise, compute_loss, LatentDiffusionDataset
import numpy as np
import time

# Load latents saved from saving_latents.py
train_jazz_latents = torch.load("latents/train_jazz_latents.pt")  # [num_pieces, latent_dim]
train_classical_latents = torch.load("latents/train_classical_latents.pt")  # [num_pieces, latent_dim]
val_jazz_latents = torch.load("latents/val_jazz_latents.pt")  # [num_pieces, latent_dim]
val_classical_latents = torch.load("latents/val_classical_latents.pt")  # [num_pieces, latent_dim]
test_jazz_latents = torch.load("latents/test_jazz_latents.pt")  # [num_pieces, latent_dim]
test_classical_latents = torch.load("latents/test_classical_latents.pt")  # [num_pieces, latent_dim]

print("Loaded latents")

# Hyperparameters for validation
batch_size = 64
latent_dim = 128
hidden_dims = [128, 256, 512]
max_noise_level = 1000
num_epochs = 5
learning_rates = [1e-3, 5e-4, 1e-4]
num_layers = [1, 2, 4]
nhead = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create datasets and dataloaders
train_dataset = LatentDiffusionDataset(train_jazz_latents, train_classical_latents)
val_dataset = LatentDiffusionDataset(val_jazz_latents, val_classical_latents)
test_dataset = LatentDiffusionDataset(test_jazz_latents, test_classical_latents)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# run transformer-based diffuser

# validation loss before training
validation_results = {}
training_results = {}
for hidden_dim in hidden_dims:
    for learning_rate in learning_rates:
        for num_layer in num_layers:
            transformer_diffuser = TransformerLatentDiffuser(latent_dim=latent_dim, conditional_dim=latent_dim, hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layer)
            transformer_diffuser.to(device)
            optimizer = torch.optim.Adam(transformer_diffuser.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                transformer_diffuser.train()
                total_loss = 0
                start_time = time.time()

                for batch_idx, (batch_style_latent, batch_content_latent) in enumerate(train_loader):
                    batch_style_latent = batch_style_latent.to(device)
                    batch_content_latent = batch_content_latent.to(device)

                    # Sample random noise level
                    t = torch.randint(1, max_noise_level + 1, (batch_style_latent.size(0),), device=device).float()

                    # Add noise to style latent
                    noisy_style_latent, noise = add_noise(batch_style_latent, t.unsqueeze(-1), max_noise_level)

                    # Forward pass
                    noise_pred = transformer_diffuser(noisy_style_latent, batch_content_latent)

                    # Compute loss
                    loss = compute_loss(noise_pred, noise)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                end = time.time()
                avg_loss = total_loss / len(train_loader)
                training_results[(hidden_dim, learning_rate, num_layer)] = avg_loss
                print(f"Transformer Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Time: {end - start_time:.2f} seconds")

                # validation
                transformer_diffuser.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_idx, (batch_style_latent, batch_content_latent) in enumerate(val_loader):
                        batch_style_latent = batch_style_latent.to(device)
                        batch_content_latent = batch_content_latent.to(device)

                        # Sample random noise level
                        t = torch.randint(1, max_noise_level + 1, (batch_style_latent.size(0),), device=device).float()

                        # Add noise to style latent
                        noisy_style_latent, noise = add_noise(batch_style_latent, t.unsqueeze(-1), max_noise_level)

                        # Forward pass
                        noise_pred = transformer_diffuser(noisy_style_latent, batch_content_latent)

                        # Compute loss
                        loss = compute_loss(noise_pred, noise)
                        val_loss += loss.item()
                
                
                avg_val_loss = val_loss / len(val_loader)
                validation_results[(hidden_dim, learning_rate, num_layer)] = avg_val_loss
                print(f"Transformer Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

# save results
with open("diffusion_training_results.txt", "w") as f:
    f.write("Training Results:\n")
    for params, loss in training_results.items():
        hidden_dim, learning_rate, num_layer = params
        f.write(f"Hidden Dim: {hidden_dim}, Learning Rate: {learning_rate}, Num Layers: {num_layer}, Training Loss: {loss:.4f}\n")
    
    f.write("\nValidation Results:\n")
    for params, loss in validation_results.items():
        hidden_dim, learning_rate, num_layer = params
        f.write(f"Hidden Dim: {hidden_dim}, Learning Rate: {learning_rate}, Num Layers: {num_layer}, Validation Loss: {loss:.4f}\n")


'''
# Run GRU-based diffuser
gru_diffuser = GRULatentDiffuser(latent_dim=latent_dim, conditional_dim=latent_dim, hidden_dim=hidden_dim)
gru_diffuser.to(device)
optimizer = torch.optim.Adam(gru_diffuser.parameters(), lr=learning_rate)

# validation loss before training
evaluate_model(gru_diffuser, val_loader, model_name="GRU Diffuser Before Training")

for epoch in range(num_epochs):
    gru_diffuser.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (batch_style_latent, batch_content_latent) in enumerate(train_loader):
        batch_style_latent = batch_style_latent.to(device)
        batch_content_latent = batch_content_latent.to(device)

        # Sample random noise level
        t = torch.randint(1, max_noise_level + 1, (batch_style_latent.size(0),), device=device).float()

        # Add noise to style latent
        noisy_style_latent, noise = add_noise(batch_style_latent, t.unsqueeze(-1), max_noise_level)

        # Forward pass
        noise_pred = gru_diffuser(noisy_style_latent, batch_content_latent)

        # Compute loss
        loss = compute_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    end = time.time()
    avg_loss = total_loss / len(train_loader)
    print(f"GRU Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Time: {end - start_time:.2f} seconds")

    # validation
    gru_diffuser.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (batch_style_latent, batch_content_latent) in enumerate(val_loader):
            batch_style_latent = batch_style_latent.to(device)
            batch_content_latent = batch_content_latent.to(device)

            # Sample random noise level
            t = torch.randint(1, max_noise_level + 1, (batch_style_latent.size(0),), device=device).float()

            # Add noise to style latent
            noisy_style_latent, noise = add_noise(batch_style_latent, t.unsqueeze(-1), max_noise_level)

            # Forward pass
            noise_pred = gru_diffuser(noisy_style_latent, batch_content_latent)

            # Compute loss
            loss = compute_loss(noise_pred, noise)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"GRU Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")


'''