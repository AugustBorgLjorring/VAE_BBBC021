import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

import numpy as np
from tqdm import tqdm
import os
import time

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from vae_model import VAE
from data_processing import load_data


# Standard VAE loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return recon_loss + kld_loss, recon_loss, kld_loss


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weights & Biases (wandb)
    wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg, resolve=True))

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data using the HDF5 data loader
    train_loader = load_data(cfg)

    # Initialize model and move to device
    model = VAE(
        input_channels=cfg.model.input_channels,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Training loop
    model.train()
    for epoch in range(cfg.train.epochs):
        train_loss = 0
        for x_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False):
            
            # Move batch to the same device as the model
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch)
            loss, recon_loss, kld_loss = loss_function(recon_batch, x_batch, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Log batch-wise metrics
            wandb.log({
                "epoch": epoch + 1,
                "batch_loss": loss.item(),
                "reconstruction_loss": recon_loss.item(),
                "kl_divergence": kld_loss.item()
            })

        # Logging epoch-wise metrics
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss})

    # Save model checkpoint with time in name of model
    model_save_path = f"experiments/models/vae_model_{int(time.time())}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # Check size is not too large for wandb 1 GB limit
    if os.path.getsize(model_save_path) < 1e9:
        # Log model as an artifact in wandb
        artifact = wandb.Artifact('vae_model', type='model')
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)
    else:
        print(f"Model file size is too large for wandb: {os.path.getsize(model_save_path)} bytes")

    # Properly finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    train_model()