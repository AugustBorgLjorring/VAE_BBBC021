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

from vae_model import VAE, BetaVAE
from data_processing import load_data

# Initialize model based on config
def initialize_model(cfg, device):
    if cfg.model.name == "VAE":
        model = VAE(
            input_channels=cfg.model.input_channels,
            latent_dim=cfg.model.latent_dim
        ).to(device)
    elif cfg.model.name == "Beta_VAE":
        model = BetaVAE(
            input_channels=cfg.model.input_channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta
        ).to(device)
    else:
        raise ValueError(f"Model type {cfg.model.name} not recognized")
    return model


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
    model = initialize_model(cfg, device)   

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Training loop (no AMP)
    model.train()
    for epoch in range(cfg.train.epochs):
        train_loss = 0.0

        for x_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False):
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch)
            loss, recon_loss, kld_loss = model.loss_function(recon_batch, x_batch, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log batch-wise metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "batch_loss": loss.item(),
                "reconstruction_loss": recon_loss.item(),
                "kl_divergence": kld_loss.item()
            })

        # Compute average epoch loss
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss})

    model_save_path = f"experiments/models/vae_model_{int(time.time())}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    artifact = wandb.Artifact('vae_model', type='model')
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    train_model()