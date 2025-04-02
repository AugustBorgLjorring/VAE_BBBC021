import torch
import torch.optim as optim

import numpy as np
import os
import time
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from vae_model import VAE, BetaVAE, Discriminator, VAEPlus
from data_processing import load_data

# Initialize model based on config
def initialize_model(cfg, device):
    model = None
    discriminator = None
    disc_optimizer = None

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
    elif cfg.model.name == "VAE+":
        # Use the standard VAE as the base for adversarial training
        model = VAEPlus(
            input_channels=cfg.model.input_channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            feature_weight=cfg.model.feature_weight,
            slope=cfg.model.adv_schedule_slope
        ).to(device)
        discriminator = model.discriminator.to(device)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.train.discriminator_lr)
    else:
        raise ValueError(f"Model type {cfg.model.name} not recognized")
    return model, discriminator, disc_optimizer

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg, resolve=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = load_data(cfg, split="train")
    val_loader = load_data(cfg, split="val")

    model, discriminator, disc_optimizer = initialize_model(cfg, device)
    print(model)
    if discriminator is not None:
        print(discriminator)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    bce_loss = torch.nn.BCELoss()

    global_step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()

        train_loss = 0.0
        disc_loss_total = 0.0

        for batch_idx, (x_batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False)):
            global_step += 1
            x_batch = x_batch.to(device)

            # Forward pass through VAE
            recon_batch, mu, logvar = model(x_batch)
            base_loss, recon_loss, kld_loss = model.loss_function(recon_batch, x_batch, mu, logvar)

            if cfg.model.name == "VAE+" and discriminator is not None:
                # Train the discriminator
                loss_disc = model.discriminator_step(
                    x_batch, recon_batch, disc_optimizer, bce_loss, device)
                # Train the VAE with feature matching
                feature_loss, gamma_values, layer_losses = model.vae_step(
                    x_batch, recon_batch, optimizer, global_step, device)
                # Compute the feature matching loss
                total_vae_loss = base_loss + model.feature_weight * feature_loss

                # Backpropagation
                optimizer.zero_grad()
                total_vae_loss.backward()
                optimizer.step()

                # Update the discriminator
                train_loss += total_vae_loss.item()
                disc_loss_total += loss_disc

                # Debugging information
                if batch_idx % 100 == 0:
                    print(f"\nStep {global_step}: Feature Matching Debugging")
                    print(f"  Gamma values: {gamma_values}")
                    print(f"  Layer-wise Feature Matching Loss: {layer_losses}")
                    print(f"  Total Feature Loss: {feature_loss.item()}")

                wandb.log({
                    "epoch": epoch + 1,
                    "batch_vae_loss": total_vae_loss.item(),
                    "reconstruction_loss": recon_loss.item(),
                    "kl_divergence": kld_loss.item(),
                    "batch_feature_loss": feature_loss.item(),
                    "batch_disc_loss": loss_disc
                })
            else:
                optimizer.zero_grad()
                base_loss.backward()
                optimizer.step()
                train_loss += base_loss.item()

                wandb.log({
                    "epoch": epoch + 1,
                    "batch_loss": base_loss.item(),
                    "reconstruction_loss": recon_loss.item(),
                    "kl_divergence": kld_loss.item()
                })

        # Validation step (without discriminator updates)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                recon_batch, mu, logvar = model(x_batch)
                loss, _, _ = model.loss_function(recon_batch, x_batch, mu, logvar)
                val_loss += loss.item()
                
        # Compute average epoch loss
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_train_loss = train_loss / len(train_loader)
        if discriminator is not None:
            avg_disc_loss = disc_loss_total / len(train_loader)
        else:
            avg_disc_loss = 0.0

        print(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss,"epoch_val_loss": avg_val_loss,"epoch_disc_loss": avg_disc_loss})
    # Save trained model
    model_save_path = f"experiments/models/vae_model_{int(time.time())}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # Upload model to W&B
    artifact = wandb.Artifact('vae_model', type='model')
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    train_model()