import os
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from vae_model import VAE, BetaVAE, VAEPlus
from data_loading import load_data

from torch.nn.utils import clip_grad_norm_

# Validate model on validation set
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_kld_loss = 0.0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x = x_batch.to(device).float()
            recon, mu, logvar = model(x)
            losses = model.loss_function(recon, x, mu, logvar)
            val_loss += losses[0].item()
            val_recon_loss += losses[1].item()
            val_kld_loss += losses[2].item()
    return val_loss / len(val_loader), val_recon_loss / len(val_loader), val_kld_loss / len(val_loader)

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'disc_optimizer_state_dict': model.discriminator.state_dict() if hasattr(model, 'discriminator') else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }

    model_save_path = f"experiments/models/vae_checkpoint_{START_TIME}/epoch_{epoch+1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(checkpoint, model_save_path)

    # Upload checkpoint to W&B
    if epoch == cfg.train.epochs - 1:
        artifact = wandb.Artifact('vae_checkpoint', type='model')
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print(f"Model checkpoint saved at {model_save_path}")

# Initialize model based on config
def initialize_model(cfg, device):
    name = cfg.model.name
    if name in ("VAE"):
        return VAE(cfg.model.input_channels, cfg.model.latent_dim).to(device)
    elif name in ("Beta_VAE"):
        return BetaVAE(cfg.model.input_channels, cfg.model.latent_dim, cfg.model.beta).to(device)
    elif name in ("VAE+"):
        model = VAEPlus(cfg.model.input_channels, cfg.model.latent_dim, cfg.model.beta, cfg.model.adv_schedule_slope).to(device)
        return model
    else:
        raise ValueError(f"Model type {cfg.model.name} not recognized")
    
START_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Wandb init
    wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg, resolve=True))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_loader = load_data(cfg, split="train")
    val_loader   = load_data(cfg, split="val")

    # Initialize model with configuration
    model = initialize_model(cfg, device)

    vae_params = (
        list(model.encoder.parameters()) +
        list(model.fc_mu.parameters()) + 
        list(model.fc_logvar.parameters()) +
        list(model.decoder_input.parameters()) + 
        list(model.decoder.parameters())
    )

    # Initialize optimizers
    vae_optimizer = optim.Adam(vae_params, lr=cfg.train.learning_rate)
    if model.adv:
        disc_optimizer = optim.SGD(model.discriminator.parameters(), lr=cfg.train.discriminator_lr, momentum=0.9)

    # Training
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0

        for x_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", leave=False):
            x_batch = x_batch.to(device)

            if not model.adv:
                recon, mu, logvar = model(x_batch)
            else:
                # Discriminator
                # Split batch into two halves for adversarial training
                half = x_batch.size(0) // 2
                batch1 = x_batch[:half]
                batch2 = x_batch[half:]

                recon, mu, logvar = model(batch1)

                # L = BCE loss of real (batch 2) and fake images
                d_loss = model.loss_discriminator(recon, batch2)
                disc_optimizer.zero_grad()
                d_loss.backward()

                # Clip gradients for discriminator to prevent exploding gradients
                clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()

                model.t += 1

            # VAE loss
            if not model.adv:
                # L = recon_loss + kld_loss
                total_loss, recon_loss, kld_loss, _, _ = model.loss_function(recon, x_batch, mu, logvar)
            else:
                # L = recon_loss + kld_loss + adv_feature_loss
                total_loss, recon_loss, kld_loss, feat_loss, gammas = model.loss_function(recon, batch1, mu, logvar)

            vae_optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients for VAE parameters to prevent exploding gradients
            clip_grad_norm_(vae_params, max_norm=1.0)

            vae_optimizer.step()

            train_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()

            # --- batch‐level logging (exactly as before + discriminator) ---
            log_dict = {
                "epoch":              epoch + 1,
                "batch_loss":         total_loss.item(),
                "reconstruction_loss": recon_loss.item(),
                "kl_divergence":       kld_loss.item(),
                "adv_feature_loss":    feat_loss.item()
            }

            # Log gamma values if applicable
            if model.adv:
                log_dict["batch_disc_loss"] = d_loss.item()
                gamma_dict = {f"gamma/layer_{i}": float(g) for i, g in enumerate(gammas)}
                log_dict.update(gamma_dict)

            # Log to wandb
            wandb.log(log_dict)
            
        # Calculate average epoch loss
        avg_train_loss = train_loss / len(train_loader)

        # Validate model
        avg_val_loss, avg_val_recon_loss, avg_val_kld_loss = validate_model(model, val_loader, device)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # W&B epoch‐level logging
        wandb.log({
            "epoch":             epoch + 1,
            "epoch_train_loss":  avg_train_loss,
            "epoch_val_loss":    avg_val_loss,
            "epoch_recon_loss":  total_recon_loss / len(train_loader),
            "epoch_kld_loss":    total_kld_loss / len(train_loader),
            "epoch_val_recon_loss": avg_val_recon_loss,
            "epoch_val_kld_loss": avg_val_kld_loss
        })

        # Save model checkpoint every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == cfg.train.epochs - 1:
            save_checkpoint(model, vae_optimizer, epoch, avg_train_loss, avg_val_loss, cfg)

if __name__ == "__main__":
    train_model()
