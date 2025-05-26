import torch
import torch.optim as optim

import numpy as np
import os
import time
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from vae_model import VAE, BetaVAE
from src.data_loading import load_data

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

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    model_save_path = f"experiments/models/vae_checkpoint_{int(time.time())}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(checkpoint, model_save_path)
    
    # Upload checkpoint to W&B
    artifact = wandb.Artifact('vae_checkpoint', type='model')
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    
    print(f"Model checkpoint saved at {model_save_path}")

# Validate model on validation set
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            recon_batch, mu, logvar = model(x_batch)
            loss, _, _ = model.loss_function(recon_batch, x_batch, mu, logvar)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def get_gradients(model, mu, logvar):
    # Compute the average gradient norm for all parameters in the encoder
    encoder_grads = [p.grad.detach().abs().mean() for p in model.encoder.parameters() if p.grad is not None]
    encoder_grad_norm = torch.stack(encoder_grads).mean() if encoder_grads else torch.tensor(0.)
    
    # Compute gradient norm for mu and logvar individually
    mu_grad_norm = mu.grad.detach().abs().mean() if mu.grad is not None else torch.tensor(0.)
    logvar_grad_norm = logvar.grad.detach().abs().mean() if logvar.grad is not None else torch.tensor(0.)
    
    # Compute the average gradient norm for all parameters in the decoder
    decoder_grads = [p.grad.detach().abs().mean() for p in model.decoder.parameters() if p.grad is not None]
    decoder_grad_norm = torch.stack(decoder_grads).mean() if decoder_grads else torch.tensor(0.)
    
    return encoder_grad_norm.item(), mu_grad_norm.item(), logvar_grad_norm.item(), decoder_grad_norm.item()

# Train model
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize Weights & Biases (wandb)
    wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg, resolve=True))

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data using the HDF5 data loader
    train_loader = load_data(cfg, split="train")
    val_loader = load_data(cfg, split="val")

    # Initialize model and move to device
    model = initialize_model(cfg, device)   
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Training loop
    model.train()
    for epoch in range(cfg.train.epochs):
        train_loss = 0.0

        # Multiple samples for Monte Carlo estimation at the end of training
        if epoch >= cfg.train.epochs - 2:
            S = 5
        else:
            S = 1

        for x_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False):
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch, S) # (S, B, C, H, W),  (B, D), (B, D)
            
            mu.retain_grad()
            logvar.retain_grad()

            loss, recon_loss, kld_loss = model.loss_function(recon_batch, x_batch, mu, logvar) # (S, B, C, H, W), (B, C, H, W), (B, D), (B, D)

            loss.backward()

            encoder_grad, mu_grad, logvar_grad, decoder_grad = get_gradients(model, mu, logvar)

            optimizer.step()

            train_loss += loss.item()

            # Log batch-wise metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "batch_loss": loss.item(),
                "reconstruction_loss": recon_loss.item(),
                "kl_divergence": kld_loss.item(),
                "grad_encoder": encoder_grad,
                "grad_mu": mu_grad,
                "grad_logvar": logvar_grad,
                "grad_decoder": decoder_grad,
            })
                 
        # Compute average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, device)
        grad_encoder, grad_mu, grad_logvar, grad_decoder = get_gradients(model, mu, logvar)
        
        # Log epoch-wise metrics to W&B
        print(f"Epoch [{epoch + 1}/{cfg.train.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss, "epoch_val_loss": avg_val_loss,
                    "grad_encoder": grad_encoder, "grad_mu": grad_mu, "grad_logvar": grad_logvar, "grad_decoder": grad_decoder})
        
        # Switch back to training mode for next epoch
        model.train()

    # Save model checkpoint
    save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_val_loss, cfg)

if __name__ == "__main__":
    train_model()
