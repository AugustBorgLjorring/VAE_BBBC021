# src/train.py
import os
import time

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import wandb

from vae_model import VAE, BetaVAE, VAEPlus
from data_processing import load_data

from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

# Helpers from your original script:
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x = x_batch.to(device).float()
            recon, mu, logvar = model(x)
            ret = model.loss_function(recon, x, mu, logvar)
            total_loss = ret[0]
            val_loss += total_loss.item()
    return val_loss / len(val_loader)

def get_gradients(model, mu, logvar):
    enc_grads = [p.grad.detach().abs().mean() for p in model.encoder.parameters() if p.grad is not None]
    grad_enc = torch.stack(enc_grads).mean().item() if enc_grads else 0.0
    grad_mu = mu.grad.detach().abs().mean().item()     if mu.grad     is not None else 0.0
    grad_logvar = logvar.grad.detach().abs().mean().item() if logvar.grad is not None else 0.0
    dec_grads = [p.grad.detach().abs().mean() for p in model.decoder.parameters() if p.grad is not None]
    grad_dec = torch.stack(dec_grads).mean().item() if dec_grads else 0.0
    return grad_enc, grad_mu, grad_logvar, grad_dec

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    path = f"experiments/models/vae_checkpoint_{int(time.time())}.pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    artifact = wandb.Artifact('vae_checkpoint', type='model')
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    print(f"Model checkpoint saved at {path}")

# Initialize model based on config
def initialize_model(cfg, device):
    name = cfg.model.name.lower()
    if name in ("vae",):
        return VAE(cfg.model.input_channels, cfg.model.latent_dim).to(device), None
    elif name in ("beta_vae", "betavae", "beta-vae"):
        return BetaVAE(cfg.model.input_channels, cfg.model.latent_dim, cfg.model.beta).to(device), None
    elif name in ("vae_plus", "vae-plus", "vae+"):
        model = VAEPlus(
            cfg.model.input_channels,
            cfg.model.latent_dim,
            cfg.model.beta,
            cfg.model.adv_schedule_slope
        ).to(device)
        return model
    else:
        raise ValueError(f"Model type {cfg.model.name} not recognized")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    # # restore cwd so data paths resolve exactly as before
    # os.chdir(get_original_cwd())
    # print(f"Working dir restored to {os.getcwd()}\n")
    print(OmegaConf.to_yaml(cfg))

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # wandb init
    wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg, resolve=True))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # data
    train_loader = load_data(cfg, split="train")
    val_loader   = load_data(cfg, split="val")

    steps_per_epoch = len(train_loader)
    cfg.model.adv_schedule_slope = steps_per_epoch * 2 
    print(cfg.model.adv_schedule_slope)
    # model + optimizers
    model = initialize_model(cfg, device)


    model.vae_modules = nn.ModuleList([
        model.encoder,
        model.decoder,
        model.fc_mu,
        model.fc_logvar,
        model.decoder_input
    ])

    vae_optimizer = optim.Adam(model.vae_modules.parameters(), lr=cfg.train.learning_rate)
    
    if model.adv:
        disc_optimizer = optim.SGD(model.discriminator.parameters(), lr=cfg.train.discriminator_lr, momentum=0.9)

    # training
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0

        # if epoch >= cfg.train.epochs -2:
        #     S = 5
        # else:
        #     S = 1

        for x_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", leave=False):
            x_batch = x_batch.to(device).float()

            recon, mu, logvar = model(x_batch)

            # 1) Discriminator step
            if model.adv:
                d_loss = model.loss_discriminator(recon, x_batch)
                disc_optimizer.zero_grad()
                d_loss.backward()

                clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()

                model.t += 1

            # 2) VAE step
            mu.retain_grad(); logvar.retain_grad()
        
            if model.adv:
                total_loss, recon_loss, kld_loss, feat_loss, gammas, feat_norm_real, feat_norm_fake = model.loss_function(recon, x_batch, mu, logvar)
            else:
                total_loss, recon_loss, kld_loss, _, _ = model.loss_function(recon, x_batch, mu, logvar)
                feat_norm_real, feat_norm_fake = 0.0, 0.0

            vae_optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.vae_modules.parameters(), max_norm=1.0)

            grad_enc, grad_mu, grad_logvar, grad_dec = get_gradients(model, mu, logvar)
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
                "adv_feature_loss":    feat_loss.item(),
                "grad_encoder":        grad_enc,
                "grad_mu":             grad_mu,
                "grad_logvar":         grad_logvar,
                "grad_decoder":        grad_dec,
                "feat_norm_real":      feat_norm_real,
                "feat_norm_fake":      feat_norm_fake
            }

            if model.adv:
                log_dict["batch_disc_loss"] = d_loss.item()

                gamma_dict = {f"gamma/layer_{i}": float(g) for i, g in enumerate(gammas)}
                log_dict.update(gamma_dict)

            wandb.log(log_dict)
        # --- end of epoch logs ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = validate_model(model, val_loader, device)

        # print to terminal
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # W&B epoch‐level logging
        wandb.log({
            "epoch":             epoch + 1,
            "epoch_train_loss":  avg_train_loss,
            "epoch_val_loss":    avg_val_loss,
            "epoch_recon_loss":  total_recon_loss / len(train_loader),
            "epoch_kld_loss":    total_kld_loss / len(train_loader)
        })

    # final checkpoint
    save_checkpoint(model, vae_optimizer, epoch, avg_train_loss, avg_val_loss, cfg)

if __name__ == "__main__":
    train_model()
