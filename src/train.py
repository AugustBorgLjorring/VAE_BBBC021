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

# Helpers from your original script:
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x = x_batch.to(device).float()
            recon, mu, logvar = model(x)
            loss, _, _, _ = model.loss_function(recon, x, mu, logvar)
            val_loss += loss.item()
    return val_loss / len(val_loader.dataset)

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
            cfg.model.feature_weight,
            cfg.model.adv_schedule_slope
        ).to(device)
        disc_opt = optim.Adam(model.discriminator.parameters(), lr=cfg.train.discriminator_lr)
        return model, disc_opt
    else:
        raise ValueError(f"Model type {cfg.model.name} not recognized")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    # restore cwd so data paths resolve exactly as before
    os.chdir(get_original_cwd())
    print(f"Working dir restored to {os.getcwd()}\n")
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

    # model + optimizers
    model, disc_optimizer = initialize_model(cfg, device)
    vae_optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # training
    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0
        disc_loss_total = 0.0

        # if epoch >= cfg.train.epochs -2:
        #     S = 5
        # else:
        #     S = 1

        for batch_idx, (x_batch,_) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", leave=False), start = 1):
            global_step += 1
            x_batch = x_batch.to(device).float()
            # 1) discriminator step
            if disc_optimizer:
                x_recon, mu, logvar = model(x_batch)
                d_loss = model.loss_discriminator(x_batch, x_recon)
                disc_optimizer.zero_grad(); d_loss.backward(); disc_optimizer.step()
                disc_loss_total += d_loss.item()
            else:
                d_loss = 0.

            # 2) VAE step
            recon, mu, logvar = model(x_batch)
            mu.retain_grad(); logvar.retain_grad()
            recon_loss, kld_loss, feat_loss, gammas = model.loss_function(recon, x_batch, mu, logvar)

            if batch_idx & 100 == 0:
                print(f"[Step {global_step}] Recon={recon_loss.item():.4f}, "
                    f"KL={kld_loss.item():.4f}, AdvFeat={d_loss.item():.4f}")
                print("    γ:", [f"{g:.3f}" for g in gammas])

            total_loss = recon_loss + model.beta * kld_loss + feat_loss


            # backward + step
            vae_optimizer.zero_grad()
            total_loss.backward()
            # capture gradients
            grad_enc, grad_mu, grad_logvar, grad_dec = get_gradients(model, mu, logvar)
            vae_optimizer.step()

            train_loss += total_loss.item()

            # --- batch‐level logging (exactly as before + discriminator) ---
            log_dict = {
                "epoch":              epoch + 1,
                "batch_loss":         total_loss.item() / x_batch.size(0),
                "reconstruction_loss": recon_loss.item() / x_batch.size(0),
                "kl_divergence":       kld_loss.item() / x_batch.size(0),
                "grad_encoder":        grad_enc,
                "grad_mu":             grad_mu,
                "grad_logvar":         grad_logvar,
                "grad_decoder":        grad_dec
            }
            if disc_optimizer:
                log_dict["batch_disc_loss"] = d_loss.item() / x_batch.size(0)

            gamma_dict = {f"gamma/layer_{i}": float(g) for i, g in enumerate(gammas)}
            log_dict.update(gamma_dict)

            wandb.log(log_dict, step=global_step)
        # --- end of epoch logs ---
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss   = validate_model(model, val_loader, device)
        avg_disc_loss  = disc_loss_total / len(train_loader) if disc_optimizer else 0.0

        # print to terminal
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

        # W&B epoch‐level logging
        wandb.log({
            "epoch":             epoch + 1,
            "epoch_train_loss":  avg_train_loss,
            "epoch_val_loss":    avg_val_loss,
            "epoch_disc_loss":   avg_disc_loss
        })

        # Monitor gammas
        # print(f"[Batch {global_step}] gammas:", gammas.detach().cpu().numpy())

    # final checkpoint
    save_checkpoint(model, vae_optimizer, epoch, avg_train_loss, avg_val_loss, cfg)

if __name__ == "__main__":
    train_model()
