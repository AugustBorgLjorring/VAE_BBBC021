import torch
from omegaconf import OmegaConf
from src.vae_model import VAE, BetaVAE


def load_model_and_cfg(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(checkpoint["config"])
    # cfg.data.train_path = "C:/BBBC021/BBBC021_cleaned_preprocessed.h5"

    # Initialize model based on configuration
    if cfg.model.name == "VAE":
        model = VAE(cfg.model.input_channels, cfg.model.latent_dim)
    elif cfg.model.name == "Beta_VAE":
        model = BetaVAE(cfg.model.input_channels, cfg.model.latent_dim, beta=cfg.model.beta)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    
    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg