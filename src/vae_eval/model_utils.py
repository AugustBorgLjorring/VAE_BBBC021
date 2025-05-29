import torch
from omegaconf import OmegaConf
from src.vae_model import VAESmall, VAEMedium, VAELarge, BetaVAE, VAEPlus


def load_model_and_cfg(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(checkpoint["config"])
    # cfg.data.train_path = "C:/BBBC021/BBBC021_cleaned_preprocessed.h5"

    # Initialize model based on configuration
    name = cfg.model.name
    if name in ("VAESmall"):
        model = VAESmall(cfg.model.input_channels, cfg.model.latent_dim)
    elif name in ("VAEMedium"):
        model = VAEMedium(cfg.model.input_channels, cfg.model.latent_dim)
    elif name in ("VAELarge"):
        model = VAELarge(cfg.model.input_channels, cfg.model.latent_dim)
    elif name in ("Beta_VAE"):
        model = BetaVAE(cfg.model.input_channels, cfg.model.latent_dim, cfg.model.beta)
    elif name in ("VAE+"):
        model = VAEPlus(cfg.model.input_channels, cfg.model.latent_dim, cfg.model.beta, cfg.model.adv_schedule_slope)

    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg