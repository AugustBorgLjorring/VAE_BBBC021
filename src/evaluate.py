import os
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra

from src.vae_model import VAE
from src.data_processing import load_data

# Avoid OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load the trained model
    model = VAE(
        input_channels=cfg.model.input_channels,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim
    )
    
    model_path = "experiments/models/vae_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load a sample batch of data
    data_loader = load_data(cfg)
    x_batch, _ = next(iter(data_loader))

    # Select the first 5 images from the batch
    num_images = 5
    images = x_batch[:num_images]

    # Pass the images through the model to get the reconstructions
    with torch.no_grad():
        reconstructions, _, _ = model(images)

    # Convert tensors to numpy arrays for visualization
    images = images.permute(0, 2, 3, 1).numpy()  # (B, C, H, W) -> (B, H, W, C)
    reconstructions = reconstructions.permute(0, 2, 3, 1).numpy()

    # Plot the original and reconstructed images
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        plt.title("Original")

        # Reconstructed images
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructions[i], cmap="gray")
        plt.axis("off")
        plt.title("Reconstructed")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
