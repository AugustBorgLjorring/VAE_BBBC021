import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=10):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.01)
        )

        self.fc_mu = nn.Linear(32 * 5 * 5, latent_dim)
        self.fc_var = nn.Linear(32 * 5 * 5, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 32 * 5 * 5)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # Reparameterization trick
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 5, 5)
        x = self.decoder(x)
        return x

    # Forward pass
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var) # (B, D)
        x_recon = self.decode(z)             # (B, C, H, W)
        return x_recon, mu, log_var

    # Loss function
    def loss_function(self, x_recon, x, mu, logvar):
        # Reconstruction: multi-var Normal with sigma=1
        mse_term = 0.5 * ((x_recon - x) ** 2).view(x.size(0), -1).sum(dim=1)  # sum over C,H,W, (B,C,H,W) -flatten-> (B, D) -sum-> (B,)
        D = x[0].numel()
        const = 0.5 * D * math.log(2 * math.pi)  # log constant per sample
        recon_loss = (mse_term + const).mean()   # mean over batch (B,) + scaler -> (B,) + (B,) -mean-> scalar

        # KL divergence between q(z|x) and N(0, I)
        kl_per_sample = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # sum over latent dim (B, D) -sum-> (B,)
        kl_loss = kl_per_sample.mean() # mean over batch (B,) -mean-> scalar

        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss, 0, 0 # mean over batch (scalar, scalar), last two are placeholders for compatibility

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.fc_mu.out_features).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]

class BetaVAE(VAE):
    def __init__(self, input_channels=3, latent_dim=10, beta=4.0):
        super(BetaVAE, self).__init__(input_channels, latent_dim)
        self.beta = beta  # Beta weight for KL divergence

    # Override the loss function
    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Apply Beta weight to KL divergence
        loss = recon_loss + self.beta * kld_loss
        return loss, recon_loss, kld_loss
