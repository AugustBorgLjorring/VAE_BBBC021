import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=10):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    # def reparameterize(self, mu, log_var):
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std
    
    def reparameterize(self, mu, log_var, S=1):
        std = torch.exp(0.5 * log_var)            # (B, D)
        B, D = mu.shape
        eps = torch.randn(S, B, D).to(mu.device)  # (S, B, D)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x

    # def forward(self, x):
    #     mu, log_var = self.encode(x)
    #     z = self.reparameterize(mu, log_var)
    #     x_recon = self.decode(z)
    #     return x_recon, mu, log_var

    def forward(self, x, S=1):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, S)       # (S, B, D)
        z_flat = z.view(S * mu.size(0), -1)           # (S*B, D)
        x_flat = self.decode(z_flat)                  # (S*B, C, H, W)
        # reshape back to (S, B, C, H, W)
        C, H, W = x_flat.shape[1:]
        x_recon = x_flat.view(S, mu.size(0), C, H, W) # (S, B, C, H, W)

        return x_recon, mu, log_var

    # def loss_function(self, recon_x, x, mu, log_var):
    #     B = x.size(0)
    #     D = x.size(1) * x.size(2) * x.size(3) 
    #     sigma = 1
    #     recon_loss = 1/(2*sigma**2) * F.mse_loss(recon_x, x, reduction='sum') + 1/(2*sigma**2) * D * B * math.log(2 * math.pi)
    #     kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #     loss = recon_loss + kld_loss
    #     return loss, recon_loss, kld_loss

    def loss_function(self, x_recon, x, mu, log_var):
        S, B, C, H, W = x_recon.shape
        D = C * H * W
        sigma = 1
        # 1) reconstruction term: average over S sum over pixels then avg over samples
        squared_error  = F.mse_loss(x_recon, x.unsqueeze(0).expand_as(x_recon), reduction='none').view(S, B, -1).sum(-1)     # (S, B) per-sample MSE
        mse_term = (1 / (2 * sigma ** 2)) * squared_error.mean(0)  # (B,)
        constant = (D / 2) * math.log(2 * math.pi * sigma ** 2)
        recon_loss = (mse_term + constant).sum()  # (B,)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # (B,)
        loss = recon_loss + kld_loss
        return loss, recon_loss, kld_loss

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