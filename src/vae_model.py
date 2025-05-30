import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VAEMedium(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        # Encoder: 4 conv layers
        # Input pictures are (B, C, H, W) = (B, 3, 68, 68)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2), # 68x68 -> 34x34
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),          # 34x34 -> 17x17
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),          # 17x17 -> 9x9
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2), # 9x9 -> 5x5
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(32 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(32 * 5 * 5, latent_dim)
        # Decoder: linear + 4 transpose convs
        self.decoder_input = nn.Linear(latent_dim, 32 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 5, 5)),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=0), # 5x5 -> 9x9
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=0), # 9x9 -> 17x17
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), # 17x17 -> 34x34
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1), # 34x34 -> 68x68
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)    # (B, 32*5*5)
        mu     = self.fc_mu(h) # (B, latent_dim)
        logvar = torch.clamp(self.fc_logvar(h), -6.0, 3.0) # (B, latent_dim)
        return mu, logvar

    # Reparameterization trick
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

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

        return total_loss, recon_loss, kl_loss, 0  # mean over batch (scalar, scalar), last is a placeholder for compatibility

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.fc_mu.out_features).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]

class VAESmall(VAEMedium):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__(in_channels, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2), # 68x68 -> 34x34
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),          # 34x34 -> 17x17
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),          # 17x17 -> 9x9
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(16 * 9 * 9, latent_dim)
        self.fc_logvar = nn.Linear(16 * 9 * 9, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 16 * 9 * 9)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 9, 9)),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=0), # 9x9 -> 17x17
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # 17x17 -> 34x34
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1), # 34x34 -> 68x68
            nn.Sigmoid()
        )

class VAELarge(VAEMedium):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__(in_channels, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2), # 68x68 -> 34x34
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),          # 34x34 -> 17x17
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),          # 17x17 -> 9x9
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 9x9 -> 5x5
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 5, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 5, 5)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=0), # 5x5 -> 9x9
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=0), # 9x9 -> 17x17
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1), # 17x17 -> 34x34
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1), # 34x34 -> 68x68
            nn.Sigmoid()
        )

class BetaVAE(VAEMedium):
    def __init__(self, in_channels=3, latent_dim=256, beta=1.0):
        super().__init__(in_channels, latent_dim)
        self.beta = beta

    def loss_function(self, x_recon, x, mu, logvar):
        _, recon_loss, kl_loss, _ = super().loss_function(x_recon, x, mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss # beta scaling of KL divergence
        return total_loss, recon_loss, kl_loss, 0  # last is a placeholder for compatibility

class VAEPlus(BetaVAE):
    def __init__(self, in_channels=3, latent_dim=256, beta=1.0, T=1, use_adverserial=True):
        super().__init__(in_channels, latent_dim, beta)
        self.T = T
        self.t = 0
        self.gamma_values = []
        self.adv = use_adverserial
        if self.adv:
            self.discriminator = Discriminator(in_channels)
    
    # Increment each feat layers gamma value every t steps (t batches)
    def gamma(self, i: int):
        return min(max((self.t/self.T) - i, 0.0), 1.0) # From Lafarge2019

    def loss_function(self, x_recon, x, mu, logvar):
        _, recon_loss, kl_loss, _ = super().loss_function(x_recon, x, mu, logvar)

        if self.adv:
            adv_fm_loss_per_sample = torch.zeros(x.size(0), device=x.device)  # (B,)
            gamma_values = []

            # Freeze D
            self.discriminator.requires_grad_(False)

            _, feats_real = self.discriminator(x)
            _, feats_fake = self.discriminator(x_recon)

            # Unfreeze D
            self.discriminator.requires_grad_(True)
            
            for i, (real, fake) in enumerate(zip(feats_real, feats_fake)):
                g = self.gamma(i)
                gamma_values.append(g)
                # Per-sample feature loss
                mse = F.mse_loss(real, fake, reduction='none')  # (B, C, H, W)
                mse = 0.5 * mse.view(x.size(0), -1).sum(dim=1)  # (B,)
                
                # Apply gamma
                adv_fm_loss_per_sample += g * mse                    # scaler + (B,) -> (B,) + (B,) -sum-> (B,)

            adv_fm_loss = adv_fm_loss_per_sample.mean()              # mean over batch (B,) -mean-> scalar

            self.gamma_values = gamma_values # Store gamma values for logging

            total_loss = recon_loss + self.beta * kl_loss + adv_fm_loss
        else:
            # Return the standard VAE loss
            adv_fm_loss = 0

            total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss, adv_fm_loss


    def loss_discriminator(self, x_recon, x):
        # Discriminator loss: binary cross-entropy between real and fake images
        real_logits, _ = self.discriminator(x)
        fake_logits, _ = self.discriminator(x_recon.detach())

        bce = F.binary_cross_entropy_with_logits
        target_real = torch.ones_like(real_logits)  # Label for real images = 1
        target_fake = torch.zeros_like(fake_logits) # Label for fake images = 0

        loss_real = bce(real_logits, target_real)
        loss_fake = bce(fake_logits, target_fake)
        return loss_real + loss_fake


# Based on: Lafarge et al., "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning" (2019)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2) # 68x68 -> 34x34
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2) # 34x34 -> 17x17
        # Conv block 3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2) # 17x17 -> 9x9
        # Conv block 4
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) # 9x9 -> 5x5
        
        # Final classifier conv → raw logit
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 5 * 5, 1)

    def forward(self, x):
        feature_maps = []
        # Block 1
        x1 = F.leaky_relu(self.conv1(x), 0.01)
        feature_maps.append(x1)
        # Block 2
        x2 = F.leaky_relu(self.conv2(x1), 0.01)
        feature_maps.append(x2)
        # Block 3
        x3 = F.leaky_relu(self.conv3(x2), 0.01)
        feature_maps.append(x3)
        # Block 4
        x4 = F.leaky_relu(self.conv4(x3), 0.01)
        feature_maps.append(x4)
        # Classifier
        x_flat = self.flatten(x4)
        logits = self.classifier(x_flat)
        return logits, feature_maps