import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math

# Based on: Lafarge et al., "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning" (2019)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        # Conv block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        
        # Final classifier conv → raw logit
        self.classifier = nn.Conv2d(256, 1, kernel_size=4)

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
        logits = self.classifier(x4).view(x4.size(0), -1).squeeze(1)
        return logits, feature_maps

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        # Encoder: 4 conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Conv2d(64,128, 5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Conv2d(128,256,5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        # Decoder: linear + 4 transpose convs
        self.decoder_input = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256,4,4)),
            nn.ConvTranspose2d(256,128,5,2,2,output_padding=1), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128,64,5,2,2,output_padding=1),  nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64,32,5,2,2,output_padding=1),   nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32,in_channels,5,2,2,output_padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        # Reconstruction: multi-var Normal with sigma=1
        mse_term = 0.5 * ((x_recon - x) ** 2).view(x.size(0), -1).sum(dim=1)  # sum over C,H,W, (B,C,H,W) -flatten-> (B, D) -sum-> (B,)
        D = x[0].numel()
        const = 0.5 * D * math.log(2 * math.pi)  # log constant per sample
        recon_loss = (mse_term + const).mean()  # mean over batch (B,) + scaler -> (B,) + (B,) -mean-> scalar

        # KL divergence between q(z|x) and N(0, I)
        kl_per_sample = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1) # sum over latent dim (B, D) -sum-> (B,)
        kl_loss = kl_per_sample.mean() # mean over batch (B,) -mean-> scalar
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss, 0, 0 # mean over batch (scalar, scalar), last two are placeholders for compatibility

class BetaVAE(VAE):
    def __init__(self, in_channels=3, latent_dim=256, beta=1.0):
        super().__init__(in_channels, latent_dim)
        self.beta = beta

    def loss_function(self, x_recon, x, mu, logvar):
        _, recon_loss, kl_loss, _, _ = super().loss_function(x_recon, x, mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss, 0, 0  # last two are placeholders for compatibility

class VAEPlus(BetaVAE):
    def __init__(self, in_channels=3, latent_dim=256,
                 beta=1.0, T=0, use_adverserial=True):
        super().__init__(in_channels, latent_dim, beta)
        self.T = T
        self.t = 0
        self.adv = use_adverserial
        if self.adv:
            self.discriminator = Discriminator(in_channels)
      
    def gamma(self, i: int):
        return min(max((self.t/self.T) - i, 0.0), 1.0) # From Lafarge2019

    def loss_function(self, x_recon, x, mu, logvar):
        _, recon_loss, kl_loss, _, _ = super().loss_function(x_recon, x, mu, logvar)
        adv_fm_loss_per_sample = torch.zeros(x.size(0), device=x.device) # (B,)
        gamma_values = []
        if self.adv:

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
                mse = mse.view(x.size(0), -1).sum(dim=1)             # (B,)
                
                # Apply gamma BEFORE reducing across batch
                adv_fm_loss_per_sample += g * mse                    # scaler + (B,) -> (B,) + (B,) -sum-> (B,)

            adv_fm_loss = adv_fm_loss_per_sample.mean()              # mean over batch (B,) -mean-> scalar

        total_loss = recon_loss + self.beta * kl_loss + adv_fm_loss

        return total_loss, recon_loss, kl_loss, adv_fm_loss, gamma_values

    def loss_discriminator(self, x_recon, x):
        real_logits, _ = self.discriminator(x)
        fake_logits, _ = self.discriminator(x_recon.detach())

        bce = F.binary_cross_entropy_with_logits
        target_real = torch.ones_like(real_logits) # Label for real images = 1
        target_fake = torch.zeros_like(fake_logits) # Label for fake images = 0

        loss_real = bce(real_logits, target_real)
        loss_fake = bce(fake_logits, target_fake)
        return loss_real + loss_fake
