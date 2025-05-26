import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math

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
        B,C,H,W = x_recon.shape
        D = C*H*W
        sigma = 1.0
        mse_term = 0.5 * torch.sum((x_recon - x)**2) / (sigma**2)
        const = B * (D/2) * math.log(2*math.pi*sigma**2)
        recon_loss = mse_term + const
        # KL divergence
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
        return recon_loss, kl_loss

class BetaVAE(VAE):
    def __init__(self, in_channels=3, latent_dim=256, beta=1.0):
        super().__init__(in_channels, latent_dim)
        self.beta = beta

    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss, kl_loss = super().loss_function(x_recon, x, mu, logvar)
        return recon_loss, self.beta * kl_loss

class VAEPlus(BetaVAE):
    def __init__(self, in_channels=3, latent_dim=256,
                 beta=1.0, feature_weight = 1.0, T=steps_per_epoch, use_adv=True):
        super().__init__(in_channels, latent_dim, beta)
        self.feature_weight = feature_weight
        self.T = T
        self.iter = 0
        self.use_adv = use_adv
        if self.use_adv:
            self.discriminator = Discriminator(in_channels)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.register_buffer("real_label", torch.tensor(1., device=device))
            self.register_buffer("fake_label", torch.tensor(0., device=device))

    def _gamma(self, layer_idx: int):
        return min(max((self.iter/self.T) - layer_idx, 0.0), 1.0)

    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss, kl_loss = super().loss_function(x_recon, x, mu, logvar)
        adv_fm_loss = torch.tensor(0., device=x.device)
        gamma_values = []
        if self.use_adv:
            # Freeze D
            for p in self.discriminator.parameters(): 
                p.requires_grad = False

            _, feats_real = self.discriminator(x)
            _, feats_fake = self.discriminator(x_recon)

            # Unfreeze D
            for p in self.discriminator.parameters():
                p.requires_grad = True

            for i, (fr, ff) in enumerate(zip(feats_real, feats_fake)):
                g = self._gamma(i)
                gamma_values.append(g)
                adv_fm_loss += g * F.mse_loss(ff, fr.detach(), reduction='sum')
            self.iter += 1
        return recon_loss, kl_loss, adv_fm_loss, gamma_values

    def loss_discriminator(self, x, x_recon):
        bce = F.binary_cross_entropy_with_logits
        real_logits, _ = self.discriminator(x)
        fake_logits, _ = self.discriminator(x_recon.detach())
        return (bce(real_logits, self.real_label.expand_as(real_logits)) +
                bce(fake_logits, self.fake_label.expand_as(fake_logits)))
