import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
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
    


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, 5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.bn4   = nn.BatchNorm2d(256)

        # After 4 layers, the shape is (256, 4, 4) for a 64×64 input => 256*4*4 = 4096
        self.flatten = nn.Flatten()
        self.fc1   = nn.Linear(256 * 4 * 4, 128)
        self.fc2   = nn.Linear(128, 1)
        
    

    def forward(self, x):
        """
        Returns:
          - out: final sigmoid(logit) for real/fake classification
          - feature_maps: list of feature map tensors (per-layer + the FC embedding)
        """
        feature_maps = []

        # --- Conv block 1 ---
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.01, inplace=True)
        feature_maps.append(x1)

        # --- Conv block 2 ---
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.01, inplace=True)
        feature_maps.append(x2)

        # --- Conv block 3 ---
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.01, inplace=True)
        feature_maps.append(x3)

        # --- Conv block 4 ---
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.01, inplace=True)
        feature_maps.append(x4)

        # --- Flatten + fully connected ---
        x5 = x4.view(x4.size(0), -1)
        x5 = F.leaky_relu(self.fc1(x5), 0.01, inplace=True)
        feature_maps.append(x5)

        # --- Final logit and output ---
        logit = self.fc2(x5)
        out = torch.sigmoid(logit)
        
        return out, feature_maps
    
class VAEPlus(BetaVAE):
    def __init__(self, input_channels=3, latent_dim=10, beta=4.0, feature_weight=1.0, slope=2500.0):
        super(VAEPlus, self).__init__(input_channels, latent_dim, beta)
        self.discriminator = Discriminator(input_channels)
        self.feature_weight = feature_weight
        self.slope = slope

    def compute_feature_matching_loss(self, real_feats, fake_feats, global_step, base_size):
        feature_loss = 0.0
        gamma_values = []
        layer_losses = []

        nb_layers = len(real_feats)
        delays = [self.slope * k for k in range(nb_layers)]

        for i in range(nb_layers):
            delay = delays[i]
            step_norm = max(0.0, float(global_step) - delay)
            w = step_norm / self.slope
            w = max(0.0, min(1.0, w))
            gamma_values.append(w)

            # L2 distance between real and fake feature maps
            distance = (real_feats[i] - fake_feats[i]).pow(2)
            layer_dist = 0.5 * base_size * distance.mean(dim=[1, 2, 3]) if distance.dim() == 4 else 0.5 * base_size * distance.mean(dim=1)
            layer_loss_val = layer_dist.mean()
            scaled_layer_loss = w * layer_loss_val
            feature_loss += scaled_layer_loss
            layer_losses.append(layer_loss_val.item())

        return feature_loss, gamma_values, layer_losses
    
    def discriminator_step(self, real_imgs, fake_imgs, disc_optimizer, bce_loss, device):
        # We train the discriminator with real and fake images
        # ---------------
        # Step 1: Train the Discriminator
        # ---------------
        self.discriminator.train()
        # Zero the gradients
        disc_optimizer.zero_grad()

        # Discriminator on real images
        d_real_logits, _ = self.discriminator(real_imgs)
        d_real = d_real_logits.view(-1)
        real_labels = torch.ones_like(d_real, device=device)
        loss_disc_real = bce_loss(d_real, real_labels)

        # Discriminator on fake images
        d_fake_logits, _ = self.discriminator(fake_imgs.detach())
        d_fake = d_fake_logits.view(-1)
        fake_labels = torch.zeros_like(d_fake, device=device)
        loss_disc_fake = bce_loss(d_fake, fake_labels)

        # Combine losses
        loss_disc = (loss_disc_real + loss_disc_fake) / 2.0
        # Backpropagation
        loss_disc.backward()
        # Update discriminator
        disc_optimizer.step()

        return loss_disc.item()
    

    def vae_step(self, real_imgs, recon_imgs, optimizer, global_step, device):
        # Set the gradients to zero
        optimizer.zero_grad()

        _, fake_feats = self.discriminator(recon_imgs)
        
        with torch.no_grad(): # Get real features without gradient tracking
            _, real_feats = self.discriminator(real_imgs)

        # Compute the feature matching loss
        base_size = real_imgs.size(2) * real_imgs.size(3) * real_imgs.size(1) # H(eight) * W(eight) * C(hannels)
        feature_loss, gamma_values, layer_losses = self.compute_feature_matching_loss(real_feats, fake_feats, global_step, base_size)

        return feature_loss, gamma_values, layer_losses