import torch
import torch.nn as nn
import torchvision.models as models

class MyVAE(nn.Module):
    def __init__(self, latent_dim=4):
        super(MyVAE, self).__init__()

        # Encoder: ResNet18 backbone
        resnet = models.resnet18(weights=None)
        self.encoder_cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder:
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),      # 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),      # 512x512
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        h = self.encoder_fc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        x = self.decoder_fc(z)
        return self.decoder_conv(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-4):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld
