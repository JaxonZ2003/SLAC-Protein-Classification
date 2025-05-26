import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import torchvision.utils as vutils


class MyVAE(nn.Module):
    def __init__(self, latent_dim=4):
        super(MyVAE, self).__init__()

        # Encoder: ResNet18 backbone
        resnet = models.resnet18(weights=False)
        self.encoder_cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC layer
        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
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
        x = self.encoder_cnn(x)              # (12, 512, 1, 1)
        x = self.flatten(x)                  # (12, 512)
        x = self.encoder_fc(x)               # (12, 256)
        mu = self.fc_mu(x)                   # (12, latent_dim)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        x = self.decoder_fc(z)               # (12, 512*8*8)
        return self.decoder_conv(x)          # (12, 3, 512, 512)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-4):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld



if __name__ == "__main__":
    data = torch.tensor(np.random.random((12, 3, 512, 512)).astype(np.float32))
    vae = MyVAE(latent_dim=4)
    recon, mu, logvar = vae(data)

    print("Input shape:       ", data.shape)
    print("Latent (mu) shape: ", mu.shape)
    print("Reconstruction:    ", recon.shape)

    os.makedirs("fake_output", exist_ok=True)
    vutils.save_image(data[:4], "fake_output/original1.png", nrow=2)
    vutils.save_image(recon[:4], "fake_output/reconstruction1.png", nrow=2)
    print("Saved images to 'fake_output/'")
