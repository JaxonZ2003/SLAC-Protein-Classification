# generates images from latent space
import torch
import torchvision.utils as vutils
from SLAC25.VAE_model import MyVAE
import os

LATENT_DIM = 4
NUM_IMAGES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = MyVAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load("vae_outputs/vae_model.pth", map_location=DEVICE))
vae.eval()

z = torch.randn(NUM_IMAGES, LATENT_DIM).to(DEVICE)
samples = vae.decode(z)

os.makedirs("vae_outputs", exist_ok=True)
vutils.save_image(samples, "vae_outputs/synthetic_samples.png", nrow=4)

print("🧪 Synthetic images saved to 'vae_outputs/synthetic_samples.png'")
