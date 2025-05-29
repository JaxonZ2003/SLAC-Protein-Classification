# generates images from latent space
import torch
import torchvision.utils as vutils
from SLAC25.models import MyVAE
import os

LATENT_DIM = 4
NUM_IMAGES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = MyVAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load("vae_outputs/vae_model.pth", map_location=DEVICE))
vae.eval()

z = torch.randn(NUM_IMAGES, LATENT_DIM).to(DEVICE)
samples = vae.decode(z)

package_root = os.path.dirname(os.path.abspath(__file__))
savedPath = os.path.join(package_root, "..", "img", "vae_outputs")
savedPath = os.path.abspath(savedPath)

os.makedirs(savedPath, exist_ok=True)
savedName = os.path.join(savedPath, "synthetic_samples.png")
vutils.save_image(samples, savedName, nrow=4)

print(f"🧪 Synthetic images saved to {savedName}")
