import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import torchvision.utils as vutils

from SLAC25.models import MyVAE


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
