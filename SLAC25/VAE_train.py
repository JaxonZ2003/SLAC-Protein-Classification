import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime

from SLAC25.dataset import ImageDataset
from SLAC25.VAE_model import MyVAE

BATCH_SIZE = 8
EPOCHS = 10
LATENT_DIM = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

from torch.utils.data import Subset
dataset = ImageDataset("../data/train_info.csv", transform=None, config=None, recordTransform=False)
dataset = Subset(dataset, list(range(10000)))
print("loaded %d images"%len(dataset))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(dataloader)

nbatches=len(dataloader)
print(f"Number of batches={nbatches}")
vae = MyVAE(latent_dim=LATENT_DIM).to(DEVICE)
print(vae)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)
ts=str(datetime.datetime.now()).replace(" ", "_")
odir=f"/scratch/slac/vae.{ts}"
os.makedirs(odir, exist_ok=True)
print(odir)

for epoch in range(EPOCHS):
    vae.train()
    total_loss = 0
    for i, (imgs, _) in enumerate(dataloader):
        print(f"processing epoch {epoch+1}, batch {i+1}/{nbatches} (batchsize={len(imgs)})", flush=True, end="\r")
        imgs = imgs.to(DEVICE, dtype=torch.float32)
        recon, mu, logvar = vae(imgs)
        loss = vae.loss_function(recon, imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i == 0:
            vutils.save_image(imgs[:4], f"{odir}/epoch{epoch+1}_input.png", nrow=2)
            vutils.save_image(recon[:4], f"{odir}/epoch{epoch+1}_recon.png", nrow=2)
    print()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.6f}")

torch.save(vae.state_dict(), f"{odir}/vae_model_{epoch+1}.pth")
print("Training complete")
