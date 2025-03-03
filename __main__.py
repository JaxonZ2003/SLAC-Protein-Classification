import os
import torch

from SLAC25.network import BaselineCNN_Wrapper
from argparse import ArgumentParser

########## Start ##########
ap = ArgumentParser()
ap.add_argument("--nepoch", type=int, default=10)
ap.add_argument("--outdir", type=str, default=None)
ap.add_argument("--lr", type=float, default=0.001)
args = ap.parse_args()

if args.outdir is None:
    
    try:
        slurm_jid = os.environ['SLURM_JOB_ID']
        slurm_jname = os.environ['SLURM_JOB_NAME']
        username = os.environ['USER']
        args.outdir = f"/scratch/slac/models/{username}.{slurm_jname}.{slurm_jid}"
    except KeyError:
        args.outdir = "./models"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# simple model
CNNWrapper = BaselineCNN_Wrapper(num_classes=4, keep_prob=0.75, num_epochs=3)
# enable testmode for smaller sample size
# enable verbose for detailed info
CNNWrapper._prepareDataLoader(testmode=True, verbose=True)
train_log = CNNWrapper.train()


########## dataset paths #########
# criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)


########## train the model #########
# train_log = fit(model, train_loader, val_loader, num_epochs=args.nepoch, 
#                 optimizer=optimizer, criterion=criterion, device=device, 
#                 lr_scheduler=lr_scheduler, outdir=args.outdir)
#test_log = test(model, test_loader, criterion, device) # will save testing for later