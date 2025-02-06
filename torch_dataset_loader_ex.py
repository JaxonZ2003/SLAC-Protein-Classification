# coding: utf-8
from torch.utils.data import DataLoader, Dataset
class TestSet(Dataset):
    def __init__(self, size=100):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        return i, [i]*10
        
ds = TestSet()
len(ds)
ds[0]
ds[1]
ds[2]
ds[10]
ds[19]
import torch
class TestSet(Dataset):
    def __init__(self, size=100):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        return torch.tensor(i), torch.tensor([i]*10)
        
ds = TestSet()
ds[0]
dl = DataLoader(ds, shuffle=True)
dl
for l, im in dl:
    break
    
#dl = DataLoader(ds, shuffle=True, )
get_ipython().run_line_magic('pinfo', 'DataLoader')
dl = DataLoader(ds, shuffle=True, batch_size=10)
for l, im in dl:
    break
    
l
im
len(ds)
dl = DataLoader(ds, shuffle=True, batch_size=10)
len(dl)
dl = DataLoader(ds, shuffle=True, batch_size=5)
len(dl)
for i, (l, im) in enumerate(dl):
    print(f"\nbatch{i+1} : ", l, im)
    
    
l
im.shape
im
