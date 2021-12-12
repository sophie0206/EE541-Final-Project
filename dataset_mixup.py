import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

ID_COL = 'Id'
TARGET_COL = 'Pawpularity'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PetDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, mode='train'):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / f'{row[ID_COL]}.jpg'
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform is not None:
            img = self.transform(image=img)['image']
        tgt = row[TARGET_COL] if self.mode == 'train' else 0
        ##=============Mix Up Augmentation=================##
        if self.mode == "train" and idx > 0 and idx%5 == 0:
            mixup_idx = random.randint(0, len(self.df)-1)
            mix_img_path = self.data_dir / f'{self.df.iloc[mixup_idx][ID_COL]}.jpg'
            mixup_img = np.array(Image.open(mix_img_path).convert('RGB'))
            if self.transform:
                mixup_img = self.transform(image=mixup_img)['image']
            mixup_tgt = self.df.iloc[mixup_idx][TARGET_COL]
        
            # Mixup the images
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            img = lam * img + (1 - lam) * mixup_img
            tgt = lam * tgt + (1 - lam) * mixup_tgt
            
        return img.float().to(device), torch.tensor(tgt).float().to(device)