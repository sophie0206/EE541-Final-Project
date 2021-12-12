import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from model import CustomModel
from dataset_mixup import PetDataset
from losses import RMSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ROOT = Path('../petfinder-pawpularity-score')
TRAIN_IMG_PATH = ROOT / 'train'
TARGET_COL = 'Pawpularity'


SEED = 42
IMG_SIZE = 512
BATCH_SIZE = 16
N_EPOCHS = 10
N_SPLITS = 5
# Tensorflow EfficientNet B0 Noisy-Student
MODEL_NAME = 'tf_efficientnet_b0_ns'

# Learning schedule, optimizer, augmentation, loss!

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def training(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    for epoch in range(N_EPOCHS):
        print('Epoch {}/{}'.format(epoch, N_EPOCHS - 1))
        print('-' * 20)

        loss_epoch = 0.0
        for inputs, labels in tqdm(dataloader):
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds.view(-1), labels)

            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print('Epoch {} Training RMSE Loss: {:.4f} '.format(epoch, loss_epoch / len(dataloader)))


def inference(model, dataloader):
    with torch.no_grad():
        model.eval()
        scores = 0.0
        final_preds = []
        for x, target in dataloader:
            preds = model(x).view(-1).detach().cpu().numpy()
            scores += mean_squared_error(target.detach().cpu().numpy(), preds, squared=False)
            final_preds = final_preds + list(preds)

        final_preds = np.array(final_preds, dtype=np.float32)
        print('Validation RMSE Loss: {:.4f} '.format(scores / len(dataloader)))
    
        return scores / len(dataloader), final_preds


def main():
    seed_everything(SEED)
    train_df = pd.read_csv(ROOT / 'train.csv')
    target = train_df[TARGET_COL]
    print(train_df.head())

    # Transforms
    train_transform = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.85, 1.1)),
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


    kfold = KFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True)

    
    # criterion = MSELossFlat()
    criterion = RMSELoss()
    score_cv = []
    oof_pred = np.zeros(len(train_df))

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        print('='*5, f'Start Fold: {fold}', '='*5)
        train_x, valid_x = train_df.loc[train_idx], train_df.loc[valid_idx]
        
        train_ds, valid_ds = PetDataset(train_x, TRAIN_IMG_PATH, train_transform), PetDataset(valid_x, TRAIN_IMG_PATH, valid_transform)
        print(len(train_ds), len(valid_ds))
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        # dls = DataLoaders(train_dl, valid_dl)

        model = CustomModel(MODEL_NAME).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6, verbose=False)

        training(model, train_dl, criterion, optimizer, scheduler)
        torch.save(model, f'./fold_{fold}_entire_model.pth')

        score, preds = inference(model, valid_dl)
        score_cv.append(score)
        oof_pred[valid_idx] = preds
        
        del model
        del optimizer
        # torch.cuda.empty_cache()

    print('Mean CV Score = {:.4f} '.format(np.mean(np.array(score_cv))))

    plt.figure(figsize=(10, 6))
    plt.hist(target, bins=30)
    plt.hist(oof_pred, bins=30)
    plt.show()

if __name__ == "__main__":
    main()