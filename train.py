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
from dataset import PetDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ROOT = Path('../petfinder-pawpularity-score')
TRAIN_IMG_PATH = ROOT / 'train'
TARGET_COL = 'Pawpularity'


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 128
N_EPOCHS = 3
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

    oof_pred = torch.zeros(len(train_df))
    criterion = MSELossFlat()
    # criterion = RMSELoss()

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        print('='*5, f'Start Fold: {fold}', '='*5)
        train_x, valid_x = train_df.loc[train_idx], train_df.loc[valid_idx]
        print(len(train_x), len(valid_x))
        train_ds, valid_ds = PetDataset(train_x, TRAIN_IMG_PATH, train_transform), PetDataset(valid_x, TRAIN_IMG_PATH, valid_transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        dls = DataLoaders(train_dl, valid_dl)
        
        model = CustomModel(MODEL_NAME).to(device)
        learner = Learner(dls, model, loss_func=criterion, metrics=rmse)
        learner.fine_tune(N_EPOCHS)
        
        pred, tgt = learner.get_preds(dl=valid_dl)
        oof_pred[valid_idx] = pred.detach().cpu().view(-1)
        
        print(f'Fold: {fold}, RMSE: {mean_squared_error(tgt, pred, squared=False)}')
        
        learner.save(f'learner_fold_{fold}')
        torch.save(learner.model.state_dict(), f'./fold_{fold}.pth')

        del model
        del optimizer
        # torch.cuda.empty_cache()

    print(f'CV Score = {mean_squared_error(target, oof_pred, squared=False)}')

    plt.figure(figsize=(10, 6))
    plt.hist(target, bins=30)
    plt.hist(oof_pred.numpy(), bins=30)

if __name__ == "__main__":
    main()