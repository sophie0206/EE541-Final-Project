import torch
import torch.nn as nn
import timm
from fastai.vision.all import *

class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.head = create_head(self.model.num_features, 1)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.act(x)
        return x