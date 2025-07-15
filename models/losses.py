#losses.py
import torch
import torchmetrics as metrics
import torch.nn as nn
from torchmetrics.image import TotalVariation
from torch.nn import MSELoss
#https://lightning.ai/docs/torchmetrics/stable/image/total_variation.html

class TVLoss(nn.Module):
    def __init__(self,tv_lambda):
        super().__init__()
        self.TV = TotalVariation(reduction='mean')
        self.MSELoss = nn.MSELoss(reduction='mean')
        self.tv_lambda = tv_lambda


    def forward(self, inputs, targets):
        MSELoss = self.MSELoss(inputs, targets)
        TVLoss = self.TV(inputs)
        return (MSELoss) + (self.tv_lambda*TVLoss)

#include more losses with whatever i can think of later

