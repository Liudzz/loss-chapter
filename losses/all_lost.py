"""
easy way to use losses
"""
from center_loss import Centerloss
import torch.nn as nn
from FocalLoss import FocalLoss


def center_loss(pred,label,num_calss,feature):
    loss = Centerloss(num_calss,feature)
    return loss(pred,label)

def Focal_loss(pred,label,num_calss,alaph=None, gamma):
    loss = Centerloss(num_calss,gamma)
    return loss(pred,label)

def L1_loss(pred,label):
    loss = nn.L1Loss(pred,label)
    return loss

def L2_loss(pred,label):
    loss = nn.MSELoss(pred,label)
    return loss

def SmoothL1_loss(pred,label):
    loss = nn.SmoothL1Loss(pred,label)
    return loss