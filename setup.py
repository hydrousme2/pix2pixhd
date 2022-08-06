import torch
import torch.nn as nn

from .config import *
from .loss import Loss  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = Loss(device=device)

def lr_lambda(epoch):
    return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (epochs - decay_after)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)