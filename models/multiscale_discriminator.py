import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator import Discriminator

class MultiscaleDiscriminator(nn.Module):
    
    def __init__(self, in_channels, base_channels=64, n_layers=3, n_discriminators=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for _ in range(n_discriminators):
            self.discriminators.append(
                Discriminator(in_channels, base_channels=base_channels, n_layers=n_layers)
            )
            
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        
    def forward(self, x):
        outputs = []
        
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                x = self.downsample(x)
                
            outputs.append(discriminator(x))
        
        return outputs
            
        
    @property
    def n_discriminators(self):
        return len(self.discriminators)