import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=False),
            
            nn.ReLU(),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=False),
        )
    
    def forward(self, x):
        return x + self.layer(x)