import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_generator import ResidualBlock

class GlobalGenerator(nn.Module):
    
    def __init__(self, in_channels, out_channels, base_channels=64, front_blocks=3, res_blocks=9):
        super().__init__()
        
        # first layer
        g1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(inplace=True)
        ]
        
        channels = base_channels
        # front-end blocks
        for _ in range(front_blocks):
            g1 += [
                nn.Conv2d(channels, channels*2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channels*2, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
            
        # res-blocks
        for _ in range(res_blocks):
            g1 += [ResidualBlock(channels)]
            
        #back-end blocks
        for _ in range(front_blocks):
            g1 += [
                nn.ConvTranspose2d(channels, channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(channels//2, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels //= 2
            
        # Output convolutional layer; will be omitted in second training phase
        self.out_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        )
        
        self.g1 = nn.Sequential(*g1)
        
    def forward(self, x):
        x = self.g1(x)
        x = self.out_layers(x)
        return x