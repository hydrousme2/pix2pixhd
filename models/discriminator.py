import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    
    def __init__(self, in_channels, base_channels=64, n_layers=3):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        
        channels = base_channels
        for _ in range(1,n_layers):
            prev_channels = channels
            channels = prev_channels * 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, channels, kernel_size=4, stride=2, padding=2),
                    nn.InstanceNorm2d(channels, affine=False),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            
        prev_channels = channels
        channels = 2 * channels
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(prev_channels, channels, kernel_size=4, stride=1, padding=2),
                nn.InstanceNorm2d(channels, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=2),
            )
        )
        
    def forward(self, x):
        outputs = []
        
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
            
        return outputs