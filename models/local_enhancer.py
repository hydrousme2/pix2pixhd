import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_generator import GlobalGenerator

class LocalEnhancer(nn.Module):
    
    def __init__(self, in_channels, out_channels, base_channels=32, g1_front_block=3, g1_res_block=9, g2_res_block=3):
        super().__init__()
        
        g1_base_channels = base_channels * 2
        
        # low res input to g1
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        
        # g1 without out_layers
        self.g1 = GlobalGenerator(in_channels, out_channels, base_channels=g1_base_channels, 
                                  front_blocks=g1_front_block, res_blocks=g1_res_block).g1
        
        # define g2
        self.g2 = nn.ModuleList()
        
        # first-layer and front-end blocks for g2
        self.g2.append(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0), 
                nn.InstanceNorm2d(base_channels, affine=False),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1), 
                nn.InstanceNorm2d(2 * base_channels, affine=False),
                nn.ReLU(inplace=True),
            )
        )
        
        # residual, backend and out-layer block
        self.g2.append(
            nn.Sequential(
                *[ResidualBlock(2 * base_channels) for _ in range(g2_res_block)],
                
                nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1), 
                nn.InstanceNorm2d(base_channels, affine=False),
                nn.ReLU(inplace=True),
                
                nn.ReflectionPad2d(3),
                nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
                nn.Tanh(),
            )
        )
        
    def forward(self, x):
        # output from g1 backend
        x_g1 = self.downsample(x)
        x_g1 = self.g1(x_g1)
        
        # output from g2 frontend
        x_g2 = self.g2[0](x)
        
        # output from g2 final
        return self.g2[1](x_g1 + x_g2)