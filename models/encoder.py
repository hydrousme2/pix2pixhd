import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, base_channels=64, n_layers=4):
        super().__init__()
        
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0), 
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        
        channels = base_channels
        
        # downsampling
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * channels),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
            
        # upsampling
        for _ in range(n_layers):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2
            
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        
        self.out_channels = out_channels
        self.layers = nn.Sequential(*layers)
        
    def instancewise_average_pooling(self, x, inst):
        
        x_mean = torch.zeros_like(x)
        classes = torch.unique(inst, return_inverse=False, return_counts=False)
        
        for i in classes:
            for b in range(x.size(0)):
                indexes = torch.nonzero(inst[b:b+1] == 1, as_tuple=False)
                for j in range(self.out_channels):
                    x_ins = x[indexes[:, 0] + b, indexes[:, 1] + j, indexes[:, 2], indexes[:, 3]]
                    mean_feat = torch.mean(x_ins).expand_as(x_ins)
                    x_mean[indexes[:, 0] + b, indexes[:, 1] + j, indexes[:, 2], indexes[:, 3]] = mean_feat
                    
        return x_mean
    
    def forward(self, x, inst):
        x = self.layers(x)
        x = self.instancewise_average_pooling(x, inst)
        return x