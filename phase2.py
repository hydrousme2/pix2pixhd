# high resolution (2048 x 1024) train phase
import torch
import torch.nn.functional as F

from .setup import *
from torch.utils.data import DataLoader
from .dataset import CityscapesDataset
from .models.local_enhancer import LocalEnhancer
from .models.multiscale_discriminator import MultiscaleDiscriminator
from .phase1 import generator1, encoder
from .train import train

# HIGH-RES (2048,1024) phase
dataloader2 = DataLoader(
    CityscapesDataset(train_dir, target_width=2048, n_classes=n_classes),
    collate_fn=CityscapesDataset.collate_fn, 
    batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
)
generator2 = LocalEnhancer(n_classes + n_features + 1, rgb_channels).to(device).apply(weights_init)
discriminator2 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels).to(device).apply(weights_init)

g2_optimizer = torch.optim.Adam(list(generator2.parameters()) + list(encoder.parameters()), lr=lr, betas=betas)
d2_optimizer = torch.optim.Adam(list(discriminator2.parameters()), lr=lr, betas=betas)
g2_scheduler = torch.optim.lr_scheduler.LambdaLR(g2_optimizer, lr_lambda)
d2_scheduler = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda)
print("Phase 2 ready")

# update g1 generator with trained g1 generator
generator2.g1 = generator1.g1

# freeze encoder 
def freeze(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    @torch.jit.script
    def forward(x, inst):
        x = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=True)
        inst = F.interpolate(inst.float(), scale_factor=0.5, recompute_scale_factor=True)
        feat = encoder(x, inst.int())
        return F.interpolate(feat, scale_factor=2.0, recompute_scale_factor=True)
    return forward

train(
    dataloader2,
    [freeze(encoder), generator2, discriminator2],
    [g2_optimizer, d2_optimizer],
    [g2_scheduler, d2_scheduler],
    device,
)