# low resolution (1024 x 512) train phase
import torch

from .setup import *
from torch.utils.data import DataLoader
from .dataset import CityscapesDataset
from .models.encoder import Encoder
from .models.global_generator import GlobalGenerator
from .models.multiscale_discriminator import MultiscaleDiscriminator
from .train import train

dataloader1 = DataLoader(
    CityscapesDataset(train_dir, target_width=1024, n_classes=n_classes),
    collate_fn=CityscapesDataset.collate_fn, 
    batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
)
encoder = Encoder(rgb_channels, n_features).to(device).apply(weights_init)
generator1 = GlobalGenerator(n_classes + n_features + 1, rgb_channels).to(device).apply(weights_init)
discriminator1 = MultiscaleDiscriminator(n_classes + 1 + rgb_channels, n_discriminators=2).to(device).apply(weights_init)

g1_optimizer = torch.optim.Adam(list(generator1.parameters()) + list(encoder.parameters()), lr=lr, betas=betas)
d1_optimizer = torch.optim.Adam(list(discriminator1.parameters()), lr=lr, betas=betas)
g1_scheduler = torch.optim.lr_scheduler.LambdaLR(g1_optimizer, lr_lambda)
d1_scheduler = torch.optim.lr_scheduler.LambdaLR(d1_optimizer, lr_lambda)
print("Phase 1 ready")

train(
    dataloader1,
    [encoder, generator1, discriminator1],
    [g1_optimizer, d1_optimizer],
    [g1_scheduler, d1_scheduler],
    device,
    load_path="",
    save_path=phase1_dir,
    resume_train=False
)