import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class VGG19(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        vgg_features = vgg19(pretrained=True).features
        
        self.f1 = nn.Sequential(*[vgg_features[x] for x in range(2)])
        self.f2 = nn.Sequential(*[vgg_features[x] for x in range(2, 7)])
        self.f3 = nn.Sequential(*[vgg_features[x] for x in range(7, 12)])
        self.f4 = nn.Sequential(*[vgg_features[x] for x in range(12, 21)])
        self.f5 = nn.Sequential(*[vgg_features[x] for x in range(21, 30)])
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x1 = self.f1(x)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        x4 = self.f4(x3)
        x5 = self.f5(x4)
        return [x1, x2, x3, x4, x5]

class Loss(nn.Module):
    
    def __init__(self, lambda1=10., lambda2=10., device="cuda", norm_weight_to_one=True):
        super().__init__()
        
        self.vgg = VGG19().to(device)
        self.vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
        lambda0 = 1.
        scale = max(lambda0, lambda1, lambda2) if norm_weight_to_one else 1.0

        self.lambda0 = lambda0 / scale
        self.lambda1 = lambda1 / scale
        self.lambda2 = lambda2 / scale
    
    def adv_loss(self, discriminator_preds, is_real):
        
        target = torch.ones_like if is_real else torch.zeros_like

        adv_loss = 0.0
        for preds in discriminator_preds:
            pred = preds[-1]
            adv_loss += F.mse_loss(pred, target(pred))
        return adv_loss
    
    def fm_loss(self, real_preds, fake_preds):
        
        fm_loss = 0.0
        for real_features, fake_features in zip(real_preds, fake_preds):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss += F.l1_loss(real_feature.detach(), fake_feature)
        return fm_loss
    
    def vgg_loss(self, x_real, x_fake):
        
        vgg_real = self.vgg(x_real)
        vgg_fake = self.vgg(x_fake)

        vgg_loss = 0.0
        for real, fake, weight in zip(vgg_real, vgg_fake, self.vgg_weights):
            vgg_loss += weight * F.l1_loss(real.detach(), fake)
        return vgg_loss
    
    def forward(self, x_real, label_map, instance_map, boundary_map, encoder, generator, discriminator):
        
        feature_map = encoder(x_real, instance_map)
        x_fake = generator(torch.cat((label_map, boundary_map, feature_map), dim=1))
        
        fake_preds_for_g = discriminator(torch.cat((label_map, boundary_map, x_fake), dim=1))
        fake_preds_for_d = discriminator(torch.cat((label_map, boundary_map, x_fake.detach()), dim=1))
        real_preds_for_d = discriminator(torch.cat((label_map, boundary_map, x_real.detach()), dim=1))
        
        g_loss = (
            self.lambda0 * self.adv_loss(fake_preds_for_g, True) + \
            self.lambda1 * self.fm_loss(real_preds_for_d, fake_preds_for_g) / discriminator.n_discriminators + \
            self.lambda2 * self.vgg_loss(x_fake, x_real)
        )
        
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )
        
        return g_loss, d_loss, x_fake.detach()