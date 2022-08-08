import torch
from tqdm import tqdm 

from .config import epochs
from .utils import save_checkpoint, show_tensor_images
from .setup import loss_fn

def train(dataloader, models, optimizers, schedulers, device, load_path, save_path, resume_train=True):
    
    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers
    resume_epoch=0

    # load model
    if resume_train and (load_path):
        checkpoint = torch.load(load_path)
        resume_epoch = checkpoint["epoch"]
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        print(f"Checkpoint restored from epoch {resume_epoch}")
    
    cur_step = 0
    display_step = 100

    mean_g_loss = 0.0
    mean_d_loss = 0.0
    
    for epoch in range(resume_epoch+1, epochs+1):
        for (x_real, labels, insts, bounds) in tqdm(dataloader, position=0, desc=f"Epoch {epoch}"):
            x_real = x_real.to(device)
            labels = labels.to(device)
            insts = insts.to(device)
            bounds = bounds.to(device)
            
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                g_loss, d_loss, x_fake = loss_fn(
                    x_real, labels, insts, bounds, encoder, generator, discriminator
                )
                
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step
            
            # display step  images
            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'
                      .format(cur_step, mean_g_loss, mean_d_loss))
                show_tensor_images(x_fake.to(x_real.dtype))
                show_tensor_images(x_real)
                mean_g_loss = 0.0
                mean_d_loss = 0.0
            cur_step += 1
            
        # model checkpoint every epoch
        save_checkpoint(epoch, encoder, generator, discriminator, 
                   g_optimizer, d_optimizer, g_scheduler, d_scheduler, save_path)
            
        g_scheduler.step()
        d_scheduler.step()