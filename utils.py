import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:1], nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def save_checkpoint(epoch, encoder, generator, discriminator, g_optimizer, d_optimizer, g_scheduler, d_scheduler, path=None):
    state = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'g_scheduler_state_dict': g_scheduler.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'd_scheduler_state_dict': d_scheduler.state_dict(),
    }
    if path is not None:
        torch.save(state, path)
    else:
        torch.save(state, f"./phase1_ep{epoch}.pth")
    print(f"<---------------------Model checkpoint at epoch {epoch}-------------------->")