import torch
import matplotlib.pyplot as plt
import torchvision
import os
from .config import Config

def save_loss_plot(train_losses, filename='loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('VAE Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.RESULT_DIR, filename))
    plt.close()

def save_image_grid(images, filename):
    """
    Saves a grid of images to the results folder.
    """
    if not os.path.exists(Config.RESULT_DIR):
        os.makedirs(Config.RESULT_DIR)
        
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(Config.RESULT_DIR, filename))
    plt.close()
