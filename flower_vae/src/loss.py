import torch
import torch.nn.functional as F

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Returns: Total Loss (Scalar), Reconstruction Loss (Scalar), KLD (Scalar)
    """
    # Reconstruction loss (Sum of Squared Errors)
    # Using sum instead of mean helps stability in VAEs
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD
