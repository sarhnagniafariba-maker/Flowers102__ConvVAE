import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from .config import Config

def get_dataloader():
    """
    Returns the DataLoader. 
    Combines train, val, and test splits for maximum data.
    """
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Download and load all splits
    train_set = datasets.Flowers102(root=Config.DATA_DIR, split='train', download=True, transform=transform)
    val_set = datasets.Flowers102(root=Config.DATA_DIR, split='val', download=True, transform=transform)
    test_set = datasets.Flowers102(root=Config.DATA_DIR, split='test', download=True, transform=transform)

    # Concatenate
    full_dataset = ConcatDataset([train_set, val_set, test_set])

    loader = DataLoader(
        full_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader
