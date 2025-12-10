import torch

class Config:
    # Paths
    DATA_DIR = './data'
    RESULT_DIR = './results'
    CHECKPOINT_DIR = './checkpoints'

    # Model Hyperparameters
    IMAGE_SIZE = 64
    LATENT_DIM = 128
    CHANNELS = 3
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random Seed
    SEED = 42
