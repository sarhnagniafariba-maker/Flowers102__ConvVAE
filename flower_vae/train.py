import os
import torch
import torch.optim as optim
from src.config import Config
from src.dataset import get_dataloader
from src.model import VAE
from src.loss import vae_loss_function
from src.utils import save_loss_plot, save_image_grid

def main():
    # 1. Setup
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    torch.manual_seed(Config.SEED)
    
    print(f"Device: {Config.DEVICE}")
    print("Initializing Data and Model...")
    
    # 2. Init components
    dataloader = get_dataloader()
    model = VAE().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    loss_history = []

    # 3. Training Loop
    print("Starting Training...")
    model.train()
    
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # End of epoch stats
        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Loss: {avg_loss:.4f}")
        
        # 4. Save snapshots every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Save Model
            torch.save(model.state_dict(), f"{Config.CHECKPOINT_DIR}/vae_epoch_{epoch+1}.pth")
            
            # Save Reconstructions
            with torch.no_grad():
                # Compare original vs recon
                sample_data = data[:8]
                recon_sample, _, _ = model(sample_data)
                comparison = torch.cat([sample_data, recon_sample])
                save_image_grid(comparison, f"recon_epoch_{epoch+1}.png")
                
                # Generate new flowers
                z = torch.randn(16, Config.LATENT_DIM).to(Config.DEVICE)
                generated = model.decoder(model.fc_decode(z))
                save_image_grid(generated, f"generated_epoch_{epoch+1}.png")

    # 5. Finalize
    save_loss_plot(loss_history)
    print("Training Complete. Check 'results/' for images and plots.")

if __name__ == "__main__":
    main()
