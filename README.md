# Convolutional VAE for Flowers102 Dataset

This repository contains a modular PyTorch implementation of a **Convolutional Variational Autoencoder (VAE)** designed to generate flower images using the [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset.

The project is structured to be clean, readable, and easy to extend. It handles the specific data split requirements of the Flowers102 dataset to maximize training data availability.

## 📌 Features

*   **Modular Design:** Code is separated into configuration, data loading, model architecture, and training logic.
*   **Convolutional Architecture:** Uses `Conv2d` and `ConvTranspose2d` layers for high-quality image reconstruction (instead of standard Linear layers).
*   **Data Handling:** Automatically combines the `train`, `val`, and `test` splits of Flowers102 to provide ~8,000 images for training (fixing the small training set issue of this specific dataset).
*   **Visualization:** Automatically saves:
    *   Reconstructed images (comparing real vs. VAE output).
    *   New generated samples (from latent noise).
    *   Training loss plots.

## 📂 Project Structure

```text
flower_vae/
│
├── train.py                
├── requirements.txt                         
├── vae_flowers.pth/            
├── results/               
└── src/
    ├── config.py          
    ├── dataset.py         
    ├── model.py           
    ├── loss.py             
    └── utils.py            
```

## 🛠️ Installation

1.  **Clone the repository** (or create the folder structure):
    ```bash
    mkdir flower_vae
    cd flower_vae
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `scipy` is required specifically for the Flowers102 dataset loader in torchvision.*

## 🚀 Usage

To start training the model, simply run the main script. The dataset will be downloaded automatically the first time you run it.

```bash
python train.py
```

### Configuration
You can adjust hyperparameters in `src/config.py` without changing the code logic:
*   `BATCH_SIZE`: Default 64
*   `LATENT_DIM`: Default 128
*   `NUM_EPOCHS`: Default 50
*   `LEARNING_RATE`: Default 1e-3

## 🧠 Model Architecture

### The Encoder
*   Takes RGB images sized **64x64**.
*   Passes through 4 Convolutional layers (increasing channels: 32 -> 64 -> 128 -> 256).
*   Flattens features to a dense vector.
*   Outputs two vectors: **Mean ($\mu$)** and **Log-Variance ($\log\sigma^2$)**.

### The Latent Space
*   Uses the **Reparameterization Trick**: $z = \mu + \sigma \cdot \epsilon$ (where $\epsilon \sim N(0,1)$).
*   This allows backpropagation through the random sampling process.

### The Decoder
*   Takes the latent vector $z$.
*   Upsamples using **Transposed Convolutions** to reconstruct the image.
*   Ends with a `Sigmoid` activation to ensure pixel values are between [0, 1].

### Loss Function (ELBO)
The model minimizes the Evidence Lower Bound:
1.  **Reconstruction Loss (MSE):** Measures pixel-wise difference between input and output.
2.  **KL Divergence:** Forces the latent distribution to approximate a Standard Normal Distribution.

## 📊 Results

During training, check the `results/` folder.
*   **`reconstruction_epoch_X.png`**: Top row is real data, bottom row is the VAE reconstruction.
*   **`generated_epoch_X.png`**: Entirely new flowers generated from random noise.
*   **`loss_plot.png`**: Generated at the end of training to show convergence.

## 📜 License
This project is open-source. Feel free to modify and use it for educational purposes.
