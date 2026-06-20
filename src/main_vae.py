import numpy as np
import matplotlib.pyplot as plt
import os

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer, VAEBottleneckLayer
from core.network import Network
from core.optimizers import SGD
from core.losses import BCE
from utils.data_loader import FontLoader, FontAugmenter

def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> int:
    binary_predicted = (predicted >= 0.5).astype(int)
    binary_expected = expected.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    errors_per_letter = np.sum(differences, axis=1)
    return np.max(errors_per_letter)

def create_vae() -> Network:
    vae = Network()
    
    # --- ENCODER ---
    vae.add(Linear(input_size=35, output_size=24))
    vae.add(ActivationLayer(Tanh()))
    
    # Proyectamos a 4 dimensiones (2 para mu, 2 para log_var)
    vae.add(Linear(input_size=24, output_size=4))
    
    # CUELLO DE BOTELLA VAE
    vae.add(VAEBottleneckLayer(latent_dim=2, kl_weight=0.0001))
    
    # --- DECODER ---
    vae.add(Linear(input_size=2, output_size=24))
    vae.add(ActivationLayer(Tanh()))
    
    vae.add(Linear(input_size=24, output_size=35))
    vae.add(ActivationLayer(Sigmoid()))
    
    return vae

def train_vae(X: np.ndarray, epochs: int = 20000, learning_rate: float = 0.05) -> Network:
    vae = create_vae()
    loss_function = BCE()
    optimizer = SGD(learning_rate=learning_rate)
    
    vae_layer = vae.layers[3]
    
    for epoch in range(epochs):
        predicted = vae.forward(X)
        rec_loss = loss_function.calculate(expected=X, predicted=predicted)
        kl_loss = vae_layer.kl_loss * vae_layer.kl_weight
        total_loss = rec_loss + kl_loss
        
        max_incorrect_pixels = evaluate_pixel_diff(X, predicted)
        
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        vae.backward(initial_gradient, optimizer)
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:05d} | Rec Loss: {rec_loss:.4f} | KL Loss: {kl_loss:.4f} | Total: {total_loss:.4f} | Max incorrect pixels: {max_incorrect_pixels}")
            
    return vae

def experiment_vae_generation(vae: Network, X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT: VAE Generation ---")
    
    mu_logvar = X
    for layer in vae.layers[:3]:
        mu_logvar = layer.forward(mu_logvar)
    mu = mu_logvar[:, :2]
    
    plt.figure(figsize=(10, 8))
    colors = ['blue']*32 + ['red']*32 + ['green']*32
    plt.scatter(mu[:, 0], mu[:, 1], c=colors, edgecolors='k', alpha=0.6)
    
    for i in range(len(mu)):
        if i % 4 == 0:
            text_label = str(labels[i % 32])
            plt.annotate(text_label, (mu[i, 0], mu[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
    plt.title("VAE Latent Space (mu)\nBlue: Normal, Red: Bold, Green: Italic")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.grid(True)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/vae_latent_space.png')
    plt.close()
    print("  [+] VAE Latent space plot saved to outputs/vae_latent_space.png")
    
    print("Generating 3 new random letters from pure N(0,1) noise...")
    Z_random = np.random.randn(3, 2)
    
    output = Z_random
    for layer in vae.layers[4:]:
        output = layer.forward(output)
        
    for i in range(3):
        image = output[i].reshape((7, 5))
        plt.figure(figsize=(3, 4))
        plt.imshow(image, cmap='gray_r', vmin=0, vmax=1)
        plt.title(f"VAE Generated Z=[{Z_random[i,0]:.2f}, {Z_random[i,1]:.2f}]")
        plt.axis('off')
        plt.savefig(f'outputs/vae_generated_random_{i}.png')
        plt.close()
        print(f"  [+] VAE generated letter saved to outputs/vae_generated_random_{i}.png")

def main():
    np.random.seed(42)
    print("Loading base dataset...")
    X_base = FontLoader.load_and_flatten('src/data/font.h')
    
    print("Augmenting fonts (Normal, Bold, Italic)...")
    X_multi = FontAugmenter.create_multipattern_dataset(X_base)
    print(f"New multi-font dataset dimensions: {X_multi.shape}")
    
    caracteres = [chr(0x60 + i) if i < 31 else 'DEL' for i in range(32)]
    labels_multi = caracteres * 3
    
    print("\n--- Training VAE ---")
    vae = train_vae(X_multi, epochs=20000, learning_rate=0.05)
    
    experiment_vae_generation(vae, X_multi, labels_multi)

if __name__ == "__main__":
    main()
