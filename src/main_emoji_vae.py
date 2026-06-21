import numpy as np
import matplotlib.pyplot as plt
import os

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer, VAEBottleneckLayer
from core.network import Network
from core.optimizers import SGD
from core.losses import BCE
from utils.emoji_loader import EmojiLoader

def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> int:
    binary_predicted = (predicted >= 0.5).astype(int)
    binary_expected = expected.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    errors_per_item = np.sum(differences, axis=1)
    return np.max(errors_per_item)

def create_emoji_vae(latent_dim: int, kl_weight: float) -> Network:
    vae = Network()
    
    # --- ENCODER ---
    vae.add(Linear(input_size=256, output_size=128))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(input_size=128, output_size=64))
    vae.add(ActivationLayer(Tanh()))
    
    # Proyectamos a 2*latent_dim (latent_dim para mu, latent_dim para log_var)
    vae.add(Linear(input_size=64, output_size=latent_dim * 2))
    
    # CUELLO DE BOTELLA VAE
    vae.add(VAEBottleneckLayer(latent_dim=latent_dim, kl_weight=kl_weight))
    
    # --- DECODER ---
    vae.add(Linear(input_size=latent_dim, output_size=64))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(input_size=64, output_size=128))
    vae.add(ActivationLayer(Tanh()))
    
    vae.add(Linear(input_size=128, output_size=256))
    vae.add(ActivationLayer(Sigmoid()))
    
    return vae

def train_emoji_vae(X: np.ndarray, epochs: int = 30000, learning_rate: float = 0.01, latent_dim: int = 10, kl_weight: float = 0.001) -> Network:
    vae = create_emoji_vae(latent_dim, kl_weight)
    loss_function = BCE()
    optimizer = SGD(learning_rate=learning_rate)
    
    vae_layer = vae.layers[5] # VAEBottleneckLayer
    
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

def experiment_emoji_generation(vae: Network, X: np.ndarray, labels: list, latent_dim: int):
    print("\n--- EXPERIMENT: VAE Emoji Generation ---")
    
    # Proyectar al espacio latente
    mu_logvar = X
    for layer in vae.layers[:5]: # Hasta antes del VAEBottleneckLayer
        mu_logvar = layer.forward(mu_logvar)
    mu = mu_logvar[:, :latent_dim]
    
    # Plotear solo las primeras 2 dimensiones del espacio latente
    plt.figure(figsize=(10, 8))
    plt.scatter(mu[:, 0], mu[:, 1], edgecolors='k', alpha=0.8)
    
    for i, label in enumerate(labels):
        plt.annotate(label, (mu[i, 0], mu[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
    plt.title("VAE Latent Space (first 2 dimensions)")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.grid(True)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/vae_emoji_latent_space.png')
    plt.close()
    print("  [+] VAE Emoji latent space plot saved to outputs/vae_emoji_latent_space.png")
    
    # Generar nuevos emojis
    print("Generating 5 new random emojis from pure N(0,1) noise...")
    Z_random = np.random.randn(5, latent_dim)
    
    output = Z_random
    for layer in vae.layers[6:]: # Desde después del VAEBottleneckLayer
        output = layer.forward(output)
        
    for i in range(5):
        image = output[i].reshape((16, 16))
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray_r', vmin=0, vmax=1)
        plt.title(f"Generated Emoji {i+1}")
        plt.axis('off')
        plt.savefig(f'outputs/vae_generated_emoji_{i+1}.png')
        plt.close()
    print("  [+] Generated emojis saved to outputs/")

def main():
    print("--- VAE Emoji Training ---")
    
    # Cargar datos
    loader = EmojiLoader()
    X, labels = loader.get_all_data()
    print(f"Loaded {len(labels)} emojis.")
    
    # Entrenar VAE
    latent_dim = 10
    kl_weight = 0.001
    vae = train_emoji_vae(X, epochs=30000, learning_rate=0.01, latent_dim=latent_dim, kl_weight=kl_weight)
    
    # Ejecutar experimentos
    experiment_emoji_generation(vae, X, labels, latent_dim)
    
    print("\n--- VAE Emoji Training Finished ---")

if __name__ == "__main__":
    main()
