import numpy as np
import os
import matplotlib.pyplot as plt

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer, VAEBottleneckLayer
from core.network import Network
from core.losses import BCE
from utils.data_loader import FontLoader

def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> int:
    binary_predicted = (predicted >= 0.5).astype(int)
    binary_expected = expected.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    return np.max(np.sum(differences, axis=1))

def create_vae(kl_weight: float) -> Network:
    vae = Network()
    vae.add(Linear(35, 24))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(24, 4))
    vae.add(VAEBottleneckLayer(latent_dim=2, kl_weight=kl_weight))
    vae.add(Linear(2, 24))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(24, 35))
    vae.add(ActivationLayer(Sigmoid()))
    return vae

def train_vae(X: np.ndarray, kl_weight: float, epochs: int = 5000):
    vae = create_vae(kl_weight)
    loss_function = BCE()
    
    for epoch in range(epochs):
        predicted = vae.forward(X)
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        vae.backward(initial_gradient, learning_rate=0.05)
            
    return vae

def main():
    """
    TEST 3: POSTERIOR COLLAPSE EN EL VAE
    Hipótesis: Un KL weight alto hace colapsar el Encoder, generando la misma letra difusa para todos.
    Un KL weight bajo anula la capacidad del VAE de tener un espacio latente contínuo Normal.
    """
    print("\n--- INICIANDO EXP3: VAE POSTERIOR COLLAPSE ---")
    np.random.seed(42)
    X = FontLoader.load_and_flatten('src/data/font.h')
    
    pesos_kl = [0.1, 0.0001, 1e-7]
    titulos = ["Alto (0.1) - Colapso", "Ideal (0.0001)", "Bajo (1e-7) - Overfit"]
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    for i, kl in enumerate(pesos_kl):
        print(f"Entrenando VAE con kl_weight={kl}...")
        vae = train_vae(X, kl_weight=kl, epochs=8000)
        
        # Muestreo desde el centro exacto de la campana (z=0,0)
        z = np.array([[0.0, 0.0]])
        output = z
        for layer in vae.layers[4:]:
            output = layer.forward(output)
            
        img = output.reshape(7, 5)
        axes[i].imshow(img, cmap='gray_r', vmin=0, vmax=1)
        axes[i].set_title(titulos[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/exp3_posterior_collapse.png')
    plt.close()
    print("Gráfica guardada en outputs/exp3_posterior_collapse.png")

if __name__ == "__main__":
    main()
