import numpy as np
import os
import matplotlib.pyplot as plt

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer, VAEBottleneckLayer
from core.network import Network
from core.losses import BCE
from utils.data_loader import FontLoader

def create_vae() -> Network:
    vae = Network()
    vae.add(Linear(35, 24))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(24, 4))
    vae.add(VAEBottleneckLayer(latent_dim=2, kl_weight=0.0001))
    vae.add(Linear(2, 24))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(24, 35))
    vae.add(ActivationLayer(Sigmoid()))
    return vae

def train_vae(X: np.ndarray, epochs: int = 15000):
    vae = create_vae()
    loss_function = BCE()
    
    for epoch in range(epochs):
        predicted = vae.forward(X)
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        vae.backward(initial_gradient, learning_rate=0.05)
            
    return vae

def main():
    """
    TEST 4: INTERPOLACIÓN LATENTE
    Hipótesis: Dado que el espacio del VAE es denso y regularizado, trazar una línea
    recta entre el espacio latente de la letra 'A' y el de la 'G' (o cualquier otra) 
    debe generar una transición morfológica suave en la imagen reconstruida.
    """
    print("\n--- INICIANDO EXP4: INTERPOLACIÓN LATENTE VAE ---")
    np.random.seed(42)
    X = FontLoader.load_and_flatten('src/data/font.h')
    
    print("Entrenando VAE...")
    vae = train_vae(X, epochs=15000)
    
    # Letra A (índice 1) y letra G (índice 7) para interpolar
    idx_start = 1 # 'a'
    idx_end = 7   # 'g'
    
    # Obtener mu de 'a' y 'g'
    mu_logvar_start = X[idx_start:idx_start+1]
    mu_logvar_end = X[idx_end:idx_end+1]
    
    for layer in vae.layers[:3]:
        mu_logvar_start = layer.forward(mu_logvar_start)
        mu_logvar_end = layer.forward(mu_logvar_end)
        
    mu_start = mu_logvar_start[:, :2]
    mu_end = mu_logvar_end[:, :2]
    
    pasos = 6
    interpolacion = np.linspace(mu_start[0], mu_end[0], pasos)
    
    fig, axes = plt.subplots(1, pasos, figsize=(12, 3))
    fig.suptitle("Interpolación en el Espacio Latente ('a' -> 'g')")
    
    for i, z_punto in enumerate(interpolacion):
        z = np.array([z_punto])
        output = z
        for layer in vae.layers[4:]:
            output = layer.forward(output)
            
        img = output.reshape(7, 5)
        axes[i].imshow(img, cmap='gray_r', vmin=0, vmax=1)
        axes[i].set_title(f"Paso {i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/exp4_latent_interpolation.png')
    plt.close()
    print("Gráfica guardada en outputs/exp4_latent_interpolation.png")

if __name__ == "__main__":
    main()
