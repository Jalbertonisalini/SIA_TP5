import numpy as np
import os
import matplotlib.pyplot as plt

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.optimizers import SGD
from core.network import Network
from core.losses import BCE
from utils.data_loader import FontLoader

def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> int:
    binary_predicted = (predicted >= 0.5).astype(int)
    binary_expected = expected.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    return np.max(np.sum(differences, axis=1))

def create_autoencoder(latent_dim: int) -> Network:
    """
    Crea un Autoencoder con un tamaño de cuello de botella variable.
    El objetivo es testear cómo afecta la capacidad de abstracción.
    """
    autoencoder = Network()
    
    # Encoder
    autoencoder.add(Linear(input_size=35, output_size=16))
    autoencoder.add(ActivationLayer(Tanh()))
    autoencoder.add(Linear(input_size=16, output_size=latent_dim))
    autoencoder.add(ActivationLayer(Tanh())) 
    
    # Decoder
    autoencoder.add(Linear(input_size=latent_dim, output_size=16))
    autoencoder.add(ActivationLayer(Tanh()))
    autoencoder.add(Linear(input_size=16, output_size=35))
    autoencoder.add(ActivationLayer(Sigmoid()))
    
    return autoencoder

def train_autoencoder(X: np.ndarray, latent_dim: int, epochs: int = 5000, learning_rate: float = 0.1):
    ae = create_autoencoder(latent_dim)
    loss_function = BCE()
    optimizer = SGD(learning_rate=learning_rate)
    
    for epoch in range(epochs):
        predicted = ae.forward(X)
        max_incorrect_pixels = evaluate_pixel_diff(X, predicted)
        
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        ae.backward(initial_gradient, optimizer)
        
        # Early stopping
        if max_incorrect_pixels == 0:
            return ae, epoch
            
    return ae, epochs

def main():
    """
    TEST 1: BARRIDO DE ARQUITECTURAS
    Hipótesis: Un cuello de botella muy restrictivo (ej 2D) fuerza pérdida de información.
    Un cuello más grande (ej 16D) permite memorización perfecta y convergencia rápida.
    """
    print("\n--- INICIANDO EXP1: BARRIDO DE BOTELLENECKS ---")
    np.random.seed(42)
    X = FontLoader.load_and_flatten('src/data/font.h')
    
    cuellos = [2, 5, 10, 16]
    idx_letra = 1 # Usaremos la 'b' para la comparación visual
    
    fig, axes = plt.subplots(1, len(cuellos) + 1, figsize=(12, 3))
    
    # Graficar la letra original primero
    img_original = X[idx_letra].reshape(7, 5)
    axes[0].imshow(img_original, cmap='gray_r', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i, dim in enumerate(cuellos):
        print(f"Entrenando AE con cuello de {dim}D...")
        ae, epocas = train_autoencoder(X, latent_dim=dim, epochs=5000)
        
        predicted = ae.forward(X)
        max_err = evaluate_pixel_diff(X, predicted)
        print(f"  Resultado {dim}D -> Epocas: {epocas}, Error Max: {max_err} píxeles")
        
        # Graficar reconstrucción de la letra de prueba
        reconstruccion = predicted[idx_letra].reshape(7, 5)
        axes[i+1].imshow(reconstruccion, cmap='gray_r', vmin=0, vmax=1)
        axes[i+1].set_title(f"AE {dim}D")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/exp1_bottlenecks.png')
    plt.close()
    print("Gráfica guardada en outputs/exp1_bottlenecks.png")

if __name__ == "__main__":
    main()
