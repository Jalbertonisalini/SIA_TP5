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

def add_noise(X: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    X_noisy = X.copy()
    for i in range(X_noisy.shape[0]):
        num_flips = int(noise_level * X.shape[1])
        indices = np.random.choice(X.shape[1], num_flips, replace=False)
        X_noisy[i, indices] = 1 - X_noisy[i, indices]
    return X_noisy

def create_dae() -> Network:
    ae = Network()
    ae.add(Linear(35, 24))
    ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 10))
    ae.add(ActivationLayer(Tanh())) 
    ae.add(Linear(10, 24))
    ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35))
    ae.add(ActivationLayer(Sigmoid()))
    return ae

def train_dae(X_clean: np.ndarray, noise_level: float, epochs: int = 5000):
    ae = create_dae()
    loss_function = BCE()
    optimizer = SGD(learning_rate=0.1)
    X_noisy = add_noise(X_clean, noise_level)
    
    for epoch in range(epochs):
        predicted = ae.forward(X_noisy)
        initial_gradient = loss_function.derivative(expected=X_clean, predicted=predicted)
        ae.backward(initial_gradient, optimizer)
        
        max_incorrect = evaluate_pixel_diff(X_clean, predicted)
        if max_incorrect <= 1:
            return ae, epoch, X_noisy, predicted
            
    # Última predicción si no converge temprano
    predicted = ae.forward(X_noisy)
    return ae, epochs, X_noisy, predicted

def main():
    """
    TEST 2: LÍMITE DE RUIDO (BREAKDOWN POINT)
    Hipótesis: Al incrementar el ruido de 10% a 50%, la red eventualmente perderá
    la capacidad matemática de deducir la letra limpia, llegando a un Breakdown Point.
    """
    print("\n--- INICIANDO EXP2: BREAKDOWN DE RUIDO DAE ---")
    np.random.seed(42)
    X = FontLoader.load_and_flatten('src/data/font.h')
    
    niveles = [0.10, 0.25, 0.40, 0.50]
    idx_letra = 12 # Probamos con la 'm' (índice 12 si a=0)
    
    fig, axes = plt.subplots(len(niveles), 3, figsize=(7, 10))
    fig.suptitle("Evolución de Reconstrucción vs Nivel de Ruido")
    
    for i, noise in enumerate(niveles):
        print(f"Entrenando DAE con {noise*100}% de ruido...")
        ae, epocas, X_noisy, predicted = train_dae(X, noise, epochs=10000)
        max_err = evaluate_pixel_diff(X, predicted)
        print(f"  Ruido: {noise*100}% -> Epocas: {epocas}, Error Max: {max_err}")
        
        # Limpia
        axes[i, 0].imshow(X[idx_letra].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[i, 0].set_title("Limpia")
        axes[i, 0].axis('off')
        
        # Con Ruido
        axes[i, 1].imshow(X_noisy[idx_letra].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Ruido {int(noise*100)}%")
        axes[i, 1].axis('off')
        
        # Predicha
        axes[i, 2].imshow(predicted[idx_letra].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicha")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/exp2_noise_limits.png')
    plt.close()
    print("Gráfica guardada en outputs/exp2_noise_limits.png")

if __name__ == "__main__":
    main()
