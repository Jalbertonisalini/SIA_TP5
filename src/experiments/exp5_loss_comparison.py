import numpy as np
import os
import matplotlib.pyplot as plt

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.optimizers import SGD
from core.network import Network
from core.losses import BCE, MSE
from utils.data_loader import FontLoader

def create_autoencoder() -> Network:
    ae = Network()
    ae.add(Linear(35, 16))
    ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 2))
    ae.add(ActivationLayer(Tanh())) 
    ae.add(Linear(2, 16))
    ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 35))
    ae.add(ActivationLayer(Sigmoid()))
    return ae

def train_with_loss(X: np.ndarray, loss_name: str, epochs: int = 8000):
    ae = create_autoencoder()
    loss_function = BCE() if loss_name == 'BCE' else MSE()
    lr = 0.1 if loss_name == 'BCE' else 0.5 # MSE necesita más LR porque sus gradientes son menores
    optimizer = SGD(learning_rate=lr)
    
    for epoch in range(epochs):
        predicted = ae.forward(X)
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        ae.backward(initial_gradient, optimizer)
            
    return ae

def main():
    """
    TEST 5: COMPARACIÓN DE FUNCIONES DE PÉRDIDA (BCE vs MSE)
    Hipótesis: La Entropía Cruzada Binaria (BCE) empuja fuertemente los valores a 0 y 1,
    creando imágenes más nítidas, mientras que el Error Cuadrático Medio (MSE) es
    más permisivo con valores intermedios (creando letras borrosas o grises).
    """
    print("\n--- INICIANDO EXP5: BCE vs MSE ---")
    np.random.seed(42)
    X = FontLoader.load_and_flatten('src/data/font.h')
    
    print("Entrenando AE con BCE...")
    ae_bce = train_with_loss(X, 'BCE')
    print("Entrenando AE con MSE...")
    ae_mse = train_with_loss(X, 'MSE')
    
    idx_letra = 15 # Letra 'p'
    
    pred_bce = ae_bce.forward(X[idx_letra:idx_letra+1]).reshape(7, 5)
    pred_mse = ae_mse.forward(X[idx_letra:idx_letra+1]).reshape(7, 5)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    axes[0].imshow(X[idx_letra].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(pred_bce, cmap='gray_r', vmin=0, vmax=1)
    axes[1].set_title("BCE (Nítido)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mse, cmap='gray_r', vmin=0, vmax=1)
    axes[2].set_title("MSE (Borroso)")
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/exp5_loss_comparison.png')
    plt.close()
    print("Gráfica guardada en outputs/exp5_loss_comparison.png")

if __name__ == "__main__":
    main()
