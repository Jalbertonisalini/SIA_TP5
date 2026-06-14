import numpy as np

# Importamos todas las piezas que construimos
from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.network import Network
from core.losses import MSE, BCE
from utils.data_loader import FontLoader
from utils.plotter import Plotter


def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> int:
    """
    Cuenta cuántos píxeles difieren entre la entrada y la reconstrucción.
    Como la salida de la Sigmoide es continua (0 a 1), umbralizamos a 0.5.
    """
    binary_predicted = (predicted >= 0.5).astype(int)
    binary_expected = expected.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    errors_per_letter = np.sum(differences, axis=1)
    return np.max(errors_per_letter)


def create_autoencoder() -> Network:
    """Instancia la arquitectura base del autoencoder 35 -> 16 -> 2 -> 16 -> 35"""
    autoencoder = Network()
    
    # --- ENCODER ---
    autoencoder.add(Linear(input_size=35, output_size=16))
    autoencoder.add(ActivationLayer(Tanh()))
    
    # CUELLO DE BOTELLA (Latent Space 2D)
    autoencoder.add(Linear(input_size=16, output_size=2))
    autoencoder.add(ActivationLayer(Tanh())) 
    
    # --- DECODER ---
    autoencoder.add(Linear(input_size=2, output_size=16))
    autoencoder.add(ActivationLayer(Tanh()))
    
    autoencoder.add(Linear(input_size=16, output_size=35))
    autoencoder.add(ActivationLayer(Sigmoid()))
    
    return autoencoder


def train_autoencoder(X: np.ndarray, epochs: int = 15000, learning_rate: float = 0.1) -> Network:
    """Entrena la red y devuelve el modelo entrenado."""
    autoencoder = create_autoencoder()
    loss_function = BCE()
    
    for epoch in range(epochs):
        predicted = autoencoder.forward(X)
        loss = loss_function.calculate(expected=X, predicted=predicted)
        max_incorrect_pixels = evaluate_pixel_diff(X, predicted)
        
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        autoencoder.backward(initial_gradient, learning_rate)
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:05d} | Loss (BCE): {loss:.4f} | Max incorrect pixels: {max_incorrect_pixels}")
            
        if max_incorrect_pixels <= 1:
            print(f"  Objective achieved! Early convergence at epoch {epoch}. Max error <= 1.")
            break
            
    return autoencoder


def experiment_full_dataset(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 1: Full Dataset (32 letters) ---")
    ae_full = train_autoencoder(X, epochs=10000, learning_rate=0.1)
    Plotter.plot_latent_space(ae_full, X, "Full Dataset", "latent_full.png", labels=labels)
    Plotter.generate_new_letter(ae_full, z_coord=[1.0, 1.0], filename="generated_letter_full.png")


def experiment_subset(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 2: Subset (5 letters) ---")
    X_sub = X[:5]
    labels_sub = labels[:5]
    ae_sub = train_autoencoder(X_sub, epochs=10000, learning_rate=0.1)
    Plotter.plot_latent_space(ae_sub, X_sub, "Subset (5 letters)", "latent_subset.png", labels=labels_sub)    
    print("\nGenerating a new letter from intermediate coordinates...")
    # Acá le inventamos una coordenada en el medio del plano
    Plotter.generate_new_letter(ae_sub, z_coord=[1.0, 1.0], filename="generated_letter.png")


def main():
    # Seed para reproducibilidad
    np.random.seed(800)
    print("Loading dataset...")
    X = FontLoader.load_and_flatten('data/font.h')
    print(f"Data dimensions: {X.shape}")
    
    # Generamos los caracteres a partir de su código ASCII base (0x60)
    caracteres = [chr(0x60 + i) if i < 31 else 'DEL' for i in range(32)]
    
    # Corremos ambos experimentos secuencialmente pasando las etiquetas
    experiment_full_dataset(X, caracteres)
    experiment_subset(X, caracteres)


if __name__ == "__main__":
    main()