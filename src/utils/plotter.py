import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.network import Network

class Plotter:
    @staticmethod
    def plot_latent_space(autoencoder: Network, X: np.ndarray, title: str, filename: str, labels: list = None):
        latent_space = X
        for layer in autoencoder.layers[:4]:
            latent_space = layer.forward(latent_space)
            
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_space[:, 0], latent_space[:, 1], c='blue', edgecolors='k')
        
        for i in range(len(latent_space)):
            text_label = str(labels[i]) if labels is not None else str(i)
            plt.annotate(text_label, (latent_space[i, 0], latent_space[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=12)
            
        plt.title(f"Latent Space 2D - {title}")
        plt.xlabel("Z1")
        plt.ylabel("Z2")
        plt.grid(True)
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/{filename}')
        plt.close() 
        print(f"  [+] Gráfico guardado en: outputs/{filename}")

    @staticmethod
    def generate_new_letter(autoencoder: Network, z_coord: list, filename: str):
        Z = np.array([z_coord])
        
        output = Z
        for layer in autoencoder.layers[4:]:
            output = layer.forward(output)
            
        image = output.reshape((7, 5))
        
        plt.figure(figsize=(3, 4))
        plt.imshow(image, cmap='gray_r', vmin=0, vmax=1)
        plt.title(f"Generated Letter at Z={z_coord}")
        plt.axis('off')
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/{filename}')
        plt.close()
        print(f"  [+] Imagen guardada en: outputs/{filename}")

    @staticmethod
    def compare_reconstruction(autoencoder: Network, original_x: np.ndarray, label: str, filename: str):
        """Pasa una letra original por la red y grafica la entrada vs la salida predicha."""
        # Aseguramos que el vector sea 2D (1, 35) para que la matemática matricial no falle
        if original_x.ndim == 1:
            original_x = original_x.reshape(1, -1)
            
        # Pasamos la letra por toda la red (Encoder + Decoder)
        reconstructed_x = autoencoder.forward(original_x)
        
        # Moldeamos ambos vectores a 7x5
        img_original = original_x.reshape((7, 5))
        img_reconstructed = reconstructed_x.reshape((7, 5))
        
        # Creamos una figura con 1 fila y 2 columnas
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        
        # Plot 1: Original
        axes[0].imshow(img_original, cmap='gray_r', vmin=0, vmax=1)
        axes[0].set_title(f"Original: '{label}'")
        axes[0].axis('off')
        
        # Plot 2: Reconstrucción Predicha
        axes[1].imshow(img_reconstructed, cmap='gray_r', vmin=0, vmax=1)
        axes[1].set_title(f"Predicha: '{label}'")
        axes[1].axis('off')
        
        # Ajustamos el layout para que no se superpongan los títulos
        plt.tight_layout()
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/{filename}')
        plt.close()
        print(f"  [+] Comparación guardada en: outputs/{filename}")