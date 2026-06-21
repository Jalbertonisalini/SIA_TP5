import matplotlib.pyplot as plt
import numpy as np
import os
from .emoji_loader import EmojiLoader

def plot_emoji_dataset():
    """
    Carga el dataset de emojis y genera un gráfico con todos ellos,
    guardándolo en el directorio 'outputs'.
    """
    loader = EmojiLoader()
    emojis, labels = loader.get_all_data()
    
    num_emojis = len(labels)
    # Calculamos un grid que se ajuste a la cantidad de emojis
    cols = 4
    rows = int(np.ceil(num_emojis / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = axes.flatten() # Aplanar el array de ejes para iterar fácilmente
    
    print(f"Generando gráfico para {num_emojis} emojis...")
    
    for i, (emoji_flat, label) in enumerate(zip(emojis, labels)):
        img = emoji_flat.reshape(16, 16)
        ax = axes[i]
        ax.imshow(img, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(label.capitalize(), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ocultar ejes sobrantes si el número de emojis no llena el grid
    for i in range(num_emojis, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout(pad=0.5)
    
    # Guardar la figura
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'emoji_dataset.png')
    plt.savefig(save_path)
    
    print(f"[+] Gráfico del dataset de emojis guardado en: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_emoji_dataset()
