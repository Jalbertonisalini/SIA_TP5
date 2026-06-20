import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Necesario para formatear los ejes

from core.network import Network

class Plotter:
    # --- Estilo Académico / Científico ---
    FACECOLOR = "#FFFFFF" 
    GRID_COLOR = "#E6E6E6"
    TITLE_COLOR = "#000000"
    TEXT_COLOR = "#333333" 
    
    SERIES_COLORS = ["#1F4E79", "#A63636", "#595959", "#2E7559"]
    
    # (El resto de tus métodos siguen acá...)

    @staticmethod
    def plot_latent_space(autoencoder: Network, X: np.ndarray, title: str, filename: str, labels: list = None):
        latent_space = X
        for layer in autoencoder.layers[:4]:
            latent_space = layer.forward(latent_space)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#FBFCFE")
        ax.set_facecolor("#FBFCFE")

        ax.scatter(
            latent_space[:, 0],
            latent_space[:, 1],
            c=Plotter.LATENT_POINT_COLOR,
            edgecolors=Plotter.LATENT_EDGE_COLOR,
            linewidths=0.9,
            s=70,
            alpha=0.95,
        )

        for i in range(len(latent_space)):
            text_label = str(labels[i]) if labels is not None else str(i)
            ax.annotate(
                text_label,
                (latent_space[i, 0], latent_space[i, 1]),
                xytext=(6, 5),
                textcoords='offset points',
                fontsize=10,
                color=Plotter.TEXT_COLOR,
            )

        ax.set_title(f"Latent Space 2D - {title}", color=Plotter.TITLE_COLOR, pad=12)
        ax.set_xlabel("Z1", color=Plotter.TEXT_COLOR)
        ax.set_ylabel("Z2", color=Plotter.TEXT_COLOR)
        ax.grid(True, color=Plotter.GRID_COLOR, linewidth=0.8, alpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(Plotter.GRID_COLOR)
        ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        fig.tight_layout()
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico guardado en: outputs/{filename}")

    @staticmethod
    def generate_new_letter(autoencoder: Network, z_coord: list, filename: str):
        Z = np.array([z_coord])
        
        output = Z
        for layer in autoencoder.layers[4:]:
            output = layer.forward(output)
            
        image = output.reshape((7, 5))
        
        fig, ax = plt.subplots(figsize=(3, 4))
        fig.patch.set_facecolor("#FBFCFE")
        ax.imshow(image, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f"Generated Letter at Z={z_coord}", color=Plotter.TITLE_COLOR, pad=10)
        ax.axis('off')
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=160, bbox_inches='tight')
        plt.close(fig)
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
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2))
        fig.patch.set_facecolor("#FBFCFE")
        
        # Plot 1: Original
        axes[0].imshow(img_original, cmap='gray_r', vmin=0, vmax=1)
        axes[0].set_title(f"Original: '{label}'", color=Plotter.TITLE_COLOR)
        axes[0].axis('off')
        
        # Plot 2: Reconstrucción Predicha
        axes[1].imshow(img_reconstructed, cmap='gray_r', vmin=0, vmax=1)
        axes[1].set_title(f"Predicha: '{label}'", color=Plotter.TITLE_COLOR)
        axes[1].axis('off')
        
        # Ajustamos el layout para que no se superpongan los títulos
        fig.tight_layout()
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Comparación guardada en: outputs/{filename}")

    @staticmethod
    def plot_architecture_comparison(resultados_loss: dict, max_epochs: int, filename: str):
        print("\nGenerando gráfico estadístico...")
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#FBFCFE")
        ax.set_facecolor("#FBFCFE")

        for (nombre, historiales), color in zip(resultados_loss.items(), Plotter.SERIES_COLORS):
            matriz_historiales = np.array(historiales)
            loss_promedio = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            epocas_x = np.arange(max_epochs)

            ax.plot(epocas_x, loss_promedio, label=f"{nombre} (Media)", color=color, linewidth=2.3)
            ax.fill_between(
                epocas_x,
                loss_promedio - loss_std,
                loss_promedio + loss_std,
                color=color,
                alpha=0.14,
            )

        ax.set_title("Convergencia Multi-Seed: Comparativa de Arquitecturas (5 Runs)", color=Plotter.TITLE_COLOR, pad=12)
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR)
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR)
        ax.legend(frameon=False, labelcolor=Plotter.TEXT_COLOR)
        ax.grid(True, linestyle='--', alpha=0.65, color=Plotter.GRID_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(Plotter.GRID_COLOR)
        ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        fig.tight_layout()

        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico de convergencia guardado en: outputs/{filename}")

    @staticmethod
    def plot_learning_rate_comparison(resultados_loss: dict, max_epochs: int, filename: str, smooth_weight: float = 0.95):
        print("\nGenerando gráfico de Learning Rates...")
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#FBFCFE")
        ax.set_facecolor("#FBFCFE")

        # 1. Aseguramos que la grilla quede por detrás de los datos dibujados
        ax.set_axisbelow(True)

        # Helper interno para suavizar la curva (como un custom hook de lógica)
        def ema_smooth(scalars, weight):
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return np.array(smoothed)

        for (lr_str, historiales), color in zip(resultados_loss.items(), Plotter.SERIES_COLORS):
            matriz_historiales = np.array(historiales)
            loss_promedio_crudo = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            epocas_x = np.arange(max_epochs)

            # 2. Aplicamos el suavizado exponencial
            loss_suavizado = ema_smooth(loss_promedio_crudo, smooth_weight)

            # 3. Dibujamos el ruido crudo de fondo (transparente) y la línea limpia por encima
            ax.plot(epocas_x, loss_promedio_crudo, color=color, alpha=0.15, linewidth=0.6) # Bajamos de 1 a 0.6
            ax.plot(epocas_x, loss_suavizado, label=f"LR = {lr_str}", color=color, linewidth=1.5) # Bajamos de 2.5 a 1.5
            
            # Franja de desviación estándar calculada sobre la línea suavizada
            ax.fill_between(
                epocas_x,
                loss_suavizado - loss_std,
                loss_suavizado + loss_std,
                color=color,
                alpha=0.14,
                edgecolor='none' # Evita que se dibuje una línea dura en el borde del sombreado
            )

        ax.set_title("Convergencia Multi-Seed: Comparativa de Learning Rates", color=Plotter.TITLE_COLOR, pad=15, fontsize=14, fontweight='bold')
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR, fontweight='semibold')
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR, fontweight='semibold')
        ax.set_ylim(0, 1.0)
        
        # 4. Formateo prolijo del eje X (ej: 5000 se lee como '5k')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))

        ax.legend(frameon=False, labelcolor=Plotter.TEXT_COLOR, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.65, color=Plotter.GRID_COLOR)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(Plotter.GRID_COLOR)
        ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        
        fig.tight_layout()

        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        # 5. DPI en 300 garantiza calidad de imprenta / PDF
        fig.savefig(filepath, dpi=300, bbox_inches='tight') 
        plt.close(fig)
        print(f"  [+] Gráfico guardado en: {filepath}")
        
    @staticmethod
    def plot_dataset_capacity_comparison(sizes: np.ndarray, mean_pixels: np.ndarray, pixeles_std: np.ndarray, loss_media: np.ndarray, loss_std: np.ndarray, filename: str):
        print("\nGenerando gráfico de Capacidad...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_facecolor("#FBFCFE")

        # Usamos colores de tu paleta personalizada para mantener coherencia
        color_pixeles = Plotter.SERIES_COLORS[2] # Naranja/Rojo (#E76F51)
        color_loss = Plotter.SERIES_COLORS[1]    # Verde/Celeste (#2A9D8F)

        # --- PANEL SUPERIOR: Error Físico (Píxeles) ---
        ax1.set_facecolor("#FBFCFE")
        ax1.set_title("Stress Test: Colapso del Espacio Latente 2D", pad=15, fontsize=14, fontweight='bold', color=Plotter.TITLE_COLOR)
        ax1.set_ylabel('Max Píxeles Incorrectos', color=color_pixeles, fontweight='bold')
        
        ax1.plot(sizes, mean_pixels, marker='o', markersize=6, color=color_pixeles, linewidth=2.0, label="Promedio Píxeles Errados")
        ax1.fill_between(sizes, np.maximum(0, mean_pixels - pixeles_std), mean_pixels + pixeles_std, color=color_pixeles, alpha=0.15, edgecolor='none')
        
        # Línea límite de viabilidad
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label="Umbral de Viabilidad (<= 1 píxel)")
        ax1.legend(loc='upper left', frameon=False, labelcolor=Plotter.TEXT_COLOR)
        ax1.grid(True, linestyle='--', alpha=0.65, color=Plotter.GRID_COLOR)

        # --- PANEL INFERIOR: Error Matemático (BCE) ---
        ax2.set_facecolor("#FBFCFE")
        ax2.set_xlabel('Cantidad de Letras en el Dataset (N)', color=Plotter.TEXT_COLOR, fontweight='semibold')
        ax2.set_ylabel('Loss Promedio (BCE)', color=color_loss, fontweight='bold')
        
        ax2.plot(sizes, loss_media, marker='s', markersize=6, color=color_loss, linewidth=2.0, label="Loss Promedio")
        ax2.fill_between(sizes, loss_media - loss_std, loss_media + loss_std, color=color_loss, alpha=0.15, edgecolor='none')
        
        ax2.legend(loc='upper left', frameon=False, labelcolor=Plotter.TEXT_COLOR)
        ax2.grid(True, linestyle='--', alpha=0.65, color=Plotter.GRID_COLOR)

        # Limpiamos bordes de ambos paneles y aplicamos color de texto global
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(Plotter.GRID_COLOR)
            ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
            ax.tick_params(colors=Plotter.TEXT_COLOR)

        fig.tight_layout()
        
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico guardado en: {filepath}")
        
    @staticmethod
    # CAMBIO: Quitamos max_epochs de los parámetros
    def plot_optimizer_comparison(resultados_loss: dict, filename: str):
        print("\nGenerando gráfico de Optimizadores (Estilo Académico Dinámico)...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)
        ax.set_axisbelow(True)

        colores = [Plotter.SERIES_COLORS[0], Plotter.SERIES_COLORS[1]] 

        for (nombre, historiales), color in zip(resultados_loss.items(), colores):
            matriz_historiales = np.array(historiales)
            loss_promedio = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            
            # CAMBIO CLAVE: El eje X mide exactamente lo que midió este optimizador
            epocas_x = np.arange(len(loss_promedio))

            ax.plot(epocas_x, loss_promedio, label=f"{nombre} (Media)", color=color, linewidth=1.5)
            
            ax.fill_between(
                epocas_x, 
                loss_promedio - loss_std, 
                loss_promedio + loss_std, 
                color=color, 
                alpha=0.10, 
                edgecolor='none'
            )

        ax.set_title("Convergencia Multi-Seed: SGD vs Adam", color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR, fontsize=11)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))

        ax.legend(frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR, loc='upper right')
        
        ax.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(Plotter.GRID_COLOR)
        ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        
        fig.tight_layout()
        
        import os
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico guardado en: {filepath}")