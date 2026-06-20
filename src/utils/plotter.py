import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from core.network import Network

class Plotter:
    """
    Clase utilitaria para la generación de gráficos estáticos y reportes visuales.
    Implementa una paleta de colores y estilos orientados a publicaciones académicas.
    """
    # --- Estilo Académico / Científico ---
    FACECOLOR = "#FFFFFF"       # Fondo blanco puro
    GRID_COLOR = "#E6E6E6"      # Grilla gris muy clara y sutil
    TITLE_COLOR = "#000000"     # Títulos en negro sólido
    TEXT_COLOR = "#333333"      # Textos en gris oscuro (cansa menos la vista)
    
    # Paleta sobria: Azul Marino (Navy), Rojo Ladrillo (Crimson), Gris Pizarra, Verde Bosque
    SERIES_COLORS = ["#1F4E79", "#A63636", "#595959", "#2E7559"]
    
    # Colores específicos para los puntos de dispersión del espacio latente
    LATENT_POINT_COLOR = "#1F4E79" 
    LATENT_EDGE_COLOR = "#FFFFFF"

    @staticmethod
    def plot_latent_space(autoencoder: Network, X: np.ndarray, title: str, filename: str, labels: list = None):
        """
        Grafica la representación bidimensional del espacio latente (Z) de las entradas.
        Calcula el 'forward pass' solo hasta la capa central del autoencoder.
        """
        latent_space = X
        for layer in autoencoder.layers[:4]:
            latent_space = layer.forward(latent_space)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)

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
        
        ax.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(Plotter.GRID_COLOR)
        ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        fig.tight_layout()
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico guardado en: outputs/{filename}")

    @staticmethod
    def generate_new_letter(autoencoder: Network, z_coord: list, filename: str):
        """
        Inyecta una coordenada específica en el espacio latente y realiza 
        el 'forward pass' a través del decodificador para generar una imagen sintética.
        """
        Z = np.array([z_coord])
        
        output = Z
        for layer in autoencoder.layers[4:]:
            output = layer.forward(output)
            
        image = output.reshape((7, 5))
        
        fig, ax = plt.subplots(figsize=(3, 4))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.imshow(image, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f"Generated Letter at Z={z_coord}", color=Plotter.TITLE_COLOR, pad=10)
        ax.axis('off')
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Imagen guardada en: outputs/{filename}")

    @staticmethod
    def compare_reconstruction(autoencoder: Network, original_x: np.ndarray, label: str, filename: str):
        """
        Pasa una letra original completa por la red (Encoder + Decoder) y 
        grafica la entrada real frente a la salida reconstruida para evaluar calidad visual.
        """
        if original_x.ndim == 1:
            original_x = original_x.reshape(1, -1)
            
        reconstructed_x = autoencoder.forward(original_x)
        
        img_original = original_x.reshape((7, 5))
        img_reconstructed = reconstructed_x.reshape((7, 5))
        
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        
        axes[0].imshow(img_original, cmap='gray_r', vmin=0, vmax=1)
        axes[0].set_title(f"Original: '{label}'", color=Plotter.TITLE_COLOR)
        axes[0].axis('off')
        
        axes[1].imshow(img_reconstructed, cmap='gray_r', vmin=0, vmax=1)
        axes[1].set_title(f"Predicha: '{label}'", color=Plotter.TITLE_COLOR)
        axes[1].axis('off')
        
        fig.tight_layout()
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Comparación guardada en: outputs/{filename}")

    @staticmethod
    def plot_architecture_comparison(resultados_loss: dict, max_epochs: int, filename: str):
        """
        Genera un gráfico comparativo de la convergencia (Loss vs Épocas) 
        para distintas topologías de red, mostrando la media y desviación estándar multi-seed.
        """
        print("\nGenerando gráfico estadístico...")
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)
        ax.set_axisbelow(True)

        for (nombre, historiales), color in zip(resultados_loss.items(), Plotter.SERIES_COLORS):
            matriz_historiales = np.array(historiales)
            loss_promedio = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            epocas_x = np.arange(max_epochs)

            ax.plot(epocas_x, loss_promedio, label=f"{nombre} (Media)", color=color, linewidth=1.5)
            ax.fill_between(
                epocas_x,
                loss_promedio - loss_std,
                loss_promedio + loss_std,
                color=color,
                alpha=0.10,
                edgecolor='none'
            )

        ax.set_title("Convergencia Multi-Seed: Comparativa de Arquitecturas (5 Runs)", color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR, fontsize=11)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))
        
        ax.legend(frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR)
        ax.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(Plotter.GRID_COLOR)
        ax.spines['bottom'].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        fig.tight_layout()

        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico de convergencia guardado en: outputs/{filename}")

    @staticmethod
    def plot_learning_rate_comparison(resultados_loss: dict, max_epochs: int, filename: str, smooth_weight: float = 0.95):
        """
        Grafica el análisis de sensibilidad frente a distintos hiperparámetros de Learning Rate.
        Aplica suavizado exponencial (EMA) nativo para estabilizar la visualización de gradientes oscilatorios.
        """
        print("\nGenerando gráfico de Learning Rates...")
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)
        ax.set_axisbelow(True)

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

            loss_suavizado = ema_smooth(loss_promedio_crudo, smooth_weight)

            ax.plot(epocas_x, loss_promedio_crudo, color=color, alpha=0.15, linewidth=0.6)
            ax.plot(epocas_x, loss_suavizado, label=f"LR = {lr_str}", color=color, linewidth=1.5)
            
            ax.fill_between(
                epocas_x,
                loss_suavizado - loss_std,
                loss_suavizado + loss_std,
                color=color,
                alpha=0.10,
                edgecolor='none' 
            )

        ax.set_title("Convergencia Multi-Seed: Comparativa de Learning Rates", color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylim(0, 1.0)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))

        ax.legend(frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)
        
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
    def plot_dataset_capacity_comparison(sizes: np.ndarray, mean_pixels: np.ndarray, pixeles_std: np.ndarray, loss_media: np.ndarray, loss_std: np.ndarray, filename: str):
        """
        Genera un panel dual (Subplots) para el Stress Test del cuello de botella.
        Muestra la degradación del error de píxeles (Panel Superior) y del error matemático BCE (Panel Inferior).
        """
        print("\nGenerando gráfico de Capacidad...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_facecolor(Plotter.FACECOLOR)

        color_pixeles = Plotter.SERIES_COLORS[1] # Rojo Ladrillo
        color_loss = Plotter.SERIES_COLORS[0]    # Azul Marino

        # --- PANEL SUPERIOR ---
        ax1.set_facecolor(Plotter.FACECOLOR)
        ax1.set_axisbelow(True)
        ax1.set_title("Stress Test: Colapso del Espacio Latente 2D", pad=15, fontsize=13, color=Plotter.TITLE_COLOR)
        ax1.set_ylabel('Max Píxeles Incorrectos', color=color_pixeles, fontsize=11)
        
        ax1.plot(sizes, mean_pixels, marker='o', markersize=6, color=color_pixeles, linewidth=1.5, label="Promedio Píxeles Errados")
        ax1.fill_between(sizes, np.maximum(0, mean_pixels - pixeles_std), mean_pixels + pixeles_std, color=color_pixeles, alpha=0.10, edgecolor='none')
        
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label="Umbral de Viabilidad (<= 1 píxel)")
        ax1.legend(loc='upper left', frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR)
        ax1.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)

        # --- PANEL INFERIOR ---
        ax2.set_facecolor(Plotter.FACECOLOR)
        ax2.set_axisbelow(True)
        ax2.set_xlabel('Cantidad de Letras en el Dataset (N)', color=Plotter.TEXT_COLOR, fontsize=11)
        ax2.set_ylabel('Loss Promedio (BCE)', color=color_loss, fontsize=11)
        
        ax2.plot(sizes, loss_media, marker='s', markersize=6, color=color_loss, linewidth=1.5, label="Loss Promedio")
        ax2.fill_between(sizes, loss_media - loss_std, loss_media + loss_std, color=color_loss, alpha=0.10, edgecolor='none')
        
        ax2.legend(loc='upper left', frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR)
        ax2.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)

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
    def plot_optimizer_comparison(resultados_loss: dict, filename: str):
        """
        Traza la convergencia con eje X dinámico para comparar algoritmos de optimización.
        El largo de cada serie se determina por la época en que la ejecución satisfizo la condición de corte.
        """
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
        
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Gráfico guardado en: {filepath}")
        
        
    @staticmethod
    def plot_subset_orthogonality_comparison(resultados_loss: dict, filename: str):
        """
        Grafica la convergencia de dos subsets distintos para demostrar 
        el impacto de la ortogonalidad de los datos en el Descenso de Gradiente.
        """
        print("\nGenerando gráfico de Calidad de Datos (Ortogonalidad)...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)
        ax.set_axisbelow(True)

        # Rojo para el difícil (Similar), Azul Marino para el fácil (Ortogonal)
        colores = [Plotter.SERIES_COLORS[1], Plotter.SERIES_COLORS[0]] 

        for (nombre, historiales), color in zip(resultados_loss.items(), colores):
            matriz_historiales = np.array(historiales)
            loss_promedio = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            
            epocas_x = np.arange(len(loss_promedio))

            ax.plot(epocas_x, loss_promedio, label=f"{nombre}", color=color, linewidth=1.5)
            ax.fill_between(
                epocas_x, 
                loss_promedio - loss_std, 
                loss_promedio + loss_std, 
                color=color, 
                alpha=0.10, 
                edgecolor='none'
            )

        ax.set_title("Impacto de la Ortogonalidad del Dataset en SGD", color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR, fontsize=11)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))

        ax.legend(frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR)
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
        
    @staticmethod
    def plot_similarity_matrix(X: np.ndarray, labels: list, filename: str):
        """
        Calcula y grafica la matriz de Similitud Coseno (Heatmap) entre todos los caracteres.
        1.0 (Color oscuro) = Caracteres idénticos.
        0.0 (Color blanco) = Caracteres ortogonales (sin píxeles compartidos).
        """
        print("\nGenerando Mapa de Calor de Ortogonalidad...")
        
        # 1. Cálculo matemático de la similitud coseno
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.where(norms == 0, 1e-10, norms)
        sim_matrix = np.dot(X_norm, X_norm.T)

        # 2. Configuración del lienzo
        fig, ax = plt.subplots(figsize=(11, 9))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)

        # 3. Dibujamos el Heatmap (Usamos 'Blues' para estilo académico)
        cax = ax.imshow(sim_matrix, cmap='Blues', vmin=0, vmax=1)

        # 4. Formateo de los ejes para mostrar las letras
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, color=Plotter.TEXT_COLOR)
        ax.set_yticklabels(labels, fontsize=9, color=Plotter.TEXT_COLOR)

        # Ocultamos las marcas de los ticks (las rayitas) pero dejamos las letras
        ax.tick_params(axis='both', which='both', length=0)

        # 5. Barra de colores lateral (Leyenda)
        cbar = fig.colorbar(cax, ax=ax, shrink=0.82, pad=0.04)
        cbar.ax.tick_params(labelsize=10, colors=Plotter.TEXT_COLOR)
        cbar.set_label('Similitud Coseno (0 = Ortogonal, 1 = Idéntico)', color=Plotter.TEXT_COLOR, fontsize=11, labelpad=15)

        ax.set_title("Matriz de Ortogonalidad y Similitud entre Caracteres", color=Plotter.TITLE_COLOR, pad=20, fontsize=14, fontweight='bold')

        # Limpiamos todos los bordes
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        
        import os
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Matriz guardada en: {filepath}")
        
    @staticmethod
    def plot_subset_orthogonality_heatmaps(X_sim: np.ndarray, labels_sim: list, X_ortho: np.ndarray, labels_ortho: list, filename: str):
        """
        Calcula y grafica dos matrices de similitud de 5x5 lado a lado.
        Permite visualizar numéricamente por qué un subset es más difícil que el otro.
        """
        print("\nGenerando Mapas de Calor para los Subsets...")
        
        # Función auxiliar para la matemática
        def calcular_similitud(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = X / np.where(norms == 0, 1e-10, norms)
            return np.dot(X_norm, X_norm.T)

        sim_matrix_sim = calcular_similitud(X_sim)
        sim_matrix_ortho = calcular_similitud(X_ortho)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.patch.set_facecolor(Plotter.FACECOLOR)

        # --- PANEL IZQUIERDO: Subset Similar (Rojos) ---
        axes[0].set_facecolor(Plotter.FACECOLOR)
        axes[0].imshow(sim_matrix_sim, cmap='Reds', vmin=0, vmax=1)
        axes[0].set_title("Subset Similar (Difícil)", color=Plotter.TITLE_COLOR, pad=15, fontsize=12, fontweight='bold')
        axes[0].set_xticks(np.arange(len(labels_sim)))
        axes[0].set_yticks(np.arange(len(labels_sim)))
        axes[0].set_xticklabels(labels_sim, fontsize=12, color=Plotter.TEXT_COLOR)
        axes[0].set_yticklabels(labels_sim, fontsize=12, color=Plotter.TEXT_COLOR)

        # --- PANEL DERECHO: Subset Ortogonal (Azules) ---
        axes[1].set_facecolor(Plotter.FACECOLOR)
        axes[1].imshow(sim_matrix_ortho, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title("Subset Ortogonal (Fácil)", color=Plotter.TITLE_COLOR, pad=15, fontsize=12, fontweight='bold')
        axes[1].set_xticks(np.arange(len(labels_ortho)))
        axes[1].set_yticks(np.arange(len(labels_ortho)))
        axes[1].set_xticklabels(labels_ortho, fontsize=12, color=Plotter.TEXT_COLOR)
        axes[1].set_yticklabels(labels_ortho, fontsize=12, color=Plotter.TEXT_COLOR)

        # Imprimimos los números adentro de los cuadrados para ambos paneles
        for i in range(len(labels_sim)):
            for j in range(len(labels_sim)):
                # Para la matriz similar (Rojos)
                color_texto_sim = "white" if sim_matrix_sim[i, j] > 0.55 else Plotter.TEXT_COLOR
                axes[0].text(j, i, f"{sim_matrix_sim[i, j]:.2f}", ha="center", va="center", color=color_texto_sim, fontsize=10)
                
                # Para la matriz ortogonal (Azules)
                color_texto_ortho = "white" if sim_matrix_ortho[i, j] > 0.55 else Plotter.TEXT_COLOR
                axes[1].text(j, i, f"{sim_matrix_ortho[i, j]:.2f}", ha="center", va="center", color=color_texto_ortho, fontsize=10)

        # Limpieza visual de ambos ejes
        for ax in axes:
            ax.tick_params(axis='both', which='both', length=0) # Saca las rayitas
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout()
        
        import os
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Matrices guardadas en: {filepath}")