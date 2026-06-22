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
        Inyecta coordenadas en el espacio latente y las pasa únicamente 
        por el Decoder para generar una imagen nueva. Soporta redes profundas.
        """
        # Formateamos la entrada a (1, 2)
        output = np.array(z_coord).reshape(1, -1)
        
        # ARREGLO: Calculamos la mitad exacta de la red dinámicamente
        mitad_red = len(autoencoder.layers) // 2
        
        # Pasamos la coordenada SOLO por las capas del Decoder
        for layer in autoencoder.layers[mitad_red:]:
            output = layer.forward(output)
            
        # Damos forma a la matriz final de 35 píxeles a 7x5
        img_reconstructed = output.reshape((7, 5))
        
        # Renderizado
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        
        ax.imshow(img_reconstructed, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f"Latente: {z_coord}", color=Plotter.TITLE_COLOR, pad=10)
        ax.axis('off')
        
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"      [+] Letra alucinada guardada en: {filepath}")
    
    
    @staticmethod
    def compare_reconstruction(autoencoder: Network, original_x: np.ndarray, label: str, model_name: str, filename: str):
        """
        Pasa una letra original completa por la red y grafica la entrada real frente a la reconstruida.
        Incluye el nombre del modelo para que la imagen sea autodescriptiva en presentaciones.
        """
        if original_x.ndim == 1:
            original_x = original_x.reshape(1, -1)
            
        reconstructed_x = autoencoder.forward(original_x)
        
        img_original = original_x.reshape((7, 5))
        img_reconstructed = reconstructed_x.reshape((7, 5))
        
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.5))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        
        # Título principal con el nombre de la arquitectura
        fig.suptitle(f"Arquitectura: {model_name}", color=Plotter.TITLE_COLOR, fontsize=14, fontweight='bold', y=0.98)
        
        axes[0].imshow(img_original, cmap='gray_r', vmin=0, vmax=1)
        axes[0].set_title(f"Original ('{label}')", color=Plotter.TITLE_COLOR, fontsize=12, pad=10)
        axes[0].axis('off')
        
        axes[1].imshow(img_reconstructed, cmap='gray_r', vmin=0, vmax=1)
        axes[1].set_title(f"Reconstrucción", color=Plotter.TITLE_COLOR, fontsize=12, pad=10)
        axes[1].axis('off')
        
        fig.tight_layout()
        # Bajamos un poquito los gráficos para que no se choquen con el título principal
        fig.subplots_adjust(top=0.82)
        
        os.makedirs('outputs', exist_ok=True)
        fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Comparación guardada en: outputs/{filename}")

    @staticmethod
    def plot_architecture_comparison(resultados_loss: dict, resultados_errores: dict, max_epochs: int, filename: str):
        print("\nGenerando gráfico estadístico de Arquitecturas...")
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)
        ax.set_axisbelow(True)

        for (nombre, historiales), color in zip(resultados_loss.items(), Plotter.SERIES_COLORS):
            matriz_historiales = np.array(historiales)
            
            loss_promedio = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            epocas_x = np.arange(max_epochs)

            bce_final_promedio = np.mean(matriz_historiales[:, -1])
            err_px_promedio = np.mean(resultados_errores[nombre])
            err_px_std = np.std(resultados_errores[nombre])

            # Etiqueta enriquecida idéntica a la Fase 3
            etiqueta = (f"{nombre}\n"
                        f"BCE: {bce_final_promedio:.4f} | Err Px: {err_px_promedio:.1f} ± {err_px_std:.1f}")

            ax.plot(epocas_x, loss_promedio, label=etiqueta, color=color, linewidth=1.5)
            ax.fill_between(
                epocas_x, loss_promedio - loss_std, loss_promedio + loss_std,
                color=color, alpha=0.10, edgecolor='none'
            )

        ax.set_title("Convergencia Multi-Seed: Comparativa de Arquitecturas", color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax.set_xlabel("Épocas", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Loss Promedio (BCE)", color=Plotter.TEXT_COLOR, fontsize=11)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))
        
        ax.legend(frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, labelcolor=Plotter.TEXT_COLOR, fontsize=9)
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
    def plot_optimizer_comparison(resultados_loss: dict, resultados_errores: dict, filename: str):
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
            
            # --- CÁLCULO DEL DESVÍO EN EL TIEMPO (Tus cálculos estaban perfectos) ---
            loss_promedio = np.mean(matriz_historiales, axis=0)
            loss_std = np.std(matriz_historiales, axis=0)
            epocas_x = np.arange(len(loss_promedio))
            
            # --- CÁLCULO DE MÉTRICAS FINALES PARA LA LEYENDA ---
            # Agarramos solo la última época ([-1]) de todas las semillas para el promedio final
            bce_final_promedio = np.mean(matriz_historiales[:, -1])
            bce_final_std = np.std(matriz_historiales[:, -1])
            
            err_px_promedio = np.mean(resultados_errores[nombre])
            err_px_std = np.std(resultados_errores[nombre])
            
            # Armamos una etiqueta limpia que muestra el nombre y debajo los promedios
            etiqueta = (f"{nombre}\n"
                        f"BCE Final: {bce_final_promedio:.4f} ± {bce_final_std:.4f}\n"
                        f"Err Píxeles: {err_px_promedio:.1f} ± {err_px_std:.1f}")

            # Dibujamos la línea principal
            ax.plot(epocas_x, loss_promedio, label=etiqueta, color=color, linewidth=1.5)
            
            # Dibujamos el área de desvío (sombra)
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

        # Ajuste en la leyenda para acomodar el texto enriquecido
        ax.legend(frameon=True, facecolor=Plotter.FACECOLOR, edgecolor=Plotter.GRID_COLOR, 
                  labelcolor=Plotter.TEXT_COLOR, loc='upper right', fontsize=9)
        
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
    def plot_single_loss_curve(loss_history: list, title: str, filename: str):
        """
        Generates a single BCE loss curve to visualize asymptotic stagnation.
        
        Args:
            loss_history (list): Array of loss values per epoch.
            title (str): Title of the plot.
            filename (str): Output filename.
        """
        print(f"\nGenerating Loss curve for {title}...")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        ax.set_facecolor(Plotter.FACECOLOR)
        ax.set_axisbelow(True)

        epochs_x = np.arange(len(loss_history))
        # Usamos el color principal (Azul Marino) de tu paleta
        ax.plot(epochs_x, loss_history, color=Plotter.SERIES_COLORS[0], linewidth=2.0)

        ax.set_title(title, color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax.set_xlabel("Epochs", color=Plotter.TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Loss (BCE)", color=Plotter.TEXT_COLOR, fontsize=11)
        
        # Formateo del eje X para miles (k)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))
        ax.grid(True, linestyle=':', alpha=0.8, color=Plotter.GRID_COLOR)
        
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']: ax.spines[spine].set_color(Plotter.GRID_COLOR)
        ax.tick_params(colors=Plotter.TEXT_COLOR)
        
        fig.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Curva guardada en: {filepath}")

    @staticmethod
    def plot_pixel_errors(original_x: np.ndarray, predicted_x: np.ndarray, label: str, filename: str):
        """
        Creates a side-by-side comparison highlighting thresholded pixel errors.
        Displays the Original, the Continuous Prediction, and a Red Error Heatmap.
        
        Args:
            original_x (np.ndarray): The ground truth letter array.
            predicted_x (np.ndarray): The raw continuous output from the network.
            label (str): The character label.
            filename (str): Output filename.
        """
        if original_x.ndim == 1: original_x = original_x.reshape(1, -1)
        if predicted_x.ndim == 1: predicted_x = predicted_x.reshape(1, -1)
        
        img_orig = original_x.reshape((7, 5))
        img_pred = predicted_x.reshape((7, 5))
        
        # Umbralizamos para ver qué decisión binaria final tomó la red
        img_pred_bin = (img_pred >= 0.5).astype(int)
        
        # Matriz de errores: 1 donde difiere de la original, 0 donde acierta
        img_errores = np.abs(img_orig - img_pred_bin)
        total_errors = np.sum(img_errores)
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 4))
        fig.patch.set_facecolor(Plotter.FACECOLOR)
        
        # Panel 1: Original
        axes[0].imshow(img_orig, cmap='gray_r', vmin=0, vmax=1)
        axes[0].set_title(f"Original: '{label}'", color=Plotter.TITLE_COLOR)
        
        # Panel 2: Predicción Continua (Para ver la inseguridad/grises)
        axes[1].imshow(img_pred, cmap='gray_r', vmin=0, vmax=1) 
        axes[1].set_title("Reconstructed", color=Plotter.TITLE_COLOR)
        
        # Panel 3: Mapa de Errores (Rojo puro donde se equivocó)
        # Usamos cmap='Reds' para que el 1 resalte fuerte
        axes[2].imshow(img_errores, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f"Errors: {int(total_errors)} pixels", color=Plotter.TITLE_COLOR, fontweight='bold')
        
        for ax in axes: ax.axis('off')
        
        fig.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Mapa de errores guardado en: {filepath}")
        
    @staticmethod
    def plot_dataset_grid(X: np.ndarray, labels: list, filename: str = "dataset_grid.png"):
        """
        Genera una plancha visual con todos los caracteres del dataset original.
        Dibuja una grilla de 4 filas x 8 columnas.
        
        Args:
            X (np.ndarray): Matriz del dataset completo (32 x 35).
            labels (list): Lista con las etiquetas de los caracteres.
            filename (str): Nombre del archivo de salida.
        """
        print("\nGenerating Dataset Grid...")
        # Creamos una grilla de 4x8. El tamaño (12, 7) da una buena proporción para PowerPoint
        fig, axes = plt.subplots(4, 8, figsize=(12, 7))
        
        # Opcional: Si querés usar el fondo oscuro de tu paleta, descomentá estas líneas
        # fig.patch.set_facecolor(Plotter.FACECOLOR)
        
        for i, ax in enumerate(axes.flat):
            if i < len(X):
                # Volvemos a armar la matriz 2D de 7x5 para la visualización
                img = X[i].reshape((7, 5))
                ax.imshow(img, cmap='gray_r', vmin=0, vmax=1)
                
                # Opcional: Color del título para que contraste si usás fondo oscuro
                # ax.set_title(f"'{labels[i]}'", color=Plotter.TITLE_COLOR, fontsize=14)
                ax.set_title(f"'{labels[i]}'", fontsize=14, fontweight='bold')
            
            # Ocultamos los ejes para que quede como una plancha limpia
            ax.axis('off')
            
        fig.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Plancha del dataset guardada en: {filepath}")
        
    @staticmethod
    def plot_capacity_limit_with_error(sizes: list, errors_mean: list, errors_std: list, similarities: list, filename: str):
        print("\nGenerating Capacity Limit Plot with Orthogonality Metric...")
        fig, ax1 = plt.subplots(figsize=(9, 5))
        
        # Eje Y izquierdo: Barras de Errores (promedio + desviación estándar)
        # Usamos errors_mean para la altura y errors_std para el "yerr" (el bigote)
        color_bar = Plotter.SERIES_COLORS[0]
        bars = ax1.bar(sizes, errors_mean, yerr=errors_std, capsize=5, 
                       color=color_bar, width=2.0, alpha=0.8, label="Error (Promedio ± Desvío)")
        
        ax1.axhline(y=1, color='red', linestyle='--', linewidth=2, label="Tolerancia Máxima (1 píxel)")
        
        ax1.set_xlabel("Tamaño del Subset (N Letras)", color=Plotter.TEXT_COLOR, fontweight='bold')
        ax1.set_ylabel("Max Píxeles de Error", color=color_bar, fontweight='bold')
        ax1.set_xticks(sizes)
        ax1.tick_params(axis='y', labelcolor=color_bar)
        
        # Anotaciones en las barras (usando el promedio)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}', # Usamos .1f porque el promedio puede tener decimales
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', color=color_bar, fontweight='bold')

        # Eje Y derecho: Curva de Similitud (Naranja)
        ax2 = ax1.twinx()
        color_line = '#E67E22' 
        ax2.plot(sizes, similarities, color=color_line, marker='o', linewidth=2.5, markersize=8, label="Similitud Máxima (Peor Par)")
        ax2.set_ylabel("Similitud (ortogonalidad)", color=color_line, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_line)
        ax2.set_ylim(0, 1.1) # Un poquito más de margen para que no toque el borde

        # Título y grilla
        ax1.set_title("Límite de Capacidad SGD y Colapso por Similitud", color=Plotter.TITLE_COLOR, pad=15, fontsize=13)
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # --- CORRECCIÓN AQUÍ ---
        # Unificamos leyendas capturando los handles y labels de ambos ejes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        # Ahora unimos los de la izquierda (ax1) con los de la derecha (ax2)
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        # -----------------------
        
        fig.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        filepath = f'outputs/{filename}'
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"  [+] Gráfico de capacidad guardado en: {filepath}")

    @staticmethod
    def plot_formula_plate(filename="formula_plate.png"):
        """Genera una imagen profesional con las fórmulas justificativas."""
        fig = plt.figure(figsize=(9, 4))
        fig.patch.set_facecolor('#F8F9F9')
        ax = plt.gca()
        ax.axis('off')
        
        # Separamos el texto descriptivo (negrita normal) de la fórmula (LaTeX puro)
        # Esto evita que el parser de Matplotlib se vuelva loco con los comandos de estilo
        items = [
            ("1. Producto Punto (Hopfield):", r"$x \cdot y = \sum_{i=1}^{35} x_i y_i$"),
            ("2. Norma L2 (Magnitud):", r"$\|x\| = \sqrt{\sum_{i=1}^{35} x_i^2}$"),
            ("3. Similitud Coseno:", r"$\text{Sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|} = \cos(\theta)$")
        ]
        
        y_pos = 0.8
        for label, formula in items:
            # Dibujamos el texto plano en negrita
            plt.text(0.05, y_pos, label, fontsize=14, color='#2C3E50', fontweight='bold')
            # Dibujamos la fórmula en la línea de abajo o al lado
            plt.text(0.40, y_pos, formula, fontsize=16, color='#1A5276')
            y_pos -= 0.25
            
        plt.text(0.05, 0.1, 
                 "Nota: La normalización elimina el sesgo por densidad de píxeles.", 
                 fontsize=11, color='#7F8C8D', fontstyle='italic')
        
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f"outputs/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [+] Placa de fórmulas guardada en: outputs/{filename}")