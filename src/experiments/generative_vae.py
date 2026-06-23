import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
from matplotlib.patches import Ellipse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer, VAEBottleneckLayer
from core.network import Network
from core.optimizers import Adam
from core.losses import BCE
from utils.emoji_loader import EmojiLoader

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.facecolor': '#f5f5f5',
    'axes.facecolor': 'white',
    'grid.color': '#e5e5e5',
    'axes.edgecolor': '#333333',
    'axes.titleweight': 'bold',
    'font.family': 'sans-serif'
})

OUTPUT_DIR = 'outputs/final_scientific_pipeline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_vae_model(latent_dim: int, kl_weight: float) -> Network:
    vae = Network()
    vae.add(Linear(input_size=256, output_size=128))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(input_size=128, output_size=64))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(input_size=64, output_size=latent_dim * 2))
    vae.add(VAEBottleneckLayer(latent_dim=latent_dim, kl_weight=kl_weight))
    
    vae.add(Linear(input_size=latent_dim, output_size=64))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(input_size=64, output_size=128))
    vae.add(ActivationLayer(Tanh()))
    vae.add(Linear(input_size=128, output_size=256))
    vae.add(ActivationLayer(Sigmoid()))
    return vae

def train_vae_model_with_early_stopping(X: np.ndarray, latent_dim: int, kl_weight: float, 
                                       max_epochs: int = 25000, patience: int = 1000):
    vae = build_vae_model(latent_dim, kl_weight)
    loss_function = BCE()
    optimizer = Adam(learning_rate=0.001)
    vae_layer = vae.layers[5]
    
    best_loss = np.inf
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        predicted = vae.forward(X)
        rec_loss = loss_function.calculate(expected=X, predicted=predicted)
        kl_loss = vae_layer.kl_loss * vae_layer.kl_weight
        current_loss = rec_loss + kl_loss
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = copy.deepcopy(vae)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        vae.backward(initial_gradient, optimizer)
        
    return best_model_state

def get_latent_distributions(vae: Network, X: np.ndarray, latent_dim: int):
    activation = X
    for layer in vae.layers[:5]:
        activation = layer.forward(activation)
    mu = activation[:, :latent_dim]
    log_var = activation[:, latent_dim:]
    sigma = np.exp(log_var / 2)
    return mu, sigma

def decode_latent_z(vae: Network, z: np.ndarray) -> np.ndarray:
    activation = z
    for layer in vae.layers[6:]:
        activation = layer.forward(activation)
    return activation

# Estructura global para registrar las trayectorias de interpolación
INTERPOLATION_LINES = []

def generate_smooth_interpolation_plot(vae, mu_data, labels_list, start_label, end_label, save_filename):
    z_start = mu_data[labels_list.index(start_label)]
    z_end = mu_data[labels_list.index(end_label)]
    
    INTERPOLATION_LINES.append({
        'start': z_start, 'end': z_end, 'label': f"{start_label} -> {end_label}"
    })
    
    num_steps = 8
    linear_steps = np.linspace(0, 1, num_steps)
    smooth_steps = 0.5 - 0.5 * np.cos(linear_steps * np.pi)
    
    interpolated_z = np.array([z_start + (z_end - z_start) * t for t in smooth_steps])
    decoded_images = decode_latent_z(vae, interpolated_z)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 3.2))
    for idx, img_flat in enumerate(decoded_images):
        img = img_flat.reshape(16, 16)
        axes[idx].imshow(img, cmap='bone_r', vmin=0, vmax=1)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_title(f"t = {linear_steps[idx]:.2f}", fontsize=9)
        
    axes[0].set_ylabel(start_label, weight='bold', fontsize=11, rotation=0, labelpad=25, va='center')
    axes[-1].set_title(end_label, weight='bold', fontsize=11)
    
    plt.suptitle(f"Metamorfosis de transición: {start_label} -> {end_label}", y=1.05, weight='bold', fontsize=13)
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.94])
    plt.savefig(os.path.join(OUTPUT_DIR, save_filename), bbox_inches='tight')
    plt.close()

print("="*60)
print("INICIANDO EXPERIMENTO DE CAPACIDAD GENERATIVA AVANZADA")
print("="*60)
print("Dataset de Entrada: Emojis binarizados (256D)")
print("Arquitectura Encoder-Decoder: Secuencial Simétrica 256-128-64-2")
print("Activaciones Ocultas: Tanh() | Activación de Reconstrucción: Sigmoid()")
print("Optimizador de Producción: Adam (Learning Rate = 0.001)")
print("Peso de regularización (Divergencia KL): beta = 0.003")
print("Espacio Latente Objetivo: Continuo Multivariado (Latent_Dim = 2D)")
print("Muestreo del Espacio Latente Puro: Variable aleatoria estocástica de N(0, I)")
print("Criterio de Parada Dinámico: Early Stopping (Paciencia = 1000 iteraciones)")
print("="*60)

loader = EmojiLoader()
X, labels = loader.get_all_data()

print("\nEntrenando VAE con convergencia dinámica por Early Stopping...")
vae_2d = train_vae_model_with_early_stopping(X, latent_dim=2, kl_weight=0.003)
mu_2d, sigma_2d = get_latent_distributions(vae_2d, X, latent_dim=2)

# --- PARTE 1: Muestreo Masivo Estocástico ---
print("\nGenerando matriz masiva de muestreo aleatorio desde N(0, I)...")
np.random.seed(42)
rows, cols = 4, 8
total_random_samples = rows * cols
z_random = np.random.randn(total_random_samples, 2)

massive_outputs = decode_latent_z(vae_2d, z_random)

fig_mass, axes_mass = plt.subplots(rows, cols, figsize=(16, 9))
for i in range(total_random_samples):
    r, c = i // cols, i % cols
    ax = axes_mass[r, c]
    img = massive_outputs[i].reshape(16, 16)
    
    ax.imshow(img, cmap='bone_r', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Z:[{z_random[i,0]:.1f}, {z_random[i,1]:.1f}]", fontsize=8)

plt.suptitle("Muestreo masivo estocástico en el espacio latente continuo", weight='bold', y=0.97, fontsize=14)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'step8a_massive_generative_grid.png'), bbox_inches='tight')
plt.close()

# --- PARTE 2: Catálogo de Transiciones Suaves Adicionales ---
print("\nGenerando catálogo expandido de transiciones morfológicas...")

transitions_to_run = [
    ('rayo', 'gotas', 'step8b_interp_rayo_gotas.png'),
    ('sorpresa', 'sonrisa', 'step8c_interp_sorpresa_sonrisa.png'),
    ('triste', 'sonrisa', 'step8d_interp_triste_sonrisa.png'),
    ('fuego', 'sol', 'step8e_interp_fuego_sol.png'),
    ('corazon', 'diamante', 'step8f_interp_corazon_diamante.png')
]

for start, end, filename in transitions_to_run:
    if start in labels and end in labels:
        generate_smooth_interpolation_plot(vae_2d, mu_2d, labels, start, end, filename)

# --- PARTE 3: Gráficos del Espacio Latente de Generación (Mapeo de Rutas) ---
print("\nGenerando mapas de análisis latente para el proceso de síntesis...")
cmap_table = plt.get_cmap('tab20')

def plot_generative_space(shadow_mode=False, filename='step8g_generative_latent_map_clean.png'):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if shadow_mode:
        for i in range(16):
            ellipse = Ellipse(xy=(mu_2d[i, 0], mu_2d[i, 1]), width=sigma_2d[i, 0]*2, height=sigma_2d[i, 1]*2,
                               edgecolor=cmap_table(i/16), facecolor=cmap_table(i/16), alpha=0.10, linestyle='--')
            ax.add_patch(ellipse)
            
    for line in INTERPOLATION_LINES:
        ax.plot([line['start'][0], line['end'][0]], [line['start'][1], line['end'][1]], 
                color='black', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
        
    ax.scatter(mu_2d[:, 0], mu_2d[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=130, label='Dataset Real', zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (mu_2d[i, 0], mu_2d[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=8, weight='bold', alpha=0.8)
        
    ax.scatter(z_random[:, 0], z_random[:, 1], color='crimson', marker='x', s=40, linewidths=1.2, alpha=0.7, label='Puntos Muestreados N(0,I)', zorder=2)
    
    title_suffix = "(Nubes 1-sigma)" if shadow_mode else "(Centros de Masa)"
    ax.set_title(f"Mapeo estructural del proceso generativo {title_suffix}", pad=20)
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.legend(loc='lower left', frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

plot_generative_space(shadow_mode=False, filename='step8g_generative_latent_map_clean.png')
plot_generative_space(shadow_mode=True, filename='step8h_generative_latent_map_shadow.png')

print("="*60)
print("PIPELINE GENERATIVO COMPLETADO. VERIFIQUE LA CARPETA DE SALIDAS.")
print("="*60)