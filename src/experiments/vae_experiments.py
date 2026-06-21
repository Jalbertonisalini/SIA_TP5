import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer, VAEBottleneckLayer
from core.network import Network
from core.optimizers import SGD, Adam
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

def smooth_curve(data, box_pts=30):
    box = np.ones(box_pts)/box_pts
    return np.convolve(data, box, mode='valid')

def build_parametrized_vae(latent_dim: int, kl_weight: float, hidden_activation_class) -> Network:
    vae = Network()
    vae.add(Linear(input_size=256, output_size=128))
    vae.add(ActivationLayer(hidden_activation_class()))
    vae.add(Linear(input_size=128, output_size=64))
    vae.add(ActivationLayer(hidden_activation_class()))
    vae.add(Linear(input_size=64, output_size=latent_dim * 2))
    vae.add(VAEBottleneckLayer(latent_dim=latent_dim, kl_weight=kl_weight))
    
    vae.add(Linear(input_size=latent_dim, output_size=64))
    vae.add(ActivationLayer(hidden_activation_class()))
    vae.add(Linear(input_size=64, output_size=128))
    vae.add(ActivationLayer(hidden_activation_class()))
    vae.add(Linear(input_size=128, output_size=256))
    vae.add(ActivationLayer(Sigmoid()))
    return vae

def train_parametrized_network(X: np.ndarray, latent_dim: int, kl_weight: float, 
                               hidden_activation_class=Tanh, optimizer_class=Adam, 
                               lr: float = 0.001, epochs: int = 10000):
    vae = build_parametrized_vae(latent_dim, kl_weight, hidden_activation_class)
    loss_function = BCE()
    optimizer = optimizer_class(learning_rate=lr)
    vae_layer = vae.layers[5]
    
    history = []
    for epoch in range(epochs):
        predicted = vae.forward(X)
        rec_loss = loss_function.calculate(expected=X, predicted=predicted)
        kl_loss = vae_layer.kl_loss * vae_layer.kl_weight
        history.append(rec_loss + kl_loss)
        
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        vae.backward(initial_gradient, optimizer)
    return vae, history

def get_mu_coordinates(vae: Network, X: np.ndarray, latent_dim: int) -> np.ndarray:
    activation = X
    for layer in vae.layers[:5]:
        activation = layer.forward(activation)
    return activation[:, :latent_dim]

def decode_latent_z(vae: Network, z: np.ndarray) -> np.ndarray:
    activation = z
    for layer in vae.layers[6:]:
        activation = layer.forward(activation)
    return activation

print("Ejecutando pipeline de experimentos...")
loader = EmojiLoader()
X, labels = loader.get_all_data()

# --- Experimento 1: Optimizadores ---
print("Ejecutando Experimento 1...")
_, hist_sgd = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, optimizer_class=SGD, lr=0.005, epochs=4000)
_, hist_adam_raw = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, optimizer_class=Adam, lr=0.001, epochs=4000)

plt.figure(figsize=(11, 5.5))
plt.plot(hist_sgd, color='orangered', label='SGD (lr=0.005)', alpha=0.6)
plt.plot(smooth_curve(hist_adam_raw, box_pts=25), color='dodgerblue', linewidth=2.5, label='Adam (lr=0.001) - Suavizado')
plt.title('Comparacion de optimizadores', pad=35)
plt.xlabel('Epocas')
plt.ylabel('Loss (BCE + KL)')
plt.yscale('log')
plt.legend(title='Configuracion: Input=256D, Ocultas=Tanh, Latente=2D, beta=0.003', title_fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step1_smooth_optimizers.png'))
plt.close()

# --- Experimento 1.b: Learning Rates ---
print("Ejecutando Experimento 1.b...")
lrs = [0.01, 0.001, 0.0001]
plt.figure(figsize=(11, 5.5))
colors_lr = ['crimson', 'dodgerblue', 'forestgreen']

for idx, lr_val in enumerate(lrs):
    _, hist_lr = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, lr=lr_val, epochs=4000)
    plt.plot(smooth_curve(hist_lr, box_pts=25), color=colors_lr[idx], linewidth=2, label=f"lr={lr_val}")

plt.title('Sensibilidad de convergencia según Learning Rate', pad=35)
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(title='Configuracion: Optimizador=Adam, Ocultas=Tanh, Latente=2D, beta=0.003', title_fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step1b_learning_rates.png'))
plt.close()

# --- Experimento 1.c: Activaciones ---
print("Ejecutando Experimento 1.c...")
plt.figure(figsize=(11, 5.5))
_, hist_tanh = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, hidden_activation_class=Tanh, lr=0.001, epochs=5000)
_, hist_sig = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, hidden_activation_class=Sigmoid, lr=0.001, epochs=5000)

plt.plot(smooth_curve(hist_tanh, box_pts=25), color='#2b5c8f', linewidth=2, label='Tanh()')
plt.plot(smooth_curve(hist_sig, box_pts=25), color='#d97d24', linewidth=2, label='Sigmoid()')
plt.title('Evaluación de funciones de activacion ocultas', pad=35)
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(title='Configuracion: Optimizador=Adam, lr=0.001, Latente=2D, beta=0.003', title_fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step1c_activation_functions.png'))
plt.close()

# --- Experimento 2: Matriz de Betas ---
print("Ejecutando Experimento 2...")
betas = [0.0, 0.001, 0.003, 0.2]
fig_betas, axes_betas = plt.subplots(2, 2, figsize=(14, 11))

for idx_b, b_val in enumerate(betas):
    ax = axes_betas[idx_b // 2, idx_b % 2]
    vae_b, _ = train_parametrized_network(X, latent_dim=2, kl_weight=b_val, lr=0.001, epochs=10000)
    mu_b = get_mu_coordinates(vae_b, X, latent_dim=2)
    
    ax.scatter(mu_b[:, 0], mu_b[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=100, alpha=0.85)
    for i, label in enumerate(labels):
         ax.annotate(label, (mu_b[i, 0], mu_b[i, 1]), xytext=(4, 4), textcoords='offset points', fontsize=8, weight='bold')
    
    subtitle = f"beta = {b_val}"
    if b_val == 0.0: subtitle += " (Autoencoder)"
    if b_val == 0.003: subtitle 
    if b_val == 0.2: subtitle 
    ax.set_title(subtitle, fontsize=11, weight='bold')
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")

plt.suptitle("Impacto de la divergencia KL en la topología del espacio latente", y=0.98, weight='bold', fontsize=14)
fig_betas.text(0.5, 0.95, "Configuracion fija: Optimizador=Adam | lr=0.001 | Epocas=10000 | Latent_Dim=2D | Activaciones=Tanh()", ha='center', fontsize=10, style='italic')
plt.tight_layout(rect=[0, 0.05, 1, 0.91])
plt.savefig(os.path.join(OUTPUT_DIR, 'step2_beta_comparison_matrix.png'), bbox_inches='tight')
plt.close()

# --- Entrenamiento de Modelos de Producción Finales ---
print("Entrenando configuraciones de control final (14000 epocas)...")
vae_2d_final, _ = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, lr=0.001, epochs=14000)
vae_3d_final, _ = train_parametrized_network(X, latent_dim=3, kl_weight=0.003, lr=0.001, epochs=14000)

# --- Experimento 3: Espacio Latente 2D Final ---
print("Ejecutando Experimento 3...")
mu_2d_f = get_mu_coordinates(vae_2d_final, X, latent_dim=2)
plt.figure(figsize=(10, 7.5))
plt.scatter(mu_2d_f[:, 0], mu_2d_f[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=140, alpha=0.9)
for i, label in enumerate(labels):
    plt.annotate(label, (mu_2d_f[i, 0], mu_2d_f[i, 1]), xytext=(6, 6), textcoords='offset points', fontsize=9, weight='bold')
plt.title('Representacion de los datos de entrada en el espacio latente (2D)', pad=35)
plt.xlabel('Z1')
plt.ylabel('Z2')
plt.legend(title='Hiperparametros: Adam, lr=0.001, Epocas=14000, beta=0.003, Ocultas=Tanh', title_fontsize=9, loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_final_latent_space_2d.png'))
plt.close()

# --- Experimento 4: Matriz de Reconstrucción Total (2D) ---
print("Ejecutando Experimento 4 (Fidelidad 2D)...")
predicted_2d = vae_2d_final.forward(X)
fig_rec, axes_rec = plt.subplots(4, 8, figsize=(15, 8.5))

for i in range(16):
    row_orig, row_rec, col = (0, 1, i) if i < 8 else (2, 3, i - 8)
    axes_rec[row_orig, col].imshow(X[i].reshape(16, 16), cmap='bone_r')
    axes_rec[row_orig, col].set_xticks([]); axes_rec[row_orig, col].set_yticks([])
    axes_rec[row_orig, col].set_title(labels[i], fontsize=9.5, weight='bold')
    if col == 0: axes_rec[row_orig, col].set_ylabel("Original", weight='bold', fontsize=11)
    
    axes_rec[row_rec, col].imshow(predicted_2d[i].reshape(16, 16), cmap='bone_r')
    axes_rec[row_rec, col].set_xticks([]); axes_rec[row_rec, col].set_yticks([])
    if col == 0: axes_rec[row_rec, col].set_ylabel("Reconstruido", weight='bold', fontsize=11, color='dodgerblue')

plt.suptitle("Fidelidad de reconstruccion completa del dataset (Espacio 2D)", y=0.98, weight='bold', fontsize=15)
fig_rec.text(0.5, 0.95, "Hiperparametros: Adam | lr=0.001 | Epocas=14000 | Latent_Dim=2D | beta=0.003 | Arquitectura=256-128-64-2-64-128-256", ha='center', fontsize=10, style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, 'step4_full_reconstruction_matrix.png'), bbox_inches='tight')
plt.close()

# --- Experimento 4b: Matriz de Reconstrucción Total (3D) ---
print("Ejecutando Experimento 4b (Fidelidad 3D)...")
predicted_3d = vae_3d_final.forward(X)
fig_rec_3d, axes_rec_3d = plt.subplots(4, 8, figsize=(15, 8.5))

for i in range(16):
    row_orig, row_rec, col = (0, 1, i) if i < 8 else (2, 3, i - 8)
    axes_rec_3d[row_orig, col].imshow(X[i].reshape(16, 16), cmap='bone_r')
    axes_rec_3d[row_orig, col].set_xticks([]); axes_rec_3d[row_orig, col].set_yticks([])
    axes_rec_3d[row_orig, col].set_title(labels[i], fontsize=9.5, weight='bold')
    if col == 0: axes_rec_3d[row_orig, col].set_ylabel("Original", weight='bold', fontsize=11)
    
    axes_rec_3d[row_rec, col].imshow(predicted_3d[i].reshape(16, 16), cmap='bone_r')
    axes_rec_3d[row_rec, col].set_xticks([]); axes_rec_3d[row_rec, col].set_yticks([])
    if col == 0: axes_rec_3d[row_rec, col].set_ylabel("Reconstruido", weight='bold', fontsize=11, color='forestgreen')

plt.suptitle("Fidelidad de reconstruccion completa del dataset (Espacio 3D)", y=0.98, weight='bold', fontsize=15)
fig_rec_3d.text(0.5, 0.95, "Hiperparametros: Adam | lr=0.001 | Epocas=14000 | Latent_Dim=3D | beta=0.003 | Arquitectura=256-128-64-3-64-128-256", ha='center', fontsize=10, style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, 'step4b_full_reconstruction_matrix_3d.png'), bbox_inches='tight')
plt.close()

# --- Experimento 5: Espacio Latente 3D Final ---
print("Ejecutando Experimento 5...")
mu_3d_f = get_mu_coordinates(vae_3d_final, X, latent_dim=3)
fig_3d = plt.figure(figsize=(11, 8.5))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.patch.set_facecolor('#f5f5f5')

ax_3d.scatter(mu_3d_f[:, 0], mu_3d_f[:, 1], mu_3d_f[:, 2], c=np.arange(16), cmap='tab20', edgecolors='k', s=130, alpha=0.9)
for i, label in enumerate(labels):
    ax_3d.text(mu_3d_f[i, 0], mu_3d_f[i, 1], mu_3d_f[i, 2], label, fontsize=9, weight='bold')

ax_3d.set_title('Representacion de los datos de entrada en el espacio latente (3D)', pad=35)
fig_3d.text(0.5, 0.93, 'Hiperparametros: Adam | lr=0.001 | Epocas=14000 | Latent_Dim=3D | beta=0.003 | Ocultas=Tanh', ha='center', fontsize=10, style='italic')
ax_3d.set_xlabel('Z1'); ax_3d.set_ylabel('Z2'); ax_3d.set_zlabel('Z3')
plt.savefig(os.path.join(OUTPUT_DIR, 'step5_latent_space_3d.png'), bbox_inches='tight')
plt.close()

# --- Experimento 6: Comparación Cruzada (Corazón) ---
print("Ejecutando Experimento 6...")
h_idx = labels.index('corazon')
br_idx = labels.index('corazon_roto')

fig_check, axes_c = plt.subplots(2, 3, figsize=(10, 6.5))
axes_c[0, 0].imshow(X[h_idx].reshape(16, 16), cmap='bone_r'); axes_c[0, 0].set_title("Original", weight='bold')
axes_c[0, 1].imshow(predicted_2d[h_idx].reshape(16, 16), cmap='bone_r'); axes_c[0, 1].set_title("Reconstruccion 2D", color='crimson', weight='bold')
axes_c[0, 2].imshow(predicted_3d[h_idx].reshape(16, 16), cmap='bone_r'); axes_c[0, 2].set_title("Reconstruccion 3D", color='forestgreen', weight='bold')

axes_c[1, 0].imshow(X[br_idx].reshape(16, 16), cmap='bone_r')
axes_c[1, 1].imshow(predicted_2d[br_idx].reshape(16, 16), cmap='bone_r')
axes_c[1, 2].imshow(predicted_3d[br_idx].reshape(16, 16), cmap='bone_r')

for r in range(2):
    for c in range(3):
        axes_c[r, c].set_xticks([]); axes_c[r, c].set_yticks([])
axes_c[0, 0].set_ylabel("Corazon liso", weight='bold', fontsize=11)
axes_c[1, 0].set_ylabel("Corazon roto", weight='bold', fontsize=11)

plt.suptitle("Analisis de capacidad resolutiva del espacio latente (2D vs. 3D)", weight='bold', y=0.99, fontsize=13)
fig_check.text(0.5, 0.02, "Parámetros: Adam | lr=0.001 | Epocas=14000 | beta=0.003", ha='center', fontsize=9.5, style='italic')
plt.tight_layout(rect=[0, 0.05, 1, 0.91])
plt.savefig(os.path.join(OUTPUT_DIR, 'step6_structural_resolution.png'))
plt.close()

# --- Experimento 7: Interpolaciones Lineales de Pares (2D) ---
print("Ejecutando Experimento 7 (Interpolaciones)...")

def generate_interpolation_plot(vae, mu_data, labels_list, start_label, end_label, save_filename):
    z_start = mu_data[labels_list.index(start_label)]
    z_end = mu_data[labels_list.index(end_label)]
    
    num_steps = 7
    steps = np.linspace(0, 1, num_steps)
    interpolated_z = np.array([z_start + (z_end - z_start) * t for t in steps])
    
    decoded_images = decode_latent_z(vae, interpolated_z)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(14, 3))
    for idx, img_flat in enumerate(decoded_images):
        img = img_flat.reshape(16, 16)
        axes[idx].imshow(img, cmap='bone_r', vmin=0, vmax=1)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_title(f"t = {steps[idx]:.2f}", fontsize=9)
        
    axes[0].set_ylabel(start_label, weight='bold', fontsize=11, rotation=0, labelpad=25, va='center')
    axes[-1].set_title(end_label, weight='bold', fontsize=11)
    
    plt.suptitle(f"Metamorfosis de transicion: {start_label} -> {end_label}", y=1.05, weight='bold', fontsize=13)
    fig.text(0.5, 0.02, "Configuracion: Adam | lr=0.001 | Epocas=14000 | Latent_Dim=2D | beta=0.003", ha='center', fontsize=9, style='italic')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])
    plt.savefig(os.path.join(OUTPUT_DIR, save_filename), bbox_inches='tight')
    plt.close()

generate_interpolation_plot(vae_2d_final, mu_2d_f, labels, 'rayo', 'gotas', 'step7a_interpolation_rayo_gotas.png')
generate_interpolation_plot(vae_2d_final, mu_2d_f, labels, 'sorpresa', 'sonrisa', 'step7b_interpolation_sorpresa_sonrisa.png')

print("Pipeline completado.")