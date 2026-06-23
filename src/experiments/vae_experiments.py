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
                               lr: float = 0.001, max_epochs: int = 20000, patience: int = 1000):
    vae = build_parametrized_vae(latent_dim, kl_weight, hidden_activation_class)
    loss_function = BCE()
    optimizer = optimizer_class(learning_rate=lr)
    vae_layer = vae.layers[5]
    
    history = []
    best_loss = np.inf
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        predicted = vae.forward(X)
        rec_loss = loss_function.calculate(expected=X, predicted=predicted)
        kl_loss = vae_layer.kl_loss * vae_layer.kl_weight
        current_loss = rec_loss + kl_loss
        history.append(current_loss)
        
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
        
    return best_model_state, history

def get_multiple_runs_stats(X: np.ndarray, num_runs: int, latent_dim: int, kl_weight: float,
                            hidden_activation_class=Tanh, optimizer_class=Adam, lr: float = 0.001, 
                            max_epochs: int = 20000, patience: int = 1000):
    all_histories = []
    best_overall_model = None
    best_overall_loss = np.inf
    max_length = 0
    
    raw_histories = []
    for r in range(num_runs):
        model, hist = train_parametrized_network(X, latent_dim, kl_weight, hidden_activation_class, optimizer_class, lr, max_epochs, patience)
        raw_histories.append(hist)
        if hist[-1] < best_overall_loss:
            best_overall_loss = hist[-1]
            best_overall_model = model
        if len(hist) > max_length:
            max_length = len(hist)
            
    padded_histories = []
    for hist in raw_histories:
        if len(hist) < max_length:
            last_val = hist[-1]
            extended = hist + [last_val] * (max_length - len(hist))
            padded_histories.append(extended)
        else:
            padded_histories.append(hist)
            
    padded_histories = np.array(padded_histories)
    mean_profile = np.mean(padded_histories, axis=0)
    std_profile = np.std(padded_histories, axis=0)
    return best_overall_model, mean_profile, std_profile

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

print("="*60)
print("INICIANDO CONFIGURACIÓN DE HIPERPARÁMETROS DEL PIPELINE")
print("="*60)
print("Dataset de Entrada: Emojis binarizados (256D)")
print("Arquitectura Encoder:  256 -> 128 -> 64 -> (Latent_Dim * 2)")
print("Arquitectura Decoder:  Latent_Dim -> 64 -> 128 -> 256")
print("Inicialización de Pesos: Xavier/Glorot Normal (std = sqrt(2 / (fan_in + fan_out)))")
print(f"Número de corridas estadísticas por experimento: {10}")
print("Criterio de parada: Early Stopping")
print("Paciencia del optimizador: 1000 iteraciones sin mejora")
print("Máximo de iteraciones de salvaguarda: 20000")
print("="*60)

loader = EmojiLoader()
X, labels = loader.get_all_data()
NUM_RUNS = 10

# --- Experimento 1: Optimizadores ---
print("\nEjecutando Experimento 1 (Comparación de Optimizadores)...")
print("  > Configuración SGD: lr=0.005 | Latente=2D | beta=0.003")
print("  > Configuración Adam: lr=0.001 | Latente=2D | beta=0.003")
_, mean_sgd, std_sgd = get_multiple_runs_stats(X, NUM_RUNS, latent_dim=2, kl_weight=0.003, optimizer_class=SGD, lr=0.005)
_, mean_adam, std_adam = get_multiple_runs_stats(X, NUM_RUNS, latent_dim=2, kl_weight=0.003, optimizer_class=Adam, lr=0.001)

plt.figure(figsize=(11, 5.5))
plt.plot(np.arange(len(mean_sgd)), mean_sgd, color='orangered', label='SGD')
plt.fill_between(np.arange(len(mean_sgd)), mean_sgd - std_sgd, mean_sgd + std_sgd, color='orangered', alpha=0.15)

smooth_mean_adam = smooth_curve(mean_adam, box_pts=25)
smooth_std_adam = smooth_curve(std_adam, box_pts=25)
plt.plot(np.arange(len(smooth_mean_adam)), smooth_mean_adam, color='dodgerblue', linewidth=2.5, label='Adam - Suavizado')
plt.fill_between(np.arange(len(smooth_mean_adam)), smooth_mean_adam - smooth_std_adam, smooth_mean_adam + smooth_std_adam, color='dodgerblue', alpha=0.15)

plt.title('Comparación de optimizadores', pad=20)
plt.xlabel('Iteraciones de entrenamiento')
plt.ylabel('Loss (BCE + KL)')
plt.yscale('log')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step1_smooth_optimizers.png'))
plt.close()

# --- Experimento 1.b: Learning Rates ---
print("\nEjecutando Experimento 1.b (Sensibilidad de Learning Rates)...")
print("  > Configuración fija: Optimizador=Adam | Latente=2D | beta=0.003")
print("  > Valores evaluados de lr: [0.01, 0.001, 0.0001]")
lrs = [0.01, 0.001, 0.0001]
plt.figure(figsize=(11, 5.5))
colors_lr = ['crimson', 'dodgerblue', 'forestgreen']

for idx, lr_val in enumerate(lrs):
    _, mean_lr, std_lr = get_multiple_runs_stats(X, NUM_RUNS, latent_dim=2, kl_weight=0.003, lr=lr_val)
    s_mean = smooth_curve(mean_lr, box_pts=25)
    s_std = smooth_curve(std_lr, box_pts=25)
    
    plt.plot(np.arange(len(s_mean)), s_mean, color=colors_lr[idx], linewidth=2, label=f"lr={lr_val}")
    plt.fill_between(np.arange(len(s_mean)), s_mean - s_std, s_mean + s_std, color=colors_lr[idx], alpha=0.12)

plt.title('Sensibilidad de convergencia según Learning Rate', pad=20)
plt.xlabel('Iteraciones de entrenamiento')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step1b_learning_rates.png'))
plt.close()

# --- Experimento 1.c: Activaciones ---
print("\nEjecutando Experimento 1.c (Evaluación de Funciones de Activación)...")
print("  > Configuración fija: Optimizador=Adam | lr=0.001 | Latente=2D | beta=0.003")
print("  > Variantes de capa oculta: Tanh() vs Sigmoid()")
plt.figure(figsize=(11, 5.5))
_, mean_tanh, std_tanh = get_multiple_runs_stats(X, NUM_RUNS, latent_dim=2, kl_weight=0.003, hidden_activation_class=Tanh, lr=0.001)
_, mean_sig, std_sig = get_multiple_runs_stats(X, NUM_RUNS, latent_dim=2, kl_weight=0.003, hidden_activation_class=Sigmoid, lr=0.001)

s_mean_tanh = smooth_curve(mean_tanh, box_pts=25)
s_std_tanh = smooth_curve(std_tanh, box_pts=25)
s_mean_sig = smooth_curve(mean_sig, box_pts=25)
s_std_sig = smooth_curve(std_sig, box_pts=25)

plt.plot(np.arange(len(s_mean_tanh)), s_mean_tanh, color='#2b5c8f', linewidth=2, label='Tanh()')
plt.fill_between(np.arange(len(s_mean_tanh)), s_mean_tanh - s_std_tanh, s_mean_tanh + s_std_tanh, color='#2b5c8f', alpha=0.12)
plt.plot(np.arange(len(s_mean_sig)), s_mean_sig, color='#d97d24', linewidth=2, label='Sigmoid()')
plt.fill_between(np.arange(len(s_mean_sig)), s_mean_sig - s_std_sig, s_mean_sig + s_std_sig, color='#d97d24', alpha=0.12)

plt.title('Evaluación de funciones de activación ocultas', pad=20)
plt.xlabel('Iteraciones de entrenamiento')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step1c_activation_functions.png'))
plt.close()

# --- Experimento 2: Matriz de Betas ---
print("\nEjecutando Experimento 2 (Estudio del Impacto del Coeficiente Beta)...")
print("  > Configuración fija: Optimizador=Adam | lr=0.001 | Latente=2D")
print("  > Coeficientes beta evaluados: [0.001, 0.003, 0.01, 0.2]")
betas = [0.001, 0.003, 0.01, 0.2]

fig_betas_clean, axes_b_clean = plt.subplots(2, 2, figsize=(14, 11))
fig_betas_shadow, axes_b_shadow = plt.subplots(2, 2, figsize=(14, 11))
fig_loss, axes_loss = plt.subplots(2, 2, figsize=(14, 11))

cmap_table = plt.get_cmap('tab20')

for idx_b, b_val in enumerate(betas):
    r_idx, c_idx = idx_b // 2, idx_b % 2
    
    vae_b, hist_b = train_parametrized_network(X, latent_dim=2, kl_weight=b_val, lr=0.001)
    mu_b, sigma_b = get_latent_distributions(vae_b, X, latent_dim=2)
    
    subtitle = f"beta = {b_val}"
    
    # 2.a Variante Limpia (Puntos limpios)
    ax_c = axes_b_clean[r_idx, c_idx]
    ax_c.scatter(mu_b[:, 0], mu_b[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=100, alpha=0.85, zorder=3)
    # Solo agregamos rótulos de nombres si beta es bajo (0.001 o 0.003)
    if b_val < 0.01:
        for i, label in enumerate(labels):
             ax_c.annotate(label, (mu_b[i, 0], mu_b[i, 1]), xytext=(4, 4), textcoords='offset points', fontsize=8, weight='bold')
    ax_c.set_title(subtitle, fontsize=11, weight='bold')
    ax_c.set_xlabel("Z1"); ax_c.set_ylabel("Z2")
    
    # 2.b Variante con Nubes (Elipses de varianza)
    ax_s = axes_b_shadow[r_idx, c_idx]
    ax_s.scatter(mu_b[:, 0], mu_b[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=100, alpha=0.85, zorder=3)
    for i, label in enumerate(labels):
         if b_val < 0.01:
              ax_s.annotate(label, (mu_b[i, 0], mu_b[i, 1]), xytext=(4, 4), textcoords='offset points', fontsize=8, weight='bold')
         ellipse = Ellipse(xy=(mu_b[i, 0], mu_b[i, 1]), width=sigma_b[i, 0]*2, height=sigma_b[i, 1]*2,
                           edgecolor=cmap_table(i/16), facecolor=cmap_table(i/16), alpha=0.12, linestyle='--')
         ax_s.add_patch(ellipse)
    ax_s.set_title(subtitle, fontsize=11, weight='bold')
    ax_s.set_xlabel("Z1"); ax_s.set_ylabel("Z2")
    
    # Curva de Loss correspondiente
    ax_loss = axes_loss[r_idx, c_idx]
    ax_loss.plot(hist_b, color='indigo', linewidth=1.8)
    ax_loss.set_title(f"Pérdida para beta = {b_val}", fontsize=11, weight='bold')
    ax_loss.set_xlabel("Iteraciones")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log')

plt.figure(fig_betas_clean.number)
plt.suptitle("Impacto de la divergencia KL en la topología del espacio latente", y=0.98, weight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step2_beta_comparison_matrix_clean.png'), bbox_inches='tight')
plt.close()

plt.figure(fig_betas_shadow.number)
plt.suptitle("Impacto de la divergencia KL en la topología del espacio latente", y=0.98, weight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step2_beta_comparison_matrix_shadow.png'), bbox_inches='tight')
plt.close()

plt.figure(fig_loss.number)
plt.suptitle("Dinámica de convergencia según parámetro beta", y=0.98, weight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step2b_beta_loss_curves.png'), bbox_inches='tight')
plt.close()

# --- Modelos de Producción Finales ---
print("\nEntrenando modelos finales de producción (Control 2D vs 3D)...")
print("  > Configuración fija: Optimizador=Adam | lr=0.001 | beta=0.003")
vae_2d_final, _ = train_parametrized_network(X, latent_dim=2, kl_weight=0.003, lr=0.001)
vae_3d_final, _ = train_parametrized_network(X, latent_dim=3, kl_weight=0.003, lr=0.001)

# --- Experimento 3: Espacio Latente 2D Final (Con y Sin Sombra) ---
print("\nEjecutando Experimento 3 (Mapeo Definitivo del Espacio 2D)...")
mu_2d_f, sigma_2d_f = get_latent_distributions(vae_2d_final, X, latent_dim=2)

# Versión 3.a: Sin Sombra (Limpia)
plt.figure(figsize=(10, 7.5))
plt.scatter(mu_2d_f[:, 0], mu_2d_f[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=140, alpha=0.9, zorder=3)
for i, label in enumerate(labels):
    plt.annotate(label, (mu_2d_f[i, 0], mu_2d_f[i, 1]), xytext=(6, 6), textcoords='offset points', fontsize=9, weight='bold')
plt.title('Representación de las muestras en el espacio latente (2D)', pad=20)
plt.xlabel('Z1'); plt.ylabel('Z2')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_final_latent_space_2d_clean.png'))
plt.close()

# Versión 3.b: Con Sombra (Elipses 1-sigma)
fig_s3, ax_s3 = plt.subplots(figsize=(10, 7.5))
ax_s3.scatter(mu_2d_f[:, 0], mu_2d_f[:, 1], c=np.arange(16), cmap='tab20', edgecolors='k', s=140, alpha=0.9, zorder=3)
for i, label in enumerate(labels):
    ax_s3.annotate(label, (mu_2d_f[i, 0], mu_2d_f[i, 1]), xytext=(6, 6), textcoords='offset points', fontsize=9, weight='bold')
    ellipse = Ellipse(xy=(mu_2d_f[i, 0], mu_2d_f[i, 1]), width=sigma_2d_f[i, 0]*2, height=sigma_2d_f[i, 1]*2,
                      edgecolor=cmap_table(i/16), facecolor=cmap_table(i/16), alpha=0.15, linestyle='--')
    ax_s3.add_patch(ellipse)
plt.title('Representación de las muestras en el espacio latente con elipses de incertidumbre', pad=20)
ax_s3.set_xlabel('Z1'); ax_s3.set_ylabel('Z2')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_final_latent_space_2d_shadow.png'))
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

plt.suptitle("Fidelidad de reconstrucción completa del dataset (Espacio 2D)", y=0.98, weight='bold', fontsize=15)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'step4_full_reconstruction_matrix.png'), bbox_inches='tight')
plt.close()

# --- Experimento 4b: Matriz de Reconstrucción Total (3D) ---
print("Ejecutando Experimento 4b (Fidelidad 3D)...")
predicted_3d = vae_3d_final.forward(X)
fig_rec_3d, axes_rec_3d = plt.subplots(4, 8, figsize=(15, 8.5))

for i in range(16):
    row_orig, row_rec, col = (0, 1, i) if i < 8 else (2, 3, i - 8)
    axes_rec_3d[row_orig, col].imshow(X[i].reshape(16, 16), cmap='bone_r')
    axes_rec_3d[row_orig, col].set_xticks([]); axes_rec_3d[row_rec, col].set_yticks([])
    axes_rec_3d[row_orig, col].set_title(labels[i], fontsize=9.5, weight='bold')
    if col == 0: axes_rec_3d[row_orig, col].set_ylabel("Original", weight='bold', fontsize=11)
    
    axes_rec_3d[row_rec, col].imshow(predicted_3d[i].reshape(16, 16), cmap='bone_r')
    axes_rec_3d[row_rec, col].set_xticks([]); axes_rec_3d[row_rec, col].set_yticks([])
    if col == 0: axes_rec_3d[row_rec, col].set_ylabel("Reconstruido", weight='bold', fontsize=11, color='forestgreen')

plt.suptitle("Fidelidad de reconstrucción completa del dataset (Espacio 3D)", y=0.98, weight='bold', fontsize=15)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'step4b_full_reconstruction_matrix_3d.png'), bbox_inches='tight')
plt.close()

# --- Experimento 5: Espacio Latente 3D Final ---
print("Ejecutando Experimento 5 (Mapeo Estructural 3D)...")
mu_3d_f, _ = get_latent_distributions(vae_3d_final, X, latent_dim=3)
fig_3d = plt.figure(figsize=(11, 8.5))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.patch.set_facecolor('#f5f5f5')

ax_3d.scatter(mu_3d_f[:, 0], mu_3d_f[:, 1], mu_3d_f[:, 2], c=np.arange(16), cmap='tab20', edgecolors='k', s=130, alpha=0.9)
for i, label in enumerate(labels):
    ax_3d.text(mu_3d_f[i, 0], mu_3d_f[i, 1], mu_3d_f[i, 2], label, fontsize=9, weight='bold')

ax_3d.set_title('Representación de las muestras en el espacio latente (3D)', pad=20)
ax_3d.set_xlabel('Z1'); ax_3d.set_ylabel('Z2'); ax_3d.set_zlabel('Z3')
plt.savefig(os.path.join(OUTPUT_DIR, 'step5_latent_space_3d.png'), bbox_inches='tight')
plt.close()

# --- Experimento 6: Comparación Cruzada (Corazón) ---
print("Ejecutando Experimento 6...")
h_idx = labels.index('corazon')
br_idx = labels.index('corazon_roto')

fig_check, axes_c = plt.subplots(2, 3, figsize=(10, 6.5))
axes_c[0, 0].imshow(X[h_idx].reshape(16, 16), cmap='bone_r'); axes_c[0, 0].set_title("Original", weight='bold')
axes_c[0, 1].imshow(predicted_2d[h_idx].reshape(16, 16), cmap='bone_r'); axes_c[0, 1].set_title("Reconstrucción 2D", color='crimson', weight='bold')
axes_c[0, 2].imshow(predicted_3d[h_idx].reshape(16, 16), cmap='bone_r'); axes_c[0, 2].set_title("Reconstrucción 3D", color='forestgreen', weight='bold')

axes_c[1, 0].imshow(X[br_idx].reshape(16, 16), cmap='bone_r')
axes_c[1, 1].imshow(predicted_2d[br_idx].reshape(16, 16), cmap='bone_r')
axes_c[1, 2].imshow(predicted_3d[br_idx].reshape(16, 16), cmap='bone_r')

for r in range(2):
    for c in range(3):
        axes_c[r, c].set_xticks([]); axes_c[r, c].set_yticks([])
axes_c[0, 0].set_ylabel("Corazón liso", weight='bold', fontsize=11)
axes_c[1, 0].set_ylabel("Corazón roto", weight='bold', fontsize=11)

plt.suptitle("Análisis de capacidad resolutiva del espacio latente (2D vs. 3D)", weight='bold', y=0.99, fontsize=13)
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig(os.path.join(OUTPUT_DIR, 'step6_structural_resolution.png'))
plt.close()

# --- Experimento 7: Interpolaciones No Lineales Suavizadas (Sinusoidales) ---
print("Ejecutando Experimento 7...")

def generate_smooth_interpolation_plot(vae, mu_data, labels_list, start_label, end_label, save_filename):
    z_start = mu_data[labels_list.index(start_label)]
    z_end = mu_data[labels_list.index(end_label)]
    
    num_steps = 7
    linear_steps = np.linspace(0, 1, num_steps)
    smooth_steps = 0.5 - 0.5 * np.cos(linear_steps * np.pi)
    
    interpolated_z = np.array([z_start + (z_end - z_start) * t for t in smooth_steps])
    decoded_images = decode_latent_z(vae, interpolated_z)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(14, 3))
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

generate_smooth_interpolation_plot(vae_2d_final, mu_2d_f, labels, 'rayo', 'gotas', 'step7a_interpolation_rayo_gotas.png')
generate_smooth_interpolation_plot(vae_2d_final, mu_2d_f, labels, 'sorpresa', 'sonrisa', 'step7b_interpolation_sorpresa_sonrisa.png')

print("="*60)
print("PIPELINE EJECUTADO CON ÉXITO. TODOS LOS ENTREGABLES GUARDADOS.")
print("="*60)