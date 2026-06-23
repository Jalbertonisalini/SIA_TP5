"""
DAE – Experimentos completos

  EXP6  : Comparación de arquitecturas DAE
  EXP6b : Barrido del cuello de botella
  EXP6c : Ruido fijo vs ruido online
"""

import numpy as np
import os
import sys
import matplotlib
import copy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.network import Network
from core.losses import BCE
from core.optimizers import Adam
from utils.data_loader import FontLoader

# ══════════════════════════════════════════════════════════════════════
# ESTILO  (mismo que AE/VAE)
# ══════════════════════════════════════════════════════════════════════

FACECOLOR  = '#FFFFFF'
GRID_COLOR = '#E6E6E6'
TEXT_COLOR = '#333333'

plt.rcParams.update({
    'figure.facecolor': FACECOLOR,
    'axes.facecolor':   FACECOLOR,
    'axes.titleweight': 'bold',
    'font.family':      'sans-serif',
})

def _apply_ae_style(ax):
    ax.set_facecolor(FACECOLOR)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=':', alpha=0.8, color=GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

SERIES_COLORS = {
    "Deep (32-16-10)":     "#C47A00",
    "Base (16-10-16)":     "#595959",
    "Wide (24-10-24)":     "#2E7559",
    "Wide (24-16-24)":     "#1F4E79",
    "Wide (24-20-24)":     "#6B4C9A",
}
COLOR_FIXED  = "#A63636"
COLOR_ONLINE = "#1F4E79"

# Arquitectura a mostrar en ejes/leyendas
ARCH_STRINGS = {
    "Deep (32-16-10)":     "35→32→16→10→16→32→35",
    "Base (16-10-16)":     "35→16→10→16→35",
    "Wide (24-10-24)":     "35→24→10→24→35",
    "Wide (24-16-24)":     "35→24→16→24→35",
    "Wide (24-20-24)":     "35→24→20→24→35",
}

# Etiquetas correctas de los 32 caracteres del dataset (0x60–0x7f)
CHAR_LABELS = ['`'] + [chr(ord('a') + i) for i in range(26)] + ['{', '|', '}', '~', 'DEL']

# ══════════════════════════════════════════════════════════════════════
# CONSTANTES COMPARTIDAS
# ══════════════════════════════════════════════════════════════════════

NOISE_LEVEL   = 0.25
MAX_EPOCHS    = 50000
PLATEAU_CHECK = 1000
PLATEAU_EPS   = 1e-5
LR            = 0.001
SEEDS         = [42, 100, 200, 500, 800, 1024, 1337, 2024, 3000, 9999]

# ══════════════════════════════════════════════════════════════════════
# HELPERS COMPARTIDOS
# ══════════════════════════════════════════════════════════════════════

def add_noise(X: np.ndarray, noise_level: float) -> np.ndarray:
    X_noisy = X.copy()
    for i in range(X_noisy.shape[0]):
        n_flip = int(noise_level * X.shape[1])
        idx = np.random.choice(X.shape[1], n_flip, replace=False)
        X_noisy[i, idx] = 1 - X_noisy[i, idx]
    return X_noisy

def pixel_error(expected: np.ndarray, predicted: np.ndarray) -> int:
    return int(np.max(np.sum(np.abs(expected.astype(int) -
                                     (predicted >= 0.5).astype(int)), axis=1)))

def rolling_mean(data, window: int = 500) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')

def plateau_stop(history: list, epoch: int) -> bool:
    if epoch >= 2 * PLATEAU_CHECK and epoch % PLATEAU_CHECK == 0:
        recent = np.mean(history[-PLATEAU_CHECK:])
        prev   = np.mean(history[-2 * PLATEAU_CHECK:-PLATEAU_CHECK])
        return prev - recent < PLATEAU_EPS
    return False

def train_with_plateau(factory_fn, X_clean: np.ndarray,
                       noise_fn) -> tuple:
    """noise_fn(X_clean) -> X_noisy  (permite online o fijo desde afuera)"""
    ae        = factory_fn() if callable(factory_fn) else factory_fn
    loss_fn   = BCE()
    optimizer = Adam(learning_rate=LR)
    history   = []
    converged = MAX_EPOCHS

    # 1. Variables para trackear el mejor estado intermedio
    best_loss = float('inf')
    best_layers = None

    for epoch in range(1, MAX_EPOCHS + 1):
        X_noisy = noise_fn(X_clean)
        pred    = ae.forward(X_noisy)
        loss    = loss_fn.calculate(expected=X_clean, predicted=pred)
        history.append(loss)
        ae.backward(loss_fn.derivative(expected=X_clean, predicted=pred), optimizer)

        # 2. Guardar checkpoint si encontramos una pérdida menor
        if loss < best_loss:
            best_loss = loss
            best_layers = copy.deepcopy(ae.layers)

        if plateau_stop(history, epoch):
            converged = epoch
            break

    # 3. Restaurar los mejores pesos antes de salir
    if best_layers is not None:
        ae.layers = best_layers

    return ae, history, converged

def encode_2d(model, X: np.ndarray) -> np.ndarray:
    """Forward hasta el cuello de botella 2D (detiene al llegar a 2 dimensiones)."""
    h = X.copy()
    for layer in model.layers:
        h = layer.forward(h)
        if h.shape[1] == 2:
            break
    return h

# ══════════════════════════════════════════════════════════════════════
# EXP6 – ARQUITECTURAS
# ══════════════════════════════════════════════════════════════════════

def create_ae_winner() -> Network:
    ae = Network()
    ae.add(Linear(35, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16,  2)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear( 2, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_ae_deep_winner() -> Network:
    ae = Network()
    ae.add(Linear(35, 32)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16,  2)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear( 2, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 32)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_deep10_dae() -> Network:
    """35 → 32 → 16 → 10 → 16 → 32 → 35  (mismo outer que AE Winner, bottleneck 10)"""
    ae = Network()
    ae.add(Linear(35, 32)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 10)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(10, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 32)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_base_dae() -> Network:
    ae = Network()
    ae.add(Linear(35, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 10)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(10, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_wide_dae() -> Network:
    ae = Network()
    ae.add(Linear(35, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 10)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(10, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_wide16_dae() -> Network:
    ae = Network()
    ae.add(Linear(35, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_wide20_dae() -> Network:
    ae = Network()
    ae.add(Linear(35, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 20)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(20, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

ARCHITECTURES = {
    "Deep (32-16-10)":     create_deep10_dae,
    "Base (16-10-16)":     create_base_dae,
    "Wide (24-10-24)":     create_wide_dae,
    "Wide (24-16-24)":     create_wide16_dae,
    "Wide (24-20-24)":     create_wide20_dae,
}


def train_arch(factory_fn, X_clean: np.ndarray) -> tuple:
    return train_with_plateau(factory_fn,
                              X_clean,
                              lambda X: add_noise(X, NOISE_LEVEL))


def run_exp6(X: np.ndarray) -> dict:
    loss_fn = BCE()
    results = {}
    for name, factory_fn in ARCHITECTURES.items():
        print(f"\n[+] {name}")
        epochs_list, px_list, bce_list, histories = [], [], [], []
        best_model, best_bce, vis_noisy = None, float('inf'), None
        for seed in SEEDS:
            print(f"   seed {seed:4d} ...", end="", flush=True)
            np.random.seed(seed)
            X_noisy_eval = add_noise(X, NOISE_LEVEL)
            model, history, conv = train_arch(factory_fn, X)
            eval_pred = model.forward(X_noisy_eval)
            px  = pixel_error(X, eval_pred)
            bce = loss_fn.calculate(expected=X, predicted=eval_pred)
            print(f"  {'OK' if conv < MAX_EPOCHS else 'NO'} ({conv} épocas)  MaxPx={px}  BCE={bce:.5f}")
            epochs_list.append(conv); px_list.append(px)
            bce_list.append(bce); histories.append(history)
            if bce < best_bce:
                best_bce = bce; best_model = model; vis_noisy = X_noisy_eval
        print(f"   → Épocas: {np.mean(epochs_list):.0f}±{np.std(epochs_list):.0f} | "
              f"MaxPx: {np.mean(px_list):.1f}±{np.std(px_list):.1f} | "
              f"BCE: {np.mean(bce_list):.5f}±{np.std(bce_list):.5f}")
        results[name] = {
            "epochs": epochs_list, "px_errors": px_list, "bce_finals": bce_list,
            "histories": histories, "best_model": best_model, "vis_noisy": vis_noisy,
        }
    return results


def plot_exp6_convergence(results: dict) -> None:
    window = 500
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(FACECOLOR)
    for name, stats in results.items():
        color  = SERIES_COLORS[name]
        hs     = stats["histories"]
        ml     = max(len(h) for h in hs)
        padded = np.array([h + [h[-1]] * (ml - len(h)) for h in hs])
        ms     = rolling_mean(np.mean(padded, axis=0), window)
        x      = np.arange(window - 1, window - 1 + len(ms))
        ax.plot(x, ms, label=ARCH_STRINGS[name], color=color, linewidth=1.5)
    ax.set_title("DAE – Convergencia por Arquitectura", fontsize=13, color=TEXT_COLOR)
    ax.set_xlabel("Épocas", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("BCE Loss (entrenamiento)", fontsize=11, color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))
    ax.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=10)
    _apply_ae_style(ax)
    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/exp6_dae_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_convergence.png")


def plot_exp6_comparison(results: dict) -> None:
    names     = list(results.keys())
    colors    = [SERIES_COLORS[n] for n in names]
    x_pos     = np.arange(len(names))
    ep_means  = [np.mean(results[n]["epochs"])     for n in names]
    ep_stds   = [np.std(results[n]["epochs"])      for n in names]
    bce_means = [np.mean(results[n]["bce_finals"]) for n in names]
    bce_stds  = [np.std(results[n]["bce_finals"])  for n in names]
    arch_labels = [ARCH_STRINGS[n] for n in names]

    os.makedirs('outputs', exist_ok=True)

    # — Figura 1: Épocas —
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_title("DAE – Épocas hasta Convergencia por Arquitectura", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("Épocas hasta convergencia", fontsize=11, color=TEXT_COLOR, fontweight='bold')
    bars = ax.bar(x_pos, ep_means, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2, zorder=3)
    ax.errorbar(x_pos, ep_means, yerr=ep_stds, fmt='none', color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    for bar, m, s in zip(bars, ep_means, ep_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 200,
                f"{m:.0f}", ha='center', fontsize=10, color=TEXT_COLOR)
    ax.axhline(y=MAX_EPOCHS, color='gray', linestyle='--', alpha=0.6,
               linewidth=1.2, label=f"Límite ({MAX_EPOCHS} épocas)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arch_labels, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, facecolor=FACECOLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    _apply_ae_style(ax)
    fig.tight_layout()
    fig.savefig('outputs/exp6_dae_epochs.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_epochs.png")

    # — Figura 2: BCE —
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_title("DAE – BCE Loss (Evaluación) por Arquitectura", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("BCE Loss (evaluación)", fontsize=11, color=TEXT_COLOR, fontweight='bold')
    bars = ax.bar(x_pos, bce_means, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2, zorder=3)
    ax.errorbar(x_pos, bce_means, yerr=bce_stds, fmt='none', color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    for bar, m, s in zip(bars, bce_means, bce_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.003,
                f"{m:.4f}", ha='center', fontsize=10, color=TEXT_COLOR)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arch_labels, fontsize=10)
    ax.set_ylim(bottom=0)
    _apply_ae_style(ax)
    fig.tight_layout()
    fig.savefig('outputs/exp6_dae_bce.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_bce.png")

    # — Figura 3: MaxPx —
    px_means  = [np.mean(results[n]["px_errors"]) for n in names]
    px_stds   = [np.std(results[n]["px_errors"])  for n in names]
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_title("DAE – Máx. Píxeles Incorrectos por Arquitectura", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("Max Píxeles Incorrectos", fontsize=11, color=TEXT_COLOR, fontweight='bold')
    bars = ax.bar(x_pos, px_means, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2, zorder=3)
    ax.errorbar(x_pos, px_means, yerr=px_stds, fmt='none', color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    for bar, m, s in zip(bars, px_means, px_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.2,
                f"{m:.1f}", ha='center', fontsize=10, color=TEXT_COLOR)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arch_labels, fontsize=10)
    ax.set_ylim(bottom=0)
    _apply_ae_style(ax)
    fig.tight_layout()
    fig.savefig('outputs/exp6_dae_px.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_px.png")


def plot_exp6_reconstructions(results: dict, X: np.ndarray, letter_indices: list) -> None:
    """Muestra un subset de letras: original + ruidosa + reconstruida por arquitectura."""
    names  = list(results.keys())
    n_cols = 1 + 2 * len(letter_indices)
    fig, axes = plt.subplots(len(names) + 1, n_cols,
                              figsize=(3.2 * n_cols, 3.0 * (len(names) + 1)))
    fig.patch.set_facecolor(FACECOLOR)

    def show(ax, img, title=None, border_color=None):
        ax.imshow(img.reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
        if title: ax.set_title(title, fontsize=26, fontweight='bold', color=TEXT_COLOR)
        if border_color:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color); spine.set_linewidth(2.5); spine.set_visible(True)

    np.random.seed(42)
    vis_shared = add_noise(X, NOISE_LEVEL)

    axes[0, 0].axis('off')
    for j, idx in enumerate(letter_indices):
        show(axes[0, 1 + j * 2], X[idx],          title=f"'{CHAR_LABELS[idx]}' original")
        show(axes[0, 2 + j * 2], vis_shared[idx], title=f"'{CHAR_LABELS[idx]}' ruidosa")

    for row, name in enumerate(names, start=1):
        axes[row, 0].axis('off')
        axes[row, 0].text(0.5, 0.5, ARCH_STRINGS[name],
                          ha='center', va='center', fontsize=21,
                          color=SERIES_COLORS[name], fontweight='bold',
                          transform=axes[row, 0].transAxes)
        pred = results[name]["best_model"].forward(vis_shared)
        for j, idx in enumerate(letter_indices):
            axes[row, 1 + j * 2].axis('off')
            show(axes[row, 2 + j * 2], pred[idx], border_color=SERIES_COLORS[name])

    fig.suptitle("DAE – Reconstrucciones por Arquitectura (mejor BCE)",
                 fontsize=20, fontweight='bold', y=1.01, color=TEXT_COLOR)
    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/exp6_dae_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_reconstructions.png")


def plot_exp6_full_reconstruction(results: dict, X: np.ndarray) -> None:
    """
    Grilla completa: todos los 32 caracteres para la arquitectura ganadora.
    Layout: 8 filas × 8 cols — filas pares = original, filas impares = reconstruida.
    """
    ranking = sorted(results.items(), key=lambda kv: np.mean(kv[1]["bce_finals"]))
    winner_name, winner_stats = ranking[0]
    model     = winner_stats["best_model"]
    vis_noisy = winner_stats["vis_noisy"]
    pred      = model.forward(vis_noisy)

    n_chars = X.shape[0]   # 32
    n_cols  = 8
    n_rows  = (n_chars // n_cols) * 2   # 8 filas

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 1.9))
    fig.patch.set_facecolor('#f5f5f5')

    for i in range(n_chars):
        col      = i % n_cols
        row_orig = (i // n_cols) * 2
        row_rec  = row_orig + 1
        label = CHAR_LABELS[i]

        axes[row_orig, col].imshow(X[i].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[row_orig, col].axis('off')
        axes[row_orig, col].set_title(f"'{label}'", fontsize=13, fontweight='bold', color=TEXT_COLOR)

        axes[row_rec, col].imshow(pred[i].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[row_rec, col].axis('off')

        if col == 0:
            axes[row_orig, col].set_ylabel("Original",     fontsize=14, fontweight='bold',
                                            rotation=0, labelpad=65, va='center', color=TEXT_COLOR)
            axes[row_rec,  col].set_ylabel("Reconstruida", fontsize=14, fontweight='bold',
                                            color=SERIES_COLORS[winner_name],
                                            rotation=0, labelpad=65, va='center')

    fig.suptitle(f"DAE – Reconstrucción Completa del Dataset\n"
                 f"Arquitectura: {ARCH_STRINGS[winner_name]}",
                 fontsize=13, fontweight='bold', y=1.01, color=TEXT_COLOR)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.97])
    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/exp6_dae_full_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_full_reconstruction.png")


def plot_exp6_latent_space(model, X: np.ndarray) -> None:
    """Espacio latente 2D del AE Winner (B=2): scatter de los 32 chars limpios."""
    color    = "#A63636"
    arch_str = "35→32→16→2→16→32→35"
    z        = encode_2d(model, X)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(FACECOLOR)

    ax.scatter(z[:, 0], z[:, 1],
               color=color, edgecolors=TEXT_COLOR,
               linewidths=0.9, s=90, alpha=0.95, zorder=2)

    for i, lbl in enumerate(CHAR_LABELS):
        ax.annotate(lbl, (z[i, 0], z[i, 1]),
                    xytext=(6, 5), textcoords='offset points',
                    fontsize=11, color=TEXT_COLOR)

    ax.set_title(f"DAE – Espacio Latente 2D\n{arch_str}",
                 fontsize=13, fontweight='bold', color=TEXT_COLOR)
    ax.set_xlabel("Z1", fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.set_ylabel("Z2", fontsize=12, fontweight='bold', color=TEXT_COLOR)
    _apply_ae_style(ax)
    fig.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/exp6_dae_latent_space.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6_dae_latent_space.png")


# ══════════════════════════════════════════════════════════════════════
# EXP6c – RUIDO FIJO vs RUIDO ONLINE
# ══════════════════════════════════════════════════════════════════════

def train_fixed(factory_fn, X_clean: np.ndarray, X_noisy_train: np.ndarray) -> tuple:
    return train_with_plateau(factory_fn,
                              X_clean,
                              lambda X: X_noisy_train)


def train_online(factory_fn, X_clean: np.ndarray) -> tuple:
    return train_with_plateau(factory_fn,
                              X_clean,
                              lambda X: add_noise(X, NOISE_LEVEL))


def run_exp6c(X: np.ndarray, factory_fn) -> dict:
    loss_fn = BCE()
    results = {"fixed": [], "online": []}
    for seed in SEEDS:
        print(f"\n  seed {seed} ...", end="", flush=True)
        np.random.seed(seed)
        X_noisy_train = add_noise(X, NOISE_LEVEL)
        np.random.seed(seed + 50000)
        X_noisy_eval  = add_noise(X, NOISE_LEVEL)

        np.random.seed(seed)
        model_f, hist_f, ep_f = train_fixed(factory_fn, X, X_noisy_train)
        pred_f = model_f.forward(X_noisy_eval)
        bce_f  = loss_fn.calculate(expected=X, predicted=pred_f)

        np.random.seed(seed)
        model_o, hist_o, ep_o = train_online(factory_fn, X)
        pred_o = model_o.forward(X_noisy_eval)
        bce_o  = loss_fn.calculate(expected=X, predicted=pred_o)

        print(f"  Fijo: {ep_f}ép BCE={bce_f:.4f} | Online: {ep_o}ép BCE={bce_o:.4f}")
        results["fixed"].append({
            "history": hist_f, "epochs": ep_f, "px": pixel_error(X, pred_f), "bce": bce_f,
            "model": model_f, "vis_noisy": X_noisy_eval, "train_noisy": X_noisy_train,
        })
        results["online"].append({
            "history": hist_o, "epochs": ep_o, "px": pixel_error(X, pred_o), "bce": bce_o,
            "model": model_o, "vis_noisy": X_noisy_eval,
        })
    return results


def plot_exp6c_convergence(results: dict) -> None:
    window = 300
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(FACECOLOR)
    for label, color, key in [("Ruido Fijo", COLOR_FIXED, "fixed"),
                               ("Ruido Online", COLOR_ONLINE, "online")]:
        hs     = [r["history"] for r in results[key]]
        ml     = max(len(h) for h in hs)
        padded = np.array([h + [h[-1]] * (ml - len(h)) for h in hs])
        ms     = rolling_mean(np.mean(padded, axis=0), window)
        ss     = rolling_mean(np.std(padded,  axis=0), window)
        x      = np.arange(window - 1, window - 1 + len(ms))
        ax.plot(x, ms, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, ms - ss, ms + ss, color=color, alpha=0.10, edgecolor='none')
    ax.set_title("DAE – Ruido Fijo vs Ruido Online", fontsize=13, color=TEXT_COLOR)
    ax.set_xlabel("Épocas", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("BCE Loss (entrenamiento)", fontsize=11, color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))
    ax.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=11)
    _apply_ae_style(ax)
    fig.tight_layout()
    os.makedirs('outputs/exp6c', exist_ok=True)
    fig.savefig('outputs/exp6c/exp6c_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6c_convergence.png")


def plot_exp6c_reconstructions(results: dict, X: np.ndarray, letter_indices: list) -> None:
    n_letters = len(letter_indices)
    n_cols    = n_letters + 1
    row_labels = [
        "Original",
        f"Ruidosa  ({int(NOISE_LEVEL*100)}%)",
        "Ruido Fijo",
        "Ruido Online",
    ]
    row_colors = ['#333333', '#333333', COLOR_FIXED, COLOR_ONLINE]

    sorted_idx  = np.argsort([r["bce"] for r in results["online"]])
    median_idx  = int(sorted_idx[len(sorted_idx) // 2])
    print(f"  Visualizando seed {SEEDS[median_idx]} (mediana BCE online)")
    model_f   = results["fixed"][median_idx]["model"]
    model_o   = results["online"][median_idx]["model"]
    vis_noisy = results["online"][median_idx]["vis_noisy"]
    pred_f    = model_f.forward(vis_noisy)
    pred_o    = model_o.forward(vis_noisy)

    imgs_per_row = [
        [X[idx].reshape(7, 5)         for idx in letter_indices],
        [vis_noisy[idx].reshape(7, 5) for idx in letter_indices],
        [pred_f[idx].reshape(7, 5)    for idx in letter_indices],
        [pred_o[idx].reshape(7, 5)    for idx in letter_indices],
    ]

    col_widths = [2.5] + [2.2] * n_letters
    fig, axes  = plt.subplots(4, n_cols, figsize=(sum(col_widths), 2.6 * 4),
                               gridspec_kw={'width_ratios': col_widths})
    fig.patch.set_facecolor('#f5f5f5')

    for row, (imgs, label, color) in enumerate(zip(imgs_per_row, row_labels, row_colors)):
        axes[row, 0].axis('off')
        axes[row, 0].text(0.95, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=12, color=color, ha='right', va='center',
                          fontweight='bold' if row >= 2 else 'normal', linespacing=1.4)
        for col, (img, idx) in enumerate(zip(imgs, letter_indices), start=1):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray_r', vmin=0, vmax=1); ax.axis('off')
            if row == 0:
                ax.set_title(f"'{CHAR_LABELS[idx]}'", fontsize=14, fontweight='bold', color=TEXT_COLOR)
            if row in (2, 3):
                pred = pred_f if row == 2 else pred_o
                bc   = "#A63636" if pixel_error(X[[idx]], pred[[idx]]) > 1 else "#2E7559"
                for spine in ax.spines.values():
                    spine.set_edgecolor(bc); spine.set_linewidth(2.5); spine.set_visible(True)

    fig.suptitle("DAE – Ruido Fijo vs Ruido Online",
                 fontsize=13, fontweight='bold', y=1.01, color=TEXT_COLOR)
    fig.tight_layout()
    os.makedirs('outputs/exp6c', exist_ok=True)
    fig.savefig('outputs/exp6c/exp6c_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6c_reconstructions.png")


# ══════════════════════════════════════════════════════════════════════
# EXP6d – LÍMITES DE RUIDO (BREAKDOWN POINT)
# ══════════════════════════════════════════════════════════════════════

NOISE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
NOISE_COLORS = ['#2E7559', '#5BA08A', '#C47A00', '#D45A00', '#A63636']


def train_noise_level(factory_fn, X_clean: np.ndarray, noise_level: float) -> tuple:
    return train_with_plateau(factory_fn,
                              X_clean,
                              lambda X: add_noise(X, noise_level))


def run_exp6d(X: np.ndarray, factory_fn) -> dict:
    loss_fn = BCE()
    results = {}
    for noise in NOISE_LEVELS:
        print(f"\n[+] Ruido: {int(noise*100)}%")
        epochs_list, px_list, bce_list, histories = [], [], [], []
        best_model, best_bce, vis_noisy = None, float('inf'), None
        for seed in SEEDS:
            print(f"   seed {seed:4d} ...", end="", flush=True)
            np.random.seed(seed)
            model, history, conv = train_noise_level(factory_fn, X, noise)
            np.random.seed(seed + 50000)
            X_noisy_eval = add_noise(X, noise)
            pred = model.forward(X_noisy_eval)
            px   = pixel_error(X, pred)
            bce  = loss_fn.calculate(expected=X, predicted=pred)
            tag  = "OK" if conv < MAX_EPOCHS else "NO"
            print(f"  {tag} ({conv} ép)  MaxPx={px}  BCE={bce:.5f}")
            epochs_list.append(conv); px_list.append(px)
            bce_list.append(bce); histories.append(history)
            if bce < best_bce:
                best_bce = bce; best_model = model; vis_noisy = X_noisy_eval
        print(f"   → Épocas: {np.mean(epochs_list):.0f}±{np.std(epochs_list):.0f} | "
              f"MaxPx: {np.mean(px_list):.1f}±{np.std(px_list):.1f} | "
              f"BCE: {np.mean(bce_list):.5f}±{np.std(bce_list):.5f}")
        results[noise] = {
            "epochs": epochs_list, "px_errors": px_list, "bce_finals": bce_list,
            "histories": histories, "best_model": best_model, "vis_noisy": vis_noisy,
        }
    return results


def plot_exp6d_metrics(results: dict) -> None:
    """Barras de BCE y MaxPx vs nivel de ruido — 2 figuras separadas."""
    x_pos     = np.arange(len(NOISE_LEVELS))
    bce_means = [np.mean(results[n]["bce_finals"]) for n in NOISE_LEVELS]
    bce_stds  = [np.std(results[n]["bce_finals"])  for n in NOISE_LEVELS]
    px_means  = [np.mean(results[n]["px_errors"])  for n in NOISE_LEVELS]
    px_stds   = [np.std(results[n]["px_errors"])   for n in NOISE_LEVELS]
    xlabels   = [f"{int(n*100)}%" for n in NOISE_LEVELS]

    os.makedirs('outputs', exist_ok=True)

    # — Figura 1: BCE —
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_title("DAE – BCE Loss vs Nivel de Ruido", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("BCE Loss (evaluación)", fontsize=11, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel("Nivel de Ruido", fontsize=11, color=TEXT_COLOR)
    bars = ax.bar(x_pos, bce_means, color=NOISE_COLORS, alpha=0.85,
                  edgecolor='white', linewidth=1.2, zorder=3)
    ax.errorbar(x_pos, bce_means, yerr=bce_stds, fmt='none',
                color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    for bar, m, s in zip(bars, bce_means, bce_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.003,
                f"{m:.4f}", ha='center', fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x_pos); ax.set_xticklabels(xlabels, fontsize=11)
    _apply_ae_style(ax)
    fig.tight_layout()
    fig.savefig('outputs/exp6d_noise_bce.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6d_noise_bce.png")

    # — Figura 2: MaxPx —
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_title("DAE – Píxeles Incorrectos vs Nivel de Ruido", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("Max Píxeles Incorrectos", fontsize=11, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel("Nivel de Ruido", fontsize=11, color=TEXT_COLOR)
    bars = ax.bar(x_pos, px_means, color=NOISE_COLORS, alpha=0.85,
                  edgecolor='white', linewidth=1.2, zorder=3)
    ax.errorbar(x_pos, px_means, yerr=px_stds, fmt='none',
                color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    for bar, m, s in zip(bars, px_means, px_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.3,
                f"{m:.1f}", ha='center', fontsize=10, color=TEXT_COLOR)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7,
               linewidth=1.2, label="Umbral viabilidad (≤ 1 px)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(x_pos); ax.set_xticklabels(xlabels, fontsize=11)
    ax.legend(fontsize=10, facecolor=FACECOLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    _apply_ae_style(ax)
    fig.tight_layout()
    fig.savefig('outputs/exp6d_noise_px.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6d_noise_px.png")


def plot_exp6d_reconstructions(results: dict, X: np.ndarray,
                                letter_indices: list) -> None:
    """Grilla: fila por nivel de ruido — original / ruidosa / reconstruida."""
    n_letters = len(letter_indices)
    n_cols    = 1 + 2 * n_letters
    n_rows    = 1 + len(NOISE_LEVELS)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.0 * n_cols, 2.8 * n_rows))
    fig.patch.set_facecolor(FACECOLOR)

    def show(ax, img, title=None, border_color=None):
        ax.imshow(img.reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
        if title: ax.set_title(title, fontsize=19, fontweight='bold', color=TEXT_COLOR)
        if border_color:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color); spine.set_linewidth(2.5); spine.set_visible(True)

    # Fila 0: originales
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, 0.5, "Original", ha='center', va='center',
                    fontsize=19, fontweight='bold', color=TEXT_COLOR,
                    transform=axes[0, 0].transAxes)
    for j, idx in enumerate(letter_indices):
        show(axes[0, 1 + j * 2], X[idx], title=f"'{CHAR_LABELS[idx]}'")
        axes[0, 2 + j * 2].axis('off')

    # Filas por nivel de ruido
    for row, (noise, color) in enumerate(zip(NOISE_LEVELS, NOISE_COLORS), start=1):
        axes[row, 0].axis('off')
        axes[row, 0].text(0.5, 0.5, f"Ruido\n{int(noise*100)}%",
                          ha='center', va='center', fontsize=19,
                          fontweight='bold', color=color,
                          transform=axes[row, 0].transAxes)
        model     = results[noise]["best_model"]
        vis_noisy = results[noise]["vis_noisy"]
        pred      = model.forward(vis_noisy)
        for j, idx in enumerate(letter_indices):
            show(axes[row, 1 + j * 2], vis_noisy[idx])
            show(axes[row, 2 + j * 2], pred[idx], border_color=color)

    # Encabezados de columnas
    for j, idx in enumerate(letter_indices):
        axes[0, 1 + j * 2].set_title("Original",     fontsize=19, fontweight='bold', color=TEXT_COLOR)
        axes[0, 2 + j * 2].set_title("Reconstruida", fontsize=19, fontweight='bold', color=TEXT_COLOR)

    fig.suptitle("DAE – Reconstrucción por Nivel de Ruido",
                 fontsize=14, fontweight='bold', y=1.01, color=TEXT_COLOR)
    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/exp6d_noise_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [+] exp6d_noise_reconstructions.png")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    X       = FontLoader.load_and_flatten('src/data/font.h')
    loss_fn = BCE()

    # ── EXP6: Arquitecturas ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXP6: DAE – COMPARACIÓN DE ARQUITECTURAS")
    print(f"Ruido={int(NOISE_LEVEL*100)}% | Adam lr={LR} | MaxÉpocas={MAX_EPOCHS} | {len(SEEDS)} Semillas")
    print("=" * 60)
    print(f"Dataset: {X.shape[0]} chars × {X.shape[1]} px")

    res6 = run_exp6(X)
    plot_exp6_convergence(res6)
    plot_exp6_comparison(res6)
    plot_exp6_reconstructions(res6, X, letter_indices=[4, 7, 13])
    plot_exp6_full_reconstruction(res6, X)

    # Entrenar AE Winner (B=2) por separado solo para el espacio latente
    print("\n[+] AE Winner (B=2) – solo para espacio latente")
    loss_fn = BCE()
    best_latent_model, best_latent_bce = None, float('inf')
    for seed in SEEDS:
        np.random.seed(seed)
        m, _, _ = train_arch(create_ae_deep_winner, X)
        X_eval = add_noise(X, NOISE_LEVEL)
        bce = loss_fn.calculate(expected=X, predicted=m.forward(X_eval))
        if bce < best_latent_bce:
            best_latent_bce = bce
            best_latent_model = m
    plot_exp6_latent_space(best_latent_model, X)

    print("\n" + "=" * 60)
    print("RANKING EXP6  (menor BCE = mejor)")
    print("=" * 60)
    ranking6 = sorted(res6.items(), key=lambda kv: np.mean(kv[1]["bce_finals"]))
    for rank, (name, stats) in enumerate(ranking6, 1):
        tag = "  ← GANADORA" if rank == 1 else ""
        print(f"  {rank}. {name:<28s}  {np.mean(stats['epochs']):6.0f} ép  "
              f"BCE={np.mean(stats['bce_finals']):.5f}±{np.std(stats['bce_finals']):.5f}{tag}")

    winner_name    = ranking6[0][0]
    winner_factory = ARCHITECTURES[winner_name]
    winner_str     = ARCH_STRINGS[winner_name]

    # ── EXP6c: Ruido Fijo vs Online ───────────────────────────────────
    print("\n" + "=" * 60)
    print("EXP6c: DAE – RUIDO FIJO vs RUIDO ONLINE")
    print(f"Arquitectura (ganadora EXP6): {winner_str}")
    print("=" * 60)

    res6c = run_exp6c(X, winner_factory)
    plot_exp6c_convergence(res6c)
    plot_exp6c_reconstructions(res6c, X, letter_indices=[1, 7, 13, 19])

    print("\n" + "=" * 60)
    print("RESUMEN EXP6c")
    print("=" * 60)
    for key, label in [("fixed", "Ruido Fijo "), ("online", "Ruido Online")]:
        pxs  = [r["px"]     for r in res6c[key]]
        bces = [r["bce"]    for r in res6c[key]]
        eps  = [r["epochs"] for r in res6c[key]]
        print(f"  {label}: {np.mean(eps):.0f}±{np.std(eps):.0f} ép  "
              f"MaxPx={np.mean(pxs):.1f}±{np.std(pxs):.1f}  "
              f"BCE={np.mean(bces):.4f}±{np.std(bces):.4f}")

    # ── EXP6d: Límites de Ruido ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXP6d: DAE – LÍMITES DE RUIDO (BREAKDOWN POINT)")
    print(f"Arquitectura (ganadora EXP6): {winner_str}  |  Niveles: {[int(n*100) for n in NOISE_LEVELS]}%")
    print("=" * 60)

    res6d = run_exp6d(X, winner_factory)
    plot_exp6d_metrics(res6d)
    plot_exp6d_reconstructions(res6d, X, letter_indices=[1, 7, 13])

    print("\n" + "=" * 60)
    print("RESUMEN EXP6d")
    print("=" * 60)
    for noise in NOISE_LEVELS:
        stats = res6d[noise]
        print(f"  {int(noise*100):2d}%  Épocas={np.mean(stats['epochs']):.0f}±{np.std(stats['epochs']):.0f}  "
              f"MaxPx={np.mean(stats['px_errors']):.1f}±{np.std(stats['px_errors']):.1f}  "
              f"BCE={np.mean(stats['bce_finals']):.5f}±{np.std(stats['bce_finals']):.5f}")


if __name__ == "__main__":
    main()
