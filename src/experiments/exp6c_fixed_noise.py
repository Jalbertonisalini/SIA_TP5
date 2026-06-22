"""
EXP6c: DAE – Ruido Fijo vs Ruido Online

Compara el mismo modelo (Wide 24-10-24) entrenado con:
  - Ruido FIJO: X_noisy generado UNA vez antes de entrenar
  - Ruido ONLINE: X_noisy nuevo en cada época (exp6)

Muestra por qué el ruido online es necesario para generalizar.
Evaluación siempre sobre ruido independiente (seed diferente).
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.network import Network
from core.losses import BCE
from core.optimizers import Adam
from utils.data_loader import FontLoader

# ── Estilo ────────────────────────────────────────────────────────────
FACECOLOR   = "#FFFFFF"
GRID_COLOR  = "#E6E6E6"
TITLE_COLOR = "#000000"
TEXT_COLOR  = "#333333"
COLOR_FIXED  = "#A63636"
COLOR_ONLINE = "#1F4E79"

# ── Constantes ────────────────────────────────────────────────────────
NOISE_LEVEL   = 0.25
MAX_EPOCHS    = 50000
PLATEAU_CHECK = 1000
PLATEAU_EPS   = 1e-4
LR            = 0.01
SEEDS         = [42, 100, 200, 500, 800, 1024, 1337, 2024, 3000, 9999]

# ── Arquitectura (ganadora del exp6) ──────────────────────────────────

def create_dae() -> Network:
    """Wide: 35 → 24 → 10 → 24 → 35"""
    ae = Network()
    ae.add(Linear(35, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 10)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(10, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

# ── Helpers ───────────────────────────────────────────────────────────

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

def rolling_mean(data, window: int = 300) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')

# ── Entrenamiento ─────────────────────────────────────────────────────

def train_fixed(X_clean: np.ndarray, X_noisy_train: np.ndarray) -> tuple:
    """Ruido FIJO: siempre el mismo X_noisy_train."""
    ae        = create_dae()
    loss_fn   = BCE()
    optimizer = Adam(learning_rate=LR)
    history   = []
    converged = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        pred = ae.forward(X_noisy_train)
        loss = loss_fn.calculate(expected=X_clean, predicted=pred)
        history.append(loss)
        grad = loss_fn.derivative(expected=X_clean, predicted=pred)
        ae.backward(grad, optimizer)

        if epoch >= 2 * PLATEAU_CHECK and epoch % PLATEAU_CHECK == 0:
            recent = np.mean(history[-PLATEAU_CHECK:])
            prev   = np.mean(history[-2 * PLATEAU_CHECK:-PLATEAU_CHECK])
            if prev - recent < PLATEAU_EPS:
                converged = epoch
                break

    return ae, history, converged


def train_online(X_clean: np.ndarray) -> tuple:
    """Ruido ONLINE: nuevo patrón de ruido en cada época."""
    ae        = create_dae()
    loss_fn   = BCE()
    optimizer = Adam(learning_rate=LR)
    history   = []
    converged = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        X_noisy = add_noise(X_clean, NOISE_LEVEL)
        pred    = ae.forward(X_noisy)
        loss    = loss_fn.calculate(expected=X_clean, predicted=pred)
        history.append(loss)
        grad    = loss_fn.derivative(expected=X_clean, predicted=pred)
        ae.backward(grad, optimizer)

        if epoch >= 2 * PLATEAU_CHECK and epoch % PLATEAU_CHECK == 0:
            recent = np.mean(history[-PLATEAU_CHECK:])
            prev   = np.mean(history[-2 * PLATEAU_CHECK:-PLATEAU_CHECK])
            if prev - recent < PLATEAU_EPS:
                converged = epoch
                break

    return ae, history, converged

# ── Experimento ───────────────────────────────────────────────────────

def run(X: np.ndarray) -> dict:
    loss_fn = BCE()
    results = {"fixed": [], "online": []}

    for seed in SEEDS:
        print(f"\n  seed {seed} ...", end="", flush=True)

        # Ruido de entrenamiento fijo (seed de entrenamiento)
        np.random.seed(seed)
        X_noisy_train = add_noise(X, NOISE_LEVEL)

        # Ruido de evaluación independiente
        np.random.seed(seed + 50000)
        X_noisy_eval = add_noise(X, NOISE_LEVEL)

        # ── Fijo ──────────────────────────────────────────────────
        np.random.seed(seed)   # reinicia RNG para mismos pesos iniciales
        model_f, hist_f, ep_f = train_fixed(X, X_noisy_train)
        pred_f  = model_f.forward(X_noisy_eval)
        px_f    = pixel_error(X, pred_f)
        bce_f   = loss_fn.calculate(expected=X, predicted=pred_f)

        # ── Online ────────────────────────────────────────────────
        np.random.seed(seed)   # mismos pesos iniciales para comparación justa
        model_o, hist_o, ep_o = train_online(X)
        pred_o  = model_o.forward(X_noisy_eval)
        px_o    = pixel_error(X, pred_o)
        bce_o   = loss_fn.calculate(expected=X, predicted=pred_o)

        print(f"  Fijo: {ep_f}ép MaxPx={px_f} BCE={bce_f:.4f} | "
              f"Online: {ep_o}ép MaxPx={px_o} BCE={bce_o:.4f}")

        results["fixed"].append({
            "history": hist_f, "epochs": ep_f,
            "px": px_f, "bce": bce_f,
            "model": model_f, "vis_noisy": X_noisy_eval,
            "train_noisy": X_noisy_train,
        })
        results["online"].append({
            "history": hist_o, "epochs": ep_o,
            "px": px_o, "bce": bce_o,
            "model": model_o, "vis_noisy": X_noisy_eval,
        })

    return results

# ── Plots ─────────────────────────────────────────────────────────────

def plot_convergence(results: dict, filename: str) -> None:
    """Curvas de entrenamiento: fijo vs online."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_facecolor(FACECOLOR)
    ax.set_axisbelow(True)
    window = 300

    for label, color, key in [("Ruido Fijo", COLOR_FIXED, "fixed"),
                               ("Ruido Online", COLOR_ONLINE, "online")]:
        histories = [r["history"] for r in results[key]]
        max_len   = max(len(h) for h in histories)
        padded    = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
        mean_s    = rolling_mean(np.mean(padded, axis=0), window)
        std_s     = rolling_mean(np.std(padded, axis=0), window)
        x         = np.arange(window - 1, window - 1 + len(mean_s))

        ax.plot(x, mean_s, label=label, color=color, linewidth=2.2)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                        color=color, alpha=0.12, edgecolor='none')

    ax.set_title(
        f"DAE – Ruido Fijo vs Ruido Online\n"
        f"Arquitectura: Wide 35→24→10→24→35  |  Ruido={int(NOISE_LEVEL*100)}%  |  "
        f"Adam lr={LR}  |  {len(SEEDS)} Semillas",
        color=TITLE_COLOR, pad=12, fontsize=13
    )
    ax.set_xlabel("Épocas", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("BCE Loss (entrenamiento)", color=TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    )
    ax.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7, color=GRID_COLOR)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    os.makedirs('outputs/exp6c', exist_ok=True)
    fig.savefig(f'outputs/exp6c/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] {filename}")


def plot_reconstructions(results: dict, X: np.ndarray,
                          letter_indices: list, filename: str) -> None:
    """
    Grid: columnas = letras, filas = [Original, Ruidosa, Fijo, Online]
    Etiquetas de fila en panel lateral izquierdo dedicado.
    """
    n_letters = len(letter_indices)
    n_cols    = n_letters + 1       # +1 para columna de etiquetas
    n_rows    = 4

    row_labels = [
        "Original",
        f"Ruidosa  ({int(NOISE_LEVEL*100)}%)",
        "Ruido Fijo\n(entrenado con ruido fijo,\nevaluado con ruido nuevo)",
        "Ruido Online\n(entrenado con ruido online,\nevaluado con ruido nuevo)",
    ]
    row_colors = [TEXT_COLOR, TEXT_COLOR, COLOR_FIXED, COLOR_ONLINE]

    # Seed del medio por BCE online (más representativa, comparación justa)
    sorted_idx = np.argsort([r["bce"] for r in results["online"]])
    median_idx = int(sorted_idx[len(sorted_idx) // 2])
    median_seed = SEEDS[median_idx]
    model_f   = results["fixed"][median_idx]["model"]
    model_o   = results["online"][median_idx]["model"]
    vis_noisy = results["online"][median_idx]["vis_noisy"]
    print(f"  Visualizando seed {median_seed} (mediana BCE online)")

    pred_f = model_f.forward(vis_noisy)
    pred_o = model_o.forward(vis_noisy)

    imgs_per_row = [
        [X[idx].reshape(7, 5)          for idx in letter_indices],
        [vis_noisy[idx].reshape(7, 5)  for idx in letter_indices],
        [pred_f[idx].reshape(7, 5)     for idx in letter_indices],
        [pred_o[idx].reshape(7, 5)     for idx in letter_indices],
    ]

    col_widths = [2.5] + [2.2] * n_letters
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(sum(col_widths), 2.6 * n_rows),
                              gridspec_kw={'width_ratios': col_widths})
    fig.patch.set_facecolor(FACECOLOR)

    for row, (imgs, label, color) in enumerate(zip(imgs_per_row, row_labels, row_colors)):
        # Columna 0: etiqueta de fila
        axes[row, 0].axis('off')
        axes[row, 0].text(0.95, 0.5, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=9, color=color,
                          ha='right', va='center',
                          fontweight='bold' if row >= 2 else 'normal',
                          linespacing=1.4)

        for col, (img, idx) in enumerate(zip(imgs, letter_indices), start=1):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray_r', vmin=0, vmax=1)
            ax.axis('off')

            if row == 0:
                char = chr(ord('a') + idx)
                ax.set_title(f"'{char}'", color=TITLE_COLOR,
                             fontsize=12, fontweight='bold')

            if row == 2:
                px = pixel_error(X[[idx]], pred_f[[idx]])
                border = "#A63636" if px > 1 else "#2E7559"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)
            if row == 3:
                px = pixel_error(X[[idx]], pred_o[[idx]])
                border = "#A63636" if px > 1 else "#2E7559"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

    fig.suptitle(
        "DAE – Comparación Ruido Fijo vs Ruido Online\n"
        "Verde = ≤1 px error  |  Rojo = >1 px error",
        color=TITLE_COLOR, fontsize=13, y=1.01
    )
    fig.tight_layout()
    os.makedirs('outputs/exp6c', exist_ok=True)
    fig.savefig(f'outputs/exp6c/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] {filename}")

# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXP6c: DAE – RUIDO FIJO vs RUIDO ONLINE")
    print(f"Arquitectura: Wide 35→24→10→24→35")
    print(f"Adam lr={LR} | MaxÉpocas={MAX_EPOCHS} | Semillas={SEEDS}")
    print("=" * 60)

    X = FontLoader.load_and_flatten('src/data/font.h')
    print(f"Dataset: {X.shape[0]} chars × {X.shape[1]} px\n")

    results = run(X)

    plot_convergence(results,     "exp6c_convergence.png")
    plot_reconstructions(results, X, letter_indices=[1, 7, 13, 19],
                          filename="exp6c_reconstructions.png")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for key, label in [("fixed", "Ruido Fijo "), ("online", "Ruido Online")]:
        pxs  = [r["px"]  for r in results[key]]
        bces = [r["bce"] for r in results[key]]
        eps  = [r["epochs"] for r in results[key]]
        print(f"  {label}: {np.mean(eps):.0f}±{np.std(eps):.0f} ép  "
              f"MaxPx={np.mean(pxs):.1f}±{np.std(pxs):.1f}  "
              f"BCE={np.mean(bces):.4f}±{np.std(bces):.4f}")


if __name__ == "__main__":
    main()
