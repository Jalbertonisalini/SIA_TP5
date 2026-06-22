"""
EXP6: DAE – Comparación de Arquitecturas

Protocolo:
  - Ruido ONLINE: nuevo patrón de ruido en cada época
  - Evaluación en ruido fijo independiente (seed)
  - Early stopping cuando max_pixel_error_eval == 0
  - Adam lr=0.001, 5 semillas

Arquitecturas:
  AE Winner  35→16→2→16→35    baseline del AE básico
  Base       35→16→10→16→35   mismo shape, bottleneck más grande
  Wide       35→24→10→24→35   más ancho
  Wide-16    35→24→16→24→35   bottleneck aún más grande
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

SERIES_COLORS = {
    "AE Winner (2)":    "#A63636",
    "Base (16-10-16)":  "#595959",
    "Wide (24-10-24)":  "#2E7559",
    "Wide (24-16-24)":  "#1F4E79",
}

# ── Constantes ────────────────────────────────────────────────────────
NOISE_LEVEL   = 0.25
MAX_EPOCHS    = 50000
PLATEAU_CHECK = 1000   # chequear plateau cada N épocas
PLATEAU_EPS   = 1e-4   # mejora mínima en BCE para no considerar plateau
LR            = 0.001
SEEDS         = [42, 100, 800, 1024, 2024]

# ── Arquitecturas ─────────────────────────────────────────────────────

def create_ae_winner() -> Network:
    """35 → 16 → 2 → 16 → 35  (ganadora del AE básico)"""
    ae = Network()
    ae.add(Linear(35, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16,  2)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear( 2, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_base_dae() -> Network:
    """35 → 16 → 10 → 16 → 35"""
    ae = Network()
    ae.add(Linear(35, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 10)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(10, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_wide_dae() -> Network:
    """35 → 24 → 10 → 24 → 35"""
    ae = Network()
    ae.add(Linear(35, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 10)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(10, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_wide16_dae() -> Network:
    """35 → 24 → 16 → 24 → 35"""
    ae = Network()
    ae.add(Linear(35, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 24)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(24, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

ARCHITECTURES = {
    "AE Winner (2)":   create_ae_winner,
    "Base (16-10-16)": create_base_dae,
    "Wide (24-10-24)": create_wide_dae,
    "Wide (24-16-24)": create_wide16_dae,
}

ARCH_FULL_NAMES = {
    "AE Winner (2)":   "AE Winner\n35→16→2→16→35",
    "Base (16-10-16)": "Base\n35→16→10→16→35",
    "Wide (24-10-24)": "Wide\n35→24→10→24→35",
    "Wide (24-16-24)": "Wide-16\n35→24→16→24→35",
}

# ── Helpers ───────────────────────────────────────────────────────────

def add_noise(X: np.ndarray, noise_level: float) -> np.ndarray:
    X_noisy = X.copy()
    for i in range(X_noisy.shape[0]):
        n_flip = int(noise_level * X.shape[1])
        idx = np.random.choice(X.shape[1], n_flip, replace=False)
        X_noisy[i, idx] = 1 - X_noisy[i, idx]
    return X_noisy

def rolling_mean(data: list, window: int = 500) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')

def pixel_error(expected: np.ndarray, predicted: np.ndarray) -> int:
    return int(np.max(np.sum(np.abs(expected.astype(int) -
                                     (predicted >= 0.5).astype(int)), axis=1)))

# ── Entrenamiento ─────────────────────────────────────────────────────

def train_dae(factory_fn, X_clean: np.ndarray) -> tuple:
    """
    Ruido ONLINE: nuevo patrón de ruido cada época.
    Early stopping por plateau: para cuando el BCE de entrenamiento
    no mejora más de PLATEAU_EPS entre ventanas consecutivas de PLATEAU_CHECK épocas.
    """
    ae        = factory_fn()
    loss_fn   = BCE()
    optimizer = Adam(learning_rate=LR)
    history   = []
    converged_epoch = MAX_EPOCHS

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
                converged_epoch = epoch
                break

    return ae, history, converged_epoch

# ── Experimento ───────────────────────────────────────────────────────

def run_comparison(X: np.ndarray) -> dict:
    loss_fn = BCE()
    results = {}

    for name, factory_fn in ARCHITECTURES.items():
        print(f"\n[+] {name}")
        epochs_list   = []
        px_list       = []
        bce_list      = []
        histories     = []
        best_model    = None
        best_bce      = float('inf')
        vis_noisy     = None

        for i, seed in enumerate(SEEDS):
            print(f"   seed {seed:4d} ...", end="", flush=True)

            np.random.seed(seed)
            X_noisy_eval = add_noise(X, NOISE_LEVEL)

            model, history, conv_epoch = train_dae(factory_fn, X)

            eval_pred = model.forward(X_noisy_eval)
            px        = pixel_error(X, eval_pred)
            bce       = loss_fn.calculate(expected=X, predicted=eval_pred)

            converged = conv_epoch < MAX_EPOCHS
            print(f"  {'OK' if converged else 'NO'} ({conv_epoch} épocas)  "
                  f"MaxPx={px}  BCE={bce:.5f}")

            epochs_list.append(conv_epoch)
            px_list.append(px)
            bce_list.append(bce)
            histories.append(history)

            if bce < best_bce:
                best_bce   = bce
                best_model = model
            if i == 0:
                vis_noisy = X_noisy_eval

        print(f"   → Épocas: {np.mean(epochs_list):.0f}±{np.std(epochs_list):.0f} | "
              f"MaxPx: {np.mean(px_list):.1f}±{np.std(px_list):.1f} | "
              f"BCE: {np.mean(bce_list):.5f}±{np.std(bce_list):.5f}")

        results[name] = {
            "epochs":     epochs_list,
            "px_errors":  px_list,
            "bce_finals": bce_list,
            "histories":  histories,
            "best_model": best_model,
            "vis_noisy":  vis_noisy,
        }

    return results

# ── Plots ─────────────────────────────────────────────────────────────

def plot_convergence(results: dict, filename: str, window: int = 500) -> None:
    """Curvas de convergencia: BCE de entrenamiento (rolling avg) por arquitectura."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_facecolor(FACECOLOR)
    ax.set_axisbelow(True)

    for name, stats in results.items():
        color     = SERIES_COLORS[name]
        histories = stats["histories"]
        max_len   = max(len(h) for h in histories)
        # Pad las historias más cortas con su último valor
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
        mean_raw = np.mean(padded, axis=0)
        std_raw  = np.std(padded, axis=0)

        mean_s = rolling_mean(mean_raw, window)
        std_s  = rolling_mean(std_raw,  window)
        x      = np.arange(window - 1, window - 1 + len(mean_s))

        ax.plot(x, mean_s, label=name, color=color, linewidth=2)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                        color=color, alpha=0.12, edgecolor='none')

    ax.set_title(
        f"DAE – Convergencia del Entrenamiento por Arquitectura\n"
        f"Ruido online {int(NOISE_LEVEL*100)}%  |  Adam lr={LR}  |  {len(SEEDS)} Semillas  "
        f"(rolling avg {window} épocas)",
        color=TITLE_COLOR, pad=12, fontsize=13
    )
    ax.set_xlabel("Épocas", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("BCE Loss (entrenamiento)", color=TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    )
    ax.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7, color=GRID_COLOR)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] {filename}")


def plot_comparison(results: dict, filename: str) -> None:
    names  = list(results.keys())
    colors = [SERIES_COLORS[n] for n in names]
    x_pos  = np.arange(len(names))

    ep_means  = [np.mean(results[n]["epochs"])     for n in names]
    ep_stds   = [np.std(results[n]["epochs"])      for n in names]
    bce_means = [np.mean(results[n]["bce_finals"]) for n in names]
    bce_stds  = [np.std(results[n]["bce_finals"])  for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.patch.set_facecolor(FACECOLOR)

    short_names = ["AE Winner\n(2)", "Base\n(16-10-16)",
                   "Wide\n(24-10-24)", "Wide\n(24-16-24)"]

    # ── Épocas ─────────────────────────────────────────────────────
    ax1.set_facecolor(FACECOLOR)
    ax1.set_axisbelow(True)
    ax1.set_title(
        f"DAE – Comparación de Arquitecturas\n"
        f"Ruido={int(NOISE_LEVEL*100)}%  |  Adam lr={LR}  |  {len(SEEDS)} Semillas",
        color=TITLE_COLOR, pad=12, fontsize=13
    )
    ax1.set_ylabel("Épocas hasta convergencia", color=TEXT_COLOR, fontsize=11)
    bars1 = ax1.bar(x_pos, ep_means, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.2, zorder=3)
    ax1.errorbar(x_pos, ep_means, yerr=ep_stds, fmt='none',
                 color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    ax1.axhline(y=MAX_EPOCHS, color='gray', linestyle='--', alpha=0.6,
                linewidth=1.2, label=f"Límite ({MAX_EPOCHS} épocas — no convergió)")
    for bar, m, s in zip(bars1, ep_means, ep_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 100,
                 f"{m:.0f}", ha='center', fontsize=9, color=TEXT_COLOR)
    ax1.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6, color=GRID_COLOR, axis='y', zorder=0)

    # ── BCE Loss ───────────────────────────────────────────────────
    ax2.set_facecolor(FACECOLOR)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("BCE Loss (evaluación)", color=TEXT_COLOR, fontsize=11)
    bars2 = ax2.bar(x_pos, bce_means, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.2, zorder=3)
    ax2.errorbar(x_pos, bce_means, yerr=bce_stds, fmt='none',
                 color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    for bar, m, s in zip(bars2, bce_means, bce_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.002,
                 f"{m:.4f}", ha='center', fontsize=9, color=TEXT_COLOR)
    ax2.grid(True, linestyle=':', alpha=0.6, color=GRID_COLOR, axis='y', zorder=0)

    for ax in [ax1, ax2]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_names, fontsize=10, color=TEXT_COLOR)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] {filename}")


def plot_reconstructions(results: dict, X: np.ndarray,
                          letter_indices: list, filename: str) -> None:
    names  = list(results.keys())
    n_cols = 1 + 2 * len(letter_indices)

    fig, axes = plt.subplots(len(names) + 1, n_cols,
                              figsize=(2.8 * n_cols, 2.4 * (len(names) + 1)))
    fig.patch.set_facecolor(FACECOLOR)

    def show(ax, img, title=None, border_color=None):
        ax.imshow(img.reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=8, color=TEXT_COLOR)
        if border_color:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
                spine.set_visible(True)

    # Fila 0: originales + ruidosas
    axes[0, 0].axis('off')
    vis_noisy_ref = results[names[0]]["vis_noisy"]
    for j, idx in enumerate(letter_indices):
        show(axes[0, 1 + j * 2], X[idx],               title="Original")
        show(axes[0, 2 + j * 2], vis_noisy_ref[idx],   title=f"Ruidosa {int(NOISE_LEVEL*100)}%")

    # Filas: reconstrucciones por arquitectura
    for row, name in enumerate(names, start=1):
        axes[row, 0].axis('off')
        axes[row, 0].text(0.5, 0.5, ARCH_FULL_NAMES[name],
                          ha='center', va='center', fontsize=8,
                          color=SERIES_COLORS[name],
                          transform=axes[row, 0].transAxes, fontweight='bold')
        model     = results[name]["best_model"]
        vis_noisy = results[name]["vis_noisy"]
        pred      = model.forward(vis_noisy)

        for j, idx in enumerate(letter_indices):
            axes[row, 1 + j * 2].axis('off')
            show(axes[row, 2 + j * 2], pred[idx],
                 border_color=SERIES_COLORS[name])

    fig.suptitle("DAE – Reconstrucciones por Arquitectura",
                 color=TITLE_COLOR, fontsize=13, y=1.01)
    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] {filename}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXP6: DAE – COMPARACIÓN DE ARQUITECTURAS")
    print("=" * 60)

    X = FontLoader.load_and_flatten('src/data/font.h')
    print(f"Dataset: {X.shape[0]} chars × {X.shape[1]} px\n")

    results = run_comparison(X)

    plot_convergence(results,    "exp6_dae_convergence.png")
    plot_comparison(results,    "exp6_dae_comparison.png")
    plot_reconstructions(results, X, letter_indices=[1, 7, 13],
                          filename="exp6_dae_reconstructions.png")

    print("\n" + "=" * 60)
    print("RANKING  (menor BCE en eval = mejor)")
    print("=" * 60)
    ranking = sorted(results.items(),
                     key=lambda kv: np.mean(kv[1]["bce_finals"]))
    for rank, (name, stats) in enumerate(ranking, 1):
        m_ep  = np.mean(stats["epochs"])
        m_bce = np.mean(stats["bce_finals"])
        s_bce = np.std(stats["bce_finals"])
        tag   = "  ← GANADORA" if rank == 1 else ""
        print(f"  {rank}. {name:<28s}  {m_ep:6.0f} ép  "
              f"BCE={m_bce:.5f}±{s_bce:.5f}{tag}")

    winner = ranking[0][0]
    print(f"\n→ Actualizar WINNING_ARCH_NAME en exp2_noise_limits.py: \"{winner}\"")


if __name__ == "__main__":
    main()
