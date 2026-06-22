"""
EXP6b: DAE – Barrido del Cuello de Botella (Bottleneck Sweep)

Fija la arquitectura externa del ganador (35→32→16→B→16→32→35)
y varía el tamaño del cuello de botella B ∈ {2, 4, 6, 8, 10}.

Protocolo: ruido online (25%), evaluación en ruido fresco, 10 semillas, Adam lr=0.01, plateau early stopping.
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.network import Network
from core.losses import BCE
from core.optimizers import Adam
from utils.data_loader import FontLoader

# =====================================================================
# ESTILO
# =====================================================================

FACECOLOR   = "#FFFFFF"
GRID_COLOR  = "#E6E6E6"
TITLE_COLOR = "#000000"
TEXT_COLOR  = "#333333"

# =====================================================================
# CONSTANTES
# =====================================================================

NOISE_LEVEL   = 0.25
MAX_EPOCHS    = 50000
PLATEAU_CHECK = 1000
PLATEAU_EPS   = 1e-4
LR            = 0.01
SEEDS         = [42, 100, 200, 500, 800, 1024, 1337, 2024, 3000, 9999]

# =====================================================================
# HELPERS
# =====================================================================

def add_noise(X: np.ndarray, noise_level: float) -> np.ndarray:
    X_noisy = X.copy()
    for i in range(X_noisy.shape[0]):
        num_flips = int(noise_level * X.shape[1])
        indices = np.random.choice(X.shape[1], num_flips, replace=False)
        X_noisy[i, indices] = 1 - X_noisy[i, indices]
    return X_noisy

def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> float:
    binary_predicted = (predicted >= 0.5).astype(int)
    differences = np.abs(expected.astype(int) - binary_predicted)
    return float(np.max(np.sum(differences, axis=1)))

# =====================================================================
# FACTORY: estructura fija, bottleneck variable
# =====================================================================

def create_deep_with_bottleneck(b: int) -> Network:
    """35 → 32 → 16 → B → 16 → 32 → 35  con B variable."""
    ae = Network()
    ae.add(Linear(35, 32));  ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 16));  ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, b));   ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(b, 16));   ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 32));  ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 35));  ae.add(ActivationLayer(Sigmoid()))
    return ae

# =====================================================================
# ENTRENAMIENTO (ruido online, mismo protocolo que exp6)
# =====================================================================

def train_dae(b: int, X_clean: np.ndarray) -> tuple:
    ae        = create_deep_with_bottleneck(b)
    loss_fn   = BCE()
    optimizer = Adam(learning_rate=LR)
    history   = []
    converged = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        X_noisy   = add_noise(X_clean, NOISE_LEVEL)
        predicted = ae.forward(X_noisy)
        loss      = loss_fn.calculate(expected=X_clean, predicted=predicted)
        history.append(loss)
        grad      = loss_fn.derivative(expected=X_clean, predicted=predicted)
        ae.backward(grad, optimizer)

        if epoch >= 2 * PLATEAU_CHECK and epoch % PLATEAU_CHECK == 0:
            recent = np.mean(history[-PLATEAU_CHECK:])
            prev   = np.mean(history[-2 * PLATEAU_CHECK:-PLATEAU_CHECK])
            if prev - recent < PLATEAU_EPS:
                converged = epoch
                break

    return ae, history, converged

# =====================================================================
# EXPERIMENTO
# =====================================================================

def run_sweep(X: np.ndarray, bottlenecks: list) -> dict:
    loss_fn = BCE()
    results = {}

    for b in bottlenecks:
        name = f"B={b}"
        print(f"\n[+] Bottleneck {name}  (35→32→16→{b}→16→32→35)")
        histories    = []
        pixel_errors = []
        bce_finals   = []
        best_model   = None
        best_loss    = float('inf')
        vis_noisy    = None

        for seed in SEEDS:
            print(f"   Seed {seed:4d} ...", end="", flush=True)
            np.random.seed(seed)
            model, history, converged = train_dae(b, X)

            np.random.seed(seed + 50000)
            X_noisy_test = add_noise(X, NOISE_LEVEL)

            pred       = model.forward(X_noisy_test)
            px_err     = evaluate_pixel_diff(X, pred)
            final_loss = loss_fn.calculate(expected=X, predicted=pred)

            tag = "OK" if converged < MAX_EPOCHS else "NO"
            print(f"  {tag} ({converged} épocas)  MaxPx={px_err:.0f}  BCE={final_loss:.5f}")
            histories.append(history)
            pixel_errors.append(px_err)
            bce_finals.append(final_loss)

            if final_loss < best_loss:
                best_loss  = final_loss
                best_model = model
                vis_noisy  = X_noisy_test

        max_len   = max(len(h) for h in histories)
        histories = [h + [h[-1]] * (max_len - len(h)) for h in histories]

        results[b] = {
            "name":        name,
            "histories":   histories,
            "pixel_errors": pixel_errors,
            "bce_finals":  bce_finals,
            "best_model":  best_model,
            "vis_noisy":   vis_noisy,
        }

        mean_px  = np.mean(pixel_errors)
        std_px   = np.std(pixel_errors)
        mean_bce = np.mean(bce_finals)
        std_bce  = np.std(bce_finals)
        print(f"   → MaxPx: {mean_px:.1f}±{std_px:.1f}  |  BCE: {mean_bce:.5f}±{std_bce:.5f}")

    return results

# =====================================================================
# PLOTS
# =====================================================================

def _bottleneck_colors(bottlenecks: list):
    """Gradiente de colores: rojo (cuello más chico) → azul (más grande)."""
    cmap = cm.get_cmap('RdYlBu')
    n = len(bottlenecks)
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_convergence(results: dict, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_facecolor(FACECOLOR)
    ax.set_axisbelow(True)

    bottlenecks = list(results.keys())
    colors = _bottleneck_colors(bottlenecks)

    for b, color in zip(bottlenecks, colors):
        mat  = np.array(results[b]["histories"])
        mean = np.mean(mat, axis=0)
        std  = np.std(mat, axis=0)
        x    = np.arange(len(mean))
        ax.plot(x, mean, label=f"B={b}", color=color, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, edgecolor='none')

    ax.set_title("DAE – Convergencia: Barrido del Cuello de Botella (35→32→16→B→16→32→35)",
                 color=TITLE_COLOR, pad=15, fontsize=13)
    ax.set_xlabel("Épocas", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Loss de Entrenamiento (BCE)", color=TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x)))
    )
    ax.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              title="Bottleneck (B)")
    ax.grid(True, linestyle=':', alpha=0.8, color=GRID_COLOR)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    fig.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Guardado: outputs/{filename}")


def plot_metrics_vs_bottleneck(results: dict, noise_level: float, filename: str) -> None:
    """
    Panel superior:  Max píxeles incorrectos vs B
    Panel inferior:  BCE Loss final vs B
    """
    bottlenecks = list(results.keys())
    colors = _bottleneck_colors(bottlenecks)

    px_means  = [np.mean(results[b]["pixel_errors"]) for b in bottlenecks]
    px_stds   = [np.std(results[b]["pixel_errors"])  for b in bottlenecks]
    bce_means = [np.mean(results[b]["bce_finals"])   for b in bottlenecks]
    bce_stds  = [np.std(results[b]["bce_finals"])    for b in bottlenecks]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor(FACECOLOR)

    x_pos = np.arange(len(bottlenecks))

    # ── Panel superior: Píxeles ─────────────────────────────────────
    ax1.set_facecolor(FACECOLOR)
    ax1.set_axisbelow(True)
    ax1.set_title(
        f"DAE – Efecto del Tamaño de Cuello de Botella\n"
        f"Estructura fija: 35→32→16→B→16→32→35  |  Ruido={int(noise_level*100)}%  |  "
        f"Adam lr=0.01  |  10 Semillas",
        color=TITLE_COLOR, pad=12, fontsize=12
    )
    ax1.set_ylabel("Max Píxeles Incorrectos", color=TEXT_COLOR, fontsize=11)
    bars1 = ax1.bar(x_pos, px_means, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.2, zorder=3)
    ax1.errorbar(x_pos, px_means, yerr=px_stds, fmt='none',
                 color=TEXT_COLOR, capsize=7, linewidth=1.8, zorder=4)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7,
                linewidth=1.2, label="Umbral viabilidad (≤ 1 px)", zorder=2)
    for bar, m, s in zip(bars1, px_means, px_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.3,
                 f"{m:.1f}±{s:.1f}", ha='center', fontsize=9, color=TEXT_COLOR)
    ax1.legend(frameon=True, facecolor=FACECOLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax1.grid(True, linestyle=':', alpha=0.6, color=GRID_COLOR, axis='y', zorder=0)

    # ── Panel inferior: BCE Loss ────────────────────────────────────
    ax2.set_facecolor(FACECOLOR)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("Tamaño del Cuello de Botella (B)", color=TEXT_COLOR, fontsize=11)
    ax2.set_ylabel("BCE Loss Final", color=TEXT_COLOR, fontsize=11)
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
        ax.set_xticklabels([f"B={b}" for b in bottlenecks], fontsize=11, color=TEXT_COLOR)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Guardado: outputs/{filename}")


def plot_reconstruction_grid(results: dict, X: np.ndarray,
                               letter_indices: list, noise_level: float,
                               filename: str) -> None:
    """
    Grid: filas = [Original, Ruidosa, B=2, B=4, …], columnas = letras elegidas.
    Todos los mejores modelos se evalúan sobre el mismo ruido de la primera seed.
    """
    bottlenecks = list(results.keys())
    n_cols = len(letter_indices)
    n_rows = 2 + len(bottlenecks)
    colors = _bottleneck_colors(bottlenecks)

    # Usamos el vis_noisy del primer bottleneck (misma seed=42)
    vis_noisy = results[bottlenecks[0]]["vis_noisy"]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.6 * n_cols, 2.4 * n_rows))
    fig.patch.set_facecolor(FACECOLOR)

    row_labels = ["Original", f"Ruidosa\n({int(noise_level*100)}%)"] + \
                 [f"B={b}" for b in bottlenecks]

    for col, idx in enumerate(letter_indices):
        axes[0, col].imshow(X[idx].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[0, col].axis('off')

        axes[1, col].imshow(vis_noisy[idx].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        axes[1, col].axis('off')

        for row_offset, b in enumerate(bottlenecks):
            pred = results[b]["best_model"].forward(vis_noisy)
            axes[2 + row_offset, col].imshow(
                pred[idx].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1
            )
            axes[2 + row_offset, col].axis('off')
            # Borde de color según bottleneck
            for spine in axes[2 + row_offset, col].spines.values():
                spine.set_edgecolor(colors[row_offset])
                spine.set_linewidth(2)
                spine.set_visible(True)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, color=TEXT_COLOR, fontsize=8.5,
                                 rotation=0, labelpad=70, va='center', ha='right')

    fig.suptitle("DAE – Reconstrucción por Tamaño de Cuello de Botella",
                 color=TITLE_COLOR, fontsize=13, y=1.01)
    fig.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    fig.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Guardado: outputs/{filename}")

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 62)
    print("EXP6b: DAE – BARRIDO DEL CUELLO DE BOTELLA")
    print("Estructura: 35 → 32 → 16 → B → 16 → 32 → 35")
    print("=" * 62)

    X = FontLoader.load_and_flatten('src/data/font.h')
    print(f"Dataset: {X.shape[0]} caracteres × {X.shape[1]} píxeles\n")

    BOTTLENECKS    = [2, 4, 6, 8, 10]
    LETTER_INDICES = [1, 7, 13]

    print(f"Bottlenecks a probar: {BOTTLENECKS}")
    print(f"Ruido={int(NOISE_LEVEL*100)}% | MaxÉpocas={MAX_EPOCHS} | Adam lr={LR} | "
          f"Semillas={SEEDS}\n")

    results = run_sweep(X, BOTTLENECKS)

    plot_convergence(results, "exp6b_bottleneck_convergence.png")
    plot_metrics_vs_bottleneck(results, NOISE_LEVEL, "exp6b_bottleneck_metrics.png")
    plot_reconstruction_grid(results, X, LETTER_INDICES, NOISE_LEVEL,
                              "exp6b_bottleneck_reconstructions.png")

    # ── Ranking ────────────────────────────────────────────────────
    ranking = sorted(results.items(), key=lambda kv: np.mean(kv[1]["bce_finals"]))

    print("\n" + "=" * 62)
    print("RANKING  (menor BCE Loss = mejor reconstrucción)")
    print("=" * 62)
    for rank, (b, stats) in enumerate(ranking, 1):
        mean_bce = np.mean(stats["bce_finals"])
        std_bce  = np.std(stats["bce_finals"])
        mean_px  = np.mean(stats["pixel_errors"])
        tag      = "  ← ÓPTIMO" if rank == 1 else ""
        print(f"  {rank}. B={b:<4d}  BCE={mean_bce:.5f}±{std_bce:.5f}  "
              f"MaxPx={mean_px:.1f}{tag}")

    best_b = ranking[0][0]
    print(f"\n→ Bottleneck óptimo: B={best_b}")
    print(f"  Actualizar exp2_noise_limits.py si B≠10.")


if __name__ == "__main__":
    main()
