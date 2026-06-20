import numpy as np
import os

# Importamos todas las piezas que construimos
from core.activations import Sigmoid, Tanh
from core.layers import Linear, ActivationLayer
from core.network import Network
from core.losses import MSE, BCE
from utils.data_loader import FontLoader
from utils.plotter import Plotter
import matplotlib.pyplot as plt
from core.optimizers import SGD, Adam


# Aseguramos que exista la carpeta outputs
os.makedirs('outputs', exist_ok=True)

def evaluate_pixel_diff(expected: np.ndarray, predicted: np.ndarray) -> int:
    """
    Cuenta cuántos píxeles difieren entre la entrada y la reconstrucción.
    Como la salida de la Sigmoide es continua (0 a 1), umbralizamos a 0.5.
    """
    binary_predicted = (predicted >= 0.5).astype(int)
    binary_expected = expected.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    errors_per_letter = np.sum(differences, axis=1)
    return np.max(errors_per_letter)


def add_noise(X: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Invierte los píxeles de la entrada aleatoriamente según el nivel de ruido."""
    X_noisy = X.copy()
    for i in range(X_noisy.shape[0]):
        num_flips = int(noise_level * X.shape[1])
        indices = np.random.choice(X.shape[1], num_flips, replace=False)
        X_noisy[i, indices] = 1 - X_noisy[i, indices]
    return X_noisy

# =====================================================================
# FACTORY: ARQUITECTURAS DEL AUTOENCODER
# =====================================================================

def create_base_ae() -> Network:
    """Arquitectura Base (Control): 35 -> 16 -> 2 -> 16 -> 35"""
    ae = Network()
    ae.add(Linear(35, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 2));  ae.add(ActivationLayer(Tanh())) 
    ae.add(Linear(2, 16));  ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_shallow_ae() -> Network:
    """Arquitectura Superficial: 35 -> 2 -> 35 (Sin capas ocultas)"""
    ae = Network()
    ae.add(Linear(35, 2));  ae.add(ActivationLayer(Tanh())) 
    ae.add(Linear(2, 35));  ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_narrow_ae() -> Network:
    """Arquitectura Angosta: 35 -> 4 -> 2 -> 4 -> 35"""
    ae = Network()
    ae.add(Linear(35, 4));  ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(4, 2));   ae.add(ActivationLayer(Tanh())) 
    ae.add(Linear(2, 4));   ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(4, 35));  ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_deep_wide_ae() -> Network:
    """Arquitectura Profunda y Ancha: 35 -> 32 -> 16 -> 2 -> 16 -> 32 -> 35"""
    ae = Network()
    ae.add(Linear(35, 32)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 16)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 2));  ae.add(ActivationLayer(Tanh())) 
    ae.add(Linear(2, 16));  ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(16, 32)); ae.add(ActivationLayer(Tanh()))
    ae.add(Linear(32, 35)); ae.add(ActivationLayer(Sigmoid()))
    return ae

def create_denoising_autoencoder() -> Network:
    """Instancia la arquitectura para el Denoising Autoencoder 35 -> 24 -> 10 -> 24 -> 35"""
    autoencoder = Network()
    
    # --- ENCODER ---
    autoencoder.add(Linear(input_size=35, output_size=24))
    autoencoder.add(ActivationLayer(Tanh()))
    
    # CUELLO DE BOTELLA
    autoencoder.add(Linear(input_size=24, output_size=10))
    autoencoder.add(ActivationLayer(Tanh())) 
    
    # --- DECODER ---
    autoencoder.add(Linear(input_size=10, output_size=24))
    autoencoder.add(ActivationLayer(Tanh()))
    
    autoencoder.add(Linear(input_size=24, output_size=35))
    autoencoder.add(ActivationLayer(Sigmoid()))
    
    return autoencoder

# =====================================================================
# MOTOR DE ENTRENAMIENTO
# =====================================================================

def train_autoencoder(autoencoder: Network, X: np.ndarray, epochs: int = 15000, learning_rate: float = 0.1, optimizer=None) -> tuple[Network, list]:
    """Entrena la red inyectada y devuelve el modelo y su historial de loss."""
    
    # Si no se pasa un optimizador, usamos SGD con el learning rate dado por parámetro
    if optimizer is None:
        optimizer = SGD(learning_rate=learning_rate)
        
    loss_function = BCE()
    loss_history = []
    
    for epoch in range(epochs):
        predicted = autoencoder.forward(X)
        loss = loss_function.calculate(expected=X, predicted=predicted)
        loss_history.append(loss)
        
        max_incorrect_pixels = evaluate_pixel_diff(X, predicted)
        
        initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
        
        # CAMBIO: Pasamos el objeto optimizer en lugar del flotante
        autoencoder.backward(initial_gradient, optimizer)
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:05d} | Loss (BCE): {loss:.4f} | Max incorrect pixels: {max_incorrect_pixels}")
            
        if max_incorrect_pixels <= 1:
            print(f"  Objective achieved! Early convergence at epoch {epoch}. Max error <= 1.")
            break
            
    return autoencoder, loss_history

def train_denoising_autoencoder(X_clean: np.ndarray, X_noisy: np.ndarray, epochs: int = 15000, learning_rate: float = 0.1) -> Network:
    """Entrena la red DAE pasando ruido pero calculando Loss con la original."""
    autoencoder = create_denoising_autoencoder()
    optimizer = SGD(learning_rate=learning_rate)
    loss_function = BCE()
    
    for epoch in range(epochs):
        predicted = autoencoder.forward(X_noisy)
        # La pérdida se calcula contra la versión LIMPIA
        loss = loss_function.calculate(expected=X_clean, predicted=predicted)
        max_incorrect_pixels = evaluate_pixel_diff(X_clean, predicted)
        
        initial_gradient = loss_function.derivative(expected=X_clean, predicted=predicted)
        autoencoder.backward(initial_gradient, optimizer)
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:05d} | Loss (BCE): {loss:.4f} | Max incorrect pixels vs clean: {max_incorrect_pixels}")
            
        if max_incorrect_pixels <= 1:
            print(f"  Objective achieved! Early convergence at epoch {epoch}. Max error <= 1.")
            break
            
    return autoencoder


# =====================================================================
# EXPERIMENTOS
# =====================================================================

# Experimento 1: Entrenamiento con el dataset completo (32 letras)

def experiment_full_dataset(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 1: Full Dataset (32 letters) ---")
    ae = create_base_ae()
    ae_full, _ = train_autoencoder(ae, X, epochs=10000, learning_rate=0.1, optimizer=Adam(learning_rate=0.01))
    Plotter.plot_latent_space(ae_full, X, "Full Dataset", "latent_full.png", labels=labels)
    Plotter.generate_new_letter(ae_full, z_coord=[0.0, 0.2], filename="generated_letter.png")


# Experimento 2: Entrenamiento con un subset reducido (5 letras)

def experiment_subset(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 2: Subset (5 letters) ---")
    X_sub = X[:5]
    labels_sub = labels[:5]
    ae = create_base_ae()
    ae_sub, _ = train_autoencoder(ae, X_sub, epochs=10000, learning_rate=0.1, optimizer=Adam(learning_rate=0.01))
    
    Plotter.plot_latent_space(ae_sub, X_sub, "Subset (5 letters)", "latent_subset.png", labels=labels_sub)    
    print("\nGenerating a new letter from intermediate coordinates...")
    Plotter.generate_new_letter(ae_sub, z_coord=[1.0, -1.0], filename="generated_letter.png")

# Experimento 3: Comparación de arquitecturas con múltiples semillas

def experiment_architectures(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 3: Multi-Seed Architecture Comparison ---")
    X_sub = X[:10]
    labels_sub = labels[:10] # Necesario para plotear el latent space correctamente
    
    arquitecturas = {
        "Base (16-2-16)": create_base_ae,
        "Shallow (2)": create_shallow_ae,
        "Narrow (4-2-4)": create_narrow_ae,
        "Deep & Wide (32-16-2-16-32)": create_deep_wide_ae
    }
    
    semillas = [42, 100, 800, 1024, 2024]
    max_epochs = 5000
    
    resultados_loss = {nombre: [] for nombre in arquitecturas.keys()}
    
    for nombre, factory_fn in arquitecturas.items():
        print(f"\n[+] Evaluando arquitectura: {nombre}")
        
        # Variables para trackear a la mejor semilla de esta arquitectura
        mejor_modelo = None
        menor_loss = float('inf')
        mejor_seed = None
        
        for seed in semillas:
            print(f"  -> Corriendo Seed: {seed}...")
            np.random.seed(seed)
            
            modelo = factory_fn() 
            modelo_entrenado, historial = train_autoencoder(modelo, X_sub, epochs=max_epochs, learning_rate=0.1, optimizer=Adam(learning_rate=0.01))
            
            # Evaluamos si esta semilla logró el menor error
            loss_final = historial[-1]
            if loss_final < menor_loss:
                menor_loss = loss_final
                mejor_modelo = modelo_entrenado
                mejor_seed = seed
            
            # Padding para el gráfico promedio
            while len(historial) < max_epochs:
                historial.append(historial[-1])
                
            resultados_loss[nombre].append(historial)
            
        print(f"  [*] Mejor corrida para {nombre}: Seed {mejor_seed} con Loss {menor_loss:.4f}")
        
        # Limpiamos el nombre para que no rompa la ruta del archivo
        nombre_limpio = nombre.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Generamos los gráficos EXCLUSIVAMENTE del modelo campeón
        if hasattr(Plotter, 'plot_latent_space'):
            Plotter.plot_latent_space(
                mejor_modelo, X_sub, 
                f"{nombre} (Best Seed: {mejor_seed})", 
                f"latent_best_{nombre_limpio}.png", 
                labels=labels_sub
            )
            
        if hasattr(Plotter, 'compare_reconstruction'):
            Plotter.compare_reconstruction(
                mejor_modelo, X_sub[1], "a", 
                f"comparativa_best_{nombre_limpio}.png"
            )

    Plotter.plot_architecture_comparison(
        resultados_loss,
        max_epochs,
        "architecture_multiseed_comparison.png",
    )

# Experimento 4: Comparación de learning rates con múltiples semillas

def experiment_learning_rates(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 4: Multi-Seed Learning Rate Comparison ---")
    X_sub = X
    
    factory_fn = create_base_ae
    
    # Definimos los LRs a testear: uno lento, el ideal, uno agresivo y uno bestial
    learning_rates = [0.01, 0.1, 1.5]
    semillas = [42, 100, 800, 1024, 2024]
    max_epochs = 20000
    
    resultados_loss = {str(lr): [] for lr in learning_rates}
    
    for lr in learning_rates:
        print(f"\n[+] Evaluando Learning Rate: {lr}")
        
        for seed in semillas:
            print(f"  -> Corriendo Seed: {seed}...")
            np.random.seed(seed)
            
            modelo = factory_fn() 
            _, historial = train_autoencoder(modelo, X_sub, epochs=max_epochs, learning_rate=lr, optimizer=Adam(learning_rate=0.01))
            
            # Padding por si el Early Stopping cortó antes
            while len(historial) < max_epochs:
                historial.append(historial[-1])
                
            resultados_loss[str(lr)].append(historial)

    Plotter.plot_learning_rate_comparison(
        resultados_loss,
        max_epochs,
        "learning_rate_multiseed_comparison.png",
    )
    
# Experimento 5: Evaluación de la capacidad del espacio latente con múltiples semillas

def experiment_dataset_size(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 5: Latent Space Capacity (Multi-Seed Stress Test) ---")
    
    sizes = [2, 5, 8, 12, 16, 24, 32]
    seeds = [42, 100, 800, 1024, 2024]
    max_epochs = 10000 
    
    mean_pixels = []
    pixeles_std = []
    loss_media = []
    loss_std = []
    
    for n in sizes:
        print(f"\n[+] Evaluando capacidad con N = {n} letras...")
        X_sub = X[:n]
        
        pixeles_por_semilla = []
        loss_por_semilla = []
        
        for seed in seeds:
            np.random.seed(seed) 
            modelo = create_base_ae()
            
            # Entrenamos
            modelo_entrenado, historial = train_autoencoder(modelo, X_sub, epochs=max_epochs, learning_rate=0.1)
            
            # Evaluamos
            prediccion_final = modelo_entrenado.forward(X_sub)
            pixeles_erroneos_max = evaluate_pixel_diff(X_sub, prediccion_final)
            loss_final = historial[-1]
            
            pixeles_por_semilla.append(pixeles_erroneos_max)
            loss_por_semilla.append(loss_final)
            
        # Calculamos estadísticas
        mean_pixels.append(np.mean(pixeles_por_semilla))
        pixeles_std.append(np.std(pixeles_por_semilla))
        loss_media.append(np.mean(loss_por_semilla))
        loss_std.append(np.std(loss_por_semilla))
        
        print(f"  [*] N={n} | Media Píxeles: {mean_pixels[-1]:.1f} (±{pixeles_std[-1]:.1f}) | Media Loss: {loss_media[-1]:.4f}")

    # Delegamos el renderizado visual a nuestra clase Plotter
    Plotter.plot_dataset_capacity_comparison(
        sizes=np.array(sizes),
        mean_pixels=np.array(mean_pixels),
        pixeles_std=np.array(pixeles_std),
        loss_media=np.array(loss_media),
        loss_std=np.array(loss_std),
        filename='dataset_capacity_multiseed.png'
    )
    
def experiment_optimizers(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 6: Optimizer Comparison (SGD vs Adam) ---")
    
    X_sub = X
    
    optimizadores = {
        "SGD Clásico (LR=0.1)": lambda: SGD(learning_rate=0.1),
        "Adam (LR=0.01)": lambda: Adam(learning_rate=0.01)
    }
    
    semillas = [42, 100, 800, 1024, 5024]
    max_epochs = 10000 # Le damos margen a SGD
    
    resultados_loss = {}
    
    for nombre, opt_factory in optimizadores.items():
        print(f"\n[+] Evaluando optimizador: {nombre}")
        
        historiales_crudos = []
        for seed in semillas:
            print(f"  -> Corriendo Seed: {seed}...")
            np.random.seed(seed)
            
            modelo = create_base_ae()
            opt = opt_factory()
            
            _, historial = train_autoencoder(modelo, X_sub, epochs=max_epochs, optimizer=opt)
            historiales_crudos.append(historial)
            
        # 1. Buscamos la época máxima REAL que alcanzó este optimizador
        max_epocas_real = max(len(h) for h in historiales_crudos)
        print(f"  [*] {nombre} finalizó todas sus corridas en la época {max_epocas_real}")
        
        # 2. Rellenamos (padding) SOLO hasta esa época máxima real
        historiales_ajustados = []
        for h in historiales_crudos:
            h_ajustado = h.copy()
            while len(h_ajustado) < max_epocas_real:
                h_ajustado.append(h_ajustado[-1])
            historiales_ajustados.append(h_ajustado)
            
        resultados_loss[nombre] = historiales_ajustados

    # Delegamos el renderizado (ya no le pasamos max_epochs)
    Plotter.plot_optimizer_comparison(
        resultados_loss,
        "optimizer_multiseed_comparison.png"
    )
    
def experiment_data_orthogonality(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 7: Data Quality (Orthogonal vs Similar Subsets) ---")
    
    Plotter.plot_similarity_matrix(X, labels, "dataset_orthogonality_matrix.png")
    
    # 1. Calculamos la matriz de Similitud Coseno
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1e-10, norms)
    sim_matrix = np.dot(X_norm, X_norm.T)
    
    subset_size = 5
    
    # --- Búsqueda del Subset SIMILAR (El desafío difícil) ---
    np.fill_diagonal(sim_matrix, -1) # Ignoramos la diagonal (similitud de una letra consigo misma)
    # Agarramos el par de letras más parecidas de todo el abecedario
    i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    similar_idx = [i, j]
    
    while len(similar_idx) < subset_size:
        # Sumamos la similitud de todas las letras contra el subset actual
        sim_sums = np.sum(sim_matrix[:, similar_idx], axis=1)
        sim_sums[similar_idx] = -np.inf # Excluimos las ya elegidas
        similar_idx.append(int(np.argmax(sim_sums)))
        
    # --- Búsqueda del Subset ORTOGONAL (El desafío fácil) ---
    np.fill_diagonal(sim_matrix, 0)
    # Empezamos con la letra más "rara" (la que menos se parece al promedio)
    ortho_idx = [int(np.argmin(np.mean(sim_matrix, axis=1)))]
    
    while len(ortho_idx) < subset_size:
        # Buscamos la letra que tenga la menor similitud máxima con las ya elegidas
        max_sims = np.max(sim_matrix[:, ortho_idx], axis=1)
        max_sims[ortho_idx] = np.inf # Excluimos las ya elegidas
        ortho_idx.append(int(np.argmin(max_sims)))

    labels_similar = [labels[idx] for idx in similar_idx]
    labels_ortho = [labels[idx] for idx in ortho_idx]
    
    print(f"  [*] Subset 'Similar' (Difícil): {labels_similar}")
    print(f"  [*] Subset 'Ortogonal' (Fácil): {labels_ortho}")

    Plotter.plot_subset_orthogonality_heatmaps(
        X_sim=X[similar_idx], labels_sim=labels_similar, 
        X_ortho=X[ortho_idx], labels_ortho=labels_ortho, 
        filename="subset_matrices_comparativas.png"
    )

    # 2. Entrenamos usando Vanilla SGD (LR=0.1) para ambos subsets
    max_epochs = 10000
    semillas = [42, 100, 800, 1024, 2024]
    
    subsets_a_evaluar = {
        f"Subset Similar {labels_similar}": X[similar_idx],
        f"Subset Ortogonal {labels_ortho}": X[ortho_idx]
    }
    
    resultados_loss = {nombre: [] for nombre in subsets_a_evaluar.keys()}
    
    for nombre, X_subset in subsets_a_evaluar.items():
        print(f"\n[+] Evaluando {nombre}...")
        
        historiales_crudos = []
        for seed in semillas:
            np.random.seed(seed)
            modelo = create_base_ae()
            # Usamos SGD explícitamente para ver cómo sufre
            opt = SGD(learning_rate=0.1)
            _, historial = train_autoencoder(modelo, X_subset, epochs=max_epochs, optimizer=opt)
            historiales_crudos.append(historial)
            
        max_epocas_real = max(len(h) for h in historiales_crudos)
        
        historiales_ajustados = []
        for h in historiales_crudos:
            h_ajustado = h.copy()
            while len(h_ajustado) < max_epocas_real:
                h_ajustado.append(h_ajustado[-1])
            historiales_ajustados.append(h_ajustado)
            
        resultados_loss[nombre] = historiales_ajustados

    # 3. Ploteamos
    Plotter.plot_subset_orthogonality_comparison(resultados_loss, "data_orthogonality_comparison.png")
    
# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================

def experiment_denoising(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 7: Denoising Autoencoder ---")
    noise_level = 0.30 # 30% de ruido (aprox 10-11 píxeles cambiados por letra)
    X_noisy = add_noise(X, noise_level=noise_level)
    
    print(f"Training DAE with {noise_level*100}% noise...")
    ae_denoising = train_denoising_autoencoder(X, X_noisy, epochs=15000, learning_rate=0.1)
    
    # Probamos con la letra 'a' (índice 1 en nuestro array de caracteres)
    idx_a = 1
    Plotter.compare_reconstruction(ae_denoising, X_noisy[idx_a], f"A (Noisy {noise_level*100}%)", "dae_reconstruction_a.png")
    
    # También probamos con otra letra para ver que funcione
    idx_z = 26
    Plotter.compare_reconstruction(ae_denoising, X_noisy[idx_z], f"Z (Noisy {noise_level*100}%)", "dae_reconstruction_z.png")


def main():
    np.random.seed(800)
    print("Loading dataset...")
    X = FontLoader.load_and_flatten('src/data/font.h')
    print(f"Data dimensions: {X.shape}")
    
    caracteres = [chr(0x60 + i) if i < 31 else 'DEL' for i in range(32)]
    
    # Comentá o descomentá los experimentos según lo que necesites correr
    # experiment_full_dataset(X, caracteres)
    # experiment_subset(X, caracteres)
    # experiment_architectures(X, caracteres)
    # experiment_learning_rates(X, caracteres)
    # experiment_dataset_size(X, caracteres)
    # experiment_optimizers(X, caracteres)
    experiment_data_orthogonality(X, caracteres)

if __name__ == "__main__":
    main()