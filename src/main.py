from xml.parsers.expat import model

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

def get_orthogonal_subset(X: np.ndarray, labels: list, n: int):
    """
    Retorna el subconjunto de tamaño N más ortogonal posible.
    """
    # 1. Matriz de Similitud Coseno
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1e-10, norms)
    sim_matrix = np.dot(X_norm, X_norm.T)
    np.fill_diagonal(sim_matrix, np.inf)
    
    # 2. Algoritmo Greedy
    i, j = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    ordered_idx = [int(i), int(j)]
    
    while len(ordered_idx) < n:
        max_sims = np.max(sim_matrix[:, ordered_idx], axis=1)
        max_sims[ordered_idx] = np.inf
        next_idx = int(np.argmin(max_sims))
        ordered_idx.append(next_idx)
        
    return X[ordered_idx], [labels[i] for i in ordered_idx], ordered_idx

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

import copy # Importante agregar esto arriba de todo en tu archivo

def train_autoencoder(autoencoder: Network, X: np.ndarray, epochs: int = 15000, 
                      optimizer=None, use_pixel_stopping: bool = True, 
                      patience: int = 0, min_delta: float = 1e-4) -> tuple[Network, list]:
    
    if optimizer is None:
        optimizer = SGD(learning_rate=0.1)
        
    loss_function = BCE()
    loss_history = []
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None # Acá vamos a guardar al campeón
    
    for epoch in range(epochs):
            predicted = autoencoder.forward(X)
            loss = loss_function.calculate(expected=X, predicted=predicted)
            loss_history.append(loss)
            
            max_incorrect_pixels = evaluate_pixel_diff(X, predicted)
            
            # --- CRITERIO 1: Éxito visual (MOVIDO ACÁ ARRIBA) ---
            if use_pixel_stopping and max_incorrect_pixels <= 1:
                print(f"  [!] Convergencia visual en época {epoch}.")
                best_model_state = copy.deepcopy(autoencoder)
                break
            
            # Si no cortó, RECIÉN AHORA calculamos gradientes y actualizamos pesos
            initial_gradient = loss_function.derivative(expected=X, predicted=predicted)
            autoencoder.backward(initial_gradient, optimizer)
            
            # --- CRITERIO 2: Estancamiento matemático (Plateau) ---
            if patience > 0:
                if (best_loss - loss) > min_delta:
                    best_loss = loss
                    patience_counter = 0 
                    best_model_state = copy.deepcopy(autoencoder)
                else:
                    patience_counter += 1 
                    
                if patience_counter >= patience:
                    print(f"  [!] Early Stopping por Plateau en época {epoch}. Restaurando mejor modelo...")
                    break
    # Si usamos patience y guardamos un modelo, lo restauramos. 
    # Si no, devolvemos el estado final.
    final_model = best_model_state if best_model_state is not None else autoencoder
    
    return final_model, loss_history


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

def experiment_phase1_baseline(X: np.ndarray, labels: list):
    """
    Phase 1: Baseline Failure Analysis.
    
    Trains a base Autoencoder (35 -> 16 -> 2 -> 16 -> 35) on the full dataset 
    (32 letters) using classical Stochastic Gradient Descent (SGD).
    
    The objective is to demonstrate empirically how a rigid optimizer, combined 
    with a severe dimensionality bottleneck, fails to converge. It generates 
    visualizations of the saturated latent space and the failed reconstruction 
    of a known character.
    
    Args:
        X (np.ndarray): The complete dataset matrix (N x 35).
        labels (list): The list of character labels for latent space plotting.
    """
    print("\n--- PHASE 1: Baseline Failure (SGD + Full Dataset) ---")
    
    # 1. Strict Hyperparameter Configuration (Required for PPT reproducibility)
    learning_rate = 0.1
    max_epochs = 200000
    seed = 800
    
    np.random.seed(seed)
    
    # 2. Model & Optimizer Initialization
    model = create_base_ae()
    optimizer = SGD(learning_rate=learning_rate)
    
    # 3. Training Phase (Early stopping disabled to observe asymptotic behavior)
    print(f"  [*] Training Baseline for {max_epochs} epochs (Early stopping disabled)...")
    trained_model, loss_history = train_autoencoder(
        autoencoder=model, 
        X=X, 
        epochs=200000, 
        optimizer=optimizer, 
        use_pixel_stopping=False, # Apagamos el de píxeles porque ya sabemos que no llega a 0
        patience=1000,            # Le damos 2000 épocas de tolerancia
        min_delta=1e-4            # Tiene que mejorar al menos 0.0001
    )
    # 4. Final Error Evaluation
    final_prediction = trained_model.forward(X)
    max_wrong_pixels = evaluate_pixel_diff(X, final_prediction)
    print(f"  [!] Final error at epoch {max_epochs}: {max_wrong_pixels} max incorrect pixels (Target: <= 1)")
    
    Plotter.plot_latent_space(
        autoencoder=trained_model, 
        X=X, 
        title=f"SGD Baseline", 
        filename="phase1_latent_space.png", 
        labels=labels
    )
    
    Plotter.plot_single_loss_curve(
        loss_history=loss_history,
        title=f"BCE Convergence",
        filename="phase1_loss_curve.png"
    )
    
    # --- NUEVO: Buscamos la PEOR letra y la graficamos ---
    
    # 1. Calculamos los errores por letra SIN hacer el np.max()
    binary_predicted = (final_prediction >= 0.5).astype(int)
    binary_expected = X.astype(int)
    differences = np.abs(binary_expected - binary_predicted)
    errors_per_letter = np.sum(differences, axis=1) # Vector con 32 números
    
    # 2. Encontramos el índice de la letra que tuvo más errores
    worst_index = int(np.argmax(errors_per_letter))
    worst_label = labels[worst_index]
    max_errors = errors_per_letter[worst_index]
    
    print(f"  [*] La peor letra fue la '{worst_label}' con {max_errors} píxeles de error.")
    
    # 3. Le pasamos ESE índice al Plotter
    prediction_worst = trained_model.forward(X[worst_index])
    Plotter.plot_pixel_errors(
        original_x=X[worst_index], 
        predicted_x=prediction_worst, 
        label=worst_label, 
        filename=f"phase1_pixel_errors_worst_case.png" # Cambiamos el nombre para no pisar la 'a'
    )
    
def experiment_phase2_capacity_limit(X: np.ndarray, labels: list):
    """
    Fase 2: Límite de Capacidad y Ortogonalidad (Versión Estadísticamente Robusta).
    """
    print("\n--- FASE 2: Límite de Capacidad (Robustez Estadística) ---")
    
    # 1. Matriz de Similitud Coseno
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1e-10, norms)
    sim_matrix = np.dot(X_norm, X_norm.T)
    
    # 2. Algoritmo Greedy para ordenar letras por Ortogonalidad
    np.fill_diagonal(sim_matrix, np.inf)
    i, j = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    ordered_idx = [int(i), int(j)]
    
    while len(ordered_idx) < len(labels):
        max_sims = np.max(sim_matrix[:, ordered_idx], axis=1)
        max_sims[ordered_idx] = np.inf
        next_idx = int(np.argmin(max_sims))
        ordered_idx.append(next_idx)

    # 3. Parámetros del experimento
    subset_sizes = [4, 8, 12, 16, 20, 24, 28, 32]
    seeds = [42, 100, 800] # Evaluamos 3 semillas para calcular el desvío
    max_epochs = 200000
    
    print("\n  [*] Orden de letras de Mayor a Menor Ortogonalidad:")
    print(f"      {[labels[i] for i in ordered_idx]}")
    
    # Estructuras para guardar resultados
    results = {size: [] for size in subset_sizes}
    sims_for_plot = [] 

    print("\n--- DETALLE DE SUBSETS Y ORTOGONALIDAD ---")
    for size in subset_sizes:
        current_subset_idx = ordered_idx[:size]
        X_subset = X[current_subset_idx]
        
        # Cálculo de Similitud Máxima Interna (determinística)
        norms_sub = np.linalg.norm(X_subset, axis=1, keepdims=True)
        X_sub_norm = X_subset / np.where(norms_sub == 0, 1e-10, norms_sub)
        sim_mat_sub = np.dot(X_sub_norm, X_sub_norm.T)
        np.fill_diagonal(sim_mat_sub, -np.inf)
        max_sim_interna = np.max(sim_mat_sub)
        sims_for_plot.append(max_sim_interna)
        
        print(f"\n[+] Evaluando N={size} (Max Similitud: {max_sim_interna:.3f})")
        
        # Corremos múltiples semillas para robustez
        for seed in seeds:
            np.random.seed(seed)
            model = create_base_ae()
            optimizer = SGD(learning_rate=0.1)
            
            trained_model, _ = train_autoencoder(
                autoencoder=model, X=X_subset, epochs=max_epochs, optimizer=optimizer,
                use_pixel_stopping=False, patience=1000
            )
            
            max_err = evaluate_pixel_diff(X_subset, trained_model.forward(X_subset))
            results[size].append(max_err)
            
        print(f"      -> Errores obtenidos con semillas {seeds}: {results[size]}")

    # 4. Cálculo de estadísticos para el gráfico
    errors_mean = [np.mean(results[s]) for s in subset_sizes]
    errors_std = [np.std(results[s]) for s in subset_sizes]
    
    # 5. Generamos el gráfico robusto
    Plotter.plot_capacity_limit_with_error(
        subset_sizes, errors_mean, errors_std, sims_for_plot, "phase2_capacity_limit_robust.png"
    )
# Experimento 2: Entrenamiento con un subset reducido (5 letras)

def experiment_subset(X: np.ndarray, labels: list):
    print(f"\n--- EXPERIMENT 2: Subset ({len(labels)} letters) ---")
    
    # Creamos y entrenamos con el subset
    ae = create_base_ae()
    ae_sub, loss_history = train_autoencoder(ae, X, epochs=200000, optimizer=SGD(learning_rate=0.1), use_pixel_stopping=False, patience=1000)
    
    # Graficamos el espacio latente del subset entrenado
    Plotter.plot_latent_space(ae_sub, X, f"Subset ({len(labels)} letters)", "latent_subset.png", labels=labels)    
    
    print("\nGenerating a new letter from intermediate coordinates...")
    Plotter.generate_new_letter(ae_sub, z_coord=[-1.0, -0.5], filename="generated_letter.png")
    Plotter.plot_single_loss_curve(
        loss_history=loss_history,
        title=f"BCE Convergence",
        filename="subset_loss_curve.png"
    )

    print("\nCalculando la peor reconstrucción del subset...")
    # 1. Hacemos la predicción final sobre el subset
    final_pred_sub = ae_sub.forward(X)
    
    # 2. Calculamos los errores absolutos píxel por píxel (ya umbralizados)
    pred_binaria = np.where(final_pred_sub >= 0.5, 1, 0)
    errores_por_letra = np.sum(np.abs(X - pred_binaria), axis=1)
    
    # 3. Agarramos el índice de la peor letra
    peor_idx = int(np.argmax(errores_por_letra))
    peor_letra_label = labels[peor_idx]
    peor_error = errores_por_letra[peor_idx]
    
    print(f"  [+] La peor letra del subset es '{peor_letra_label}' con {peor_error} píxeles de error.")
    Plotter.plot_pixel_errors(X[peor_idx], final_pred_sub[peor_idx], peor_letra_label, filename="phase2_subset_worst_letter.png")

# Experimento 3: Comparación de arquitecturas con múltiples semillas

def experiment_architectures(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT 4: Multi-Seed Architecture Comparison ---")
    
    # Usamos el dataset completo para la prueba definitiva de arquitecturas
    X_full = X
    labels_full = labels
    
    arquitecturas = {
        "Base (16-2-16)": create_base_ae,
        "Shallow (2)": create_shallow_ae,
        "Narrow (4-2-4)": create_narrow_ae,
        "Deep & Wide (32-16-2-16-32)": create_deep_wide_ae
    }
    
    semillas = [42, 100, 800, 1024, 2024, 3030, 4040, 5050, 6060, 7070]
    max_epochs = 5000
    
    resultados_loss = {nombre: [] for nombre in arquitecturas.keys()}
    resultados_errores = {nombre: [] for nombre in arquitecturas.keys()}
    
    for nombre, factory_fn in arquitecturas.items():
        print(f"\n[+] Evaluando arquitectura: {nombre}")
        
        mejor_modelo = None
        menor_loss = float('inf')
        mejor_seed = None
        
        for seed in semillas:
            np.random.seed(seed)
            
            modelo = factory_fn() 
            # Limpiamos el learning_rate suelto para no pisarnos con el de Adam
            modelo_entrenado, historial = train_autoencoder(
                modelo, X_full, epochs=max_epochs, 
                optimizer=Adam(learning_rate=0.001),
                use_pixel_stopping=False, patience=1000
            )
            
            # Tracking de BCE
            loss_final = historial[-1]
            if loss_final < menor_loss:
                menor_loss = loss_final
                mejor_modelo = modelo_entrenado
                mejor_seed = seed
                
            # Tracking del Error Físico
            final_pred = modelo_entrenado.forward(X_full)
            max_err = evaluate_pixel_diff(X_full, final_pred)
            resultados_errores[nombre].append(max_err)
            
            # Padding
            while len(historial) < max_epochs:
                historial.append(historial[-1])
                
            resultados_loss[nombre].append(historial)
            
        print(f"  [*] Mejor corrida para {nombre}: Seed {mejor_seed} con Loss {menor_loss:.4f}")
        
        nombre_limpio = nombre.replace(' ', '_').replace('(', '').replace(')', '')
        
        if hasattr(Plotter, 'plot_latent_space'):
            Plotter.plot_latent_space(
                mejor_modelo, X_full, 
                f"{nombre} (Best Seed: {mejor_seed})", 
                f"latent_best_{nombre_limpio}.png", 
                labels=labels_full
            )
            
        if hasattr(Plotter, 'compare_reconstruction'):
            Plotter.compare_reconstruction(
                mejor_modelo, 
                X_full[1], # Índice 1 = Letra 'b'
                "a",       # Etiqueta visual
                nombre,    # <--- NUEVO: Pasamos el nombre de la arquitectura (ej: "Shallow (2)")
                f"comparativa_best_{nombre_limpio}.png"
            )

    Plotter.plot_architecture_comparison(
        resultados_loss,
        resultados_errores,
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
    
def experiment_phase3_optimizers(X: np.ndarray, labels: list):
    print("\n--- FASE 3: SGD vs Adam (Dataset Completo) ---")
    
    # Volvemos al dataset original completo (32 letras)
    X_full = X 
    
    optimizadores = {
        "SGD Clásico (LR=0.1)": lambda: SGD(learning_rate=0.1),
        "Adam (LR=0.001)": lambda: Adam(learning_rate=0.001)
    }
    
    #usamos 10 semillas para tener una estadística más robusta en esta fase final
    semillas = [42, 100, 800, 1024, 2024, 3030, 4040, 5050, 6060, 7070]
    max_epochs = 200000 
    
    resultados_loss = {}
    resultados_errores = {nombre: [] for nombre in optimizadores.keys()}
    
    for nombre, opt_factory in optimizadores.items():
        print(f"\n[+] Evaluando optimizador: {nombre}")
        
        historiales_crudos = []
        for seed in semillas:
            print(f"  -> Corriendo Seed: {seed}...")
            np.random.seed(seed)
            
            modelo = create_base_ae()
            opt = opt_factory()
            
            # Entrenamos habilitando el corte por píxeles
            trained_model, historial = train_autoencoder(
                modelo, X_full, epochs=max_epochs, optimizer=opt,
                use_pixel_stopping=False, patience=1000
            )
            historiales_crudos.append(historial)
            
            # NUEVO: Medimos el error físico al terminar
            final_pred = trained_model.forward(X_full)
            max_err = evaluate_pixel_diff(X_full, final_pred)
            resultados_errores[nombre].append(max_err)
            
        # 1. Buscamos la época máxima REAL que alcanzó este optimizador
        max_epocas_real = max(len(h) for h in historiales_crudos)
        print(f"  [*] {nombre} finalizó todas sus corridas (Max época: {max_epocas_real})")
        print(f"  [*] Errores máximos por seed: {resultados_errores[nombre]}")
        
        # 2. Rellenamos (padding) SOLO hasta esa época máxima real
        historiales_ajustados = []
        for h in historiales_crudos:
            h_ajustado = h.copy()
            while len(h_ajustado) < max_epocas_real:
                h_ajustado.append(h_ajustado[-1])
            historiales_ajustados.append(h_ajustado)
            
        resultados_loss[nombre] = historiales_ajustados

    # Imprimimos el veredicto final en consola para tu tabla del PPT
    print("\n--- RESUMEN DE ERRORES MÁXIMOS (Píxeles) ---")
    for nombre in optimizadores.keys():
        promedio_err = np.mean(resultados_errores[nombre])
        std_err = np.std(resultados_errores[nombre])
        print(f"{nombre}: Promedio {promedio_err:.1f} ± {std_err:.1f} píxeles")

# Delegamos el renderizado
    Plotter.plot_optimizer_comparison(
        resultados_loss,
        resultados_errores, # <--- ¡Faltaba pasarle esto!
        "phase3_optimizer_comparison.png"
    )
    
def experiment_play_generative(X: np.ndarray, labels: list):
    print("\n--- EXPERIMENT: Generación Pura desde Espacio Latente ---")
    
    # Usamos la semilla ganadora que te dio ese espacio latente amontonado pero efectivo
    seed_ganadora = 3030
    np.random.seed(seed_ganadora)
    
    print("  [+] Entrenando arquitectura Deep & Wide (Campeona) en modo rápido...")
    modelo = create_deep_wide_ae()
    
    # Entrenamos con Adam usando la configuración ganadora
    modelo_entrenado, _ = train_autoencoder(
        modelo, X, epochs=200000, 
        optimizer=Adam(learning_rate=0.001),
        use_pixel_stopping=False, patience=1000
    )
    
    print("\n  [+] Buscando coordenadas exactas de la 'h'...")
    
    # 1. Encontramos el índice de la 'h' en la lista de etiquetas
    try:
        idx_h = labels.index('h')
        # 2. Pasamos la letra 'h' SOLO por el encoder (la primera mitad de las capas)
        mitad = len(modelo_entrenado.layers) // 2
        latent_h = X[idx_h].reshape(1, -1)
        for layer in modelo_entrenado.layers[:mitad]:
            latent_h = layer.forward(latent_h)
            
        coord_h = latent_h.flatten().tolist()
        print(f"      -> Coordenada exacta de 'h': {coord_h}")
        
        # 3. Probamos generar con esa coordenada exacta
        Plotter.generate_new_letter(modelo_entrenado, coord_h, "generada_h_exacta.png")
        
    except ValueError:
        print("      -> Error: No se encontró la letra 'h' en las etiquetas.")
    
    # --- ZONA DE JUEGO ---
    # Acá ponés las coordenadas (Z1, Z2) que quieras probar mirando tu gráfico.
    coordenadas_a_probar = {
        "centro_vacio": [-1.0, 1.0],       # La zona muerta del medio
        "zona_vocales": [0.76,-1.0],    # Bien adentro del clúster de a, c, e, o
        "zona_palitos": [-0.70, -0.75],   # Cerca de la 'h' y la 'k'
        "intermedio": [0.50, -0.25]       # Mitad de camino hacia la 'p'
    }
    
    print("\n  [+] Inyectando coordenadas en el Decoder...")
    for nombre, coord in coordenadas_a_probar.items():
        filename = f"generada_{nombre}.png"
        
        # Llamamos a tu Plotter para que hackee el Decoder
        Plotter.generate_new_letter(
            modelo_entrenado, 
            z_coord=coord, 
            filename=filename
        )
        print(f"      -> Coordenada {coord} generada como '{filename}'")
        
    print("\n  [*] ¡Experimento finalizado! Revisá la carpeta outputs.")
    Plotter.plot_latent_space(
        modelo_entrenado, X, 
        f"Espacio Latente (Seed {seed_ganadora})", 
        "final_latent_space.png", 
        labels=labels
    )
    print("  [+] Mapa latente guardado en 'outputs/final_latent_space.png'.")
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
    X = FontLoader.load_and_flatten('data/font.h')
    print(f"Data dimensions: {X.shape}")
    
    caracteres = [chr(0x60 + i) if i < 31 else 'DEL' for i in range(32)]
    
    Plotter.plot_dataset_grid(X, caracteres, filename="dataset_completo.png")
    Plotter.plot_formula_plate(filename="formula_plate.png")
    
    X_sub, labels_sub, _ = get_orthogonal_subset(X, caracteres, n=20)
    
    # Comentá o descomentá los experimentos según lo que necesites correr
    # experiment_phase1_baseline(X, caracteres)
    # experiment_phase2_capacity_limit(X, caracteres)
    # experiment_subset(X_sub, labels_sub)
    # experiment_architectures(X, caracteres)
    # experiment_learning_rates(X, caracteres)
    # experiment_phase3_optimizers(X, caracteres)
    experiment_play_generative(X, caracteres)

if __name__ == "__main__":
    main()