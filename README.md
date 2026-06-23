# SIA_TP5: Autoencoders desde Cero (AE, DAE y VAE)

Este proyecto implementa **Autoencoders**, **Denoising Autoencoders (DAE)** y **Variational Autoencoders (VAE)** completamente desde cero en NumPy, incluyendo toda la arquitectura de red neuronal, capas personalizadas, funciones de pérdida y optimizadores. Se utiliza un dataset de fuentes pixeladas y emojis para entrenar y experimentar con diferentes configuraciones de autoencoders.

## Descripción General

Este proyecto explora tres arquitecturas fundamentales de autoencoders:

### Autoencoder Clásico (AE)
Red neuronal que comprime datos en un cuello de botella y luego los reconstruye. Aprende representaciones comprimidas de datos de entrada.

### Denoising Autoencoder (DAE)
Autoencoder que se entrena para reconstruir datos limpios a partir de versiones ruidosas. Útil para limpiar y desruir datos corrupted.

### Variational Autoencoder (VAE)
Modelo generativo que incorpora componentes probabilísticos. Aprende una distribución latente permitiendo:
- **Generación de nuevos datos**: Muestreando del espacio latente
- **Interpolación suave**: Moviéndose entre representaciones latentes
- **Análisis de la incertidumbre**: Representación como distribuciones

## Estructura del Proyecto

```
src/
├── main.py                      # Autoencoder clásico (referencia)
├── main_vae.py                  # VAE estándar
├── main_emoji_vae.py            # VAE especializado en emojis
│
├── core/                         # Implementación desde cero
│   ├── network.py               # Orquestador de capas (Forward/Backward)
│   ├── layers.py                # Capas: Linear, Activation, VAEBottleneck
│   ├── activations.py           # Funciones de activación (Sigmoid, Tanh)
│   ├── losses.py                # Funciones de pérdida (BCE, MSE)
│   └── optimizers.py            # Optimizadores (SGD, Adam)
│
├── data/                         # Manejo de datos
│   ├── font.h                   # Dataset de fuente pixelada (5x7)
│   └── emojis.py                # Carga de emojis
│
├── utils/                        # Utilidades
│   ├── data_loader.py           # Cargador de datos de fuentes
│   ├── emoji_loader.py          # Cargador de emojis
│   ├── plotter.py               # Visualización estándar
│   └── emoji_plotter.py         # Visualización de emojis
│
└── experiments/                  # Suite completa de experimentos
    ├── dae_main.py              # Denoising Autoencoder
    ├── exp1_bottleneck.py       # Experimento 1: Análisis del cuello de botella
    ├── exp2_noise_limits.py     # Experimento 2: Límites de ruido
    ├── exp3_posterior_collapse.py # Experimento 3: Colapso posterior
    ├── exp4_latent_interpolation.py # Experimento 4: Interpolación en espacio latente
    ├── exp5_loss_comparison.py  # Experimento 5: Comparación de pérdidas
    ├── exp6_dae_architectures.py # Experimento 6: Arquitecturas DAE
    ├── exp6b_bottleneck_sweep.py # Experimento 6b: Barrido de cuello de botella
    ├── exp6c_fixed_noise.py     # Experimento 6c: Ruido fijo
    ├── generative_vae.py        # VAE generativo
    └── vae_experiments.py       # Utilidades para experimentos
```

## Características Principales

### 1. **Implementación Desde Cero**
- Red neuronal construida con NumPy puro
- Propagación Forward y Backward implementadas manualmente
- No se utilizan frameworks como TensorFlow o PyTorch

### 2. **Arquitecturas Soportadas**
- **Autoencoder clásico**: Compresión de datos
- **Denoising Autoencoder (DAE)**: Reconstrucción de datos ruidosos
- **Variational Autoencoder (VAE)**: Modelo generativo con espacio latente probabilístico

### 3. **Componentes Core**
- **Capas**: Linear, ActivationLayer, VAEBottleneckLayer
- **Activaciones**: Sigmoid, Tanh (con derivadas)
- **Pérdidas**: Binary Cross-Entropy (BCE), Mean Squared Error (MSE)
- **Optimizadores**: SGD, Adam

### 4. **Suite Experimental Completa**
Investigación exhaustiva sobre:
- **Autoencoders clásicos**: Análisis del cuello de botella y compresión
- **Denoising Autoencoders**: Robustez al ruido, límites de ruido, arquitecturas alternativas
- **Variational Autoencoders**: Colapso posterior, interpolación latente, comparación de pérdidas
- **Comparativas**: Análisis cruzado entre AE, DAE y VAE
- Generación de nuevos datos y visualización del espacio latente

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Jalbertonisalini/SIA_TP5.git
cd SIA_TP5

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias
```
numpy>=1.26.0
matplotlib>=3.8.0
```

## Uso

### Entrenar un Autoencoder Clásico
```bash
cd src
python main.py
```

### Entrenar un Denoising Autoencoder (DAE)
```bash
cd src/experiments
python dae_main.py
```

### Entrenar un VAE Básico
```bash
cd src
python main_vae.py
```

### Entrenar un VAE con Emojis
```bash
python main_emoji_vae.py
```

### Ejecutar Experimentos Específicos
```bash
# Experimento 1: Análisis del cuello de botella
python experiments/exp1_bottleneck.py

# Experimento 4: Interpolación en espacio latente
python experiments/exp4_latent_interpolation.py

# Experimento 6c: Ruido fijo
python experiments/exp6c_fixed_noise.py
```

## Conceptos Clave

### Autoencoder Clásico
Arquitectura básica con:
- **Encoder**: Comprime entrada en representación latente
- **Decoder**: Reconstruye datos desde representación latente
- **Pérdida**: Solo basada en diferencia de reconstrucción

### Denoising Autoencoder (DAE)
Autoencoder entrenado con datos ruidosos:
- **Entrada**: Datos originales + ruido agregado
- **Salida**: Datos limpios originales
- **Objetivo**: Aprender a desruir y reconstruir robustamente

### Variational Autoencoder (VAE)
Arquitectura probabilística con:
1. **Encoder**: Codifica a distribución (μ, σ)
2. **Muestreo**: Extrae z ~ N(μ, σ)
3. **Decoder**: Reconstruye desde z
4. **KL Loss**: Regulariza la distribución latente

### KL Divergence
En VAE, se minimizan dos pérdidas:
- **Reconstruction Loss**: ¿Qué tan bien se reconstruye la entrada?
- **KL Loss**: ¿Qué tan cerca está la distribución latente de una normal estándar?

### Colapso Posterior
Problema en VAE donde el codificador ignora el espacio latente. Los experimentos exploran formas de evitarlo.

## Resultados y Outputs

Los resultados se guardan en:
- `outputs/` - Resultados de los experimentos

Cada experimento genera:
- Imágenes de reconstrucciones
- Gráficos de pérdida
- Visualizaciones del espacio latente
- Interpolaciones entre puntos latentes

## Autor
**Proyecto de la materia SIA (Sistemas Inteligentes Autónomos)**

Implementación educativa para entender en profundidad cómo funcionan los Autoencoders, desde los fundamentos matemáticos hasta la implementación práctica, cubriendo las tres arquitecturas principales: AE, DAE y VAE.
