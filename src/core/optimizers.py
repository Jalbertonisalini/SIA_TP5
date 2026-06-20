import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Contrato base para los algoritmos de optimización.
    """
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer_id: int, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Recibe el parámetro actual y su gradiente, y devuelve el parámetro actualizado.
        """
        pass

class SGD(Optimizer):
    """
    Descenso de Gradiente Estocástico clásico.
    Actualiza los pesos usando únicamente el gradiente actual y el learning rate.
    """
    def update(self, layer_id: int, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return param - self.learning_rate * grad

class Adam(Optimizer):
    """
    Optimizador Adam (Adaptive Moment Estimation).
    Calcula tasas de aprendizaje adaptativas para cada parámetro basándose 
    en estimaciones de primer y segundo momento (inercia y varianza) de los gradientes.
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Diccionarios para guardar el estado (inercia y varianza) de cada parámetro
        self.m = {}
        self.v = {}
        self.t = {} # Contador de épocas independiente para cada parámetro

    def update(self, layer_id: int, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        key = f"{layer_id}_{param_name}"
        
        # Inicializamos el historial en cero la primera vez que vemos un parámetro
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)
            self.t[key] = 0
            
        self.t[key] += 1
        t = self.t[key]
        
        # 1. Momento de 1er orden (Inercia)
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        
        # 2. Momento de 2do orden (Varianza / Fricción)
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
        
        # 3. Corrección de sesgo para que no arranque lento en la época 1
        m_hat = self.m[key] / (1 - self.beta1 ** t)
        v_hat = self.v[key] / (1 - self.beta2 ** t)
        
        # 4. Actualización final con Learning Rate adaptativo
        paso = (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)
        return param - paso