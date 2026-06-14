import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """Interfaz base para todas las funciones de activación."""
    
    @abstractmethod
    def calculate(self, x: np.ndarray) -> np.ndarray:
        """Calcula el forward pass de la activación."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada para el backward pass."""
        pass

class Sigmoid(ActivationFunction):
    """
    Función Sigmoide. 
    Crucial para la última capa del decodificador para mapear salidas entre 0 y 1 y representar los píxeles binarios de font.h.
    """
    def calculate(self, x: np.ndarray) -> np.ndarray:
        # np.clip evita desbordamientos numéricos (overflow) con exp
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.calculate(x)
        return sig * (1.0 - sig)

class Tanh(ActivationFunction):
    """
    Tangente Hiperbólica.
    """
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2