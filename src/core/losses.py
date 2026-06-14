import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """Contrato base para las funciones de pérdida que guían el aprendizaje."""
    
    @abstractmethod
    def calculate(self, expected: np.ndarray, predicted: np.ndarray) -> float:
        """Calcula el error escalar total de la predicción."""
        pass

    @abstractmethod
    def derivative(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Calcula el gradiente inicial dE/dY para arrancar el backpropagation.
        """
        pass

class MSE(LossFunction):
    """
    Error Cuadrático Medio (Mean Squared Error).
    L(X, X') = ||X - X'||^2
    """
    def calculate(self, expected: np.ndarray, predicted: np.ndarray) -> float:
        return np.mean(np.power(expected - predicted, 2))

    def derivative(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        # La derivada de (Y - Y')^2 es proporcional a 2*(Y' - Y) / N
        return 2 * (predicted - expected) / expected.size

class BCE(LossFunction):
    """
    Entropía Cruzada Binaria (Binary Cross Entropy).
    """
    def calculate(self, expected: np.ndarray, predicted: np.ndarray) -> float:
        # Agregamos un epsilon minúsculo para evitar calcular el logaritmo de 0
        epsilon = 1e-9
        pred_clipped = np.clip(predicted, epsilon, 1.0 - epsilon)
        
        # Ecuación de la Diapositiva 18 (Capsule.pdf)
        error = -np.mean(
            expected * np.log(pred_clipped) + (1.0 - expected) * np.log(1.0 - pred_clipped)
        )
        return float(error)

    def derivative(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        pred_clipped = np.clip(predicted, epsilon, 1.0 - epsilon)
        
        # Derivada analítica de BCE respecto a la predicción
        # dE/dY' = (Y' - Y) / (Y' * (1 - Y'))
        return (pred_clipped - expected) / (pred_clipped * (1.0 - pred_clipped) * expected.size)