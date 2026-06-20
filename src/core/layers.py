import numpy as np
from abc import ABC, abstractmethod
from core.optimizers import Optimizer

from core.activations import ActivationFunction

class Layer(ABC):
    """
    Contrato base para cualquier capa de la red neuronal.
    Garantiza que todas las capas puedan integrarse en el pipeline secuencial.
    """
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Toma los datos de la capa anterior, realiza el cálculo y devuelve la salida hacia la capa siguiente.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """
        Toma el gradiente del error proveniente de la capa siguiente,
        delega la actualización de sus parámetros (si tiene pesos/sesgos) al optimizador, 
        y devuelve el gradiente respecto a su entrada para continuar la propagación.
        """
        pass
    
class Linear(Layer):
    """
    Representa la transformación afín: XW + b (Codificador) o ZV + p (Decodificador).
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        
        # Inicialización de pesos escalada (Xavier/Glorot) para estabilidad
        std = np.sqrt(2.0 / (input_size + output_size))
        
        # Matriz de pesos W (input_size x output_size)
        self.weights = np.random.randn(input_size, output_size) * std
        
        # Vector de sesgo b (1 x output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        
        # Ecuación: XW + b
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        # Derivada respecto a la entrada (dE/dX = dE/dZ * W^T)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Derivadas respecto a pesos y sesgos (dE/dW = X^T * dE/dZ)
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Actualización de parámetros delegada al Optimizador
        self.weights = optimizer.update(id(self), "weights", self.weights, weights_gradient)
        self.bias = optimizer.update(id(self), "bias", self.bias, bias_gradient)

        return input_gradient

class ActivationLayer(Layer):
    """
    Representa la función h() sobre una capa lineal.
    """
    def __init__(self, activation_fn: ActivationFunction):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        # Aplicación de h(XW + b)
        return self.activation_fn.calculate(self.input)

    def backward(self, output_gradient: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        # Regla de la cadena: dE/dX = dE/dZ * h'(X)
        return output_gradient * self.activation_fn.derivative(self.input)