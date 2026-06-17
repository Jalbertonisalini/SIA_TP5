import numpy as np
from abc import ABC, abstractmethod

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
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Toma el gradiente del error proveniente de la capa siguiente,
        actualiza sus parámetros (si tiene pesos/sesgos), y devuelve 
        el gradiente respecto a su entrada para continuar la propagación.
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

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Derivada respecto a la entrada (dE/dX = dE/dZ * W^T)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Derivadas respecto a pesos y sesgos (dE/dW = X^T * dE/dZ)
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Actualización de parámetros (Descenso por gradiente estándar)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

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

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Regla de la cadena: dE/dX = dE/dZ * h'(X)
        return output_gradient * self.activation_fn.derivative(self.input)

class VAEBottleneckLayer(Layer):
    """
    Capa de muestreo para el VAE (Reparameterization Trick).
    Espera una entrada de tamaño 2*latent_dim.
    """
    def __init__(self, latent_dim: int, kl_weight: float = 0.001):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.mu = None
        self.log_var = None
        self.epsilon = None
        self.kl_loss = 0.0

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        
        self.mu = input_data[:, :self.latent_dim]
        self.log_var = input_data[:, self.latent_dim:]
        
        self.epsilon = np.random.randn(*self.mu.shape)
        z = self.mu + np.exp(self.log_var / 2) * self.epsilon
        
        # KL loss promedio por muestra
        self.kl_loss = -0.5 * np.mean(np.sum(1 + self.log_var - np.square(self.mu) - np.exp(self.log_var), axis=1))
        
        return z

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # output_gradient es de tamaño (batch_size, latent_dim)
        dRec_dmu = output_gradient
        dRec_dlog_var = output_gradient * 0.5 * np.exp(self.log_var / 2) * self.epsilon
        
        # El BCE original está promediado por tamaño total (batch * 35).
        # Ajustamos el gradiente KL acordemente o mediante kl_weight.
        dKL_dmu = self.mu * self.kl_weight
        dKL_dlog_var = 0.5 * (np.exp(self.log_var) - 1.0) * self.kl_weight
        
        grad_mu = dRec_dmu + dKL_dmu
        grad_log_var = dRec_dlog_var + dKL_dlog_var
        
        input_gradient = np.concatenate([grad_mu, grad_log_var], axis=1)
        return input_gradient