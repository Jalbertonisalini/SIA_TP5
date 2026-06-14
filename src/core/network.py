import numpy as np
from typing import List
from core.layers import Layer

class Network:
    """
    Orquestador de la red neuronal. Encadena las capas y maneja el flujo de información Forward y Backward.
    """
    def __init__(self):
        self.layers: List[Layer] = []

    def add(self, layer: Layer) -> None:
        """Agrega una capa secuencial a la red."""
        self.layers.append(layer)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Pasa los datos de entrada a través de todas las capas secuencialmente. La salida de una capa es la entrada de la siguiente.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss_gradient: np.ndarray, learning_rate: float) -> None:
        """
        Propaga el error hacia atrás, desde la última capa hasta la primera.
        Cada capa actualiza sus pesos internamente.
        """
        # Empezamos con el gradiente del error de la capa de salida
        gradient = loss_gradient
        
        # Recorremos las capas en orden inverso
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Alias para forward"""
        return self.forward(input_data)