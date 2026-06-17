import numpy as np
import re

class FontLoader:
    """
    Se encarga de parsear el archivo font.h de C y convertirlo en 
    una matriz matemática de Numpy lista para el Autoencoder.
    """
    @staticmethod
    def load_and_flatten(filepath: str) -> np.ndarray:
        caracteres = []
        
        with open(filepath, 'r') as file:
            lineas = file.readlines()
            
        for linea in lineas:
            # Cortamos el string justo donde empieza el comentario '//' 
            # para ignorar la basura y el código hex del final de la línea.
            linea_limpia = linea.split('//')[0]
            
            # Buscamos que sea una línea de la matriz
            if '{' in linea_limpia and '0x' in linea_limpia:
                hex_values = re.findall(r'0x[0-9a-fA-F]{2}', linea_limpia)
                
                # Ahora sí, de la llave solo vamos a extraer 7 valores
                if len(hex_values) == 7:
                    caracter_aplanado = []
                    
                    for hex_val in hex_values:
                        entero = int(hex_val, 16)
                        binario = format(entero, '05b')
                        fila_pixeles = [float(bit) for bit in binario]
                        caracter_aplanado.extend(fila_pixeles)
                        
                    caracteres.append(caracter_aplanado)
                    
        return np.array(caracteres)

class FontAugmenter:
    """Genera variantes de fuentes a partir de la fuente base."""
    @staticmethod
    def generate_bold(X: np.ndarray) -> np.ndarray:
        # Engrosar la letra horizontalmente haciendo un OR con la misma letra desplazada 1 pixel a la derecha
        X_bold = np.zeros_like(X)
        for i in range(X.shape[0]):
            img = X[i].reshape(7, 5)
            img_shifted = np.roll(img, shift=1, axis=1)
            img_shifted[:, 0] = 0 # No hacer wrap-around
            bold_img = np.logical_or(img, img_shifted).astype(float)
            X_bold[i] = bold_img.flatten()
        return X_bold

    @staticmethod
    def generate_italic(X: np.ndarray) -> np.ndarray:
        # Desplazar las filas superiores a la derecha
        X_italic = np.zeros_like(X)
        for i in range(X.shape[0]):
            img = X[i].reshape(7, 5).copy()
            # Fila 0, 1 desplazadas 2 px
            img[0:2] = np.roll(img[0:2], shift=2, axis=1)
            img[0:2, 0:2] = 0
            # Fila 2, 3 desplazadas 1 px
            img[2:4] = np.roll(img[2:4], shift=1, axis=1)
            img[2:4, 0] = 0
            X_italic[i] = img.flatten()
        return X_italic

    @staticmethod
    def create_multipattern_dataset(X_base: np.ndarray):
        X_bold = FontAugmenter.generate_bold(X_base)
        X_italic = FontAugmenter.generate_italic(X_base)
        return np.vstack([X_base, X_bold, X_italic])