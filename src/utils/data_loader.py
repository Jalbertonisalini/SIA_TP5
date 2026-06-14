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