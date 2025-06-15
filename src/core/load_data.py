import os
import pandas as pd
from typing import List

class LoadData:
    """
    Clase para cargar rutas de imágenes y sus etiquetas desde un directorio estructurado por categorías.
    """

    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    def __init__(self, base_path: str, categories: List[str]):
        """
        Inicializa el cargador con el directorio base y las categorías esperadas.
        
        :param base_path: Ruta raíz donde se encuentran las carpetas de categorías.
        :param categories: Lista de nombres de carpetas que representan cada clase.
        """
        self.base_path = base_path
        self.categories = categories

    def load_images(self) -> pd.DataFrame:
        """
        Recorre las carpetas de categorías para obtener las rutas completas de las imágenes y sus etiquetas.
        
        :return: DataFrame con dos columnas: 'image_path' y 'label'.
        """
        image_paths = []
        labels = []

        for category in self.categories:
            category_path = os.path.join(self.base_path, category)
            if not os.path.isdir(category_path):
                print(f"Warning: No se encontró la carpeta para la categoría '{category}' en '{category_path}'")
                continue
            
            # Recorrer solo archivos con extensiones válidas
            files = [f for f in os.listdir(category_path) if f.lower().endswith(self.VALID_EXTENSIONS)]
            if not files:
                print(f"Warning: No se encontraron imágenes con extensiones válidas en '{category_path}'")

            for image_name in files:
                image_path = os.path.join(category_path, image_name)
                image_paths.append(image_path)
                labels.append(category)
        
        df = pd.DataFrame({
            "image_path": image_paths,
            "label": labels
        })

        return df