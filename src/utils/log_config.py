# src/utils/log_config.py

import logging
import os
from datetime import datetime

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """
    Configura el sistema de logging para el proyecto.

    :param log_dir: Directorio donde se guardarán los archivos de log.
    :param log_level: Nivel de logging mínimo a mostrar (ej. logging.INFO, logging.DEBUG).
    """
    # Asegura que el directorio de logs exista
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define el formato del log
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Obtiene la fecha y hora actual para el nombre del archivo de log
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'brain_tumor_classification_{current_time}.log')

    # Configuración del logger raíz
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename),  # Guarda los logs en un archivo
            logging.StreamHandler()             # Muestra los logs en la consola
        ]
    )

    # Opcional: Configurar un logger específico para TensorFlow para controlar su verbosidad
    # Puedes ajustar esto si los logs de TF son demasiado ruidosos o no se muestran.
    # Por lo general, os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' en main.py es suficiente.
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.ERROR) # O logging.WARNING, para ver menos mensajes de TF

    logger = logging.getLogger(__name__)
    logger.info("Configuración de logging completada.")
    logger.info(f"Los logs se guardarán en: {log_filename}")

# Ejemplo de uso (esto no se ejecuta automáticamente, es solo para demostrar)
if __name__ == '__main__':
    setup_logging()
    test_logger = logging.getLogger("TestLogger")
    test_logger.debug("Este es un mensaje de depuración.")
    test_logger.info("Este es un mensaje de información.")
    test_logger.warning("Este es un mensaje de advertencia.")
    test_logger.error("Este es un mensaje de error.")
    test_logger.critical("Este es un mensaje crítico.")
    
