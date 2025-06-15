# src/core/evaluate.py

import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging
import matplotlib.pyplot as plt # Importar matplotlib
import seaborn as sns          # Importar seaborn

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.class_names = label_encoder.classes_ # Obtener nombres de las clases

    def evaluate(self, test_ds, save_path=None):
        """
        Evalúa el modelo en el dataset de prueba y genera un reporte de clasificación,
        matriz de confusión y, opcionalmente, guarda las visualizaciones.

        Args:
            test_ds: tf.data.Dataset de prueba.
            save_path (str, optional): Directorio donde se guardarán las imágenes
                                       de la matriz de confusión. Si es None, no se guardan.
        """
        logger.info("Starting model evaluation...")

        y_true = []
        y_pred = []
        
        # Iterar sobre el dataset de prueba para obtener predicciones y etiquetas verdaderas
        for images, labels in test_ds:
            predictions = self.model.predict(images, verbose=0) # verbose=0 para no imprimir cada predicción
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Generar Reporte de Clasificación
        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=2)
        logger.info(f"\n--- Classification Report ---\n{report}")

        # Generar Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\n--- Confusion Matrix ---\n{cm}")

        # Guardar la Matriz de Confusión como imagen
        if save_path:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicción')
            plt.ylabel('Etiqueta Verdadera')
            plt.title('Matriz de Confusión')
            
            # Crear el directorio si no existe
            os.makedirs(save_path, exist_ok=True)
            cm_filepath = os.path.join(save_path, 'confusion_matrix.png')
            plt.savefig(cm_filepath)
            logger.info(f"Matriz de confusión guardada en: {cm_filepath}")
            plt.close() # Cierra la figura para liberar memoria

        logger.info("Model evaluation complete.")