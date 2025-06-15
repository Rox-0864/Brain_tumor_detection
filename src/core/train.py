# src/core/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
import os # ¡Asegúrate de importar os!
from datetime import datetime # ¡Asegúrate de importar datetime!

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Clase para entrenar modelos de TensorFlow, incluyendo fases de pre-entrenamiento
    y ajuste fino (fine-tuning).
    """

    def __init__(self, model, num_classes, model_dir='models'):
        """
        Inicializa el entrenador del modelo.

        :param model: tf.keras.Model, el modelo YA CONSTRUIDO y COMPILADO que se va a entrenar.
        :param num_classes: Entero, el número de clases de salida.
        :param model_dir: String, directorio para guardar los checkpoints del modelo.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError("El 'model' proporcionado debe ser una instancia de tf.keras.Model.")
        if not hasattr(model, 'optimizer') or model.optimizer is None:
             logger.warning("El modelo no parece estar compilado. Asegúrate de compilarlo antes de pasarlo al ModelTrainer.")

        self.model = model
        self.num_classes = num_classes
        self.model_dir = model_dir
        tf.io.gfile.makedirs(self.model_dir) # Asegurarse de que el directorio exista
        self.history = None
        self.history_fine = None # Para almacenar el historial de fine-tuning

    def train(self, train_ds, valid_ds, epochs, initial_epoch=0, callbacks=None, phase_name=""):
        """
        Entrena el modelo.

        :param train_ds: tf.data.Dataset, dataset de entrenamiento.
        :param valid_ds: tf.data.Dataset, dataset de validación.
        :param epochs: Entero, número total de epochs para entrenar en esta fase.
        :param initial_epoch: Entero, epoch desde el que empezar (útil para continuar entrenamiento).
        :param callbacks: Lista de tf.keras.callbacks.Callback, callbacks adicionales.
        :param phase_name: String, nombre de la fase de entrenamiento (ej. "Pre-entrenamiento", "Fine-tuning").
        :return: History object.
        """
        if callbacks is None:
            callbacks = []

        # Checkpoint para guardar el mejor modelo basado en la precisión de validación
        # Se guarda con un nombre específico de la fase (Pretraining o Finetuning)
        checkpoint_filepath = os.path.join(self.model_dir, f"best_model_{phase_name}_epoch{{epoch:02d}}.h5")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint_callback)

        # Early Stopping para detener el entrenamiento si la precisión de validación no mejora
        early_stopping_callback = EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # Espera 10 epochs sin mejora antes de detener
            mode='max',
            verbose=1,
            restore_best_weights=True # Restaura los pesos del mejor epoch
        )
        callbacks.append(early_stopping_callback)

        # ReduceLROnPlateau para reducir la tasa de aprendizaje si la precisión de validación se estanca
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,  # Reduce la tasa de aprendizaje a un 20%
            patience=5,  # Espera 5 epochs sin mejora
            mode='max',
            min_lr=1e-7, # Tasa de aprendizaje mínima
            verbose=1
        )
        callbacks.append(reduce_lr_on_plateau)

        logger.info(f"Iniciando entrenamiento para {epochs} epochs ({phase_name}) desde epoch {initial_epoch}...")
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=valid_ds,
            callbacks=callbacks,
            verbose=2
        )
        logger.info(f"Entrenamiento de {phase_name} finalizado.")
        return history

    def compile_model_for_finetuning(self, learning_rate):
        """
        Recompila el modelo con una tasa de aprendizaje más baja para fine-tuning.
        """
        logger.info(f"Recompilando el modelo para ajuste fino con learning rate: {learning_rate}")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Ajusta si tu problema es binario (sigmoid)
            metrics=['accuracy']
        )
        logger.info("Modelo recompilado exitosamente.")

    # --- ¡NUEVO MÉTODO PARA GUARDAR EL MODELO FINAL! ---
    def save_final_model(self, filename_prefix="model"):
        """
        Guarda el estado actual del modelo en un archivo H5.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.model_dir, f"{filename_prefix}_{timestamp}_final.h5")
        self.model.save(filepath)
        logger.info(f"Modelo final guardado en: {filepath}")
        return filepath
    # ----------------------------------------------------

    # Puedes mantener los métodos de plot_training_history si quieres.
    # Los métodos train_complete y fine_tune no son necesarios aquí si main.py orquesta las fases.