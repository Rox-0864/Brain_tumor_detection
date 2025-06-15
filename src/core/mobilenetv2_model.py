# src/core/mobilenetv2_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy # O BinaryCrossentropy

class MobileNetV2Model:
    def __init__(self, input_shape, num_classes, learning_rate=1e-3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.base_model = None
        self.model = None

    def build_model(self):
        """
        Construye el modelo MobileNetV2 con capas de clasificación personalizadas.
        El modelo base se inicializa como NO ENTRENABLE.
        """
        self.base_model = MobileNetV2(
            include_top=False, # No incluir las capas de clasificación de ImageNet
            weights='imagenet', # Usar pesos pre-entrenados de ImageNet
            input_shape=self.input_shape # Asegura el tamaño de entrada correcto
        )
        # --- ¡CLAVE! Congelar el modelo base al principio ---
        self.base_model.trainable = False

        inputs = layers.Input(shape=self.input_shape)
        # Usar el modelo base en modo de inferencia (training=False) para asegurar que se comporte como congelado
        x = self.base_model(inputs, training=False) 
        x = layers.GlobalAveragePooling2D()(x)
        # Puedes añadir capas Dense adicionales si lo necesitas, por ejemplo:
        # x = layers.Dense(128, activation='relu')(x)
        # x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x) # 'softmax' para múltiples clases

        self.model = Model(inputs, outputs)

        # Compilar el modelo con el optimizador y la función de pérdida.
        # Asegúrate de usar SparseCategoricalCrossentropy si tus etiquetas son enteros (0, 1, 2, ...)
        # Y BinaryCrossentropy si es un problema de 2 clases con salida sigmoid.
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=SparseCategoricalCrossentropy(), # Ajusta aquí si es problema binario
            metrics=['accuracy']
        )
        print("Modelo MobileNetV2 construido y compilado (base congelada).")
        return self.model

    def unfreeze_base_model(self, num_layers_to_unfreeze=None):
        """
        Controla si las capas del modelo base se descongelan o permanecen congeladas.
        Si num_layers_to_unfreeze es None o 0, el modelo base permanece completamente congelado.
        """
        if self.base_model is None:
            raise ValueError("El modelo base no ha sido construido. Llama a 'build_model()' primero.")

        if num_layers_to_unfreeze is None or num_layers_to_unfreeze == 0:
            # Asegura que todas las capas del modelo base permanezcan congeladas
            for layer in self.base_model.layers:
                layer.trainable = False
            print("Todas las capas del modelo base (MobileNetV2) permanecen congeladas.")
        else:
            # Este bloque NO debería activarse si quieres solo entrenar las capas superiores
            # Sin embargo, lo mantengo por si decides experimentar en el futuro con más VRAM.
            # Aquí se descongelarían las últimas 'num_layers_to_unfreeze' capas.
            num_layers = len(self.base_model.layers)
            actual_layers_to_unfreeze = min(num_layers_to_unfreeze, num_layers)

            for layer in self.base_model.layers[:-actual_layers_to_unfreeze]:
                layer.trainable = False
            for layer in self.base_model.layers[-actual_layers_to_unfreeze:]:
                layer.trainable = True
            print(f"Las últimas {actual_layers_to_unfreeze} capas de MobileNetV2 han sido descongeladas (ADVERTENCIA: esto puede causar OOM en GT 1030).")