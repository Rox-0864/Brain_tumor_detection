# src/core/resnet50_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class ResNet50Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model = None # Inicializa como None
        self.model = None      # Inicializa como None

    def build_model(self):
        self.base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        self.base_model.trainable = False  # Congelar el modelo base

        inputs = layers.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, x)

        self.model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        # *** ASEGÚRATE DE QUE ESTA LÍNEA ESTÉ PRESENTE ***
        return self.model

    def unfreeze_base_model(self, num_layers_to_unfreeze=None):
        if self.base_model is None:
            raise ValueError("El modelo base no ha sido construido. Llama a 'build_model()' primero.")

        if num_layers_to_unfreeze is None:
            for layer in self.base_model.layers:
                layer.trainable = True
            print("Todas las capas del modelo base (ResNet50) han sido descongeladas.")
        else:
            # Asegurarse de no ir más allá del inicio de las capas
            num_layers = len(self.base_model.layers)
            actual_layers_to_unfreeze = min(num_layers_to_unfreeze, num_layers)

            for layer in self.base_model.layers[:-actual_layers_to_unfreeze]:
                layer.trainable = False
            for layer in self.base_model.layers[-actual_layers_to_unfreeze:]:
                layer.trainable = True
            print(f"Las últimas {actual_layers_to_unfreeze} capas de ResNet50 han sido descongeladas.")

# Ejemplo de uso (esto no se ejecuta automáticamente, es solo para demostrar)
if __name__ == '__main__':
    # Este bloque solo se ejecuta si corres este archivo directamente
    # Para probarlo, necesitarías un dataset
    print("Este archivo define la clase ResNet50Model.")
    print("Normalmente es importado y usado por main.py.")
    # Ejemplo muy básico
    input_shape = (224, 224, 3)
    num_classes = 4
    resnet_model = ResNet50Model(input_shape, num_classes)
    model = resnet_model.build_model()
    model.summary()