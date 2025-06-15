import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import logging

# Importa las funciones de preprocesamiento de cada modelo
# Asegúrate de que estas rutas sean correctas.
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input


AUTOTUNE = tf.data.AUTOTUNE
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, img_size=(128, 128), batch_size=1, random_state=42): # <-- Ajustado para tu GPU
        self.img_size = img_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.label_encoder = None
        self.model_preprocess_func = None # Se asignará en main.py

    def _load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        return img

    def _augment_image(self, img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        # Puedes añadir más aumentaciones aquí
        return img

    def _preprocess_input(self, img):
        if self.model_preprocess_func:
            return self.model_preprocess_func(img)
        else:
            logger.warning("No se especificó una función de preprocesamiento de modelo. Usando normalización 0-1.")
            return img / 255.0

    @tf.function
    def load_and_preprocess_image(self, image_path, label, augment=False):
        img = self._load_image(image_path)
        # tf.print("DEBUG PREPROCESS: Shape de imagen individual después de resize:", tf.shape(img), output_stream=tf.sys.stderr) # Descomenta para depurar si necesitas

        if augment:
            img = self._augment_image(img)
            # tf.print("DEBUG PREPROCESS: Shape de imagen individual después de augmentation:", tf.shape(img), output_stream=tf.sys.stderr) # Descomenta para depurar si necesitas

        img = self._preprocess_input(img)
        # tf.print("DEBUG PREPROCESS: Shape de imagen individual después de preprocess_input:", tf.shape(img), output_stream=tf.sys.stderr) # Descomenta para depurar si necesitas

        label = tf.cast(label, tf.int32)
        return img, label

    def _create_dataset(self, df, shuffle=True, augment=False):
        paths = df['image_path'].values
        labels = df['category_encoded'].values.astype(int)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        logger.debug(f"Dataset.from_tensor_slices paths shape: {paths.shape}, labels shape: {labels.shape}")

        # Pasa la función load_and_preprocess_image como método con su contexto
        ds = ds.map(lambda p, l: self.load_and_preprocess_image(p, l, augment), num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))

        ds = ds.batch(self.batch_size).prefetch(AUTOTUNE)

        # DEBUGGING: Muestra la forma de un lote para verificar
        try:
            sample_batch = next(iter(ds.take(1)))
            logger.debug(f"Shape del lote de imágenes después del batching: {sample_batch[0].shape}")
            logger.debug(f"Shape del lote de etiquetas después del batching: {sample_batch[1].shape}")
        except tf.errors.OutOfRangeError:
            logger.warning("El dataset está vacío o no se pudo tomar un lote de muestra.")
        except Exception as e:
            logger.error(f"Error al obtener lote de muestra del dataset: {e}")


        return ds

    def process_data(self, data_df):
        logger.info("Starting data preprocessing steps...")

        # 1. Codificar etiquetas
        self.label_encoder = LabelEncoder()
        data_df['category_encoded'] = self.label_encoder.fit_transform(data_df['label'])
        num_classes = len(self.label_encoder.classes_)
        logger.info(f"Detected {num_classes} classes: {self.label_encoder.classes_}")

        # 2. Dividir datos
        train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=self.random_state, stratify=data_df['category_encoded'])
        train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=self.random_state, stratify=train_df['category_encoded']) # 0.25 de 0.8 es 0.2

        logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(valid_df)}, Test samples: {len(test_df)}")

        # 3. Balancear el conjunto de entrenamiento con oversampling
        X_train = train_df['image_path'].values.reshape(-1, 1)
        y_train = train_df['category_encoded'].values
        
        oversampler = RandomOverSampler(random_state=self.random_state)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
        
        train_df_resampled = pd.DataFrame({
            'image_path': X_train_resampled.flatten(),
            'category_encoded': y_train_resampled
        })
        logger.info(f"Train samples after oversampling: {len(train_df_resampled)}")
        logger.debug(f"Class distribution after oversampling:\n{train_df_resampled['category_encoded'].value_counts()}")

        # 4. Crear datasets de TensorFlow
        # Aquí es donde el método _create_dataset se encarga de todo.
        train_ds = self._create_dataset(train_df_resampled, shuffle=True, augment=True)
        valid_ds = self._create_dataset(valid_df, shuffle=False, augment=False)
        test_ds = self._create_dataset(test_df, shuffle=False, augment=False)

        logger.info("Data preprocessing completed.")
        return train_ds, valid_ds, test_ds, self.label_encoder