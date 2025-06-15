# /home/rosy/ProyectosDL/Brain_tumor/main.py

import os
# from datetime import datetime # No necesitas importarla aquí si ya la usas en ModelTrainer
import pandas as pd
import tensorflow as tf
from src.core.preprocess import DataPreprocessor

# --- IMPORTACIONES DE MODELOS Y PREPROCESAMIENTO ---
from src.core.resnet50_model import ResNet50Model # O MobileNetV2Model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input # O mobilenetv2_preprocess_input

from src.core.mobilenetv2_model import MobileNetV2Model # Asegúrate de que ambas estén importadas si cambias
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input


from src.core.train import ModelTrainer
from src.core.evaluate import ModelEvaluator
from src.utils.log_config import setup_logging
import logging

# Set TF logging level to suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    """
    Main function to run the brain tumor classification pipeline.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Variable para almacenar la ruta del modelo guardado
    final_model_saved_path = None # Inicializamos a None

    try:
        # --- Configuration ---
        IMG_SIZE = (124, 124) # Mantén este tamaño, o (128,128) si hay OOM en pre-entrenamiento
        BATCH_SIZE = 2        # ¡ESENCIAL! Solo 1 por lote. Cualquier cosa más grande puede fallar.
        RANDOM_STATE = 42
        BASE_DATA_DIR = 'Brain_tumor/dataset_images' # Asegúrate de que esta ruta sea correcta
        
        PRETRAINING_EPOCHS = 10 # Pocos epochs en la fase inicial (capas congeladas)
        FINETUNING_EPOCHS = 5  # 0 para no hacer fine-tuning, o el número que quieras para activarlo
        FINETUNING_LEARNING_RATE = 1e-5 # Tasa de aprendizaje mucho más baja para fine-tuning
        
        # Cuántas capas descongelar para fine-tuning.
        # Con 2GB de VRAM y ResNet50/MobileNetV2, empieza en 0.
        LAYERS_TO_UNFREEZE = 0 # <-- ¡CLAVE para tu GPU!

        # --- SELECCIÓN DEL MODELO Y SU PREPROCESAMIENTO ASOCIADO ---
        # *** DESCOMENTA SOLO UNA DE ESTAS OPCIONES ***

        # Opción 1: Usar ResNet50
        SELECTED_MODEL_CLASS = ResNet50Model
        SELECTED_PREPROCESS_FUNC = resnet_preprocess_input
        
        # Opción 2: Usar MobileNetV2
        #SELECTED_MODEL_CLASS = MobileNetV2Model
        #SELECTED_PREPROCESS_FUNC = mobilenetv2_preprocess_input
        # -----------------------------------------------------------

        logger.info("Starting brain tumor classification pipeline.")
        logger.info(f"Configuration: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
                    f"Pretraining Epochs={PRETRAINING_EPOCHS}, Finetuning Epochs={FINETUNING_EPOCHS}")
        logger.info(f"Selected Model: {SELECTED_MODEL_CLASS.__name__}")


        # --- 1. Load Data ---
        logger.info("Loading data...")
        data = []
        for label_dir in os.listdir(BASE_DATA_DIR):
            label_path = os.path.join(BASE_DATA_DIR, label_dir)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        data.append({
                            'image_path': os.path.join(label_path, img_file),
                            'label': label_dir
                        })
        data_df = pd.DataFrame(data)
        if data_df.empty:
            raise ValueError(f"No images found in {BASE_DATA_DIR}. Please check the data path and structure.")
        logger.info(f"Found {len(data_df)} images.")
        logger.debug(f"Data Head:\n{data_df.head()}")

        # --- 2. Preprocess Data ---
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(img_size=IMG_SIZE, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
        preprocessor.model_preprocess_func = SELECTED_PREPROCESS_FUNC
        train_ds, valid_ds, test_ds, label_encoder = preprocessor.process_data(data_df)
        logger.info("Data preprocessing complete.")

        # --- 3. Build Model ---
        logger.info(f"Building {SELECTED_MODEL_CLASS.__name__} model...")
        num_classes = len(label_encoder.classes_)
        model_builder = SELECTED_MODEL_CLASS(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
        
        # EL MODELO SE CONSTRUYE Y COMPILA AQUÍ (en su propia clase)
        model = model_builder.build_model() # build_model ya compila el modelo con Adam()
        
        model.summary(expand_nested=True)
        logger.info("Model built successfully.")

        # --- 4. Train Model - Phase 1: Pre-training (Frozen Base) ---
        logger.info("Starting model training - Phase 1: Pre-training (Frozen Base)...")
        
        # Pasar el modelo ya CONSTRUIDO y COMPILADO al ModelTrainer
        trainer = ModelTrainer(model, num_classes, model_dir='models')
        
        # La fase 1 de entrenamiento
        history1 = trainer.train(train_ds, valid_ds, epochs=PRETRAINING_EPOCHS, initial_epoch=0, phase_name="Pretraining")
        trainer.history = history1 # Guardar el historial de la primera fase

        # --- 5. Train Model - Phase 2: Fine-tuning (Unfrozen Base) ---
        # Solo ejecutar esta fase si se especifican epochs de fine-tuning
        if FINETUNING_EPOCHS > 0:
            logger.info("Starting model training - Phase 2: Fine-tuning (Unfrozen Base)...")
            
            # Descongelar las capas del modelo base (usa el model_builder original)
            model_builder.unfreeze_base_model(num_layers_to_unfreeze=LAYERS_TO_UNFREEZE)
            
            # Recompilar el modelo para fine-tuning con una tasa de aprendizaje más baja
            trainer.compile_model_for_finetuning(learning_rate=FINETUNING_LEARNING_RATE)
            
            # Muestra el resumen del modelo después de descongelar para verificar que las capas están entrenables
            model.summary(expand_nested=True) # Verás más capas como "trainable"

            # Entrenar la fase de fine-tuning
            history2 = trainer.train(train_ds, valid_ds, epochs=PRETRAINING_EPOCHS + FINETUNING_EPOCHS, initial_epoch=PRETRAINING_EPOCHS, phase_name="Finetuning")
            trainer.history_fine = history2 # Guardar el historial de la segunda fase
        else:
            logger.info("Fase de Fine-tuning omitida (FINETUNING_EPOCHS = 0).")

        # --- 6. Guardar el Modelo Final ---
        # El modelo actual de 'trainer.model' es el que tiene los pesos finales
        final_model_saved_path = trainer.save_final_model(filename_prefix=f"{SELECTED_MODEL_CLASS.__name__.lower()}_final_model")
        logger.info(f"Ruta del modelo guardado para evaluación independiente: {final_model_saved_path}")


        # --- 7. Evaluate Model (Después de todo el entrenamiento) ---
        logger.info("Evaluating final model...")
        evaluator = ModelEvaluator(trainer.model, label_encoder) 
        # Exporta la matriz de confusión aquí, en la misma carpeta que el modelo
        evaluator.evaluate(test_ds, save_path=os.path.dirname(final_model_saved_path))
        logger.info("Model evaluation complete.")

        # --- SECCIÓN PARA EVALUAR UN MODELO GUARDADO INDEPENDIENTEMENTE ---
        # Esta sección se puede ejecutar INDEPENDIENTEMENTE o después del entrenamiento.
        # Usa la ruta que se acaba de guardar o una ruta de un modelo ya existente.
        
        # Si 'final_model_saved_path' no se estableció (ej. si hubo un error antes de guardar),
        # puedes poner una ruta manual aquí para probar modelos ya guardados.
        # Si prefieres solo evaluarlo después del entrenamiento, puedes comentar esta sección.
        MODEL_TO_LOAD_PATH_INDEPENDENT = final_model_saved_path # Usamos la ruta del modelo que acabamos de guardar
        # Si quisieras probar un modelo ANTERIOR, cambiarías esta línea a:
        # MODEL_TO_LOAD_PATH_INDEPENDENT = "models/resnet50_final_model_20250612_080250_final.h5" # <<-- ¡EJEMPLO!

        EVAL_OUTPUT_DIR = "evaluation_results_independent" # Directorio para guardar las imágenes de esta evaluación independiente

        logger.info("\n--- Iniciando evaluación INDEPENDIENTE de un modelo cargado (opcional) ---")
        if MODEL_TO_LOAD_PATH_INDEPENDENT and os.path.exists(MODEL_TO_LOAD_PATH_INDEPENDENT):
            logger.info(f"Cargando modelo para evaluación independiente desde: {MODEL_TO_LOAD_PATH_INDEPENDENT}")
            
            # Cargar el modelo
            # custom_objects es importante si tienes capas o funciones de pérdida/métrica personalizadas
            loaded_model = tf.keras.models.load_model(MODEL_TO_LOAD_PATH_INDEPENDENT)
            
            # Re-procesar los datos para obtener el test_ds (es necesario para la evaluación)
            preprocessor_eval = DataPreprocessor(img_size=IMG_SIZE, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
            preprocessor_eval.model_preprocess_func = SELECTED_PREPROCESS_FUNC 
            # IMPORTANTE: Asegúrate de que label_encoder_eval sea el mismo que se usó en el entrenamiento.
            # Aquí lo recreamos usando data_df, que debería ser consistente.
            _ , _ , test_ds_eval, label_encoder_eval = preprocessor_eval.process_data(data_df)

            # Evaluar el modelo cargado y guardar los resultados
            logger.info("Ejecutando evaluación independiente...")
            independent_evaluator = ModelEvaluator(loaded_model, label_encoder_eval)
            independent_evaluator.evaluate(test_ds_eval, save_path=EVAL_OUTPUT_DIR)
            logger.info(f"Resultados de evaluación independiente guardados en: {EVAL_OUTPUT_DIR}")
        else:
            logger.warning(f"No se encontró el modelo en {MODEL_TO_LOAD_PATH_INDEPENDENT}. No se realizó evaluación independiente.")

        return 0

    except Exception as e:
        logger.exception(f"An exception occurred during pipeline execution: {e}")
        return 1

if __name__ == "__main__":
    status = main()
    exit(status)