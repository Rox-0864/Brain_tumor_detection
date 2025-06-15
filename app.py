import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow import keras
# Importar ambas funciones de preprocesamiento, el usuario elegirá cuál usar
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
import logging

# Configurar un logger básico para Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Deep Learning para Detección de Tumor Cerebral",
    layout="wide",
)

# --- CONSTANTES ---
# IMPORTANTE: ACTUALIZA ESTA RUTA A LA DE TU MEJOR MODELO GUARDADO
# Por ejemplo: "models/resnet50_final_model_20250612_080250_final.h5"
MODEL_FILENAME = "models/resnet50model_final_model_20250614_135532_final.h5" # <<-- ¡ACTUALIZA ESTO!
IMG_SIZE = (124, 124) # Tamaño de imagen con el que entrenaste tu modelo
CLASS_NAMES = ['Healthy', 'Tumor'] # Tus nombres de clase


# --- FUNCIONES ---
@st.cache_resource # Carga el modelo una sola vez para mejorar el rendimiento
def load_model(model_path):
    """
    Carga el modelo de Keras desde la ruta especificada.
    """
    if not os.path.exists(model_path):
        logger.error(f"Error: El archivo del modelo no se encontró en {model_path}")
        st.error(f"¡Oops! No se pudo encontrar el archivo del modelo en la ruta especificada: **{model_path}**.")
        st.error("Por favor, asegúrate de que el archivo `.h5` de tu modelo esté en la carpeta `models/` y de que la ruta en `MODEL_FILENAME` sea correcta.")
        return None
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Modelo cargado exitosamente desde {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo desde {model_path}: {e}")
        st.error(f"Hubo un error al cargar el modelo. Verifica que el archivo no esté corrupto. Error: {e}")
        return None

def preprocess_image(image: Image.Image, target_size: tuple, preprocess_func):
    """
    Preprocesa la imagen para que sea compatible con el modelo.
    Redimensiona, convierte a array, expande dimensiones y aplica el preprocesamiento específico del modelo.
    """
    img = image.resize(target_size)
    img_array = np.array(img).astype(np.float32) # Asegurar tipo float32
    img_array_expanded = np.expand_dims(img_array, axis=0) # Añadir dimensión de lote
    return preprocess_func(img_array_expanded)

# --- CARGA DEL MODELO UNA VEZ ---
model = load_model(MODEL_FILENAME)

# --- TÍTULO ---
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>🧠 Deep Learning para Detección de Tumor Cerebral</h2><br>",
    unsafe_allow_html=True
)

# --- DISEÑO EN 3 COLUMNAS ---
col1, col_mid, col2 = st.columns([1, 0.1, 1])

with col1:
    st.markdown("#### ⚙️ Configuración y Carga")
    
    # Selector para el tipo de preprocesamiento del modelo
    selected_preprocess_type = st.selectbox(
        "Selecciona el tipo de preprocesamiento del modelo:",
        ("ResNet50", "MobileNetV2"),
        help="Elige el preprocesamiento que corresponde al modelo que entrenaste (ResNet50 o MobileNetV2)."
    )

    # Asignar la función de preprocesamiento correcta
    if selected_preprocess_type == "ResNet50":
        selected_preprocess_func = resnet_preprocess_input
        logger.info("Preprocesamiento seleccionado: ResNet50")
    else: # MobileNetV2
        selected_preprocess_func = mobilenetv2_preprocess_input
        logger.info("Preprocesamiento seleccionado: MobileNetV2")

    uploaded_file = st.file_uploader("📤 Sube una imagen de resonancia magnética", type=["png", "jpg", "jpeg"])
    
    # Solo muestra el botón de predicción si hay un archivo cargado y el modelo cargó bien
    predict_btn = False
    if uploaded_file and model:
        predict_btn = st.button("🔍 Predecir")

with col2:
    st.markdown("#### 🖼️ Imagen y Resultado")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Imagen de RM cargada', use_container_width=True)
            logger.info("Imagen cargada y mostrada.")

            if predict_btn and model:
                with st.spinner('Realizando predicción...'):
                    # Preprocesar la imagen con la función seleccionada
                    img_preprocessed = preprocess_image(image, IMG_SIZE, selected_preprocess_func)
                    
                    # Realizar la predicción
                    prediction_probs = model.predict(img_preprocessed)
                    
                    # Interpretar la salida del modelo (asumiendo softmax con 2 clases)
                    predicted_index = np.argmax(prediction_probs[0])
                    predicted_class = CLASS_NAMES[predicted_index]
                    confidence = prediction_probs[0][predicted_index] * 100

                # Mostrar resultados
                st.markdown("---")
                st.markdown("#### 🧾 Resultado del diagnóstico")
                if predicted_class == 'Tumor':
                    st.error(f"**Predicción:** ¡Tumor detectado! ⚠️")
                else:
                    st.success(f"**Predicción:** Saludable (No Tumor) ✅")
                
                st.markdown(f"**Confianza:** `{confidence:.2f}%`")
                st.markdown(f"Probabilidades de las clases: `Healthy: {prediction_probs[0][0]:.4f}`, `Tumor: {prediction_probs[0][1]:.4f}`")
                logger.info(f"Predicción: {predicted_class} con confianza: {confidence:.2f}%")
                
        except Exception as e:
            logger.error(f"Error procesando la imagen o realizando la predicción: {e}")
            st.error(f"Ocurrió un error al procesar la imagen o al realizar la predicción. Por favor, intenta con otra imagen. Error: {e}")
    elif not uploaded_file:
        st.info("Sube una imagen de resonancia magnética para empezar.")

# --- ESTILO PARA IMAGEN ---
st.markdown("""
<style>
/* Estilo para que la imagen se adapte bien sin exceder su tamaño natural */
img {
    max-width: 100%;
    height: auto;
    object-fit: contain;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 1.1em;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}
h2 {
    color: #4CAF50;
    font-weight: bold;
}
.stMarkdown h4 {
    color: #333;
    font-size: 1.2em;
    border-bottom: 2px solid #eee;
    padding-bottom: 5px;
    margin-top: 20px;
}
.stSpinner > div > div {
    color: #4CAF50 !important;
}
</style>
""", unsafe_allow_html=True)