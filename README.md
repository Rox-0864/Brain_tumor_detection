Brain Tumor Classification with ResNet50 and MobileNetV2
Proyecto de clasificación de imágenes médicas para detectar tumores cerebrales usando modelos CNN preentrenados ResNet50 y MobileNetV2.

Contenido
src/core/load_data.py: carga y lectura de imágenes.

src/core/preprocess.py: procesamiento y creación de datasets.

src/core/train.py: definición, entrenamiento y fine-tuning de modelos.

main.py: orquestación general del pipeline.

requirements.txt: dependencias necesarias.

setup.sh: script para crear y activar entorno virtual e instalar dependencias.

Requisitos
Python 3.8 a 3.10 recomendado (compatible con TensorFlow 2.8)

GPU opcional, pero se puede ejecutar en CPU con buen rendimiento para pequeños lotes.

Linux/macOS o Windows (adaptar setup.sh o crear .bat para Windows).

Instalación
Clona el repositorio.

Ejecuta el script para preparar el entorno (Linux/macOS):

bash
Copiar
Editar
chmod +x setup.sh
./setup.sh
Para Windows, crea y activa un entorno virtual manualmente y ejecuta:

powershell
Copiar
Editar
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Uso
Para ejecutar el pipeline completo:

bash
Copiar
Editar
python main.py
Esto:

Carga y preprocesa las imágenes.

Construye y entrena los modelos (fase inicial + fine-tuning).

Guarda el mejor modelo.

Evalúa el modelo en test.

Muestra gráficos de entrenamiento.

Modelos disponibles
ResNet50: modelo principal, más pesado, mayor capacidad.

MobileNetV2: modelo ligero, optimizado para equipos con recursos limitados.

Puedes modificar en src/core/train.py qué modelo usar.

Configuración y parámetros
Tamaño de imagen por defecto: 224x224.

Batch size recomendado: 12 (puedes ajustarlo si tienes poca memoria).

Épocas iniciales y fine-tuning ajustables en main.py.

Early stopping con paciencia 3 para evitar sobreentrenamiento.

Recomendaciones para hardware limitado
Usa MobileNetV2 (trainer = ModelTrainer(model_name='mobilenetv2', ...)).

Reduce batch size a 8 o menos.

Reduce tamaño de imagen a 160x160 (modificar en DataPreprocessor).

Ejecuta menos épocas (5-7).

Asegúrate de tener las versiones compatibles de TensorFlow 2.8, Keras 2.8.

Dependencias principales (requirements.txt)
txt
Copiar
Editar
tensorflow==2.8.0
keras==2.8.0
keras-preprocessing==1.1.2
numpy
pandas
scikit-learn
matplotlib
Estructura del proyecto
css
Copiar
Editar
Brain_Tumor_Project/
├── main.py
├── requirements.txt
├── setup.sh
├── models/
│   └── (modelos guardados .h5)
├── src/
│   └── core/
│       ├── load_data.py
│       ├── preprocess.py
│       ├── train.py
│       └── utils.py  # (si tienes utilidades)
└── dataset_images/
    ├── Healthy/
    └── Tumor/
Contacto
Para dudas o mejoras, contáctame.