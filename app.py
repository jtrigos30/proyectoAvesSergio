import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# --- Ruta y descarga del modelo desde Google Drive ---
MODEL_PATH = "model_Sergio_v2_os.keras"
FILE_ID = "1RbJjbe6bWn-rXbxIwHijYoNezuB1vQl6"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo desde Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    with st.spinner("Cargando modelo..."):
        modelo = tf.keras.models.load_model(MODEL_PATH)
    return modelo

# --- Preprocesamiento para VGG16 ---
def preparar_imagen_vgg16(imagen):
    imagen = imagen.convert("RGB")  # Asegurar que tenga 3 canales
    imagen = imagen.resize((224, 224))  # Tamaño requerido por VGG16
    matriz = np.array(imagen).astype(np.float32) / 255.0  # Normalizar
    matriz = np.expand_dims(matriz, axis=0)  # Añadir dimensión de batch
    return matriz

# --- Etiquetas del modelo ---
etiquetas = [
    'ZANATE MAYOR', 'ZANATE CARIBEÑO', 'ZANATE SP.', 'CHANGO VENTRIRROJO', 'CHANGO ORIOLINO',
    'VARILLERO CAPUCHINO', 'CHIPE ARROYERO', 'CHIPE CHARQUERO',
    'CHIPE ARROYERO/CHARQUERO', 'CHIPE ALAS AMARILLAS'
]

# --- Título de la aplicación ---
st.title("🦜 Clasificación de Aves con VGG16 + Keras")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    imagen_preparada = preparar_imagen_vgg16(imagen)

    # --- Cargar modelo e inferencia ---
    modelo = cargar_modelo()
    salida_predicha = modelo.predict(imagen_preparada)

    clase = int(np.argmax(salida_predicha))
    confianza = float(np.max(salida_predicha))

    st.success(f"🧠 Predicción: *{etiquetas[clase]}*")
    st.info(f"📊 Confianza del modelo: *{confianza*100:.2f}%*")

    # --- Visualización opcional ---
    if st.checkbox("Mostrar probabilidades por clase"):
        st.bar_chart(salida_predicha[0])
