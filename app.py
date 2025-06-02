import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# --- Ruta y descarga del modelo desde Google Drive ---
MODEL_PATH = "model_Sergio_v2_os.keras"
DOWNLOAD_URL = "https://drive.google.com/file/d/1RbJjbe6bWn-rXbxIwHijYoNezuB1vQl6/view?usp=sharing"

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo desde Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, fuzzy=True, quiet=False)
        # Verificación del tamaño del archivo
        st.write("✅ Modelo descargado. Tamaño:", os.path.getsize(MODEL_PATH), "bytes")
    with st.spinner("Cargando modelo..."):
        modelo = tf.keras.models.load_model(MODEL_PATH)
    return modelo

# --- Preprocesamiento para VGG16 ---
def preparar_imagen_vgg16(imagen):
    imagen = imagen.convert("RGB")
    imagen = imagen.resize((224, 224))
    matriz = np.array(imagen).astype(np.float32) / 255.0
    matriz = np.expand_dims(matriz, axis=0)
    return matriz

# --- Etiquetas del modelo ---
etiquetas = [
    'ZANATE MAYOR', 'ZANATE CARIBEÑO', 'ZANATE SP.', 'CHANGO VENTRIRROJO', 'CHANGO ORIOLINO',
    'VARILLERO CAPUCHINO', 'CHIPE ARROYERO', 'CHIPE CHARQUERO',
    'CHIPE ARROYERO/
