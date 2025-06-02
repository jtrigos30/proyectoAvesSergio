import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os
from io import BytesIO

# --- Ruta y descarga del modelo desde Google Drive ---
MODEL_PATH = "model_Sergio_v2_os.keras"
DOWNLOAD_URL = "https://drive.google.com/uc?id=1RbJjbe6bWn-rXbxIwHijYoNezuB1vQl6"

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo desde Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
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
    'CHIPE ARROYERO/CHARQUERO', 'CHIPE ALAS AMARILLAS'
]

# --- Título de la aplicación ---
st.title("🦜 Clasificación de Aves con VGG16 + Keras")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    try:
        imagen = Image.open(BytesIO(archivo_imagen.read()))
        st.image(imagen, caption="Imagen cargada")

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

    except Exception as e:
        st.error(f"No se pudo cargar o procesar la imagen. Detalles: {e}")

