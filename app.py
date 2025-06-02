import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import os
import requests

# --- Configuración del modelo ---
MODEL_PATH = "model_Sergio_v2_os.keras"
FILE_ID = "1RbJjbe6bWn-rXbxIwHijYoNezuB1vQl6"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# --- Función para descargar archivos grandes de Google Drive ---
def descargar_modelo_desde_drive(destination):
    st.write("Descargando el modelo desde Google Drive...")

    session = requests.Session()
    response = session.get(DOWNLOAD_URL, stream=True)
    
    # Verificar si Google requiere confirmación por tamaño del archivo
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            params = {'id': FILE_ID, 'confirm': confirm_token}
            response = session.get(DOWNLOAD_URL, params=params, stream=True)
            break

    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32 * 1024):
                if chunk:
                    f.write(chunk)
        st.write("✅ Modelo descargado correctamente.")
    else:
        st.error("❌ Error al descargar el modelo.")
        raise RuntimeError("No se pudo descargar el modelo desde Google Drive.")

# --- Carga y validación del modelo ---
@st.cache_resource
def cargar_modelo():
    st.write("Verificando si el modelo existe en ruta:", MODEL_PATH)
    
    if not os.path.exists(MODEL_PATH):
        descargar_modelo_desde_drive(MODEL_PATH)

    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        st.write(f"Tamaño del archivo del modelo: {size} bytes")
        if size < 10000:
            st.warning("⚠️ El archivo parece corrupto. Eliminando y reintentando descarga...")
            os.remove(MODEL_PATH)
            descargar_modelo_desde_drive(MODEL_PATH)

    else:
        st.error("❌ El archivo del modelo no existe tras la descarga.")
        raise FileNotFoundError("Fallo en la descarga del modelo.")

    st.write("🧠 Cargando el modelo...")
    modelo = tf.keras.models.load_model(MODEL_PATH)
    st.write("✅ Modelo cargado exitosamente.")
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

# --- Interfaz de la aplicación ---
st.title("🦜 Clasificación de Aves con VGG16 + Keras")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    try:
        imagen = Image.open(archivo_imagen)
        st.image(imagen, caption="Imagen cargada")

        imagen_preparada = preparar_imagen_vgg16(imagen)
        modelo = cargar_modelo()
        salida_predicha = modelo.predict(imagen_preparada)

        clase = int(np.argmax(salida_predicha))
        confianza = float(np.max(salida_predicha))

        st.success(f"🧠 Predicción: *{etiquetas[clase]}*")
        st.info(f"📊 Confianza del modelo: *{confianza*100:.2f}%*")

        if st.checkbox("Mostrar probabilidades por clase"):
            st.bar_chart(salida_predicha[0])

    except UnidentifiedImageError:
        st.error("❌ No se pudo cargar la imagen: formato no reconocido o archivo corrupto.")
    except Exception as e:
        st.error(f"❌ No se pudo procesar la imagen. Detalles: {e}")

