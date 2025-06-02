import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import gdown
import os

# --- ID del archivo en Google Drive y nombre local ---
MODEL_ID = "1RbJjbe6bWn-rXbxIwHijYoNezuB1vQl6"
MODEL_PATH = "model_Sergio_v2_os.keras"

@st.cache_resource
def cargar_modelo():
    st.write("🔍 Verificando si el modelo existe en ruta:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.warning("📦 Modelo no encontrado. Descargando desde Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        st.write(f"📏 Tamaño del archivo: {size} bytes")
        if size < 100000:  # Menos de 100 KB es claramente inválido
            st.warning("⚠️ El archivo parece corrupto. Eliminando y reintentando descarga...")
            os.remove(MODEL_PATH)
            with st.spinner("Reintentando descarga del modelo..."):
                gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
            st.success("✅ Modelo descargado correctamente.")
    else:
        st.error("❌ No se pudo encontrar ni descargar el modelo.")
        return None

    st.info("🧠 Cargando el modelo...")
    modelo = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Modelo cargado exitosamente.")
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

# --- Título de la app ---
st.title("🦜 Clasificación de Aves con VGG16 + Keras")
st.write("Sube una imagen de un ave para predecir su especie.")

# --- Subida de imagen ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    try:
        imagen = Image.open(archivo_imagen)
        st.image(imagen, caption="Imagen cargada")

        imagen_preparada = preparar_imagen_vgg16(imagen)
        modelo = cargar_modelo()
        if modelo is None:
            st.error("No se pudo cargar el modelo.")
        else:
            salida_predicha = modelo.predict(imagen_preparada)
            clase = int(np.argmax(salida_predicha))
            confianza = float(np.max(salida_predicha))

            st.success(f"🧠 Predicción: *{etiquetas[clase]}*")
            st.info(f"📊 Confianza: *{confianza*100:.2f}%*")

            if st.checkbox("Mostrar probabilidades por clase"):
                st.bar_chart(salida_predicha[0])

    except UnidentifiedImageError:
        st.error("❌ Imagen no válida o corrupta.")
    except Exception as e:
        st.error(f"❌ No se pudo procesar la imagen. Detalles: {e}")


