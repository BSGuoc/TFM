import streamlit as st
from PIL import Image
import numpy as np
# install gdown
from os import path
import urllib.request
import requests
## from keras.models import load_model
from tensorflow.keras.models import load_model

modelo_local = 'modelo.h5'

url = "https://modelo2.s3.eu-west-3.amazonaws.com/modelo_FINAL.h5"

response = requests.get(url)
with open(modelo_local, "wb") as file:
    file.write(response.content)
    
model = load_model(modelo_local)

# Función para realizar la predicción
def predecir_imagen(imagen):
    
    # Preprocesar la imagen
    img = imagen.resize((216, 216))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    test_predict = model.predict(img_array)
    predicho = np.argmax(test_predict, axis=1)
    probabil=test_predict
    cell_dict_or = {0: "Benigno", 1: "Maligno", 2: "Normal"}
    predicho = cell_dict_or[predicho[0]]

    return predicho, probabil

# Configuración de la página


st.title("Predicción de imagen de CT")
st.header("Con esta app se pretende predecir una imágen de CT de pulmón, con el fin de predecir si es un caso maligno o benigno")
st.subheader(" Solo tienes que cargar o arrastrar una imagen y pinchar en el botón de predecir que aparecerá abajo")



# Subir una imagen
imagen = st.file_uploader("Subir una imagen", type=["jpg", "jpeg", "png"])

if imagen is not None:
    # Mostrar la imagen
    st.image(imagen, caption="Imagen subida", use_column_width=True)

    # Realizar la predicción cuando se presiona el botón
    if st.button("Predecir"):
        # Convertir la imagen de BytesIO a PIL Image
        imagen_pil = Image.open(imagen)

        # Realizar la predicción
        resultado,probabil = predecir_imagen(imagen_pil)

        # Mostrar el resultado
        st.header("Resultado de la Predicción:")
        st.write(f"Clasificación: {resultado}")
        st.write(f"Probabilidad estimada: {round(100*np.max(probabil), 2)}%")
        st.markdown("---")

