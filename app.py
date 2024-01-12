{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[],"authorship_tag":"ABX9TyOH54euFepuTWGXObVur2To"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"code","execution_count":2,"metadata":{"id":"6I1HbbuGOgiN","colab":{"base_uri":"https://localhost:8080/"},"executionInfo":{"status":"ok","timestamp":1702317425483,"user_tz":-60,"elapsed":255,"user":{"displayName":"Marcos García Rebés","userId":"18419758289050575083"}},"outputId":"52be0cab-418c-479f-931a-c5a7d52f6e20"},"outputs":[{"output_type":"stream","name":"stdout","text":["Writing app.py\n"]}],"source":["# Guarda este script como app.py\n","\n","%%writefile app.py\n","!pip install Streamlit\n","import streamlit as st\n","import tensorflow as tf\n","from PIL import Image\n","import numpy as np\n","\n","# Cargar el modelo\n","\n","\n","# Función para realizar la predicción\n","def predecir_imagen(imagen):\n","    # Preprocesar la imagen\n","    img = tf.keras.preprocessing.image.load_img(imagen, target_size=(216, 216))\n","    img_array = tf.keras.preprocessing.image.img_to_array(img)\n","    img_array = tf.expand_dims(img_array, 0)  # Añade una dimensión adicional para batch\n","    img_array /= 255.0\n","\n","    # Realizar la predicción\n","    prediccion = model.predict(img_array)\n","    return prediccion\n","\n","# Configuración de la aplicación Streamlit\n","st.title(\"Aplicación de Predicción de Imágenes\")\n","\n","# Subir una imagen\n","imagen = st.file_uploader(\"Subir una imagen\", type=[\"jpg\", \"jpeg\", \"png\"])\n","\n","if imagen is not None:\n","    # Mostrar la imagen\n","    st.image(imagen, caption=\"Imagen subida\", use_column_width=True)\n","\n","    # Realizar la predicción cuando se presiona el botón\n","    if st.button(\"Predecir\"):\n","        # Realizar la predicción\n","        resultado = predecir_imagen(imagen)\n","\n","        # Mostrar el resultado\n","        st.write(\"Resultado de la predicción:\")\n","        st.write(resultado)"]}]}