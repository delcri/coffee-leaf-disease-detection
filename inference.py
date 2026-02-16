import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# === Configuración ===
#IMAGE_PATH = "DATA/test/images/1016_jpg.rf.6eb2a2e19fccef14b16083ca386dcbf2.jpg"  # cambia por la imagen que quieras
IMAGE_PATH ="DATA/test/images/bicho_mineiro7_jpg.rf.f38a8700e2bbedd2ea739d15e873d449.jpg"
IMG_SIZE = (224, 224)  # mismo tamaño usado en el entrenamiento
MODEL_PATH = "pavicnet_mcv2.h5"  # o "pavicnet_mcv2.keras"

# === Cargar el modelo entrenado ===
model = keras.models.load_model(MODEL_PATH)

# === Definir nombres de las clases (ajusta si son diferentes) ===
#class_names = ["clase_0", "clase_1", "clase_2", "clase_3"]  # cámbialo por tus clases reales
class_names = ["Roya", "Bicho_mineiro", "Manchas", "Saludable"]


# === Cargar y preprocesar imagen ===
img = keras.utils.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalizar [0,1]

# === Realizar predicción ===
predictions = model.predict(img_array)
pred_class = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions)

# === Mostrar resultados ===
print(f"Predicción: {class_names[pred_class]} (confianza: {confidence:.2f})")

plt.imshow(img)
plt.title(f"Predicción: {class_names[pred_class]} ({confidence:.2f})")
plt.axis("off")
plt.show()
