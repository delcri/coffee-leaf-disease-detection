import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Cargar modelo
model = tf.keras.models.load_model("pavicnet_mcv2.h5")

# Diccionario de clases (ajústalo a tus carpetas reales)
class_names = ["Cercospora", "Saludable", "Araña roja",  "Coffee_rust"]

# Cargar imagen de prueba
#img_path = "classification_dataset/test/Coffee__cercospora_leaf_spot/DSC_0018.jpg"  # <-- cambia a tu imagen
img_path = "/home/anderson/Downloads/1_615.jpg"  # <-- cambia a tu imagen

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predicción
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"✅ Imagen: {img_path}")
print(f"🔎 Predicción: {predicted_class} (confianza: {confidence:.2f})")
