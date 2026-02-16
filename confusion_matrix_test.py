import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "pavicnet_mcv2.h5"
TEST_DIR = "/home/anderson/Downloads/testeo_saludables"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Ajusta según las carpetas reales de test
CLASS_NAMES = ["coffee_rust", "saludable"]

# ==============================
# CARGAR MODELO
# ==============================
model = load_model(MODEL_PATH)

# ==============================
# CARGAR DATASET DE TEST
# ==============================
test_dataset = image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    shuffle=False,
    batch_size=BATCH_SIZE
)

# ==============================
# PREDICCIONES
# ==============================
y_true = []
for _, labels in test_dataset:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

y_pred = model.predict(test_dataset, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# ==============================
# REPORTE DE CLASIFICACIÓN
# ==============================
print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))
