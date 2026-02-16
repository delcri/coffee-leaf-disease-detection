from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

# === Config ===
YOLO_WEIGHTS = "runs/detect/train3/weights/best.pt"   # Ajusta ruta si es diferente
CLASSIFIER   = "pavicnet_mcv2.h5"                     # Clasificador entrenado
SOURCE_DIR   = "DATA/test/images"                     # Carpeta con imágenes a procesar
OUT_DIR      = "results_batch"                        # Carpeta de salida
IMG_SIZE     = (224, 224)
CONF = 0.25
IOU = 0.45

# Cargar modelos
detector = YOLO(YOLO_WEIGHTS)
clf = tf.keras.models.load_model(CLASSIFIER)

# Clases de tu clasificador (ajústalas si cambian)
class_names = ['Roya', 'Bicho_mineiro', 'Manchas', 'Saludable']

# Crear carpeta de salida
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Procesar imágenes
images = list(Path(SOURCE_DIR).glob("*.jpg")) + list(Path(SOURCE_DIR).glob("*.png"))

for img_path in images:
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Detección con YOLO
    res = detector.predict(source=str(img_path), conf=CONF, iou=IOU, save=False, verbose=False)[0]

    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Clasificación
        crop_resized = cv2.resize(crop, IMG_SIZE)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        tensor = crop_rgb.astype(np.float32) / 255.0
        pred = clf.predict(np.expand_dims(tensor, 0), verbose=0)[0]
        cls_id = int(np.argmax(pred))
        cls_name = class_names[cls_id]
        conf_cls = float(np.max(pred))

        # Dibujar resultados
        label = f"{cls_name} {conf_cls:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,255,0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # Guardar imagen con resultados
    out_path = Path(OUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"✅ Procesada: {img_path.name} -> {out_path}")

print("\n🎉 Todas las imágenes fueron procesadas. Resultados en:", OUT_DIR)
