import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Mapeo manual de nombres
# ---------------------------
label_mapping = {
    "Coffee__cercospora_leaf_spot": "cercospora",
    "Coffee__red_spider_mite": "araña_roja",
    "Coffee__healthy": "saludable",
    "Coffee__rust": "roya"
}

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocesa la imagen para el clasificador"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def main(args):
    # ---------------------------
    # Cargar modelos
    # ---------------------------
    print("🔁 Cargando modelo YOLO...")
    detector = YOLO("yolov8n.pt")  # o tu modelo entrenado

    print(f"🔁 Cargando modelo clasificador: {args.classifier_model}")
    classifier = load_model(args.classifier_model)

    class_names = ["cercospora", "araña_roja", "saludable", "roya"]

    # ---------------------------
    # Preparar carpetas de salida
    # ---------------------------
    crops_dir = os.path.join(args.output_dir, "crops")
    overlays_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    y_trues, y_preds = [], []

    # ---------------------------
    # Procesar imágenes
    # ---------------------------
    for root, dirs, files in os.walk(args.images_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(root, file)
            folder_name = os.path.basename(root)

            # Mapeo de la etiqueta real
            true_label = label_mapping.get(folder_name, None)

            # Cargar imagen
            img = cv2.imread(img_path)
            if img is None:
                continue

            results = detector(img)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # recorte
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_path = os.path.join(crops_dir, f"{file}")
                    cv2.imwrite(crop_path, crop)

                    # Clasificación
                    crop_array = preprocess_image(crop_path)
                    preds = classifier.predict(crop_array, verbose=0)
                    pred_class = class_names[np.argmax(preds)]

                    # Guardar overlay
                    conf = float(np.max(preds))
                    label_text = f"{pred_class} ({conf:.2f})"
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(overlay, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    overlay_path = os.path.join(overlays_dir, f"{file}")
                    cv2.imwrite(overlay_path, overlay)

                    # Guardar etiquetas para métricas
                    if true_label in class_names:
                        y_trues.append(true_label)
                        y_preds.append(pred_class)

    # ---------------------------
    # Métricas
    # ---------------------------
    if y_trues and y_preds:
        print("📊 Reporte de clasificación:")
        print(classification_report(y_trues, y_preds, target_names=class_names))

        cm = confusion_matrix(y_trues, y_preds, labels=class_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicción")
        plt.ylabel("Etiqueta real")
        plt.title("Matriz de confusión")
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        plt.close()
    else:
        print("⚠️ No se encontraron etiquetas reales válidas para calcular métricas (y_trues vacío).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="Carpeta con imágenes de test")
    parser.add_argument("--classifier_model", type=str, required=True, help="Modelo de clasificación (PavicNet .h5)")
    parser.add_argument("--output_dir", type=str, default="integration_results", help="Carpeta de salida")

    args = parser.parse_args()
    main(args)
