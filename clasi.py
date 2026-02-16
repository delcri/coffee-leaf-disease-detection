import os
from ultralytics import YOLO
from PIL import Image

# Cargar el mejor modelo
model = YOLO("runs/detect/train/weights/best.pt")

# Carpeta con imágenes originales
input_dir = "/content/dpcl-bracol-for-region-detect/test/images"
output_dir = "/content/cropped_dataset"
os.makedirs(output_dir, exist_ok=True)

# Detectar y recortar regiones
for img_name in os.listdir(input_dir):
    results = model.predict(os.path.join(input_dir, img_name), conf=0.25)
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        img = Image.open(os.path.join(input_dir, img_name))
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(os.path.join(output_dir, f"{img_name}_{i}.jpg"))
