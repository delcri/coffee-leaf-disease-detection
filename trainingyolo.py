from ultralytics import YOLO

# Cargar modelo base
model = YOLO("yolov8n.pt")

# Entrenar usando el dataset
model.train(
    data="/home/anderson/cafe/DATA/data.yaml",  # Ruta correcta
    epochs=300,
    imgsz=640,
    batch=16,
    optimizer="Adam",
    patience=20
)
0