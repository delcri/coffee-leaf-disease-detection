import os
import cv2

# Rutas
base_dir = "/home/anderson/cafe/DATA"
splits = ["train", "valid", "test"]

# Clases en orden como en tu data.yaml
class_names = ["roya", "cercospora", "phoma", "bicho_mineiro"]

# Carpeta de salida
output_dir = "/home/anderson/cafe/classification_dataset"
os.makedirs(output_dir, exist_ok=True)

for split in splits:
    img_dir = os.path.join(base_dir, split, "images")
    lbl_dir = os.path.join(base_dir, split, "labels")

    for cls in class_names:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        img_file = lbl_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, lbl_file)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(lbl_path, "r") as f:
            for i, line in enumerate(f):
                cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                cls = int(cls)

                # Convertir a pixeles
                x_c, y_c, bw, bh = x_c*w, y_c*h, bw*w, bh*h
                x1, y1 = int(x_c - bw/2), int(y_c - bh/2)
                x2, y2 = int(x_c + bw/2), int(y_c + bh/2)

                # Recorte con límites seguros
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                out_path = os.path.join(output_dir, split, class_names[cls],
                                        f"{img_file.replace('.jpg','')}_{i}.jpg")
                cv2.imwrite(out_path, crop)

print("✅ Dataset de clasificación generado en:", output_dir)
