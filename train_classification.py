import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# 1. Cargar datasets
# =========================
train_dir = "/home/anderson/cafe/classification_dataset/train"
val_dir   = "/home/anderson/cafe/classification_dataset/valid"
test_dir  = "/home/anderson/cafe/classification_dataset/test"

# Crear datasets desde las carpetas
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=16
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=16
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=16
)

# Guardar nombres de clases ANTES de normalizar
class_names = train_dataset.class_names
print("📂 Clases detectadas:", class_names)

# Normalizar imágenes a [0,1]
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset   = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset  = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# =========================
# 2. Definir modelo PavicNet-MCv2
# =========================
def PavicNet_MCv2(input_shape=(224,224,3), num_classes=4):
    model = models.Sequential()

    # Bloques convolucionales
    model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(4,4)))

    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())

    # Capas densas con Dropout
    for units in [128, 64, 32, 16]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))

    # Capa de salida dinámica según clases
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# =========================
# 3. Crear y compilar modelo
# =========================
model = PavicNet_MCv2(input_shape=(224,224,3), num_classes=len(class_names))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # etiquetas enteras
              metrics=['accuracy'])

# =========================
# 4. Entrenar modelo
# =========================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=300,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]
)

# =========================
# 5. Evaluar en test
# =========================
test_loss, test_acc = model.evaluate(test_dataset)
print(f"✅ Precisión en test: {test_acc*100:.2f}%")

# =========================
# 6. Guardar modelo
# =========================
model.save("pavicnet_mcv2.h5")
print("✅ Entrenamiento finalizado y modelo guardado en pavicnet_mcv2.h5")
