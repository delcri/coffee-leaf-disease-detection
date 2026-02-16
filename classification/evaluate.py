import tensorflow as tf
from tensorflow.keras import layers, models
import os

# -------------------
# Configuración
# -------------------
DATA_DIR = "/home/anderson/cafe/DATA"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 300

# -------------------
# Cargar datasets
# -------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Guardar nombres de clases antes de mapear
class_names = train_dataset.class_names
num_classes = len(class_names)
print("Clases detectadas:", class_names)

# Normalización de imágenes
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Prefetching para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------
# Definir PavicNet-MCv2
# -------------------
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

    # Capas densas con dropout
    for units in [128, 64, 32, 16]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))

    # Capa final
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# -------------------
# Compilar y entrenar
# -------------------
model = PavicNet_MCv2(input_shape=(224,224,3), num_classes=num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
)

# -------------------
# Guardar modelo
# -------------------
model.save("pavicnet_mcv2.h5")
print("Modelo guardado como pavicnet_mcv2.h5")

# -------------------
# Evaluar en test
# -------------------
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc*100:.2f}%")
