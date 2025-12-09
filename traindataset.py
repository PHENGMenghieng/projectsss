# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models

# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# DATASET_DIR = "dataset"

# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True
# )

# train_gen = train_datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     subset='training'
# )

# val_gen = train_datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     subset='validation'
# )

# # Build model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model.trainable = False

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(6, activation='softmax')   # 0–5
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_gen, validation_data=val_gen, epochs=10)

# # Save model
# model.save("hand_numbers_model.keras")

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping
# import os

# # ----------------------------
# # CONFIG
# # ----------------------------
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# DATASET_DIR = "dataset"   # Your dataset folder with subfolders 0-5
# NUM_CLASSES = 6
# EPOCHS = 25

# # ----------------------------
# # DATA GENERATORS
# # ----------------------------
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     brightness_range=[0.7, 1.3],
#     horizontal_flip=True
# )

# train_gen = train_datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     subset='training',
#     class_mode='categorical'
# )

# val_gen = train_datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     subset='validation',
#     class_mode='categorical'
# )

# # ----------------------------
# # MODEL
# # ----------------------------
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# # Freeze all layers except last few for fine-tuning
# base_model.trainable = True
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')  # 0–5
# ])

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # ----------------------------
# # CALLBACKS
# # ----------------------------
# early_stop = EarlyStopping(
#     monitor='val_accuracy',
#     patience=5,
#     restore_best_weights=True
# )

# # ----------------------------
# # TRAIN
# # ----------------------------
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS,
#     callbacks=[early_stop]
# )

# # ----------------------------
# # SAVE KERAS MODEL
# # ----------------------------
# model.save("hand_numbers_model.keras")
# print("✅ Keras model saved as hand_numbers_model.keras")

# # ----------------------------
# # CONVERT TO TFLITE
# # ----------------------------
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open("model_numbers.tflite", "wb") as f:
#     f.write(tflite_model)

# print("✅ TFLite model saved as model_numbers.tflite")







# another version
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_DIR = "dataset"   # Your dataset folder with subfolders 0-5
NUM_CLASSES = 6
EPOCHS = 25

# ----------------------------
# DATA GENERATORS
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# ----------------------------
# MODEL (Load if exists)
# ----------------------------
if os.path.exists("hand_numbers_model.keras"):
    print("🔄 Loading existing model to continue training...")
    model = load_model("hand_numbers_model.keras")
else:
    print("⚠️ No previous model found, creating new one...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')  # 0–5
    ])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# CALLBACKS
# ----------------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ----------------------------
# SAVE KERAS MODEL
# ----------------------------
model.save("hand_numbers_model.keras")
print("✅ Keras model saved as hand_numbers_model.keras")

# ----------------------------
# CONVERT TO TFLITE
# ----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model_numbers.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as model_numbers.tflite")
