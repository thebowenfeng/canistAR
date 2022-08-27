from tabnanny import verbose
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

training_dataset_path = r"C:\Users\Mango\Documents\Datasets\training_images"
validation_images_path = r"C:\Users\Mango\Documents\Datasets\validation_images"
test_images_path = r'C:\Users\Mango\Documents\Datasets\test_image'
checkpoint_filepath = r'C:\Users\Mango\Documents\Github_repo\Graffiti-CNN\checkpoints\CNN\checkpoint'

# normal_images = tf.keras.utils.image_dataset_from_directory(
#     test_images_path,
#     color_mode='rgb',
#     image_size=(200,200)
# )

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dataset_path,
    target_size=(250, 250),
    batch_size=5
)

validation_generator = validation_datagen.flow_from_directory(
    validation_images_path,
    target_size=(250, 250),
    batch_size=5,
)

# def create_CNN_model():
#     model = tf.keras.models.Sequential([
#         # First Convolution
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(250, 250, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         # Second Convolution
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         # Third Convolution
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         # Flatten
#         tf.keras.layers.Flatten(),
#         # Dense layer
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=RMSprop(lr=0.001),
#         metrics=['accuracy']
#     )
#     print("model created")
#     model.summary()
#     return model

# model = create_CNN_model()

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='max',
#     save_best_only=True,
#     verbose=1
# )


# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=5,
#     verbose=1,
#     validation_data=validation_generator,
#     callbacks=[model_checkpoint_callback]
# )


# Load and predict
# model.load_weights(checkpoint_filepath)

# result = model.predict(normal_images)
# print("results:")
# print(result)