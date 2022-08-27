import os
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timeit
from helper_functions import *


train_img_path = r"C:\Users\Mango\Documents\Datasets\training_images\violent"
#checkpoint_filepath = r'C:\Users\Mango\Documents\Github_repo\Graffiti-CNN\checkpoints\AE\checkpoint'
checkpoint_filepath = r'D:\Github Repositories\Graffiti-CNN\checkpoints\AE\checkpoint'
test_image = r'D:\Github Repositories\Graffiti-CNN\test\images4.jpg'
#THRESHOLD = 0.37

#0.05128128045738449
#0.05128073412351329
#0.051278865198170986
#0.05127886520061225
#0.05128128049283738
#0.05276601297434547
#0.052765910094652445
#0.052765395665541975
#0.05276601304729084
#0.05276427698316248
#0.02870819494854607

# TF_GPU_ALLOCATOR=cuda_malloc_async

# limit gpu memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def load_data():
    # import and format image from folder
    # x_train, x_test = parse_images(train_img_path)
    # np.save("x_train.npy", x_train)
    # np.save("x_test.npy", x_test)


    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")

    print("training data: " + str(x_train.shape))
    print("test data: " + str(x_test.shape))

    # adjust size of the input data
    partial_x_train, temp_train = np.split(x_train, [int(len(x_train) * 0.1)], axis=0)
    partial_x_test, temp_test = np.split(x_test, [int(len(x_test) * 0.5)], axis=0)
    x_train = partial_x_train
    x_test = partial_x_test
    print("reduced training data: " + str(x_train.shape))
    print("reduced test data: " + str(x_test.shape))

    return x_train, x_test


def parse_images(path):
    images = []
    i = 0
    print("------start importing images-----")
    start_timer = timeit.default_timer()
    for file in os.listdir(path):
        ext = os.path.splitext(file)[1]
        if ext in [".jpg", ".png", ".jpeg"]:
            try:
                img = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (200, 200))
                images.append(img / 255)
            except:
                pass
        i += 1
        if i % 500 == 0:
            print("#" + str(i))

    images = np.array(images)
    x_test, x_train = np.split(images, [int(len(images) * 0.166)], axis=0)
    end_timer = timeit.default_timer()
    print("import finished, taken " + str(end_timer - start_timer) + "s")
    return x_train, x_test

class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(200, 200, 3)),
            # Block 1
            layers.Conv2D(32, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # Block 2
            layers.Conv2D(64, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # Block 3
            layers.Conv2D(64, (3, 3), strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])
        self.decoder = tf.keras.Sequential([
            # Block 1
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # Block 2
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # Block 3
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            # output
            layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# SSIM loss function
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def retrain_model(autoencoder, x_train, x_test):
    # create autoencoder model
    optimize = tf.keras.optimizers.Adam(lr=0.0005)
    autoencoder.compile(optimizer=optimize, loss=SSIMLoss)

    # create check point
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # train the model
    hist = autoencoder.fit(
        x_train, 
        x_train, 
        epochs=160, 
        batch_size=32,
        verbose=1,
        validation_data=(x_test, x_test),
        shuffle=True, 
        callbacks=[model_checkpoint_callback]
    )
    return hist

def load_checkpoint(autoencorder):
    autoencorder.load_weights(checkpoint_filepath)

def plot_hist(hist):
    # plot the recent taining history
    plt.plot(hist.history["loss"], label="Training Loss")
    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

def find_threshold(autoencoder, x_train):
    # find the threshold
    reconstructions = autoencoder.predict(x_train)
    train_loss = tf.keras.losses.mae(reconstructions, x_train)
    threshold = np.mean(train_loss) + np.std(train_loss)
    print("sd: ",  np.std(train_loss))
    print("Threshold: ", threshold)
    return threshold

def predict_image(image):
    autoencoder = AnomalyDetector()
    reconstructions = autoencoder.predict(image)
    train_loss = tf.keras.losses.mae(reconstructions, image)
    return np.mean(train_loss) >= THRESHOLD


autoencoder = AnomalyDetector()

# x_train, x_test = load_data()
# hist = retrain_model(autoencoder, x_train, x_test)
# plot_hist(hist)


load_checkpoint(autoencoder)
image = base64_to_array(image_base64_string(test_image).decode('utf-8'))
reconstructions = autoencoder.predict(image)
train_loss = tf.keras.losses.mae(reconstructions, image)
print(np.mean(train_loss))


# find_threshold(autoencoder, x_test)
