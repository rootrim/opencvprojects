import os
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.api.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.api.models import Sequential
from keras.api.utils import image_dataset_from_directory, to_categorical
from sklearn.model_selection import train_test_split


def main():
    pass


def load_data(path):
    images = []
    labels = []
    for label in os.listdir(path):
        for image_path in os.listdir(os.path.join(path, label)):
            image = cv.imread(os.path.join(path, label, image_path))
            image = cv.resize(image, (128, 128))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


def plot_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis("off")
    plt.show()


def preprocess_data(images, labels):
    images = images / 255.0
    labels = to_categorical(labels)
    return images, labels


if __name__ == "__main__":
    main()
