import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.api.utils import to_categorical
from keras.api.preprocessing.image import ImagaeDataGenerator
import pickle


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
