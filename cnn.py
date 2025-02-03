import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from keras.api.models import Sequential
from keras.api.utils import to_categorical

# Define paths for the dataset
path = "usagedata/"
train_path = os.path.join(path, "Training")
validation_path = os.path.join(path, "Validation")
test_path = os.path.join(path, "Test")

# Get class names and number of classes
class_names = sorted(os.listdir(train_path))
no_of_classes = len(class_names)


def main():
    print("Program has just started".center(100, "-"))
    print("Total classes: ", no_of_classes)

    print("Loading images".center(100, "-"))
    X_train, y_train = import_images(train_path)
    X_validation, y_validation = import_images(validation_path)
    X_test, y_test = import_images(test_path)

    print("Total images in training set: ", len(X_train))
    print("Total images in validation set: ", len(X_validation))
    print("Total images in testing set: ", len(X_test))

    print("Preprocessing images".center(100, "-"))
    X_train = np.array(list(map(preProcess, X_train)))
    X_validation = np.array(list(map(preProcess, X_validation)))
    X_test = np.array(list(map(preProcess, X_test)))

    print("Preprocessing done".center(100, "-"))

    print("Data Augmentation".center(100, "-"))
    y_train = to_categorical(y_train, no_of_classes)
    y_validation = to_categorical(y_validation, no_of_classes)
    y_test = to_categorical(y_test, no_of_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (X_validation, y_validation)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare_dataset(ds, augment=False):
        ds = ds.cache()
        if augment:
            ds = ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    data_augmentation = Sequential(
        [
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
            RandomZoom(0.2),
        ]
    )

    train_ds = prepare_dataset(train_dataset, augment=True)
    validation_ds = prepare_dataset(validation_dataset)
    test_ds = prepare_dataset(test_dataset)

    print("Data Augmentation done".center(100, "-"))
    print("Building model".center(100, "-"))

    model = Sequential(
        [
            # Input layer
            Input(shape=(100, 100, 3)),
            # First convolutional layer
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Second convolutional layer
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Third convolutional layer
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Fourth convolutional layer
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Flatten layer
            Flatten(),
            # Fully connected layer
            Dense(512, activation="relu"),
            Dropout(0.5),  # Adding dropout for regularization
            # Output layer
            Dense(no_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    )

    hist = model.fit(
        train_ds,
        epochs=50,
        validation_data=validation_ds,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True,
    )

    print("Model has built".center(100, "-"))
    print("Saving model".center(100, "-"))

    model.save("myamazingmodel.keras")

    print("Model saved".center(100, "-"))
    print("Program has just ended".center(100, "-"))

    return train_ds, validation_ds, test_ds


def import_images(directory):
    images = []
    classNo = []
    class_list = sorted(os.listdir(directory))
    for label, class_name in enumerate(class_list):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv.imread(img_path)
                img = cv.resize(img, (100, 100))
                images.append(img)
                classNo.append(label)
    images = np.array(images)
    classNo = np.array(classNo)
    return images, classNo


def preProcess(img):
    img = img / 255.0
    return img


if __name__ == "__main__":
    main()
