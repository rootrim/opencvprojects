from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2 as cv
from floatingwindow import sliding_window
from pressureThing import non_maxi_suppression
from piramyd import image_pyramid
from typing import Final
from pprint import pprint
from openFuncs import imshow


WIDTH: Final = 600
HEIGHT: Final = 600
PYR_SCALE: Final = 1.5
WIN_STEP: Final = 16
ROI_SIZE: Final = (200, 150)
INPUT_SIZE: Final = (224, 224)


def main():

    print("Model is loading...")
    model = ResNet50(weights="imagenet", include_top=True)

    orig = cv.imread("Photos/saint_bernard.jpg")
    orig = cv.resize(orig, (WIDTH, HEIGHT))
    (W, H) = orig.shape[:2]

    pyramid = image_pyramid(orig, PYR_SCALE, ROI_SIZE)

    rois = []
    locs = []

    for img in pyramid:
        scale = W / float(img.shape[1])

        for x, y, roiOrig in sliding_window(img, WIN_STEP, ROI_SIZE):

            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)

            roi = cv.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            rois.append(roi)
            locs.append((x, y, x + w, y + h))

    rois = np.array(rois, dtype="float32")

    print("Classifying...")
    preds = model.predict(rois)
    preds = imagenet_utils.decode_predictions(preds, top=1)

    labels = {}
    min_prob = 0.9

    for i, p in enumerate(preds):
        (_, label, prob) = p[0]

        if prob >= min_prob:
            box = locs[i]
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L
    pprint(labels)

    for label in labels.keys():
        clone = orig.copy()
        for box, prob in labels[label]:
            (startX, startY, endX, endY) = box
            cv.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        imshow(clone)
        clone = orig.copy()
        boxes = np.array([p[0] for p in labels[label]])
        probs = np.array([p[1] for p in labels[label]])

        boxes = non_maxi_suppression(boxes, probs)
        for startX, startY, endX, endY in boxes:
            cv.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.putText(
                clone, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
            )
        imshow(clone)


if __name__ == "__main__":
    main()
