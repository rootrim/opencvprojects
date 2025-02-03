from pprint import pprint

# from typing import Final

import cv2 as cv
import numpy as np
from keras.applications import imagenet_utils
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from openFuncs import imshow
from pressureThing import non_maxi_suppression


def main():
    model = ResNet50(weights="imagenet")
    image = cv.imread("Photos/catalouge.png")
    image = cv.resize(image, (400, 400))
    (H, W) = image.shape[:2]

    rects = selectivSearch(image)
    print(len(rects))

    proposals = []
    boxes = []

    for x, y, w, h in rects:

        if w / float(W) < 0.1 or h / float(H) < 0.1:
            continue

        roi = image[y : y + h, x : x + w]
        roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
        roi = cv.resize(roi, (224, 224))

        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        proposals.append(roi)
        boxes.append((x, y, w, h))

    proposals = np.array(proposals)

    preds = model.predict(proposals)
    preds = imagenet_utils.decode_predictions(preds, top=1)

    labels = {}
    min_prob = 0.9
    # pprint(preds)

    for i, p in enumerate(preds):
        (_, label, prob) = p[0]

        if prob >= min_prob:
            (x, y, w, h) = boxes[i]
            box = (x, y, x + w, y + h)
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L

    clone = image.copy()

    for label in labels.keys():
        for box, prob in labels[label]:
            boxes = np.array([p[0] for p in labels[label]])
            probs = np.array([p[1] for p in labels[label]])
            boxes = non_maxi_suppression(boxes, probs)

            for startX, startY, endX, endY in boxes:
                cv.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv.putText(
                    clone,
                    label,
                    (startX, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
    clone = cv.cvtColor(clone, cv.COLOR_BGR2RGB)
    imshow(clone)


def selectivSearch(img):
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects


if __name__ == "__main__":
    main()
