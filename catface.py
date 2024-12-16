import cv2 as cv
import openFuncs as of
import numpy as np


def main():
    res: np.ndarray

    catCascade = cv.CascadeClassifier("Data/haarcascade_frontalcatface.xml")

    strayCatto = cv.imread("Photos/cats.jpg", 0)

    catRect = catCascade.detectMultiScale(strayCatto, minNeighbors=2)

    for x, y, w, h in catRect:
        res = cv.rectangle(strayCatto, (x, y), (x + w, y + h), (255, 256, 255), 5)

    of.imshow(res, "gray")


if __name__ == "__main__":
    main()
