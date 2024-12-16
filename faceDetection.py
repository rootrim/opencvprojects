import cv2 as cv
import openFuncs as of
import numpy as np


def main():
    cam = cv.VideoCapture(0)
    faceCascade = cv.CascadeClassifier("Data/haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cam.read()
        thepicture = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if not ret:
            break

        faceRect = faceCascade.detectMultiScale(thepicture, minNeighbors=5)

        detectedSigma: np.ndarray

        for x, y, w, h in faceRect:
            detectedSigma = cv.rectangle(
                thepicture,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                3,
            )

        cv.imshow("Cam", detectedSigma)

        if cv.waitKey(20) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
